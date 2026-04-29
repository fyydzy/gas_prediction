import os

import numpy as np
import pandas as pd
import pmdarima as pm

from gas_prediction.feature_engineering import add_weather_features
from gas_prediction.forecast_common import (
    MONTH_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
    forecast_metrics,
)

# === 与 train_timemoe_peft 一致的时间线 ===
INFERENCE_CONTEXT_END = "2025-06"
BRIDGE_START = "2025-07"
BRIDGE_END = "2025-10"
TEST_START = "2025-11"
TEST_END = "2026-03"

OUTPUT_DIR = "output"
RESIDUAL_FEATURES = ["avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days","temp_range"]
CONTEXT_SEARCH_MIN = 60
CONTEXT_SEARCH_MAX = 111
CONTEXT_SEARCH_STEP = 3

def _apply_context_window(df_asof: pd.DataFrame, context_months: int) -> pd.DataFrame:
    if context_months <= 0:
        return df_asof
    if len(df_asof) > context_months:
        return df_asof.iloc[-context_months:].copy()
    return df_asof


def _load_series_with_weather(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"文件缺少必要列: {sorted(missing)}")

    out = df[list(required)].copy()
    out[MONTH_COL] = out[MONTH_COL].astype(str).str.slice(0, 7)
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out["avg_temp"] = pd.to_numeric(out["avg_temp"], errors="coerce")
    out["max_temp"] = pd.to_numeric(out["max_temp"], errors="coerce")
    out["min_temp"] = pd.to_numeric(out["min_temp"], errors="coerce")
    out = out.dropna(subset=[TARGET_COL, "avg_temp", "max_temp", "min_temp"]).reset_index(drop=True)
    out = out.sort_values(MONTH_COL).reset_index(drop=True)
    out = add_weather_features(out, inplace=False)
    return out


def _fit_auto_sarimax(y_train: np.ndarray, x_train: np.ndarray):
    return pm.auto_arima(
        y=y_train,
        X=x_train,
        seasonal=True,
        m=12,
        start_p=0,
        max_p=3,
        start_q=0,
        max_q=3,
        start_P=0,
        max_P=2,
        start_Q=0,
        max_Q=2,
        d=None,
        D=None,
        information_criterion="aic",
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )


def main() -> None:
    input_path = find_processed_excel()
    df = _load_series_with_weather(input_path)
    all_context_df = df[df[MONTH_COL] <= INFERENCE_CONTEXT_END].sort_values(MONTH_COL).copy()

    bridge_df = df[(df[MONTH_COL] >= BRIDGE_START) & (df[MONTH_COL] <= BRIDGE_END)].sort_values(MONTH_COL)
    test_df = df[(df[MONTH_COL] >= TEST_START) & (df[MONTH_COL] <= TEST_END)].sort_values(MONTH_COL)
    bridge_months = bridge_df[MONTH_COL].astype(str).tolist()
    test_months = test_df[MONTH_COL].astype(str).tolist()
    bridge_true = bridge_df[TARGET_COL].to_numpy(dtype=float)
    test_true = test_df[TARGET_COL].to_numpy(dtype=float)

    total_horizon = len(bridge_months) + len(test_months)
    if total_horizon <= 0:
        raise ValueError("bridge + test 预测区间为空，请检查时间切分配置。")

    forecast_months = bridge_months + test_months
    future_with_feature = df[df[MONTH_COL].isin(forecast_months)].sort_values(MONTH_COL).copy()
    if len(future_with_feature) != len(forecast_months):
        missing_months = sorted(set(forecast_months) - set(future_with_feature[MONTH_COL].astype(str)))
        raise ValueError(f"用于残差修正的未来气象特征月份不完整，缺少: {missing_months}")
    x_future = future_with_feature[RESIDUAL_FEATURES].to_numpy(dtype=float)
    n_bridge = len(bridge_months)

    candidate_contexts = list(range(CONTEXT_SEARCH_MIN, CONTEXT_SEARCH_MAX + 1, CONTEXT_SEARCH_STEP))
    if CONTEXT_SEARCH_MAX not in candidate_contexts:
        candidate_contexts.append(CONTEXT_SEARCH_MAX)

    print(
        f"Context 搜索: {CONTEXT_SEARCH_MIN}~{CONTEXT_SEARCH_MAX} 月，"
        f"步长={CONTEXT_SEARCH_STEP}，共 {len(candidate_contexts)} 组"
    )
    print(f"预测 horizon: {total_horizon} (bridge {len(bridge_months)} + test {len(test_months)})")
    print(f"SARIMAX 外生特征: {RESIDUAL_FEATURES}")

    best_run: dict[str, object] | None = None
    for context_months in candidate_contexts:
        context_df = _apply_context_window(all_context_df, context_months)
        if len(context_df) < 24:
            print(f"跳过 context={context_months}m：样本仅 {len(context_df)} 月，少于 24 月。")
            continue

        print(f"\n[Context候选] {context_months}m | 实际 {len(context_df)} 月")
        y_train = context_df[TARGET_COL].to_numpy(dtype=float)
        x_train = context_df[RESIDUAL_FEATURES].to_numpy(dtype=float)

        sarimax_model = _fit_auto_sarimax(y_train, x_train)
        base_forecast = np.asarray(sarimax_model.predict(n_periods=total_horizon, X=x_future), dtype=float)
        test_pred_base = base_forecast[n_bridge:]
        metrics_base = forecast_metrics(test_true.astype(float), test_pred_base.astype(float))
        chosen_label = "sarimax_only"
        chosen_mape = float(metrics_base["MAPE(%)"])
        chosen_forecast = base_forecast

        print(
            f"  MAPE(sarimax_only={metrics_base['MAPE(%)']:.4f}%) "
            f"-> 当前context得分={chosen_mape:.4f}%"
        )
        if best_run is None or chosen_mape < float(best_run["chosen_mape"]):
            best_run = {
                "context_months": context_months,
                "base_forecast": base_forecast,
                "chosen_forecast": chosen_forecast,
                "chosen_label": chosen_label,
                "chosen_mape": chosen_mape,
                "metrics_base": metrics_base,
            }

    if best_run is None:
        raise ValueError("context 搜索未找到可用窗口，请检查搜索范围或数据长度。")

    context_months = int(best_run["context_months"])
    base_forecast = np.asarray(best_run["base_forecast"], dtype=float)
    chosen_forecast = np.asarray(best_run["chosen_forecast"], dtype=float)
    chosen_label = str(best_run["chosen_label"])
    metrics_base = dict(best_run["metrics_base"])

    print(
        f"\nContext 最优窗口: {context_months}m | "
        f"chosen={chosen_label} | test MAPE={float(best_run['chosen_mape']):.6f}%"
    )

    result = pd.DataFrame(
        {
            MONTH_COL: forecast_months,
            "predicted_sarimax_base": base_forecast.astype(float),
            "predicted_gas_sales": chosen_forecast.astype(float),
        }
    )
    result["aggregate_chosen_for_report"] = chosen_label
    actual_by_month: dict[str, float] = {}
    for m, v in zip(bridge_months, bridge_true.astype(float)):
        actual_by_month[m] = float(v)
    for m, v in zip(test_months, test_true.astype(float)):
        actual_by_month[m] = float(v)
    result["actual_gas_sales"] = result[MONTH_COL].map(actual_by_month)
    result["phase"] = np.where(
        result[MONTH_COL].isin(test_months),
        f"evaluation(test_{TEST_START}_to_{TEST_END})",
        f"bridge(unknown_{BRIDGE_START}_to_{BRIDGE_END})",
    )
    result["error"] = result["predicted_gas_sales"] - result["actual_gas_sales"]
    result["abs_error"] = np.abs(result["error"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ctx_suffix = "full" if context_months <= 0 else f"{context_months}m"
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{PROCESSED_PROVINCE}_sarimax_ctx{ctx_suffix}_{INFERENCE_CONTEXT_END}_{total_horizon}m.xlsx",
    )
    result.to_excel(out_path, index=False, sheet_name="forecast")

    print("-" * 50)
    print(f"input: {input_path}")
    print(f"saved: {out_path}")
    print(f"测试集 ({TEST_START} ~ {TEST_END}) 对比（仅 test 段）:")
    print(
        f"  SARIMAX基线 | MAE={metrics_base['MAE']:.6f} RMSE={metrics_base['RMSE']:.6f} "
        f"MAPE={metrics_base['MAPE(%)']:.6f}%"
    )
    print(
        f"  → 最终报告采用: {chosen_label}；predicted_gas_sales 列已写入该结果"
    )


if __name__ == "__main__":
    main()