import os

import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gas_prediction.feature_engineering1 import add_weather_features
from gas_prediction.forecast_common1 import (
    DATE_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
)

# 旬度时间线：date 均为上/中/下旬起始日。
INFERENCE_CONTEXT_END = "2025-06-21"
BRIDGE_START = "2025-07-01"
BRIDGE_END = "2025-10-21"
TEST_START = "2025-11-01"
TEST_END = "2026-03-21"

OUTPUT_DIR = "output1"
RESIDUAL_FEATURES = ["min_temp", "HDD", "extreme_cold_days"]
CONTEXT_SEARCH_MIN = 180
CONTEXT_SEARCH_MAX = 333
CONTEXT_SEARCH_STEP = 9


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if np.any(mask):
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return float("nan")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE(%)": _mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _apply_context_window(df_asof: pd.DataFrame, context_steps: int) -> pd.DataFrame:
    if context_steps <= 0:
        return df_asof
    if len(df_asof) > context_steps:
        return df_asof.iloc[-context_steps:].copy()
    return df_asof


def _load_series_with_weather(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {DATE_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"文件缺少必要列: {sorted(missing)}")

    columns = [DATE_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days"]
    out = df[columns].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out["avg_temp"] = pd.to_numeric(out["avg_temp"], errors="coerce")
    out["max_temp"] = pd.to_numeric(out["max_temp"], errors="coerce")
    out["min_temp"] = pd.to_numeric(out["min_temp"], errors="coerce")
    out["HDD"] = pd.to_numeric(out["HDD"], errors="coerce")
    out["extreme_cold_days"] = pd.to_numeric(out["extreme_cold_days"], errors="coerce")
    out = out.dropna(subset=columns).reset_index(drop=True)
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    out = add_weather_features(out, inplace=False)
    return out


def _fit_auto_sarimax(y_train: np.ndarray, x_train: np.ndarray):
    return pm.auto_arima(
        y=y_train,
        X=x_train,
        seasonal=True,
        m=36,
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
    all_context_df = df[df[DATE_COL] <= INFERENCE_CONTEXT_END].sort_values(DATE_COL).copy()

    bridge_df = df[(df[DATE_COL] >= BRIDGE_START) & (df[DATE_COL] <= BRIDGE_END)].sort_values(DATE_COL)
    test_df = df[(df[DATE_COL] >= TEST_START) & (df[DATE_COL] <= TEST_END)].sort_values(DATE_COL)
    bridge_dates = bridge_df[DATE_COL].astype(str).tolist()
    test_dates = test_df[DATE_COL].astype(str).tolist()
    bridge_true = bridge_df[TARGET_COL].to_numpy(dtype=float)
    test_true = test_df[TARGET_COL].to_numpy(dtype=float)

    total_horizon = len(bridge_dates) + len(test_dates)
    if total_horizon <= 0:
        raise ValueError("bridge + test 预测区间为空，请检查时间切分配置。")

    forecast_dates = bridge_dates + test_dates
    future_with_feature = df[df[DATE_COL].isin(forecast_dates)].sort_values(DATE_COL).copy()
    if len(future_with_feature) != len(forecast_dates):
        missing_dates = sorted(set(forecast_dates) - set(future_with_feature[DATE_COL].astype(str)))
        raise ValueError(f"用于 SARIMAX 外生变量的未来气象特征旬度日期不完整，缺少: {missing_dates}")
    x_future = future_with_feature[RESIDUAL_FEATURES].to_numpy(dtype=float)
    n_bridge = len(bridge_dates)

    candidate_contexts = list(range(CONTEXT_SEARCH_MIN, CONTEXT_SEARCH_MAX + 1, CONTEXT_SEARCH_STEP))
    if CONTEXT_SEARCH_MAX not in candidate_contexts:
        candidate_contexts.append(CONTEXT_SEARCH_MAX)

    print(
        f"Context 搜索: {CONTEXT_SEARCH_MIN}~{CONTEXT_SEARCH_MAX} 旬，"
        f"步长={CONTEXT_SEARCH_STEP}，共 {len(candidate_contexts)} 组"
    )
    print(f"预测 horizon: {total_horizon} (bridge {len(bridge_dates)} + test {len(test_dates)})")
    print(f"SARIMAX 外生特征: {RESIDUAL_FEATURES}")

    best_run: dict[str, object] | None = None
    for context_steps in candidate_contexts:
        context_df = _apply_context_window(all_context_df, context_steps)
        if len(context_df) < 72:
            print(f"跳过 context={context_steps}旬：样本仅 {len(context_df)} 旬，少于 72 旬。")
            continue

        print(f"\n[Context候选] {context_steps}旬 | 实际 {len(context_df)} 旬")
        y_train = context_df[TARGET_COL].to_numpy(dtype=float)
        x_train = context_df[RESIDUAL_FEATURES].to_numpy(dtype=float)

        sarimax_model = _fit_auto_sarimax(y_train, x_train)
        base_forecast = np.asarray(sarimax_model.predict(n_periods=total_horizon, X=x_future), dtype=float)
        test_pred_base = base_forecast[n_bridge:]
        metrics_base = _regression_metrics(test_true.astype(float), test_pred_base.astype(float))
        chosen_label = "sarimax_only"
        chosen_mape = float(metrics_base["MAPE(%)"])
        chosen_forecast = base_forecast

        print(
            f"  MAPE(sarimax_only={metrics_base['MAPE(%)']:.4f}%) "
            f"-> 当前context得分={chosen_mape:.4f}%"
        )
        if best_run is None or chosen_mape < float(best_run["chosen_mape"]):
            best_run = {
                "context_steps": context_steps,
                "base_forecast": base_forecast,
                "chosen_forecast": chosen_forecast,
                "chosen_label": chosen_label,
                "chosen_mape": chosen_mape,
                "metrics_base": metrics_base,
            }

    if best_run is None:
        raise ValueError("context 搜索未找到可用窗口，请检查搜索范围或数据长度。")

    context_steps = int(best_run["context_steps"])
    base_forecast = np.asarray(best_run["base_forecast"], dtype=float)
    chosen_forecast = np.asarray(best_run["chosen_forecast"], dtype=float)
    chosen_label = str(best_run["chosen_label"])
    metrics_base = dict(best_run["metrics_base"])

    print(
        f"\nContext 最优窗口: {context_steps}旬 | "
        f"chosen={chosen_label} | test MAPE={float(best_run['chosen_mape']):.6f}%"
    )

    result = pd.DataFrame(
        {
            DATE_COL: forecast_dates,
            "predicted_sarimax_base": base_forecast.astype(float),
            "predicted_gas_sales": chosen_forecast.astype(float),
        }
    )
    result["aggregate_chosen_for_report"] = chosen_label
    actual_by_date: dict[str, float] = {}
    for d, v in zip(bridge_dates, bridge_true.astype(float)):
        actual_by_date[d] = float(v)
    for d, v in zip(test_dates, test_true.astype(float)):
        actual_by_date[d] = float(v)
    result["actual_gas_sales"] = result[DATE_COL].map(actual_by_date)
    result["phase"] = np.where(
        result[DATE_COL].isin(test_dates),
        f"evaluation(test_{TEST_START}_to_{TEST_END})",
        f"bridge(unknown_{BRIDGE_START}_to_{BRIDGE_END})",
    )
    result["error"] = result["predicted_gas_sales"] - result["actual_gas_sales"]
    result["abs_error"] = np.abs(result["error"])
    result["mape_pct"] = np.where(
        result["actual_gas_sales"] != 0,
        np.abs(result["error"] / result["actual_gas_sales"]) * 100.0,
        np.nan,
    )
    metrics_df = pd.DataFrame([metrics_base])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ctx_suffix = "full" if context_steps <= 0 else f"{context_steps}t"
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{PROCESSED_PROVINCE}_sarimax_ctx{ctx_suffix}_{INFERENCE_CONTEXT_END}_{total_horizon}t.xlsx",
    )
    with pd.ExcelWriter(out_path) as writer:
        result.to_excel(writer, index=False, sheet_name="forecast")
        metrics_df.to_excel(writer, index=False, sheet_name="metrics")

    print("-" * 50)
    print(f"input: {input_path}")
    print(f"saved: {out_path}")
    print(f"测试集 ({TEST_START} ~ {TEST_END}) 对比（仅 test 段）:")
    for name, value in metrics_base.items():
        print(f"  {name}: {value:.6f}")
    print(
        f"  → 最终报告采用: {chosen_label}；predicted_gas_sales 列已写入该结果"
    )


if __name__ == "__main__":
    main()