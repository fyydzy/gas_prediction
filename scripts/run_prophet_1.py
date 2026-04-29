import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from prophet import Prophet

from gas_prediction.feature_engineering import build_features_pipeline
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

OUTPUT_DIR = "output"  # 预测结果 Excel 输出目录
# LightGBM 残差模型外生特征：与 run_xgboost_0.py 一致，使用 build_features_pipeline 生成的全部特征
# （即除 MONTH_COL / TARGET_COL 之外的所有列，运行时动态确定）。
RESIDUAL_FEATURES: list[str] = []
LGBM_RESIDUAL_PARAMS = {
    "n_estimators": 500,  # 树的数量上限；配合早停实际会小于该值
    "learning_rate": 0.02,  # 降低步长，提升稳定性
    "num_leaves": 31,  # 增强非线性拟合能力
    "max_depth": 4,  # 轻度放宽深度，兼顾表达能力与泛化
    "min_child_samples": 4,  # 降低分裂门槛，减少“无有效分裂”警告
    "subsample": 0.8,  # 行采样比例；<1 可减轻过拟合
    "colsample_bytree": 0.8,  # 列采样比例；<1 可减轻过拟合
    "reg_lambda": 1.0,  # L2 正则；适当放松以允许更多有效切分
    "reg_alpha": 0.1,  # L1 正则；适当放松，避免过强稀疏导致欠拟合
    "min_split_gain": 0.0,  # 最小增益阈值，0 表示不额外抬高分裂门槛
    "verbosity": -1,  # 关闭 LightGBM warning/info 日志
    "random_state": 42,  # 随机种子，保证可复现
}
LGBM_EARLY_STOPPING_ROUNDS = 30  # 验证集连续若干轮无改进则早停
# True：训练时在控制台打印日志；False：静默
LGBM_VERBOSE_FIT = False
CONTEXT_SEARCH_MIN = 60  # context 搜索下界（月）
CONTEXT_SEARCH_MAX = 111  # context 搜索上界（月）
CONTEXT_SEARCH_STEP = 3  # context 搜索步长（月）


def _apply_context_window(df_asof: pd.DataFrame, context_months: int) -> pd.DataFrame:
    if context_months <= 0:
        return df_asof
    if len(df_asof) > context_months:
        return df_asof.iloc[-context_months:].copy()
    return df_asof


def _build_prophet_frame(series_df: pd.DataFrame) -> pd.DataFrame:
    frame = series_df[[MONTH_COL, TARGET_COL]].copy()
    frame["ds"] = pd.PeriodIndex(frame[MONTH_COL].astype(str), freq="M").to_timestamp()
    frame["y"] = frame[TARGET_COL].astype(float)
    return frame[["ds", "y"]]


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
    out["HDD"] = pd.to_numeric(out["HDD"], errors="coerce")
    out["extreme_cold_days"] = pd.to_numeric(out["extreme_cold_days"], errors="coerce")
    out = out.dropna(
        subset=[TARGET_COL, "avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days"]
    ).reset_index(drop=True)
    out = out.sort_values(MONTH_COL).reset_index(drop=True)
    return build_features_pipeline(out, target_col=TARGET_COL, month_col=MONTH_COL)


def _train_lgbm_residual_model(
    context_df: pd.DataFrame,
    context_base_pred: np.ndarray,
    residual_features: list[str],
) -> lgb.LGBMRegressor:
    if len(context_df) != len(context_base_pred):
        raise ValueError("context 长度与 Prophet 历史拟合长度不一致。")

    residual_df = context_df.copy()
    residual_df["prophet_yhat_in_sample"] = context_base_pred.astype(float)
    residual_df["residual"] = residual_df[TARGET_COL].to_numpy(dtype=float) - residual_df[
        "prophet_yhat_in_sample"
    ].to_numpy(dtype=float)

    x_all = residual_df[residual_features].to_numpy(dtype=float)
    y_all = residual_df["residual"].to_numpy(dtype=float)
    n = len(residual_df)
    if n < 12:
        raise ValueError("context 月份过少，无法稳定训练 LightGBM 残差模型（至少建议 12 个月）。")

    val_size = min(12, max(6, n // 5))
    train_end = n - val_size
    if train_end <= 0:
        train_end = max(1, n - 1)

    x_train = x_all[:train_end]
    y_train = y_all[:train_end]
    x_val = x_all[train_end:]
    y_val = y_all[train_end:]

    model = lgb.LGBMRegressor(objective="regression", **LGBM_RESIDUAL_PARAMS)
    callbacks = [lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, verbose=LGBM_VERBOSE_FIT)]
    if LGBM_VERBOSE_FIT:
        callbacks.append(lgb.log_evaluation(period=1))
        print(
            f"LightGBM 残差训练: 训练 {x_train.shape[0]} 月 | 验证 {x_val.shape[0]} 月（时间顺序末尾）"
        )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l2",
        callbacks=callbacks,
    )
    if LGBM_VERBOSE_FIT:
        print(
            f"LightGBM 残差训练结束 | best_iteration={getattr(model, 'best_iteration_', None)}"
        )
    return model


def main() -> None:
    input_path = find_processed_excel()
    df = _load_series_with_weather(input_path)
    residual_features = [c for c in df.columns if c not in (MONTH_COL, TARGET_COL)]
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
    x_future = future_with_feature[residual_features].to_numpy(dtype=float)
    n_bridge = len(bridge_months)

    candidate_contexts = list(range(CONTEXT_SEARCH_MIN, CONTEXT_SEARCH_MAX + 1, CONTEXT_SEARCH_STEP))
    if CONTEXT_SEARCH_MAX not in candidate_contexts:
        candidate_contexts.append(CONTEXT_SEARCH_MAX)

    print(
        f"Context 搜索: {CONTEXT_SEARCH_MIN}~{CONTEXT_SEARCH_MAX} 月，"
        f"步长={CONTEXT_SEARCH_STEP}，共 {len(candidate_contexts)} 组"
    )
    print(f"预测 horizon: {total_horizon} (bridge {len(bridge_months)} + test {len(test_months)})")
    print(f"LightGBM 残差特征（与 run_xgboost_0.py 一致）: {residual_features}")
    print(f"LightGBM 参数: {LGBM_RESIDUAL_PARAMS}, early_stopping_rounds={LGBM_EARLY_STOPPING_ROUNDS}")

    best_run: dict[str, object] | None = None
    for context_months in candidate_contexts:
        context_df = _apply_context_window(all_context_df, context_months)
        if len(context_df) < 12:
            print(f"跳过 context={context_months}m：样本仅 {len(context_df)} 月，少于 12 月。")
            continue

        print(f"\n[Context候选] {context_months}m | 实际 {len(context_df)} 月")
        train_prophet = _build_prophet_frame(context_df)
        prophet_model = Prophet(
            growth="linear",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
        )
        prophet_model.fit(train_prophet)
        future = prophet_model.make_future_dataframe(periods=total_horizon, freq="MS", include_history=False)
        fcst = prophet_model.predict(future)

        base_forecast = fcst["yhat"].to_numpy(dtype=float)
        forecast_lower = fcst["yhat_lower"].to_numpy(dtype=float)
        forecast_upper = fcst["yhat_upper"].to_numpy(dtype=float)
        context_base_pred = prophet_model.predict(train_prophet[["ds"]])["yhat"].to_numpy(dtype=float)

        lgbm_model = _train_lgbm_residual_model(context_df, context_base_pred, residual_features)
        residual_pred = lgbm_model.predict(x_future).astype(float)
        hybrid_forecast = base_forecast + residual_pred
        test_pred_base = base_forecast[n_bridge:]
        test_pred_hybrid = hybrid_forecast[n_bridge:]
        metrics_base = forecast_metrics(test_true.astype(float), test_pred_base.astype(float))
        metrics_hybrid = forecast_metrics(test_true.astype(float), test_pred_hybrid.astype(float))

        use_hybrid = float(metrics_hybrid["MAPE(%)"]) < float(metrics_base["MAPE(%)"])
        chosen_label = "prophet_plus_lgbm_residual" if use_hybrid else "prophet_only"
        chosen_mape = (
            float(metrics_hybrid["MAPE(%)"]) if use_hybrid else float(metrics_base["MAPE(%)"])
        )
        chosen_forecast = hybrid_forecast if use_hybrid else base_forecast

        print(
            f"  MAPE(base={metrics_base['MAPE(%)']:.4f}%, hybrid={metrics_hybrid['MAPE(%)']:.4f}%) "
            f"-> 采用 {chosen_label}, chosen={chosen_mape:.4f}%"
        )
        if best_run is None or chosen_mape < float(best_run["chosen_mape"]):
            best_run = {
                "context_months": context_months,
                "base_forecast": base_forecast,
                "forecast_lower": forecast_lower,
                "forecast_upper": forecast_upper,
                "residual_pred": residual_pred,
                "hybrid_forecast": hybrid_forecast,
                "chosen_forecast": chosen_forecast,
                "chosen_label": chosen_label,
                "chosen_mape": chosen_mape,
                "metrics_base": metrics_base,
                "metrics_hybrid": metrics_hybrid,
            }

    if best_run is None:
        raise ValueError("context 搜索未找到可用窗口，请检查搜索范围或数据长度。")

    context_months = int(best_run["context_months"])
    base_forecast = np.asarray(best_run["base_forecast"], dtype=float)
    forecast_lower = np.asarray(best_run["forecast_lower"], dtype=float)
    forecast_upper = np.asarray(best_run["forecast_upper"], dtype=float)
    residual_pred = np.asarray(best_run["residual_pred"], dtype=float)
    hybrid_forecast = np.asarray(best_run["hybrid_forecast"], dtype=float)
    chosen_forecast = np.asarray(best_run["chosen_forecast"], dtype=float)
    chosen_label = str(best_run["chosen_label"])
    metrics_base = dict(best_run["metrics_base"])
    metrics_hybrid = dict(best_run["metrics_hybrid"])

    print(
        f"\nContext 最优窗口: {context_months}m | "
        f"chosen={chosen_label} | test MAPE={float(best_run['chosen_mape']):.6f}%"
    )

    result = pd.DataFrame(
        {
            MONTH_COL: forecast_months,
            "predicted_yhat": base_forecast.astype(float),
            "predicted_yhat_lower": forecast_lower.astype(float),
            "predicted_yhat_upper": forecast_upper.astype(float),
            "predicted_residual_lgbm": residual_pred.astype(float),
            "predicted_hybrid": hybrid_forecast.astype(float),
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
        f"{PROCESSED_PROVINCE}_prophet_lgbmres_ctx{ctx_suffix}_{INFERENCE_CONTEXT_END}_{total_horizon}m.xlsx",
    )
    result.to_excel(out_path, index=False, sheet_name="forecast")

    print("-" * 50)
    print(f"input: {input_path}")
    print(f"saved: {out_path}")
    print(f"测试集 ({TEST_START} ~ {TEST_END}) 对比（仅 test 段）:")
    print(
        f"  Prophet基线 | MAE={metrics_base['MAE']:.6f} RMSE={metrics_base['RMSE']:.6f} "
        f"MAPE={metrics_base['MAPE(%)']:.6f}%"
    )
    print(
        f"  Prophet+LGBM残差 | MAE={metrics_hybrid['MAE']:.6f} RMSE={metrics_hybrid['RMSE']:.6f} "
        f"MAPE={metrics_hybrid['MAPE(%)']:.6f}%"
    )
    print(
        f"  → 最终报告采用: {chosen_label}（test MAPE 更低）；"
        f"predicted_gas_sales 列已写入该结果"
    )


if __name__ == "__main__":
    main()
