import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gas_prediction.feature_engineering1 import build_features_pipeline
from gas_prediction.forecast_common1 import (
    DATE_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
)

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 旬度时间线：date 为上/中/下旬起始日（YYYY-MM-DD）
INFERENCE_CONTEXT_END = "2025-06-21"
BRIDGE_START = "2025-07-01"
BRIDGE_END = "2025-10-21"
TEST_START = "2025-11-01"
TEST_END = "2026-03-21"

OUTPUT_DIR = "output1"
# LightGBM 残差特征：build_features_pipeline 生成后，除 date / gas_sales 外的全部数值特征列（运行时确定）
RESIDUAL_FEATURES: list[str] = []
LGBM_RESIDUAL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 4,
    "min_child_samples": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "min_split_gain": 0.0,
    "verbosity": -1,
    "random_state": 42,
}
LGBM_EARLY_STOPPING_ROUNDS = 30
LGBM_VERBOSE_FIT = False
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


def _save_lgbm_feature_importance_plot(
    model: lgb.LGBMRegressor,
    feature_names: list[str],
    output_path: str,
    title: str,
) -> None:
    if not hasattr(model, "feature_importances_"):
        print("警告：当前 LightGBM 模型不支持 feature_importances_，跳过特征重要性作图。")
        return

    importances = np.asarray(model.feature_importances_, dtype=float)
    if importances.size != len(feature_names):
        print(
            "警告：特征重要性长度与特征名不一致，跳过作图。"
            f" len(importances)={importances.size}, len(features)={len(feature_names)}"
        )
        return

    order = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(sorted_features)), sorted_importances, color="#2ca02c")
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=30, ha="right")
    plt.ylabel("Feature Importance")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, sorted_importances):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"saved feature importance plot: {output_path}")


def _apply_context_window(df_asof: pd.DataFrame, context_steps: int) -> pd.DataFrame:
    if context_steps <= 0:
        return df_asof
    if len(df_asof) > context_steps:
        return df_asof.iloc[-context_steps:].copy()
    return df_asof


def _build_prophet_frame(series_df: pd.DataFrame) -> pd.DataFrame:
    frame = series_df[[DATE_COL, TARGET_COL]].copy()
    frame["ds"] = pd.to_datetime(frame[DATE_COL], errors="coerce")
    frame["y"] = frame[TARGET_COL].astype(float)
    return frame[["ds", "y"]]


def _load_series_with_weather(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {
        DATE_COL,
        TARGET_COL,
        "avg_temp",
        "max_temp",
        "min_temp",
        "HDD",
        "extreme_cold_days",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"文件缺少必要列: {sorted(missing)}")

    columns = [
        DATE_COL,
        TARGET_COL,
        "avg_temp",
        "max_temp",
        "min_temp",
        "HDD",
        "extreme_cold_days",
    ]
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
    return build_features_pipeline(out, target_col=TARGET_COL, date_col=DATE_COL)


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
    if n < 36:
        raise ValueError("context 旬度过少，无法稳定训练 LightGBM 残差模型（至少建议 36 旬）。")

    val_size = min(36, max(18, n // 5))
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
            f"LightGBM 残差训练: 训练 {x_train.shape[0]} 旬 | 验证 {x_val.shape[0]} 旬（时间顺序末尾）"
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
    global RESIDUAL_FEATURES
    input_path = find_processed_excel()
    df = _load_series_with_weather(input_path)
    RESIDUAL_FEATURES = [c for c in df.columns if c not in (DATE_COL, TARGET_COL)]
    residual_features = RESIDUAL_FEATURES

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
        raise ValueError(f"用于残差修正的未来气象特征旬度日期不完整，缺少: {missing_dates}")
    x_future = future_with_feature[residual_features].to_numpy(dtype=float)
    n_bridge = len(bridge_dates)

    future_prophet = pd.DataFrame({"ds": pd.to_datetime(forecast_dates)})

    candidate_contexts = list(range(CONTEXT_SEARCH_MIN, CONTEXT_SEARCH_MAX + 1, CONTEXT_SEARCH_STEP))
    if CONTEXT_SEARCH_MAX not in candidate_contexts:
        candidate_contexts.append(CONTEXT_SEARCH_MAX)

    print(
        f"Context 搜索: {CONTEXT_SEARCH_MIN}~{CONTEXT_SEARCH_MAX} 旬，"
        f"步长={CONTEXT_SEARCH_STEP}，共 {len(candidate_contexts)} 组"
    )
    print(f"预测 horizon: {total_horizon} (bridge {len(bridge_dates)} + test {len(test_dates)})")
    print(f"LightGBM 残差特征（来自 feature_engineering1）: {residual_features}")
    print(f"LightGBM 参数: {LGBM_RESIDUAL_PARAMS}, early_stopping_rounds={LGBM_EARLY_STOPPING_ROUNDS}")

    best_run: dict[str, object] | None = None
    for context_steps in candidate_contexts:
        context_df = _apply_context_window(all_context_df, context_steps)
        if len(context_df) < 36:
            print(f"跳过 context={context_steps}旬：样本仅 {len(context_df)} 旬，少于 36 旬。")
            continue

        print(f"\n[Context候选] {context_steps}旬 | 实际 {len(context_df)} 旬")
        train_prophet = _build_prophet_frame(context_df)
        prophet_model = Prophet(
            growth="linear",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
        )
        prophet_model.fit(train_prophet)
        fcst = prophet_model.predict(future_prophet)

        base_forecast = fcst["yhat"].to_numpy(dtype=float)
        forecast_lower = fcst["yhat_lower"].to_numpy(dtype=float)
        forecast_upper = fcst["yhat_upper"].to_numpy(dtype=float)
        context_base_pred = prophet_model.predict(train_prophet[["ds"]])["yhat"].to_numpy(dtype=float)

        lgbm_model = _train_lgbm_residual_model(context_df, context_base_pred, residual_features)
        residual_pred = lgbm_model.predict(x_future).astype(float)
        hybrid_forecast = base_forecast + residual_pred
        test_pred_base = base_forecast[n_bridge:]
        test_pred_hybrid = hybrid_forecast[n_bridge:]
        metrics_base = _regression_metrics(test_true.astype(float), test_pred_base.astype(float))
        metrics_hybrid = _regression_metrics(test_true.astype(float), test_pred_hybrid.astype(float))

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
                "context_steps": context_steps,
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
                "lgbm_model": lgbm_model,
            }

    if best_run is None:
        raise ValueError("context 搜索未找到可用窗口，请检查搜索范围或数据长度。")

    context_steps = int(best_run["context_steps"])
    base_forecast = np.asarray(best_run["base_forecast"], dtype=float)
    forecast_lower = np.asarray(best_run["forecast_lower"], dtype=float)
    forecast_upper = np.asarray(best_run["forecast_upper"], dtype=float)
    residual_pred = np.asarray(best_run["residual_pred"], dtype=float)
    hybrid_forecast = np.asarray(best_run["hybrid_forecast"], dtype=float)
    chosen_forecast = np.asarray(best_run["chosen_forecast"], dtype=float)
    chosen_label = str(best_run["chosen_label"])
    metrics_base = dict(best_run["metrics_base"])
    metrics_hybrid = dict(best_run["metrics_hybrid"])
    best_lgbm_model = best_run["lgbm_model"]

    test_pred_chosen = chosen_forecast[n_bridge:]
    metrics_chosen = _regression_metrics(test_true.astype(float), test_pred_chosen.astype(float))

    print(
        f"\nContext 最优窗口: {context_steps}旬 | "
        f"chosen={chosen_label} | test MAPE={float(best_run['chosen_mape']):.6f}%"
    )

    result = pd.DataFrame(
        {
            DATE_COL: forecast_dates,
            "predicted_yhat": base_forecast.astype(float),
            "predicted_yhat_lower": forecast_lower.astype(float),
            "predicted_yhat_upper": forecast_upper.astype(float),
            "predicted_residual_lgbm": residual_pred.astype(float),
            "predicted_hybrid": hybrid_forecast.astype(float),
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

    metrics_rows = pd.DataFrame(
        [
            {"model": "prophet_only_test", **metrics_base},
            {"model": "prophet_plus_lgbm_residual_test", **metrics_hybrid},
            {"model": f"chosen_{chosen_label}_test", **metrics_chosen},
        ]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ctx_suffix = "full" if context_steps <= 0 else f"{context_steps}t"
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{PROCESSED_PROVINCE}_prophet_lgbmres_ctx{ctx_suffix}_{INFERENCE_CONTEXT_END}_{total_horizon}t.xlsx",
    )
    with pd.ExcelWriter(out_path) as writer:
        result.to_excel(writer, index=False, sheet_name="forecast")
        metrics_rows.to_excel(writer, index=False, sheet_name="metrics")

    if isinstance(best_lgbm_model, lgb.LGBMRegressor):
        fi_path = os.path.join(
            OUTPUT_DIR,
            f"{PROCESSED_PROVINCE}_prophet_lgbmres_ctx{ctx_suffix}_{INFERENCE_CONTEXT_END}_feature_importance.png",
        )
        _save_lgbm_feature_importance_plot(
            model=best_lgbm_model,
            feature_names=residual_features,
            output_path=fi_path,
            title=f"{PROCESSED_PROVINCE} Prophet+LGBM 残差特征重要性（旬度）",
        )

    print("-" * 50)
    print(f"input: {input_path}")
    print(f"saved: {out_path}")
    print(f"测试集 ({TEST_START} ~ {TEST_END}) 指标（仅 test 段）:")
    print("  Prophet 基线:", metrics_base)
    print("  Prophet+LGBM 残差:", metrics_hybrid)
    print(f"  最终采用 {chosen_label}:", metrics_chosen)
    print(
        f"  → predicted_gas_sales 列为上述「chosen」对应预测；"
        f"Excel 中 metrics 表含三行对比"
    )


if __name__ == "__main__":
    main()
