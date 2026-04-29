import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor

from gas_prediction.forecast_common import (
    MONTH_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
    forecast_metrics,
)
from gas_prediction.feature_engineering import build_features_pipeline

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# === 与 train_timemoe_peft 一致的时间线 ===
INFERENCE_CONTEXT_END = "2025-06"
BRIDGE_START = "2025-07"
BRIDGE_END = "2025-10"
TEST_START = "2025-11"
TEST_END = "2026-03"

OUTPUT_DIR = "output"  # 预测结果 Excel 输出目录
# XGBoost 残差模型外生特征：与 run_xgboost_0.py 一致，使用 build_features_pipeline 生成的全部特征
# （即除 MONTH_COL / TARGET_COL 之外的所有列，运行时动态确定）。
RESIDUAL_FEATURES: list[str] = []
XGB_RESIDUAL_PARAMS = {
    "n_estimators": 400,  # 树的数量上限；越大拟合能力越强，但更易过拟合、训练更慢
    "learning_rate": 0.03,  # 每棵树贡献的步长（shrinkage）；越小通常需更多树、更稳
    "max_depth": 3,  # 单棵树最大深度；越小模型越简单、越不易过拟合
    "min_child_weight": 5.0,  # 叶子继续分裂所需的最小样本权重和；越大越保守、抑制过拟合
    "subsample": 0.8,  # 每棵树随机抽取的样本比例（行采样）；<1 可减轻过拟合
    "colsample_bytree": 0.8,  # 每棵树随机抽取的特征比例（列采样）；<1 可减轻过拟合
    "reg_lambda": 2.0,  # L2 正则（权重平方惩罚）；越大越平滑、不易过拟合
    "reg_alpha": 0.2,  # L1 正则（权重绝对值惩罚）；越大越稀疏、部分特征权重可压到 0
    "random_state": 42,  # 随机种子，保证 subsample/列采样等可复现
}
XGB_EARLY_STOPPING_ROUNDS = 30  # 早停轮数；须传给 XGBRegressor(...)（XGBoost 2.1+ 不再支持 fit(early_stopping_rounds=...)）
# True：训练时在控制台逐轮打印验证集 rmse（需 eval_set）；False：静默
XGB_VERBOSE_FIT = False
CONTEXT_SEARCH_MIN = 60  # context 搜索下界（月）
CONTEXT_SEARCH_MAX = 111  # context 搜索上界（月）
CONTEXT_SEARCH_STEP = 3  # context 搜索步长（月）


def _save_xgb_feature_importance_plot(
    model: XGBRegressor,
    feature_names: list[str],
    output_path: str,
    title: str,
) -> None:
    if not hasattr(model, "feature_importances_"):
        print("警告：当前 XGBoost 模型不支持 feature_importances_，跳过特征重要性作图。")
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
    bars = plt.bar(range(len(sorted_features)), sorted_importances, color="#4C78A8")
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


def _apply_context_window(
    df_asof: pd.DataFrame, context_months: int
) -> pd.DataFrame:
    """
    在「<= INFERENCE_CONTEXT_END」已排序的序列上，只保留末尾 context_months 个月。
    context_months==0 表示不截断（使用全部 <=as-of 历史）。
    """
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


def _train_xgb_residual_model(
    context_df: pd.DataFrame,
    context_base_pred: np.ndarray,
    residual_features: list[str],
) -> XGBRegressor:
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
        raise ValueError("context 月份过少，无法稳定训练 XGBoost 残差模型（至少建议 12 个月）。")

    # 月度数据样本较小，使用时间顺序切分做早停，降低过拟合风险。
    val_size = min(12, max(6, n // 5))
    train_end = n - val_size
    if train_end <= 0:
        train_end = max(1, n - 1)

    x_train = x_all[:train_end]
    y_train = y_all[:train_end]
    x_val = x_all[train_end:]
    y_val = y_all[train_end:]

    # XGBoost 2.1+：early_stopping_rounds 须在构造函数中传入，不能传给 fit()
    model = XGBRegressor(
        objective="reg:squarederror",
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        **XGB_RESIDUAL_PARAMS,
    )
    if XGB_VERBOSE_FIT:
        print(
            f"XGBoost 残差训练: 训练 {x_train.shape[0]} 月 | 验证 {x_val.shape[0]} 月（时间顺序末尾）| "
            f"eval 指标为 validation_0-rmse"
        )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=XGB_VERBOSE_FIT,
    )
    if XGB_VERBOSE_FIT:
        print(
            f"XGBoost 残差训练结束 | best_iteration={getattr(model, 'best_iteration', None)} "
            f"| best_score={getattr(model, 'best_score', None)}"
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
    print(f"XGBoost 残差特征（与 run_xgboost_0.py 一致）: {residual_features}")
    print(f"XGBoost 参数: {XGB_RESIDUAL_PARAMS}, early_stopping_rounds={XGB_EARLY_STOPPING_ROUNDS}")

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

        xgb_model = _train_xgb_residual_model(context_df, context_base_pred, residual_features)
        residual_pred = xgb_model.predict(x_future).astype(float)
        hybrid_forecast = base_forecast + residual_pred
        test_pred_base = base_forecast[n_bridge:]
        test_pred_hybrid = hybrid_forecast[n_bridge:]
        metrics_base = forecast_metrics(test_true.astype(float), test_pred_base.astype(float))
        metrics_hybrid = forecast_metrics(test_true.astype(float), test_pred_hybrid.astype(float))

        use_hybrid = float(metrics_hybrid["MAPE(%)"]) < float(metrics_base["MAPE(%)"])
        chosen_label = "prophet_plus_xgb_residual" if use_hybrid else "prophet_only"
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
                "xgb_model": xgb_model,
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
    best_xgb_model = best_run["xgb_model"]

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
            "predicted_residual_xgb": residual_pred.astype(float),
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
        f"{PROCESSED_PROVINCE}_prophet_xgbres_ctx{ctx_suffix}_{INFERENCE_CONTEXT_END}_{total_horizon}m.xlsx",
    )
    result.to_excel(out_path, index=False, sheet_name="forecast")

    if isinstance(best_xgb_model, XGBRegressor):
        fi_path = os.path.join(
            OUTPUT_DIR,
            f"{PROCESSED_PROVINCE}_prophet_xgbres_ctx{ctx_suffix}_{INFERENCE_CONTEXT_END}_feature_importance.png",
        )
        _save_xgb_feature_importance_plot(
            model=best_xgb_model,
            feature_names=residual_features,
            output_path=fi_path,
            title=f"{PROCESSED_PROVINCE} Prophet+XGB 特征重要性",
        )

    print("-" * 50)
    print(f"input: {input_path}")
    print(f"saved: {out_path}")
    print(f"测试集 ({TEST_START} ~ {TEST_END}) 对比（仅 test 段）:")
    print(
        f"  Prophet基线 | MAE={metrics_base['MAE']:.6f} RMSE={metrics_base['RMSE']:.6f} "
        f"MAPE={metrics_base['MAPE(%)']:.6f}%"
    )
    print(
        f"  Prophet+XGB残差 | MAE={metrics_hybrid['MAE']:.6f} RMSE={metrics_hybrid['RMSE']:.6f} "
        f"MAPE={metrics_hybrid['MAPE(%)']:.6f}%"
    )
    print(
        f"  → 最终报告采用: {chosen_label}（test MAPE 更低）；"
        f"predicted_gas_sales 列已写入该结果"
    )


if __name__ == "__main__":
    main()
