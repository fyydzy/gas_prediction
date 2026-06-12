import os
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from gas_prediction.feature_engineering1 import build_features_pipeline
from gas_prediction.forecast_common1 import (
    DATE_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
)

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 旬度时间切分：训练集 date<=TRAIN_END；测试集 [TEST_START, TEST_END] 闭区间（旬起始日，YYYY-MM-DD）。
TRAIN_END = "2025-06-21"
TEST_START = "2025-11-01"
TEST_END = "2026-03-21"

OUTPUT_ROOT = "output1"
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "xgboost")

# 仅用下列特征建模
MODEL_FEATURE_COLS: list[str] = [
    # "avg_temp",
    # "max_temp",
    "min_temp",
    "HDD",
    "extreme_cold_days",
    "temp_range",
    "time_index",
    # "month_sin",
    # "month_cos",
    # "tenday_in_month",
    # "is_heating_season",
    # "spring_rework_peak",
    "Lag_36",
    "HDD_squared",
    "HDD_cross_Lag_36",
    # "HDD_cross_HeatingSeason",
    # "ColdDays_cross_Lag_36",
]


# --- 随机网格（笛卡尔积 972 组，再随机抽 RANDOM_SEARCH_N_TRIALS 组；验证集 MAPE 最小者胜）---
# GRID_MAX_DEPTH：单棵树最大深度；越小越不易过拟合、表达能力越弱。
# GRID_LEARNING_RATE：每棵树对残差的步长；越小通常需更多树但更稳，常与 n_estimators/早停一起权衡。
# GRID_REG_ALPHA：L1 正则系数；越大权重越稀疏、抑制过拟合，过大易欠拟合。
# GRID_REG_LAMBDA：L2 正则系数；越大叶子权重被拉平、抑制过拟合。
# GRID_MIN_CHILD_WEIGHT：分裂所需子节点权重和的最小值；越大分裂越保守、树更浅。
# GRID_SUBSAMPLE：每棵树训练时的行采样比例；<1 引入随机性、减轻过拟合。
# GRID_COLSAMPLE_BYTREE：每棵树建树时的列采样比例；<1 减轻特征共线时的过拟合。
# RANDOM_SEARCH_N_TRIALS：从全组合中无放回随机抽取的试验次数。
# RANDOM_SEARCH_SEED：抽样与模型随机性的种子（与 random_state 配合保证可复现）。
GRID_MAX_DEPTH = [3, 4, 5]
GRID_LEARNING_RATE = [0.02, 0.03, 0.05]
GRID_REG_ALPHA = [0.0, 0.1, 0.2]
GRID_REG_LAMBDA = [0.5, 1.0, 2.0]
GRID_MIN_CHILD_WEIGHT = [3.0, 5.0, 7.0]
GRID_SUBSAMPLE = [0.7, 0.8]
GRID_COLSAMPLE_BYTREE = [0.7, 0.8]
RANDOM_SEARCH_N_TRIALS = 200
RANDOM_SEARCH_SEED = 42

# --- 不参与网格、对所有试验固定的基础参数 ---
# n_estimators：提升树棵数上限；配合早停实际有效棵数常小于该值。
# objective：回归目标，squarederror 为平方损失。
# random_state：样本与特征子采样、列采样等随机过程种子。
BASE_XGB_PARAMS = {
    "n_estimators": 800,
    "objective": "reg:squarederror",
    "random_state": 42,
}

# EARLY_STOPPING_ROUNDS：验证集指标连续若干轮（加树轮次）无提升则停止训练，并回退到最优轮次。
EARLY_STOPPING_ROUNDS = 50


def _load_and_build_features() -> pd.DataFrame:
    input_path = find_processed_excel()
    raw_df = pd.read_excel(input_path)
    required = {
        DATE_COL,
        TARGET_COL,
        "avg_temp",
        "max_temp",
        "min_temp",
        "HDD",
        "extreme_cold_days",
    }
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {sorted(missing)}")

    model_input_cols = [
        DATE_COL,
        TARGET_COL,
        "avg_temp",
        "max_temp",
        "min_temp",
        "HDD",
        "extreme_cold_days",
    ]
    raw_df = raw_df[model_input_cols].copy()
    raw_df[DATE_COL] = pd.to_datetime(raw_df[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")
    raw_df[TARGET_COL] = pd.to_numeric(raw_df[TARGET_COL], errors="coerce")
    raw_df["avg_temp"] = pd.to_numeric(raw_df["avg_temp"], errors="coerce")
    raw_df["max_temp"] = pd.to_numeric(raw_df["max_temp"], errors="coerce")
    raw_df["min_temp"] = pd.to_numeric(raw_df["min_temp"], errors="coerce")
    raw_df["HDD"] = pd.to_numeric(raw_df["HDD"], errors="coerce")
    raw_df["extreme_cold_days"] = pd.to_numeric(raw_df["extreme_cold_days"], errors="coerce")
    raw_df = raw_df.dropna(subset=list(required)).reset_index(drop=True)

    return build_features_pipeline(raw_df, target_col=TARGET_COL, date_col=DATE_COL)


def _plot_feature_importance(model: XGBRegressor, feature_names: list[str]) -> str:
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_OUTPUT_DIR, "xgboost_feature_importance.png")

    importances = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(sorted_features)), sorted_importances, color="#4C78A8")
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=45, ha="right")
    ax.set_title("XGBoost 特征重要性")
    ax.set_ylabel("Importance")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, sorted_importances):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


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


def _grid_search_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[dict, int, pd.DataFrame]:
    rows: list[dict] = []
    best_params: dict | None = None
    best_iter = 0
    best_mape = float("inf")

    all_combos = list(
        product(
            GRID_MAX_DEPTH,
            GRID_LEARNING_RATE,
            GRID_REG_ALPHA,
            GRID_REG_LAMBDA,
            GRID_MIN_CHILD_WEIGHT,
            GRID_SUBSAMPLE,
            GRID_COLSAMPLE_BYTREE,
        )
    )
    total_combos = len(all_combos)
    n_trials = min(RANDOM_SEARCH_N_TRIALS, total_combos)
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)
    chosen_idx = rng.choice(total_combos, size=n_trials, replace=False)
    sampled_combos = [all_combos[i] for i in chosen_idx]

    print(f"参数组合总数={total_combos}，随机抽样评估={n_trials}（seed={RANDOM_SEARCH_SEED}）")

    for (
        max_depth,
        learning_rate,
        reg_alpha,
        reg_lambda,
        min_child_weight,
        subsample,
        colsample_bytree,
    ) in sampled_combos:
        params = dict(BASE_XGB_PARAMS)
        params.update(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }
        )
        model = XGBRegressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_pred = model.predict(X_val).astype(float)
        val_mape = _mape(y_val.astype(float), val_pred)
        best_iteration = int(getattr(model, "best_iteration", params["n_estimators"] - 1))
        rows.append(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "best_iteration": best_iteration,
                "val_mape_pct": val_mape,
            }
        )
        if val_mape < best_mape:
            best_mape = val_mape
            best_params = params
            best_iter = best_iteration

    if best_params is None:
        raise ValueError("网格搜索失败，未找到有效参数。")
    grid_df = pd.DataFrame(rows).sort_values("val_mape_pct").reset_index(drop=True)
    return best_params, best_iter, grid_df


def main():
    print("正在读取数据并构建特征...")
    df_features = _load_and_build_features()

    train_df = df_features[df_features[DATE_COL] <= TRAIN_END].copy()
    test_df = df_features[
        (df_features[DATE_COL] >= TEST_START) & (df_features[DATE_COL] <= TEST_END)
    ].copy()

    missing_feats = [c for c in MODEL_FEATURE_COLS if c not in df_features.columns]
    if missing_feats:
        raise ValueError(
            f"特征工程结果中缺少下列建模列: {missing_feats}；请检查 feature_engineering1 流水线。"
        )

    X_train_full = train_df[MODEL_FEATURE_COLS].copy()
    y_train_full = train_df[TARGET_COL].to_numpy(dtype=float)
    X_test = test_df[MODEL_FEATURE_COLS].copy()
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    if X_train_full.empty:
        raise ValueError("训练集为空，请检查时间范围和数据。")
    if X_test.empty:
        raise ValueError("测试集为空，请检查时间范围和数据。")

    # 用训练集末尾一段做 early stopping 验证集（保持时间顺序）
    n = len(X_train_full)
    val_size = min(36, max(18, n // 5))
    split_idx = n - val_size
    if split_idx <= 0:
        split_idx = max(1, n - 1)

    X_train = X_train_full.iloc[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full.iloc[split_idx:]
    y_val = y_train_full[split_idx:]

    print("开始轻量网格搜索（训练内验证集）...")
    best_params, best_iter, grid_df = _grid_search_xgb(X_train, y_train, X_val, y_val)
    print(f"网格搜索完成，共 {len(grid_df)} 组。最优验证 MAPE={grid_df.iloc[0]['val_mape_pct']:.4f}%")
    print(f"最优参数: {best_params} | best_iteration={best_iter}")

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    grid_path = os.path.join(PLOT_OUTPUT_DIR, "xgboost_grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print(f"网格搜索结果已保存: {grid_path}")

    # 使用最优参数在完整训练集重训（不再 early stopping，树数固定为最优轮数+1）
    final_params = dict(best_params)
    final_params["n_estimators"] = max(best_iter + 1, 50)
    model = XGBRegressor(**final_params)
    model.fit(X_train_full, y_train_full, verbose=False)
    print(f"已用完整训练集重训，n_estimators={final_params['n_estimators']}")

    test_pred = model.predict(X_test).astype(float)

    result_df = pd.DataFrame(
        {
            DATE_COL: test_df[DATE_COL].astype(str).values,
            "actual_gas_sales": y_test.astype(float),
            "predicted_gas_sales": test_pred,
        }
    )
    result_df["error"] = result_df["predicted_gas_sales"] - result_df["actual_gas_sales"]
    result_df["abs_error"] = np.abs(result_df["error"])
    result_df["mape_pct"] = np.where(
        result_df["actual_gas_sales"] != 0,
        np.abs(result_df["error"] / result_df["actual_gas_sales"]) * 100.0,
        np.nan,
    )

    metrics = _regression_metrics(y_test.astype(float), test_pred.astype(float))
    metrics_df = pd.DataFrame([metrics])

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    excel_path = os.path.join(
        OUTPUT_ROOT,
        f"{PROCESSED_PROVINCE}_xgboost_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    with pd.ExcelWriter(excel_path) as writer:
        result_df.to_excel(writer, index=False, sheet_name="test_forecast")
        metrics_df.to_excel(writer, index=False, sheet_name="metrics")
    print(f"测试集预测结果已保存: {excel_path}")

    fi_path = _plot_feature_importance(model, X_train_full.columns.tolist())
    print(f"特征重要性图已保存: {fi_path}")

    print(f"\nXGBoost 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

