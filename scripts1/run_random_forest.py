import os
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "random_forest")

# 仅用下列特征建模（与 run_catboost 一致，便于对比；可按需取消下一行注释加入 spring_rework_peak）
MODEL_FEATURE_COLS: list[str] = [
    "avg_temp",
    # "max_temp",
    "min_temp",
    "HDD",
    "extreme_cold_days",
    # "temp_range",
    "time_index",
    # "month_sin",
    "month_cos",
    # "tenday_in_month",
    # "is_heating_season",
    # "spring_rework_peak",
    "Lag_36",
    "HDD_squared",
    "HDD_cross_Lag_36",
    "HDD_cross_HeatingSeason",
    "ColdDays_cross_Lag_36",
]

# --- 随机网格（972 组抽 RANDOM_SEARCH_N_TRIALS 组；验证集 MAPE 最小者胜）---
# 随机森林无 boosting 式早停；仅在训练子集上 fit，用末尾验证集算 MAPE 选参。
# GRID_N_ESTIMATORS：森林中决策树的棵数；越多通常方差越小、训练越慢，过大易在噪声上过拟合。
# GRID_MAX_DEPTH：单棵树最大深度；None 表示不限制（由 min_samples_* 等约束停止生长）。
# GRID_MIN_SAMPLES_LEAF：叶子节点最少样本数；越大叶子越大、模型越平滑。
# GRID_MIN_SAMPLES_SPLIT：内部节点再划分所需最少样本数；越大越难分裂、树更浅。
# GRID_MAX_FEATURES：每次分裂考虑的特征数；sqrt/log2 为启发式，0.6 表示最多 60% 列。
# GRID_MAX_SAMPLES：自助采样时每棵树使用的样本比例；<1 类似行子采样，增大树间差异。
# RANDOM_SEARCH_N_TRIALS / RANDOM_SEARCH_SEED：随机抽样次数与种子。
GRID_N_ESTIMATORS = [300, 500, 800]
GRID_MAX_DEPTH = [3, 4, 5, None]
GRID_MIN_SAMPLES_LEAF = [1, 2, 4]
GRID_MIN_SAMPLES_SPLIT = [2, 5, 10]
GRID_MAX_FEATURES = ["sqrt", "log2", 0.6]
GRID_MAX_SAMPLES = [0.7, 0.8, 1.0]
RANDOM_SEARCH_N_TRIALS = 200
RANDOM_SEARCH_SEED = 42

# --- 不参与网格的基础参数 ---
# random_state：自助抽样与特征子集抽样的随机种子。
# n_jobs：并行建树的线程数；-1 表示用满可用 CPU。
# bootstrap：True 为每棵树有放回抽样训练子样本（标准随机森林）。
BASE_RF_PARAMS = {
    "random_state": 42,
    "n_jobs": -1,
    "bootstrap": True,
}


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


def _plot_feature_importance(model: RandomForestRegressor, feature_names: list[str]) -> str:
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_OUTPUT_DIR, "random_forest_feature_importance.png")

    importances = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(sorted_features)), sorted_importances, color="#9467bd")
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=45, ha="right")
    ax.set_title("随机森林 特征重要性")
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


def _grid_search_rf(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    best_params: dict | None = None
    best_mape = float("inf")

    all_combos = list(
        product(
            GRID_N_ESTIMATORS,
            GRID_MAX_DEPTH,
            GRID_MIN_SAMPLES_LEAF,
            GRID_MIN_SAMPLES_SPLIT,
            GRID_MAX_FEATURES,
            GRID_MAX_SAMPLES,
        )
    )
    total_combos = len(all_combos)
    n_trials = min(RANDOM_SEARCH_N_TRIALS, total_combos)
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)
    chosen_idx = rng.choice(total_combos, size=n_trials, replace=False)
    sampled_combos = [all_combos[i] for i in chosen_idx]

    print(f"参数组合总数={total_combos}，随机抽样评估={n_trials}（seed={RANDOM_SEARCH_SEED}）")

    for (
        n_estimators,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        max_features,
        max_samples,
    ) in sampled_combos:
        if min_samples_split <= min_samples_leaf:
            continue
        params = dict(BASE_RF_PARAMS)
        params.update(
            {
                "n_estimators": int(n_estimators),
                "max_depth": max_depth,
                "min_samples_leaf": int(min_samples_leaf),
                "min_samples_split": int(min_samples_split),
                "max_features": max_features,
                "max_samples": float(max_samples),
            }
        )
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val).astype(float)
        val_mape = _mape(y_val.astype(float), val_pred)
        rows.append(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth is not None else "None",
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
                "max_samples": max_samples,
                "val_mape_pct": val_mape,
            }
        )
        if val_mape < best_mape:
            best_mape = val_mape
            best_params = params

    if best_params is None:
        raise ValueError("网格搜索失败，未找到有效参数（可能需放宽 min_samples_split / leaf 约束）。")
    grid_df = pd.DataFrame(rows).sort_values("val_mape_pct").reset_index(drop=True)
    return best_params, grid_df


def main() -> None:
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

    n = len(X_train_full)
    val_size = min(36, max(18, n // 5))
    split_idx = n - val_size
    if split_idx <= 0:
        split_idx = max(1, n - 1)

    X_train = X_train_full.iloc[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full.iloc[split_idx:]
    y_val = y_train_full[split_idx:]

    print("开始随机网格搜索（训练子集拟合、末尾验证集 MAPE）...")
    best_params, grid_df = _grid_search_rf(X_train, y_train, X_val, y_val)
    print(f"网格搜索完成，共 {len(grid_df)} 组。最优验证 MAPE={grid_df.iloc[0]['val_mape_pct']:.4f}%")
    print(f"最优参数: {best_params}")

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    grid_path = os.path.join(PLOT_OUTPUT_DIR, "random_forest_grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print(f"网格搜索结果已保存: {grid_path}")

    model = RandomForestRegressor(**best_params)
    model.fit(X_train_full, y_train_full)
    print(
        "已用完整训练集重训，"
        f"n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}"
    )

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
        f"{PROCESSED_PROVINCE}_random_forest_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    with pd.ExcelWriter(excel_path) as writer:
        result_df.to_excel(writer, index=False, sheet_name="test_forecast")
        metrics_df.to_excel(writer, index=False, sheet_name="metrics")
    print(f"测试集预测结果已保存: {excel_path}")

    fi_path = _plot_feature_importance(model, X_train_full.columns.tolist())
    print(f"特征重要性图已保存: {fi_path}")

    print(f"\n随机森林 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
