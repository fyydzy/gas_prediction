import os
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from gas_prediction.feature_engineering import build_features_pipeline
from gas_prediction.forecast_common import (
    MONTH_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
)

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 时间切分：训练集仅含 MONTH_COL<=TRAIN_END；测试集为 [TEST_START, TEST_END] 闭区间月份字符串。
TRAIN_END = "2025-06"
TEST_START = "2025-11"
TEST_END = "2026-03"

PLOT_OUTPUT_DIR = os.path.join("output", "svr")

# 定义沿用的筛选特征（若缺失会在运行时抛错提示）。
SELECTED_FEATURES = [
    "HDD_cross_Lag_12",
    "Lag_12",
    "ColdDays_cross_Lag_12",
    "HDD_cross_HeatingSeason",
    "time_index",
    "min_temp",
    "is_heating_season",
    "HDD_squared",
]

# --- 随机网格（全组合 972 组抽 RANDOM_SEARCH_N_TRIALS 组；验证集 MAPE 最小者胜）---
# GRID_C：误差惩罚系数；越大越贴合训练数据，过大可能过拟合。
# GRID_EPSILON：epsilon-insensitive 管道宽度；越大越鲁棒、预测更平滑。
# GRID_GAMMA：RBF 核影响范围；值越大决策边界越局部。
# GRID_TOL：优化停止阈值；越小优化更精细但更慢。
# GRID_SHRINKING：是否使用 shrinking 启发式以加速优化。
# GRID_CACHE_SIZE：核矩阵缓存大小（MB），影响训练速度而非目标函数。
GRID_C = [0.1, 1.0, 10.0]
GRID_EPSILON = [0.01, 0.1, 0.2]
GRID_GAMMA = ["scale", 0.1, 0.01]
GRID_TOL = [1e-3, 5e-4]
GRID_SHRINKING = [True, False]
GRID_CACHE_SIZE = [200, 500, 1000]
RANDOM_SEARCH_N_TRIALS = 200
RANDOM_SEARCH_SEED = 42

# --- 不参与网格的基础参数 ---
# kernel：核函数类型；此处固定 rbf 以建模非线性。
# max_iter：优化迭代上限；-1 表示不限制，由收敛准则控制停止。
BASE_SVR_PARAMS = {
    "kernel": "rbf",
    "max_iter": -1,
}


def _load_and_build_features() -> pd.DataFrame:
    input_path = find_processed_excel()
    raw_df = pd.read_excel(input_path)
    required = {MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {sorted(missing)}")

    raw_df = raw_df.copy()
    raw_df[MONTH_COL] = raw_df[MONTH_COL].astype(str).str.slice(0, 7)
    raw_df[TARGET_COL] = pd.to_numeric(raw_df[TARGET_COL], errors="coerce")
    raw_df["avg_temp"] = pd.to_numeric(raw_df["avg_temp"], errors="coerce")
    raw_df["max_temp"] = pd.to_numeric(raw_df["max_temp"], errors="coerce")
    raw_df["min_temp"] = pd.to_numeric(raw_df["min_temp"], errors="coerce")
    raw_df = raw_df.dropna(
        subset=[MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp"]
    ).reset_index(drop=True)

    return build_features_pipeline(raw_df, target_col=TARGET_COL, month_col=MONTH_COL)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if np.any(mask):
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return float("nan")


def _grid_search_svr(
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
            GRID_C,
            GRID_EPSILON,
            GRID_GAMMA,
            GRID_TOL,
            GRID_SHRINKING,
            GRID_CACHE_SIZE,
        )
    )
    total_combos = len(all_combos)
    n_trials = min(RANDOM_SEARCH_N_TRIALS, total_combos)
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)
    chosen_idx = rng.choice(total_combos, size=n_trials, replace=False)
    sampled_combos = [all_combos[i] for i in chosen_idx]

    print(f"参数组合总数={total_combos}，随机抽样评估={n_trials}（seed={RANDOM_SEARCH_SEED}）")

    for c, epsilon, gamma, tol, shrinking, cache_size in sampled_combos:
        params = dict(BASE_SVR_PARAMS)
        params.update(
            {
                "C": float(c),
                "epsilon": float(epsilon),
                "gamma": gamma,
                "tol": float(tol),
                "shrinking": bool(shrinking),
                "cache_size": int(cache_size),
            }
        )
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(**params)),
            ]
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val).astype(float)
        val_mape = _mape(y_val.astype(float), val_pred)
        rows.append(
            {
                "C": c,
                "epsilon": epsilon,
                "gamma": gamma,
                "tol": tol,
                "shrinking": shrinking,
                "cache_size_mb": cache_size,
                "val_mape_pct": val_mape,
            }
        )
        if val_mape < best_mape:
            best_mape = val_mape
            best_params = params

    if best_params is None:
        raise ValueError("网格搜索失败，未找到有效参数。")
    grid_df = pd.DataFrame(rows).sort_values("val_mape_pct").reset_index(drop=True)
    return best_params, grid_df


def main() -> None:
    print("正在读取数据并构建特征...")
    df_features = _load_and_build_features()

    missing_feats = [f for f in SELECTED_FEATURES if f not in df_features.columns]
    if missing_feats:
        raise ValueError(f"特征工程后缺少所需特征: {missing_feats}")

    train_df = df_features[df_features[MONTH_COL] <= TRAIN_END].copy()
    test_df = df_features[
        (df_features[MONTH_COL] >= TEST_START) & (df_features[MONTH_COL] <= TEST_END)
    ].copy()

    X_train_full = train_df[SELECTED_FEATURES].copy()
    y_train_full = train_df[TARGET_COL].to_numpy(dtype=float)
    X_test = test_df[SELECTED_FEATURES].copy()
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    if X_train_full.empty:
        raise ValueError("训练集为空，请检查时间范围和数据。")
    if X_test.empty:
        raise ValueError("测试集为空，请检查时间范围和数据。")

    n = len(X_train_full)
    val_size = min(12, max(6, n // 5))
    split_idx = n - val_size
    if split_idx <= 0:
        split_idx = max(1, n - 1)

    X_train = X_train_full.iloc[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full.iloc[split_idx:]
    y_val = y_train_full[split_idx:]

    print(f"开始随机网格搜索（训练内验证集，特征数={len(SELECTED_FEATURES)}）...")
    best_params, grid_df = _grid_search_svr(X_train, y_train, X_val, y_val)
    print(f"网格搜索完成，共 {len(grid_df)} 组。最优验证 MAPE={grid_df.iloc[0]['val_mape_pct']:.4f}%")
    print(f"最优参数: {best_params}")

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    grid_path = os.path.join(PLOT_OUTPUT_DIR, "svr_grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print(f"网格搜索结果已保存: {grid_path}")

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(**best_params)),
        ]
    )
    final_model.fit(X_train_full, y_train_full)
    print("已用完整训练集重训 SVR。")

    test_pred = final_model.predict(X_test).astype(float)

    result_df = pd.DataFrame(
        {
            MONTH_COL: test_df[MONTH_COL].astype(str).values,
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

    os.makedirs("output", exist_ok=True)
    excel_path = os.path.join(
        "output",
        f"{PROCESSED_PROVINCE}_svr_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    result_df.to_excel(excel_path, index=False, sheet_name="test_forecast")
    print(f"测试集预测结果已保存: {excel_path}")

    if len(y_test) > 0 and not np.any(y_test == 0):
        mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
        print(f"\nSVR 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
        print(f"Test MAPE: {mape:.4f}%")
    else:
        print("测试集包含 0 或为空，跳过 MAPE 输出。")


if __name__ == "__main__":
    main()