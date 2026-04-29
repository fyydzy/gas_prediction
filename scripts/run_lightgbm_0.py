import os
import warnings
from itertools import product

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

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

PLOT_OUTPUT_DIR = os.path.join("output", "lightgbm")

# --- 随机网格（与 XGBoost 脚本同量级：972 组抽 RANDOM_SEARCH_N_TRIALS 组；验证集 MAPE 最小者胜）---
# GRID_MAX_DEPTH：树的最大深度；限制结构复杂度，常与 num_leaves（此处未搜）联动。
# GRID_LEARNING_RATE：每轮梯度提升的步长；越小一般需更多轮但更稳。
# GRID_REG_ALPHA / GRID_REG_LAMBDA：L1 / L2 正则，含义与 XGBoost 的 reg_alpha/reg_lambda 类似。
# GRID_MIN_CHILD_SAMPLES：叶子最少样本数；越大越保守、抑制对小样本的划分。
# GRID_SUBSAMPLE：行采样比例（bagging_fraction）；<1 减轻过拟合。
# GRID_COLSAMPLE_BYTREE：列采样比例（feature_fraction）；<1 减轻过拟合。
# RANDOM_SEARCH_N_TRIALS / RANDOM_SEARCH_SEED：随机抽样的次数与种子。
GRID_MAX_DEPTH = [3, 4, 5]
GRID_LEARNING_RATE = [0.02, 0.03, 0.05]
GRID_REG_ALPHA = [0.0, 0.1, 0.2]
GRID_REG_LAMBDA = [0.5, 1.0, 2.0]
GRID_MIN_CHILD_SAMPLES = [5, 10, 20]
GRID_SUBSAMPLE = [0.7, 0.8]
GRID_COLSAMPLE_BYTREE = [0.7, 0.8]
RANDOM_SEARCH_N_TRIALS = 200
RANDOM_SEARCH_SEED = 42

# --- 不参与网格的基础参数 ---
# n_estimators：迭代次数上限；早停后脚本会用 best_iteration_+1 在全长训练集上重训。
# objective：回归任务默认平方损失意义下的 regression。
# random_state：特征与样本子采样等随机种子。
# verbosity：-1 关闭 LightGBM 自身日志输出。
BASE_LGBM_PARAMS = {
    "n_estimators": 800,
    "objective": "regression",
    "random_state": 42,
    "verbosity": -1,
}

# EARLY_STOPPING_ROUNDS：验证集 loss 连续多轮无改善则停止，并记录 best_iteration_。
EARLY_STOPPING_ROUNDS = 50


def _load_and_build_features() -> pd.DataFrame:
    input_path = find_processed_excel()
    raw_df = pd.read_excel(input_path)
    required = {MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp"}
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


def _plot_feature_importance(model: LGBMRegressor, feature_names: list[str]) -> str:
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_OUTPUT_DIR, "lightgbm_feature_importance.png")

    importances = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(sorted_features)), sorted_importances, color="#2ca02c")
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=45, ha="right")
    ax.set_title("LightGBM 特征重要性")
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
    plt.show()
    return plot_path


def _grid_search_lgbm(
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
            GRID_MIN_CHILD_SAMPLES,
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

    callbacks = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]

    for (
        max_depth,
        learning_rate,
        reg_alpha,
        reg_lambda,
        min_child_samples,
        subsample,
        colsample_bytree,
    ) in sampled_combos:
        params = dict(BASE_LGBM_PARAMS)
        params.update(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "min_child_samples": int(min_child_samples),
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
            }
        )
        model = LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=callbacks,
        )
        val_pred = model.predict(X_val).astype(float)
        val_mape = _mape(y_val.astype(float), val_pred)
        best_iteration = int(getattr(model, "best_iteration_", params["n_estimators"] - 1))
        rows.append(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "min_child_samples": min_child_samples,
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


def main() -> None:
    print("正在读取数据并构建特征...")
    df_features = _load_and_build_features()

    train_df = df_features[df_features[MONTH_COL] <= TRAIN_END].copy()
    test_df = df_features[
        (df_features[MONTH_COL] >= TEST_START) & (df_features[MONTH_COL] <= TEST_END)
    ].copy()

    drop_cols = [MONTH_COL, TARGET_COL]
    X_train_full = train_df.drop(columns=drop_cols)
    y_train_full = train_df[TARGET_COL].to_numpy(dtype=float)
    X_test = test_df.drop(columns=drop_cols)
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

    print("开始随机网格搜索（训练内验证集）...")
    best_params, best_iter, grid_df = _grid_search_lgbm(X_train, y_train, X_val, y_val)
    print(f"网格搜索完成，共 {len(grid_df)} 组。最优验证 MAPE={grid_df.iloc[0]['val_mape_pct']:.4f}%")
    print(f"最优参数: {best_params} | best_iteration={best_iter}")

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    grid_path = os.path.join(PLOT_OUTPUT_DIR, "lightgbm_grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print(f"网格搜索结果已保存: {grid_path}")

    final_params = dict(best_params)
    final_params["n_estimators"] = max(best_iter + 1, 50)
    model = LGBMRegressor(**final_params)
    model.fit(X_train_full, y_train_full)
    print(f"已用完整训练集重训，n_estimators={final_params['n_estimators']}")

    test_pred = model.predict(X_test).astype(float)

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
        f"{PROCESSED_PROVINCE}_lightgbm_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    result_df.to_excel(excel_path, index=False, sheet_name="test_forecast")
    print(f"测试集预测结果已保存: {excel_path}")

    fi_path = _plot_feature_importance(model, X_train_full.columns.tolist())
    print(f"特征重要性图已保存: {fi_path}")

    if len(y_test) > 0 and not np.any(y_test == 0):
        mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
        print(f"\nLightGBM 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
        print(f"Test MAPE: {mape:.4f}%")
    else:
        print("测试集包含 0 或为空，跳过 MAPE 输出。")


if __name__ == "__main__":
    main()
