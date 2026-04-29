import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

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

TRAIN_END = "2025-06"
TEST_START = "2025-11"
TEST_END = "2026-03"
PLOT_OUTPUT_DIR = os.path.join("output", "ridge")


def plot_ridge_path_and_train(X_train, y_train, feature_names):
    print("\n1. 标准化数据并计算 Ridge 系数路径...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Ridge 的 alpha 网格（对数刻度）
    alphas_path = np.logspace(-4, 4, 200)
    coefs_path = np.zeros((len(feature_names), len(alphas_path)), dtype=float)
    for j, a in enumerate(alphas_path):
        model = Ridge(alpha=float(a), random_state=42)
        model.fit(X_scaled, y_train)
        coefs_path[:, j] = model.coef_

    print("2. 进行 5 折交叉验证，寻找全局最佳惩罚力度 Alpha...")
    ridge_cv = RidgeCV(alphas=alphas_path, cv=5)
    ridge_cv.fit(X_scaled, y_train)
    best_alpha = float(ridge_cv.alpha_)
    print(f"找到最佳 Alpha: {best_alpha:.6f}")

    n_features = coefs_path.shape[0]
    legend_cols = 4
    legend_rows = int(np.ceil((n_features + 1) / legend_cols))
    fig_height = 8 + legend_rows * 0.45
    fig, ax = plt.subplots(figsize=(14, fig_height))

    log_alphas = np.log10(alphas_path)
    for i in range(coefs_path.shape[0]):
        ax.plot(log_alphas, coefs_path[i], linewidth=2, label=feature_names[i])

    ax.axvline(
        x=np.log10(best_alpha),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"最佳 Alpha 截断线 (Alpha={best_alpha:.6f})",
    )

    ax.set_xlabel("Log10(Alpha) - 惩罚力度 (向右滑动惩罚越大)", fontsize=14)
    ax.set_ylabel("特征权重 (Coefficients)", fontsize=14)
    ax.set_title("Ridge 变量选择全过程：特征系数随惩罚力度的衰减路径图", fontsize=16, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_cols,
        fontsize=10,
        frameon=False,
    )
    fig.subplots_adjust(bottom=min(0.10 + legend_rows * 0.06, 0.55))

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_OUTPUT_DIR, "ridge_feature_selection_path.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"图表已保存: {plot_path}")
    plt.show()

    return ridge_cv, scaler


def main():
    print("正在读取数据并构建特征...")
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

    df_features = build_features_pipeline(raw_df, target_col=TARGET_COL, month_col=MONTH_COL)

    train_df = df_features[df_features[MONTH_COL] <= TRAIN_END].copy()
    test_df = df_features[(df_features[MONTH_COL] >= TEST_START) & (df_features[MONTH_COL] <= TEST_END)].copy()

    drop_cols = [MONTH_COL, TARGET_COL]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    if X_train.empty:
        raise ValueError("训练集为空，请检查时间范围和数据。")
    if X_test.empty:
        raise ValueError("测试集为空，请检查时间范围和数据。")

    feature_names = X_train.columns.tolist()
    best_ridge, fitted_scaler = plot_ridge_path_and_train(X_train, y_train, feature_names)

    print("\n" + "=" * 50)
    print("Ridge 最终特征权重:")
    coef_dict = dict(zip(feature_names, best_ridge.coef_))
    for feat, weight in sorted(coef_dict.items(), key=lambda item: abs(item[1]), reverse=True):
        print(f"  {feat}: {weight:.4f}")
    print("=" * 50)

    X_test_scaled = fitted_scaler.transform(X_test)
    test_pred = best_ridge.predict(X_test_scaled)

    result_df = pd.DataFrame(
        {
            MONTH_COL: test_df[MONTH_COL].astype(str).values,
            "actual_gas_sales": y_test.astype(float),
            "predicted_gas_sales": test_pred.astype(float),
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
        f"{PROCESSED_PROVINCE}_ridge_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    result_df.to_excel(excel_path, index=False, sheet_name="test_forecast")
    print(f"测试集预测结果已保存: {excel_path}")

    if len(y_test) > 0 and not np.any(y_test == 0):
        mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
        print(f"\nRidge 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
        print(f"Test MAPE: {mape:.4f}%")
    else:
        print("测试集包含 0 或为空，跳过 MAPE 输出。")


if __name__ == "__main__":
    main()

