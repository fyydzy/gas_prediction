import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, enet_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# 旬度时间切分：训练集 date<=TRAIN_END；测试集为 [TEST_START, TEST_END] 闭区间。
TRAIN_END = "2025-06-21"
TEST_START = "2025-11-01"
TEST_END = "2026-03-21"
OUTPUT_ROOT = "output1"
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "elasticnet")


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


def plot_enet_path_and_train(X_train, y_train, feature_names):
    print("\n1. 标准化数据并计算 ElasticNet 系数路径...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("2. 进行 5 折交叉验证，联合寻找最佳 Alpha 与 l1_ratio...")
    l1_ratio_grid = [0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    enet_cv = ElasticNetCV(
        l1_ratio=l1_ratio_grid,
        cv=5,
        random_state=42,
        max_iter=20000,
        n_alphas=200,
    ).fit(X_scaled, y_train)
    best_alpha = float(enet_cv.alpha_)
    best_l1_ratio = float(enet_cv.l1_ratio_)
    print(f"找到最佳 Alpha: {best_alpha:.6f}")
    print(f"找到最佳 l1_ratio: {best_l1_ratio:.3f}")

    alphas_path, coefs_path, _ = enet_path(
        X_scaled,
        y_train,
        l1_ratio=best_l1_ratio,
        eps=1e-4,
        n_alphas=200,
    )

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
    ax.set_title(
        f"ElasticNet 路径图：特征系数衰减（l1_ratio={best_l1_ratio:.3f}）",
        fontsize=16,
        fontweight="bold",
    )
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
    plot_path = os.path.join(PLOT_OUTPUT_DIR, "elasticnet_feature_selection_path.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"图表已保存: {plot_path}")
    plt.close(fig)

    return enet_cv, scaler


def main():
    print("正在读取数据并构建特征...")
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
    raw_df = raw_df.dropna(subset=model_input_cols).reset_index(drop=True)

    df_features = build_features_pipeline(raw_df, target_col=TARGET_COL, date_col=DATE_COL)

    train_df = df_features[df_features[DATE_COL] <= TRAIN_END].copy()
    test_df = df_features[(df_features[DATE_COL] >= TEST_START) & (df_features[DATE_COL] <= TEST_END)].copy()

    drop_cols = [DATE_COL, TARGET_COL]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    if X_train.empty:
        raise ValueError("训练集为空，请检查时间范围和数据。")
    if X_test.empty:
        raise ValueError("测试集为空，请检查时间范围和数据。")

    feature_names = X_train.columns.tolist()
    best_enet, fitted_scaler = plot_enet_path_and_train(X_train, y_train, feature_names)

    print("\n" + "=" * 50)
    print("ElasticNet 最终特征权重:")
    coef_dict = dict(zip(feature_names, best_enet.coef_))
    for feat, weight in sorted(coef_dict.items(), key=lambda item: abs(item[1]), reverse=True):
        print(f"  {feat}: {weight:.4f}")
    print("=" * 50)

    X_test_scaled = fitted_scaler.transform(X_test)
    test_pred = best_enet.predict(X_test_scaled)

    result_df = pd.DataFrame(
        {
            DATE_COL: test_df[DATE_COL].astype(str).values,
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
    metrics = _regression_metrics(y_test.astype(float), test_pred.astype(float))
    metrics_df = pd.DataFrame([metrics])

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    excel_path = os.path.join(
        OUTPUT_ROOT,
        f"{PROCESSED_PROVINCE}_elasticnet_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    with pd.ExcelWriter(excel_path) as writer:
        result_df.to_excel(writer, index=False, sheet_name="test_forecast")
        metrics_df.to_excel(writer, index=False, sheet_name="metrics")
    print(f"测试集预测结果已保存: {excel_path}")

    print(f"\nElasticNet 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

