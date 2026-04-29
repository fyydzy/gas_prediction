import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, lasso_path
import warnings

warnings.filterwarnings('ignore')

# 解决 Matplotlib 运行时的中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 引入项目内特征工程与数据读取
from gas_prediction.feature_engineering import build_features_pipeline
from gas_prediction.forecast_common import (
    MONTH_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
)

# === 严格复用你的时间线逻辑 ===
TRAIN_END = "2025-06"
TEST_START = "2025-11"
TEST_END = "2026-03"
PLOT_OUTPUT_DIR = os.path.join("output", "lasso")

def plot_lasso_path_and_train(X_train, y_train, feature_names):
    """
    核心制图函数：计算并绘制 Lasso 系数衰减路径，同时返回最佳模型
    """
    print("\n1. 标准化数据并计算 Lasso 系数路径...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 计算路径：尝试 200 个不同的 alpha，看看系数是如何变化的
    alphas_path, coefs_path, _ = lasso_path(X_scaled, y_train, eps=1e-4, n_alphas=200)

    print("2. 进行 5 折交叉验证，寻找全局最佳的惩罚力度 Alpha...")
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_alphas=200).fit(X_scaled, y_train)
    best_alpha = lasso_cv.alpha_
    print(f"找到最佳 Alpha: {best_alpha:.4f}")

    # ================= 绘制变量选择图 =================
    n_features = coefs_path.shape[0]
    legend_cols = 4
    legend_rows = int(np.ceil((n_features + 1) / legend_cols))  # +1 给红色 alpha 线
    fig_height = 8 + legend_rows * 0.45
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # 将 X 轴转换为对数坐标，方便观察
    log_alphas = np.log10(alphas_path)
    
    # 遍历画出每一个特征的“衰减折线”（全部特征都显示在图例中）
    for i in range(coefs_path.shape[0]):
        ax.plot(log_alphas, coefs_path[i], linewidth=2, label=feature_names[i])

    # 画一条红色的垂直虚线，代表交叉验证选出来的“最佳截断点”
    ax.axvline(
        x=np.log10(best_alpha),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'最佳 Alpha 截断线 (Alpha={best_alpha:.4f})',
    )

    # 图表美化设置
    ax.set_xlabel('Log10(Alpha) - 惩罚力度 (向右滑动惩罚越大)', fontsize=14)
    ax.set_ylabel('特征权重 (Coefficients)', fontsize=14)
    ax.set_title('Lasso 变量选择全过程：特征系数随惩罚力度的衰减路径图', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    # 图例放在底部多列显示，避免被裁剪
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_cols,
        fontsize=10,
        frameon=False,
    )
    fig.subplots_adjust(bottom=min(0.10 + legend_rows * 0.06, 0.55))
    
    # 自动保存为高清图片（output/lasso 下）
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_OUTPUT_DIR, "lasso_feature_selection_path.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {plot_path}")
    
    # 弹出窗口显示图表
    plt.show()
    # ===================================================

    return lasso_cv, scaler

def main():
    # 1. 读取真实数据并生成特征
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
    raw_df = raw_df.dropna(subset=[MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp"]).reset_index(drop=True)

    df_features = build_features_pipeline(raw_df, target_col=TARGET_COL, month_col=MONTH_COL)

    # 2. 严格切分时间线 (必须在截断前的数据上做特征选择)
    train_df = df_features[df_features[MONTH_COL] <= TRAIN_END].copy()
    test_df = df_features[(df_features[MONTH_COL] >= TEST_START) & 
                          (df_features[MONTH_COL] <= TEST_END)].copy()

    drop_cols = [MONTH_COL, TARGET_COL]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[TARGET_COL].values
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COL].values

    if X_train.empty:
        raise ValueError("训练集为空，请检查时间范围和数据。")
    if X_test.empty:
        raise ValueError("测试集为空，请检查时间范围和数据。")

    feature_names = X_train.columns.tolist()

    # 3. 传入核心制图函数
    best_lasso, fitted_scaler = plot_lasso_path_and_train(X_train, y_train, feature_names)

    # 4. 打印最终存活的特征名单
    print("\n" + "="*50)
    print("经过红线截断后，最终存活特征及权重:")
    coef_dict = dict(zip(feature_names, best_lasso.coef_))
    surviving_features = {k: v for k, v in coef_dict.items() if abs(v) > 1e-5}

    if surviving_features:
        for feat, weight in sorted(surviving_features.items(), key=lambda item: abs(item[1]), reverse=True):
            print(f"  {feat}: {weight:.2f}")
    else:
        print("  无存活特征（系数均接近 0）。")
    print(f"总共测试了 {len(feature_names)} 个特征，剔除了 {len(feature_names) - len(surviving_features)} 个。")
    print("="*50)

    # 5. 在未知的测试集上进行终极大考
    # 预测前必须使用训练集的 scaler 进行标准化
    X_test_scaled = fitted_scaler.transform(X_test)
    test_pred = best_lasso.predict(X_test_scaled)

    # 6. 保存测试段预测结果到 Excel
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
        f"{PROCESSED_PROVINCE}_lasso_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    result_df.to_excel(excel_path, index=False, sheet_name="test_forecast")
    print(f"测试集预测结果已保存: {excel_path}")
    
    # 避免测试集为 0 导致 MAPE 报错
    if len(y_test) > 0 and not np.any(y_test == 0):
        mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
        print(f"\nLasso 特征筛选版 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
        print(f"Test MAPE: {mape:.4f}%")
    else:
        print("测试集包含 0 或为空，跳过 MAPE 输出。")

if __name__ == "__main__":
    main()