import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from gas_prediction.forecast_common import MONTH_COL, PROCESSED_PROVINCE, TARGET_COL, find_processed_excel

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"
OUTDIR = os.path.join(OUTPUT_DIR, "time_analysis")

# 仅分析最近多少个月；<=0 表示使用全量。
CONTEXT_LEN = 60

# 异常月份判定：|residual| > 1.5 * std(residual)
RESID_STD_MULT = 1.5


def analyze_residuals_vs_weather(df: pd.DataFrame, target_col: str = TARGET_COL, month_col: str = MONTH_COL) -> pd.DataFrame:
    print("开始进行 STL 残差-气象归因分析（适配当前月度数据）...")
    df_analysis = df.copy()

    # 1) 基础清洗：月份/目标列
    df_analysis[month_col] = df_analysis[month_col].astype(str).str.slice(0, 7)
    df_analysis[target_col] = pd.to_numeric(df_analysis[target_col], errors="coerce")
    df_analysis = df_analysis.dropna(subset=[month_col, target_col]).sort_values(month_col).reset_index(drop=True)

    if CONTEXT_LEN > 0 and len(df_analysis) > CONTEXT_LEN:
        df_analysis = df_analysis.iloc[-CONTEXT_LEN:].copy()

    # 2) STL 分解得到残差（按月，周期=12）
    y = df_analysis[target_col].astype(float).to_numpy()
    if len(y) < 24:
        raise ValueError("样本量过少（<24），STL 分解与同月正常值统计会不稳定。")

    stl = STL(y, period=12, robust=True)
    res = stl.fit()
    df_analysis["sales_residual"] = res.resid.astype(float)
    df_analysis["trend"] = res.trend.astype(float)
    df_analysis["seasonal"] = res.seasonal.astype(float)

    # 3) 计算同月正常值并得到 anomaly
    df_analysis["month_idx"] = pd.to_datetime(df_analysis[month_col]).dt.month

    weather_cols = ["HDD", "extreme_cold_days", "avg_temp", "min_temp"]
    missing = sorted(set(weather_cols) - set(df_analysis.columns))
    if missing:
        raise ValueError(f"数据中缺少气象列: {missing}。请先运行 process_all.py 生成这些列。")

    monthly_normals = df_analysis.groupby("month_idx")[weather_cols].mean().reset_index()
    monthly_normals = monthly_normals.rename(
        columns={
            "HDD": "HDD_normal",
            "extreme_cold_days": "cold_days_normal",
            "avg_temp": "avg_temp_normal",
            "min_temp": "min_temp_normal",
        }
    )
    df_analysis = pd.merge(df_analysis, monthly_normals, on="month_idx", how="left")

    df_analysis["HDD_anomaly"] = df_analysis["HDD"] - df_analysis["HDD_normal"]
    df_analysis["cold_days_anomaly"] = df_analysis["extreme_cold_days"] - df_analysis["cold_days_normal"]
    df_analysis["avg_temp_anomaly"] = df_analysis["avg_temp"] - df_analysis["avg_temp_normal"]
    df_analysis["min_temp_anomaly"] = df_analysis["min_temp"] - df_analysis["min_temp_normal"]

    # 4) 相关性（皮尔逊相关系数）
    corr_hdd = df_analysis["HDD_anomaly"].corr(df_analysis["sales_residual"])
    corr_cold = df_analysis["cold_days_anomaly"].corr(df_analysis["sales_residual"])
    corr_avg = df_analysis["avg_temp_anomaly"].corr(df_analysis["sales_residual"])
    corr_min = df_analysis["min_temp_anomaly"].corr(df_analysis["sales_residual"])

    print("\n归因诊断结果（皮尔逊相关系数）：")
    print(f"- HDD 异常度 vs 残差: {corr_hdd:.3f}")
    print(f"- 极端低温天数异常度 vs 残差: {corr_cold:.3f}")
    print(f"- 平均气温异常度 vs 残差: {corr_avg:.3f}")
    print(f"- 最低气温异常度 vs 残差: {corr_min:.3f}")

    # 5) 四宫格图（保存）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(16, 10))
    features_to_plot = [
        ("HDD_anomaly", "HDD Anomaly（偏离同月均值）", corr_hdd, "red"),
        ("cold_days_anomaly", "极端低温天数 Anomaly", corr_cold, "red"),
        ("avg_temp_anomaly", "平均气温 Anomaly（实际-同月均值）", corr_avg, "blue"),
        ("min_temp_anomaly", "最低气温 Anomaly（实际-同月均值）", corr_min, "blue"),
    ]

    for i, (col, xlabel, corr, color) in enumerate(features_to_plot, 1):
        plt.subplot(2, 2, i)
        x = pd.to_numeric(df_analysis[col], errors="coerce").to_numpy(dtype=float)
        y_resid = pd.to_numeric(df_analysis["sales_residual"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y_resid)
        x = x[mask]
        y_resid = y_resid[mask]
        plt.scatter(x, y_resid, alpha=0.6)
        if len(x) >= 2 and float(np.std(x)) > 0:
            a, b = np.polyfit(x, y_resid, deg=1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 50)
            ys = a * xs + b
            plt.plot(xs, ys, color=color)

        # 在散点图上标注异常月份（同下方异常表一致的口径：|residual| > 1.5 * std）
        std_resid_for_plot = float(df_analysis["sales_residual"].std())
        if std_resid_for_plot > 0:
            anom_mask = np.abs(df_analysis["sales_residual"]) > RESID_STD_MULT * std_resid_for_plot
            anom_df = df_analysis.loc[anom_mask, [month_col, col, "sales_residual"]].copy()
            anom_df[col] = pd.to_numeric(anom_df[col], errors="coerce")
            anom_df["sales_residual"] = pd.to_numeric(anom_df["sales_residual"], errors="coerce")
            anom_df = anom_df.dropna(subset=[col, "sales_residual"])
            if not anom_df.empty:
                plt.scatter(
                    anom_df[col].to_numpy(dtype=float),
                    anom_df["sales_residual"].to_numpy(dtype=float),
                    color="red",
                    s=35,
                    zorder=5,
                )
                for _, r in anom_df.iterrows():
                    plt.annotate(
                        str(r[month_col])[:7],
                        xy=(float(r[col]), float(r["sales_residual"])),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                        color="red",
                    )
        plt.title(f"{col} vs Sales Residual\\nCorrelation: {corr:.2f}")
        plt.xlabel(xlabel)
        plt.ylabel("Sales Residual")
        plt.axhline(0, color="gray", linestyle="--")
        plt.axvline(0, color="gray", linestyle="--")

    plt.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    out_png = os.path.join(OUTDIR, f"{PROCESSED_PROVINCE}_stl_residual_weather_attribution.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    # 6) 异常月份表（同你之前的标准：1.5 倍标准差）
    std_resid = float(df_analysis["sales_residual"].std())
    if std_resid > 0:
        anomalies = df_analysis[np.abs(df_analysis["sales_residual"]) > RESID_STD_MULT * std_resid].copy()
    else:
        anomalies = df_analysis.iloc[0:0].copy()

    report_cols = [
        month_col,
        "sales_residual",
        "HDD_anomaly",
        "cold_days_anomaly",
        "avg_temp_anomaly",
        "min_temp_anomaly",
    ]
    anomalies = anomalies[report_cols].sort_values("sales_residual", ascending=False)

    out_xlsx = os.path.join(OUTDIR, f"{PROCESSED_PROVINCE}_stl_residual_weather_attribution.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_analysis.to_excel(writer, index=False, sheet_name="all")
        anomalies.to_excel(writer, index=False, sheet_name="anomalies")

    print("\n输出已保存：")
    print(f"- 归因图: {out_png}")
    print(f"- 明细与异常月: {out_xlsx}")

    return df_analysis


def main() -> None:
    path = find_processed_excel()
    df = pd.read_excel(path)
    analyze_residuals_vs_weather(df)


if __name__ == "__main__":
    main()