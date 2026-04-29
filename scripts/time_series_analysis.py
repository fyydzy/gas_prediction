import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from gas_prediction.forecast_common import MONTH_COL, PROCESSED_PROVINCE, TARGET_COL, find_processed_excel

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"
DECOMPOSE_DIR = os.path.join(OUTPUT_DIR, "time_analysis")
SEASONAL_PERIOD = 12  # 月度数据按 12 期分解
# 仅使用最近多少个月做分解；<=0 表示使用全量历史数据。
CONTEXT_LEN = 69
# 异常值判定：残差绝对值 > 1.5 倍标准差
RESID_OUTLIER_STD_MULT = 1.5


def _load_monthly_series() -> pd.Series:
    input_path = find_processed_excel()
    df = pd.read_excel(input_path)
    required = {MONTH_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {sorted(missing)}")

    df = df.copy()
    df[MONTH_COL] = df[MONTH_COL].astype(str).str.slice(0, 7)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[MONTH_COL, TARGET_COL]).sort_values(MONTH_COL).reset_index(drop=True)

    series = df.set_index(pd.PeriodIndex(df[MONTH_COL], freq="M").to_timestamp())[TARGET_COL].astype(float)
    series = series.asfreq("MS")
    series = series.interpolate(limit_direction="both")
    return series


def _plot_decompose(result, out_png: str) -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("月度时间序列分解（趋势 / 季节 / 残差）", y=0.98)

    # 在残差子图中高亮极端值并标注对应月份
    axes = fig.get_axes()
    if axes:
        resid_ax = axes[-1]
        resid = pd.Series(result.resid, index=result.observed.index).dropna()
        if not resid.empty:
            std = float(np.std(resid.values))
            if std <= 0:
                outlier_series = resid.iloc[0:0]
            else:
                threshold = RESID_OUTLIER_STD_MULT * std
                outlier_series = resid[np.abs(resid.values) > threshold]

            if not outlier_series.empty:
                resid_ax.scatter(
                    outlier_series.index,
                    outlier_series.values,
                    color="red",
                    s=30,
                    zorder=5,
                    label="残差极端值",
                )
                for x, y in outlier_series.items():
                    resid_ax.annotate(
                        x.strftime("%Y-%m"),
                        xy=(x, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        color="red",
                        fontsize=8,
                    )
                resid_ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(DECOMPOSE_DIR, exist_ok=True)

    ts = _load_monthly_series()
    if CONTEXT_LEN > 0 and len(ts) > CONTEXT_LEN:
        ts = ts.iloc[-CONTEXT_LEN:].copy()
    if len(ts) < SEASONAL_PERIOD * 2:
        raise ValueError(f"样本量不足，至少需要 {SEASONAL_PERIOD * 2} 个观测点才能稳定分解。")

    stl = STL(ts, period=SEASONAL_PERIOD, robust=True)
    result = stl.fit()

    out_png = os.path.join(DECOMPOSE_DIR, f"{PROCESSED_PROVINCE}_stl_decompose.png")
    _plot_decompose(result, out_png)

    out_xlsx = os.path.join(DECOMPOSE_DIR, f"{PROCESSED_PROVINCE}_stl_decompose_components.xlsx")
    comp_df = pd.DataFrame(
        {
            MONTH_COL: ts.index.to_period("M").astype(str),
            "observed": result.observed.values,
            "trend": result.trend.values,
            "seasonal": result.seasonal.values,
            "resid": result.resid.values,
        }
    )
    comp_df.to_excel(out_xlsx, index=False, sheet_name="decompose")

    print("时间序列分解完成：")
    print(f"- 分解图: {out_png}")
    print(f"- 分解数据: {out_xlsx}")


if __name__ == "__main__":
    main()
