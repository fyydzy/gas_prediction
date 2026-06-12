"""
汇总 output1 下若干树/线性模型的测试集预测结果：
- 一张图：真实销量 vs 各模型预测（按旬 date 对齐）
- 按自然月加总：每月内各旬销量相加得到「月总真实 / 月总预测」，再在该月度序列上算 MAE/MSE/RMSE/MAPE/R²；
  同时输出按旬粒度的整体指标作对照。
- 简要文字分析（控制台 + 写入 txt）

默认将所有产出放在「output1 下的专用子目录」（如 output1/model_comparison/），与各模型原始 xlsx 分开。
SARIMAX 单独约定：文件名为 `{省}_sarimax_*.xlsx`，工作表 `forecast`（其它模型为 `{省}_{model}_test_*.xlsx` + `test_forecast`）。

依赖：pandas、numpy、matplotlib、openpyxl、sklearn
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gas_prediction.forecast_common1 import DATE_COL, PROCESSED_PROVINCE

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_ORDER = [
    "lasso",
    "lightgbm",
    "random_forest",
    "ridge",
    "elasticnet",
    "xgboost",
    "catboost",
    "sarimax"
]

# 图例与曲线颜色（与 MODEL_ORDER 条数匹配或更多；不足时 plot 里会取模循环）
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#aec7e8",
    "#ffbb78",
]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float("nan")
    if len(y_true) >= 2 and np.var(y_true) > 1e-12:
        try:
            r2 = float(r2_score(y_true, y_pred))
        except ValueError:
            r2 = float("nan")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE(%)": _mape(y_true, y_pred),
        "R2": r2,
    }


def _find_model_files(output_dir: Path, province: str) -> dict[str, Path]:
    """匹配各模型结果 xlsx。树/线性脚本多为 {省}_{model}_test_*.xlsx；SARIMAX 为 {省}_sarimax_*.xlsx。"""
    out: dict[str, Path] = {}
    for key in MODEL_ORDER:
        if key == "sarimax":
            hits = sorted(output_dir.glob(f"{province}_sarimax_*.xlsx"))
        else:
            hits = sorted(output_dir.glob(f"{province}_{key}_test_*.xlsx"))
        if hits:
            out[key] = hits[-1]
    return out


def _load_model_forecast(path: Path, model_key: str) -> pd.DataFrame:
    """统一读成 date / actual_gas_sales / predicted_gas_sales。SARIMAX 使用 sheet forecast。"""
    sheet = "forecast" if model_key == "sarimax" else "test_forecast"
    df = pd.read_excel(path, sheet_name=sheet)
    if DATE_COL not in df.columns:
        raise ValueError(f"{path.name} 缺少列 {DATE_COL}")
    for c in ("actual_gas_sales", "predicted_gas_sales"):
        if c not in df.columns:
            raise ValueError(f"{path.name} 缺少列 {c}")
    out = df[[DATE_COL, "actual_gas_sales", "predicted_gas_sales"]].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out["actual_gas_sales"] = pd.to_numeric(out["actual_gas_sales"], errors="coerce")
    out["predicted_gas_sales"] = pd.to_numeric(out["predicted_gas_sales"], errors="coerce")
    out = out.dropna(subset=[DATE_COL, "actual_gas_sales", "predicted_gas_sales"])
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    return out


def _merge_predictions(paths: dict[str, Path]) -> tuple[pd.DataFrame, str | None]:
    """按 date 内连接各模型预测列；校验各文件中的 actual 与首文件一致。"""
    names = [k for k in MODEL_ORDER if k in paths]
    if not names:
        raise ValueError("paths 中无已知模型键")
    m = _load_model_forecast(paths[names[0]], names[0])
    m = m.rename(columns={"predicted_gas_sales": f"pred__{names[0]}"})
    warns: list[str] = []
    for nm in names[1:]:
        d = _load_model_forecast(paths[nm], nm)
        d = d.rename(columns={"predicted_gas_sales": f"pred__{nm}", "actual_gas_sales": "actual_r"})
        m = m.merge(d[[DATE_COL, "actual_r", f"pred__{nm}"]], on=DATE_COL, how="inner")
        diff = (m["actual_gas_sales"] - m["actual_r"]).abs().max()
        if diff > 1e-4:
            warns.append(f"{nm}: actual 与基准最大偏差 {float(diff):.6g}")
        m = m.drop(columns=["actual_r"])
    m = m.sort_values(DATE_COL).reset_index(drop=True)
    if m.empty:
        raise ValueError("各模型结果按 date 内连接后为空，请检查测试日期范围是否一致。")
    return m, ("；".join(warns) if warns else None)


def _monthly_sum_detail_and_overall(
    wide: pd.DataFrame, model_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    自然月内对 actual、各模型 pred 分别求和，得到月度总量序列；
    - detail：每月 × 每模型的月总和与单月相对误差；
    - overall：各模型在「月总真实 vs 月总预测」K 个点上的回归指标（K=测试段跨月数）。
    """
    w = wide.copy()
    w["year_month"] = pd.to_datetime(w[DATE_COL]).dt.to_period("M")
    n_by = w.groupby("year_month", sort=True).size().rename("n_tendays_in_month")
    agg_dict: dict[str, str] = {"actual_gas_sales": "sum"}
    for mn in model_names:
        agg_dict[f"pred__{mn}"] = "sum"
    mt = w.groupby("year_month", sort=True).agg(agg_dict)
    mt = mt.join(n_by, how="left")

    y_vec = mt["actual_gas_sales"].to_numpy(dtype=float)
    detail_rows: list[dict] = []
    for i, ym in enumerate(mt.index):
        ym_str = str(ym)
        actual_sum = float(mt["actual_gas_sales"].iloc[i])
        nt = int(mt["n_tendays_in_month"].iloc[i])
        for mn in model_names:
            pred_sum = float(mt[f"pred__{mn}"].iloc[i])
            err = pred_sum - actual_sum
            abs_err = abs(err)
            mape_m = (abs(err) / actual_sum * 100.0) if abs(actual_sum) > 1e-12 else float("nan")
            detail_rows.append(
                {
                    "year_month": ym_str,
                    "model": mn,
                    "n_tendays_in_month": nt,
                    "actual_month_sum": actual_sum,
                    "predicted_month_sum": pred_sum,
                    "error_month": err,
                    "abs_error_month": abs_err,
                    "MAPE_month_on_total(%)": mape_m,
                    "MAE": abs_err,
                    "MSE": err**2,
                    "RMSE": abs_err,
                    "MAPE(%)": mape_m,
                    "R2": float("nan"),
                }
            )

    overall_rows: list[dict] = []
    for mn in model_names:
        p_vec = mt[f"pred__{mn}"].to_numpy(dtype=float)
        overall_rows.append({"model": mn, **_metrics_block(y_vec, p_vec)})

    return pd.DataFrame(detail_rows), pd.DataFrame(overall_rows)


def _build_analysis(
    wide: pd.DataFrame,
    model_names: list[str],
    monthly_sum_detail: pd.DataFrame,
    overall_monthly: pd.DataFrame,
    overall_tenday: pd.DataFrame,
    align_note: str | None,
) -> str:
    lines: list[str] = []
    n_months = int(monthly_sum_detail["year_month"].nunique()) if not monthly_sum_detail.empty else 0

    lines.append("【按月加总后的整体指标（主）】")
    lines.append(
        f"每月将旬度 actual / 各模型 pred 在日历月内求和，得到 {n_months} 个「月总量」样本点，再算 MAE/MSE/RMSE/MAPE/R²。"
    )
    om = overall_monthly.set_index("model").sort_values("MAE")
    lines.append("按 MAE（月总量序列）从小到大: " + ", ".join(f"{i}(MAE={r['MAE']:.4f})" for i, r in om.iterrows()))
    lines.append(f"月总量口径下相对最优: {om.index[0]}；相对最弱: {om.index[-1]}。")

    lines.append("\n【按旬粒度的整体指标（对照）】")
    ot = overall_tenday.set_index("model").sort_values("MAE")
    lines.append("按 MAE（逐旬）从小到大: " + ", ".join(f"{i}(MAE={r['MAE']:.4f})" for i, r in ot.iterrows()))

    if align_note:
        lines.append(f"\n对齐提示: {align_note}")

    if not monthly_sum_detail.empty:
        mape_pivot = monthly_sum_detail.pivot_table(
            index="year_month", columns="model", values="MAPE_month_on_total(%)"
        )
        lines.append("\n【各月「月总量」相对误差 MAPE(%)（单月：|月总预测-月总真实|/月总真实）】")
        lines.append(mape_pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比 output1 中多模型测试集预测并出图；指标按「自然月加总」与「逐旬」两种口径输出"
    )
    parser.add_argument("--output-dir", type=str, default="output1", help="结果目录（默认 output1）")
    parser.add_argument("--province", type=str, default=PROCESSED_PROVINCE, help="文件名前缀省份")
    parser.add_argument("--out-prefix", type=str, default="model_comparison", help="输出文件前缀")
    parser.add_argument(
        "--result-subdir",
        type=str,
        default="model_comparison1",
        help="在 output-dir 下新建的子文件夹名，用于存放对比图、指标 Excel、分析 txt（默认 model_comparison）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {output_dir}")

    paths = _find_model_files(output_dir, args.province)
    missing = [k for k in MODEL_ORDER if k not in paths]
    if missing:
        print("警告: 未找到以下模型文件（将跳过）: " + ", ".join(missing))
    if not paths:
        raise FileNotFoundError(
            f"在 {output_dir} 下未找到形如 {args.province}_<model>_test_*.xlsx 的文件。"
        )

    wide, align_note = _merge_predictions(paths)
    model_names = [k for k in MODEL_ORDER if k in paths]

    # ---------- 图 ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    x = pd.to_datetime(wide[DATE_COL])
    ax.plot(x, wide["actual_gas_sales"], color="black", linewidth=2.4, label="真实值", marker="o", markersize=4)
    for i, mn in enumerate(model_names):
        c = COLORS[i % len(COLORS)]
        ax.plot(
            x,
            wide[f"pred__{mn}"],
            color=c,
            linewidth=1.5,
            alpha=0.9,
            label=mn.replace("_", " "),
            marker=".",
            markersize=5,
        )
    ax.set_xlabel("旬 (date)")
    ax.set_ylabel("燃气销量")
    ax.set_title(f"测试集：真实 vs 多模型预测（{args.province}）")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    plot_dir = output_dir / args.result_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{args.out_prefix}_{args.province}_test_actual_vs_pred.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存: {plot_path}")

    # ---------- 按月加总明细 + 月总量整体指标；另保留逐旬整体指标 ----------
    monthly_sum_detail, overall_monthly = _monthly_sum_detail_and_overall(wide, model_names)
    overall_tenday_rows = []
    y_t = wide["actual_gas_sales"].to_numpy(dtype=float)
    for mn in model_names:
        overall_tenday_rows.append(
            {"model": mn, **_metrics_block(y_t, wide[f"pred__{mn}"].to_numpy(dtype=float))}
        )
    overall_tenday_df = pd.DataFrame(overall_tenday_rows)

    excel_path = plot_dir / f"{args.out_prefix}_{args.province}_metrics.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        wide.to_excel(writer, index=False, sheet_name="test_wide")
        monthly_sum_detail.to_excel(writer, index=False, sheet_name="monthly_sum_detail")
        overall_monthly.to_excel(writer, index=False, sheet_name="test_overall_monthly_sum")
        overall_tenday_df.to_excel(writer, index=False, sheet_name="test_overall_by_tenday")

    print(f"表格已保存: {excel_path}")

    analysis = _build_analysis(
        wide,
        model_names,
        monthly_sum_detail,
        overall_monthly,
        overall_tenday_df,
        align_note,
    )
    print("\n" + analysis)
    txt_path = plot_dir / f"{args.out_prefix}_{args.province}_analysis.txt"
    txt_path.write_text(analysis, encoding="utf-8")
    print(f"\n分析已写入: {txt_path}")


if __name__ == "__main__":
    main()
