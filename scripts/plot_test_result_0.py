import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 运行示例：
# uv run scripts/plot_test_result_0.py --province 北京 --all-models --model-keyword prophet_xgbres

PROVINCE = "内蒙古"  # 这里可直接改省份
MODEL_KEYWORD = "prophet"  # 可选：如 "prophet_xgbres"、"timesfm_xgbres"，留空表示不限制
OUTPUT_DIR = "output"

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _select_result_file(output_dir: Path, province: str, model_keyword: str) -> Path:
    candidates = []
    for path in output_dir.glob("*.xlsx"):
        name = path.name
        if name.startswith("~$"):
            continue
        if province not in name:
            continue
        if model_keyword and model_keyword not in name:
            continue
        candidates.append(path)

    if not candidates:
        hint = f"，模型关键字={model_keyword}" if model_keyword else ""
        raise FileNotFoundError(f"未找到省份={province}{hint} 的结果文件，请检查 output 目录和参数。")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    print(f"选中文件: {chosen}")
    if len(candidates) > 1:
        print("检测到多个候选文件，按最后修改时间选择最新文件。")
    return chosen


def _collect_result_files(output_dir: Path, province: str) -> list[Path]:
    candidates = []
    for path in output_dir.glob("*.xlsx"):
        name = path.name
        if name.startswith("~$"):
            continue
        if province not in name:
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"未找到省份={province} 的结果文件。")
    candidates.sort(key=lambda p: p.name.lower())
    return candidates


def _model_label_from_filename(path: Path, province: str) -> str:
    stem = path.stem
    prefix = f"{province}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem


def _short_model_name(path: Path, province: str) -> str:
    """条形图等处的短名称，避免 ctx/日期/9m 等长文件名。"""
    stem = _model_label_from_filename(path, province)
    s = stem.lower()
    if s.startswith("chronos"):
        return "Chronos"
    if "prophet_xgbres" in s:
        return "Prophet"
    if "prophet_lgbmres" in s:
        return "Prophet+LGBM"
    if s.startswith("prophet"):
        return "Prophet"
    if "timesfm_xgbres" in s:
        return "TimesFM+XGB"
    if "timesfm" in s:
        return "TimesFM"
    if "timemoe" in s:
        return "TimeMoE"
    return stem.split("_")[0].title() if "_" in stem else stem


def _annotate_sales_labels(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    *,
    offset_xy_points: tuple[float, float],
    va: str,
) -> None:
    """在折线点上标注数值，用像素偏移避免真实值/预测值挤在一起。"""
    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.0f}",
            xy=(float(xi), float(yi)),
            xytext=offset_xy_points,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color=color,
            clip_on=True,
        )


def _load_test_segment(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"month", "actual_gas_sales", "predicted_gas_sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"结果文件缺少必要列: {sorted(missing)}")

    if "phase" in df.columns:
        test_df = df[df["phase"].astype(str).str.contains("evaluation", na=False)].copy()
    else:
        test_df = df.copy()

    test_df["month"] = test_df["month"].astype(str)
    test_df["actual_gas_sales"] = pd.to_numeric(test_df["actual_gas_sales"], errors="coerce")
    test_df["predicted_gas_sales"] = pd.to_numeric(test_df["predicted_gas_sales"], errors="coerce")
    test_df = test_df.dropna(subset=["actual_gas_sales", "predicted_gas_sales"]).copy()
    test_df = test_df.sort_values("month").reset_index(drop=True)

    if test_df.empty:
        raise ValueError("测试段为空，无法作图。请确认 phase 列或数据内容。")

    test_df["mape_pct"] = (
        np.abs(test_df["actual_gas_sales"] - test_df["predicted_gas_sales"])
        / np.where(test_df["actual_gas_sales"] != 0, test_df["actual_gas_sales"], np.nan)
        * 100.0
    )
    return test_df


def _plot_comparison(test_df: pd.DataFrame, province: str, out_path: Path) -> None:
    x = np.arange(len(test_df))
    months = test_df["month"].tolist()
    actual = test_df["actual_gas_sales"].to_numpy(dtype=float)
    pred = test_df["predicted_gas_sales"].to_numpy(dtype=float)
    mape = test_df["mape_pct"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    # 上图：真实值 vs 预测值（含数值标注）
    actual_color = "#1f77b4"
    pred_color = "#ff7f0e"
    axes[0].plot(x, actual, marker="o", color=actual_color, label="真实值")
    axes[0].plot(x, pred, marker="s", color=pred_color, label="预测值")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(months, rotation=30)
    axes[0].set_title(f"{province}预测值与真实值比较")
    axes[0].set_xlabel("月份")
    axes[0].set_ylabel("销量")
    axes[0].grid(alpha=0.3)
    axes[0].margins(y=0.15)
    axes[0].legend()
    _annotate_sales_labels(
        axes[0], x, actual, actual_color, offset_xy_points=(-8, 10), va="bottom"
    )
    _annotate_sales_labels(
        axes[0], x, pred, pred_color, offset_xy_points=(8, -10), va="top"
    )

    # 下图：各月 MAPE
    bars = axes[1].bar(x, mape, color="#4C78A8")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(months, rotation=30)
    axes[1].set_title(f"{province} 测试段各月 MAPE(%)")
    axes[1].set_xlabel("月份")
    axes[1].set_ylabel("MAPE（%）")
    axes[1].grid(axis="y", alpha=0.3)
    for rect, val in zip(bars, mape):
        axes[1].text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.savefig(out_path, dpi=180)
    print(f"已保存图像: {out_path}")
    plt.show()


def _plot_single_line_multi_model_bars(
    province: str,
    line_df: pd.DataFrame,
    model_avg_mapes: list[tuple[str, float, Path]],
    out_path: Path,
) -> None:
    x = np.arange(len(line_df))
    months = line_df["month"].tolist()
    actual = line_df["actual_gas_sales"].to_numpy(dtype=float)
    pred = line_df["predicted_gas_sales"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    left, right = axes

    # 上图：单模型真实值 vs 预测值（含数值标注）
    actual_color = "#1f77b4"
    pred_color = "#ff7f0e"
    left.plot(x, actual, marker="o", color=actual_color, label="真实值")
    left.plot(x, pred, marker="s", color=pred_color, label="预测值")
    left.set_xticks(x)
    left.set_xticklabels(months, rotation=30)
    left.set_title(f"{province}预测值与真实值比较")
    left.set_xlabel("月份")
    left.set_ylabel("销量")
    left.grid(alpha=0.3)
    left.margins(y=0.15)
    left.legend(loc="best")
    _annotate_sales_labels(
        left, x, actual, actual_color, offset_xy_points=(-8, 10), va="bottom"
    )
    _annotate_sales_labels(
        left, x, pred, pred_color, offset_xy_points=(8, -10), va="top"
    )

    # 下图：多个模型平均 MAPE 对比条形图
    labels = [m[0] for m in model_avg_mapes]
    values = np.asarray([m[1] for m in model_avg_mapes], dtype=float)
    x_bar = np.arange(len(labels))
    bars = right.bar(x_bar, values, color="#4C78A8")
    right.set_xticks(x_bar)
    right.set_xticklabels(labels, rotation=45, ha="right")
    right.set_title(f"{province}多模型月均MAPE(%)对比")
    right.set_xlabel("模型")
    right.set_ylabel("月均MAPE（%）")
    right.grid(axis="y", alpha=0.3)
    for rect, val in zip(bars, values):
        right.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.savefig(out_path, dpi=180)
    print(f"已保存图像: {out_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制测试段：真实值/预测值 + 各月MAPE")
    parser.add_argument("--province", type=str, default=PROVINCE, help="省份名称，如 北京")
    parser.add_argument(
        "--model-keyword",
        type=str,
        default=MODEL_KEYWORD,
        help="文件名关键字过滤，如 prophet_xgbres；留空表示不限制",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="结果目录，默认 output",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="开启后左图只画一个模型，右图画多模型月均MAPE对比",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.all_models:
        # 左图使用一个模型（若给 model-keyword 就按关键字选，否则取最新）
        line_source = _select_result_file(output_dir, args.province, args.model_keyword)
        line_df = _load_test_segment(line_source)

        # 右图汇总该省多个模型平均 MAPE
        files = _collect_result_files(output_dir, args.province)
        model_avg_mapes: list[tuple[str, float, Path]] = []
        for path in files:
            try:
                test_df = _load_test_segment(path)
            except Exception as exc:  # pragma: no cover
                print(f"跳过文件 {path.name}，原因: {exc}")
                continue
            short = _short_model_name(path, args.province)
            avg_mape = float(np.nanmean(test_df["mape_pct"].to_numpy(dtype=float)))
            model_avg_mapes.append((short, avg_mape, path))

        if not model_avg_mapes:
            raise ValueError("没有可用于多模型 MAPE 对比的数据。")

        # 同一短名称多文件时只保留平均 MAPE 最优的一条，避免柱状图标签重叠
        best_by_short: dict[str, tuple[float, Path]] = {}
        for short, avg_mape, path in model_avg_mapes:
            if short not in best_by_short or avg_mape < best_by_short[short][0]:
                best_by_short[short] = (avg_mape, path)
        model_avg_mapes = [(s, v[0], v[1]) for s, v in best_by_short.items()]

        model_avg_mapes.sort(key=lambda x: x[1])
        for label, avg_mape, path in model_avg_mapes:
            print(f"{label} ({path.name}) | Avg MAPE={avg_mape:.4f}%")

        suffix = f"_{args.model_keyword}" if args.model_keyword else ""
        out_path = output_dir / f"{args.province}{suffix}_single_line_multi_model_bars.png"
        _plot_single_line_multi_model_bars(args.province, line_df, model_avg_mapes, out_path)
    else:
        source_path = _select_result_file(output_dir, args.province, args.model_keyword)
        test_df = _load_test_segment(source_path)
        suffix = f"_{args.model_keyword}" if args.model_keyword else ""
        out_path = output_dir / f"{args.province}{suffix}_test_compare_mape.png"
        _plot_comparison(test_df, args.province, out_path)


if __name__ == "__main__":
    main()

