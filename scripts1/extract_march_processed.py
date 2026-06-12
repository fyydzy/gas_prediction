"""
按 feature_engineering1 流水线构造特征，导出 2021-2026 年 3 月上旬样本。

旬起始日为每月 1 日，故 3 月上旬对应 date 为每年 YYYY-03-01。
输出包含天然气销量 gas_sales 及特征工程后的对应特征列。

默认：河北 -> output1/河北_2021_2026_march_shangxun_features.xlsx

用法：
  set PYTHONPATH=src
  uv run scripts1/extract_march_processed.py
  uv run scripts1/extract_march_processed.py --province 河北 --output output1/custom.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gas_prediction.feature_engineering1 import build_features_pipeline
from gas_prediction.forecast_common1 import (
    DATE_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
)


START_YEAR = 2021
END_YEAR = 2026

MODEL_INPUT_COLS = [
    DATE_COL,
    TARGET_COL,
    "avg_temp",
    "max_temp",
    "min_temp",
    "HDD",
    "extreme_cold_days",
]


def _load_and_build_features(province: str) -> tuple[Path, pd.DataFrame]:
    input_path = Path(find_processed_excel(province))
    raw_df = pd.read_excel(input_path)

    missing = set(MODEL_INPUT_COLS) - set(raw_df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {sorted(missing)}")

    model_df = raw_df[MODEL_INPUT_COLS].copy()
    model_df[DATE_COL] = pd.to_datetime(model_df[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in MODEL_INPUT_COLS:
        if col != DATE_COL:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna(subset=MODEL_INPUT_COLS).reset_index(drop=True)

    features_df = build_features_pipeline(model_df, target_col=TARGET_COL, date_col=DATE_COL)
    return input_path, features_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 feature_engineering1 构造特征后，提取 2021-2026 年 3 月上旬样本到新表"
    )
    parser.add_argument("--province", type=str, default=PROCESSED_PROVINCE, help="省份名，对应 {省}.xlsx")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出 xlsx 路径；默认 output1/{省}_2021_2026_march_shangxun_features.xlsx",
    )
    args = parser.parse_args()

    input_path, df_features = _load_and_build_features(args.province)

    dates = pd.to_datetime(df_features[DATE_COL], errors="coerce")
    # 3 月上旬：旬度表以旬起始日表示，上旬为每月 1 日 -> 3 月仅 YYYY-03-01
    march_shangxun_mask = (
        (dates.dt.year >= START_YEAR)
        & (dates.dt.year <= END_YEAR)
        & (dates.dt.month == 3)
        & (dates.dt.day == 1)
    )
    out = df_features.loc[march_shangxun_mask].copy()
    out["_sort_date"] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out = out.sort_values("_sort_date").drop(columns=["_sort_date"]).reset_index(drop=True)
    out.insert(0, "calendar_year", pd.to_datetime(out[DATE_COL], errors="coerce").dt.year)

    out_path = (
        Path(args.output)
        if args.output
        else Path("output1") / f"{args.province}_{START_YEAR}_{END_YEAR}_march_shangxun_features.xlsx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(out_path, index=False, sheet_name="march_shangxun")

    print(f"读取: {input_path}")
    print(f"特征工程后总行数: {len(df_features)} -> {START_YEAR}-{END_YEAR} 年 3 月上旬（03-01）行数: {len(out)}")
    print(f"已写入: {out_path.resolve()}")


if __name__ == "__main__":
    main()
