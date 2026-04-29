"""销量验证：数据加载、切分、指标与结果表"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

AS_OF_MONTH = "2025-06"
VAL_START = "2025-11"
VAL_END = "2026-03"
TARGET_COL = "gas_sales"
MONTH_COL = "month"

# 要跑哪个省：对应 `data/processed_data/{PROCESSED_PROVINCE}.xlsx`
PROCESSED_PROVINCE = "河北"


def month_range(start: str, end: str) -> list[str]:
    return list(pd.period_range(start=start, end=end, freq="M").astype(str))


def find_processed_excel(province: str | None = None) -> str:
    prov = PROCESSED_PROVINCE if province is None else province
    candidates = [
        os.path.join("data", "processed_data", f"{prov}.xlsx"),
        os.path.join("processed_data", f"{prov}.xlsx"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"未找到 {prov} 的 processed 文件，请确认 data/processed_data/{prov}.xlsx 存在。"
    )


def load_gas_series(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {MONTH_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"文件缺少必要列: {missing}")

    out = df[[MONTH_COL, TARGET_COL]].copy()
    out[MONTH_COL] = out[MONTH_COL].astype(str).str.slice(0, 7)
    out = out.sort_values(MONTH_COL).reset_index(drop=True)
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return out


def split_asof_bridge(
    df: pd.DataFrame,
    as_of_month: str = AS_OF_MONTH,
    val_start: str = VAL_START,
    val_end: str = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """截至 as_of 为训练集；返回验证段、桥接月份列表（as_of 次月至验证前一月）。"""
    train = df[df[MONTH_COL] <= as_of_month].copy()
    val = df[(df[MONTH_COL] >= val_start) & (df[MONTH_COL] <= val_end)].copy()
    bridge_start = str((pd.Period(as_of_month, freq="M") + 1).strftime("%Y-%m"))
    bridge_end = str((pd.Period(val_start, freq="M") - 1).strftime("%Y-%m"))
    bridge_months = month_range(bridge_start, bridge_end)
    if train.empty:
        raise ValueError("训练集为空，请检查数据。")
    if val.empty:
        raise ValueError("验证集为空，请检查月份范围。")
    return train, val, bridge_months


def split_asof_forecast9(
    df: pd.DataFrame,
    as_of_month: str = AS_OF_MONTH,
    val_start: str = VAL_START,
    val_end: str = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """两阶段里的「一次性预测 9 个月」：forecast_months = 桥接 + 验证。"""
    train, val, bridge_months = split_asof_bridge(df, as_of_month, val_start, val_end)
    forecast_months = bridge_months + list(val[MONTH_COL].astype(str).values)
    return train, val, forecast_months


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    non_zero = np.where(y_true != 0, y_true, np.nan)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / non_zero)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}


def finalize_validation_table_9m(
    forecast_months: list[str],
    predictions: np.ndarray,
    val_df: pd.DataFrame,
    val_start: str = VAL_START,
    val_end: str = VAL_END,
    month_col: str = MONTH_COL,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """合并预测与真实值，打 phase / error；返回 (result, y_true, y_pred) 仅评估段。"""
    forecast_df = pd.DataFrame(
        {
            month_col: forecast_months,
            "predicted_gas_sales": predictions[: len(forecast_months)],
        }
    )
    result = forecast_df.merge(
        val_df[[month_col, target_col]],
        on=month_col,
        how="left",
    ).rename(columns={target_col: "actual_gas_sales"})
    result["phase"] = np.where(
        (result[month_col] >= val_start) & (result[month_col] <= val_end),
        "evaluation",
        "bridge(unknown_at_asof)",
    )
    result["error"] = result["predicted_gas_sales"] - result["actual_gas_sales"]
    result["abs_error"] = np.abs(result["error"])

    eval_mask = result["phase"] == "evaluation"
    y_true = result.loc[eval_mask, "actual_gas_sales"].to_numpy(dtype=float)
    y_pred = result.loc[eval_mask, "predicted_gas_sales"].to_numpy(dtype=float)
    return result, y_true, y_pred


def build_result_two_stage(
    bridge_months: list[str],
    bridge_pred: np.ndarray,
    val_months: np.ndarray,
    val_pred: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """两阶段预测：桥接段 + 评估段拼成一张表。"""
    bridge_result = pd.DataFrame(
        {
            MONTH_COL: bridge_months,
            "actual_gas_sales": np.nan,
            "predicted_gas_sales": bridge_pred[: len(bridge_months)],
            "phase": "bridge(unknown_at_asof)",
        }
    )
    eval_result = pd.DataFrame(
        {
            MONTH_COL: val_months,
            "actual_gas_sales": y_true,
            "predicted_gas_sales": val_pred,
            "error": val_pred - y_true,
            "abs_error": np.abs(val_pred - y_true),
            "phase": "evaluation",
        }
    )
    return pd.concat([bridge_result, eval_result], ignore_index=True)
