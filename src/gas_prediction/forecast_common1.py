"""旬度销量验证：数据加载、切分、指标与结果表"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

AS_OF_DATE = "2025-06-21"
VAL_START = "2025-11-01"
VAL_END = "2026-03-21"
TARGET_COL = "gas_sales"
DATE_COL = "date"
MONTH_COL = DATE_COL  # 兼容旧脚本中的 MONTH_COL 导入，旬度版本实际使用 date。

# 要跑哪个省：对应 `data/processed_data1/{PROCESSED_PROVINCE}.xlsx`
PROCESSED_PROVINCE = "河北"


def _normalize_tenday_start(value: str | pd.Timestamp) -> pd.Timestamp:
    date = pd.to_datetime(value)
    if date.day <= 10:
        day = 1
    elif date.day <= 20:
        day = 11
    else:
        day = 21
    return pd.Timestamp(year=date.year, month=date.month, day=day)


def _add_one_tenday(date: pd.Timestamp) -> pd.Timestamp:
    if date.day == 1:
        return pd.Timestamp(year=date.year, month=date.month, day=11)
    if date.day == 11:
        return pd.Timestamp(year=date.year, month=date.month, day=21)
    next_month = date + pd.DateOffset(months=1)
    return pd.Timestamp(year=next_month.year, month=next_month.month, day=1)


def _date_str(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m-%d")


def tenday_range(start: str, end: str) -> list[str]:
    current = _normalize_tenday_start(start)
    end_date = _normalize_tenday_start(end)
    dates = []
    while current <= end_date:
        dates.append(_date_str(current))
        current = _add_one_tenday(current)
    return dates


def month_range(start: str, end: str) -> list[str]:
    """兼容旧函数名；旬度版本返回上/中/下旬起始日期列表。"""
    return tenday_range(start, end)


def find_processed_excel(province: str | None = None) -> str:
    prov = PROCESSED_PROVINCE if province is None else province
    candidates = [
        os.path.join("data", "processed_data1", f"{prov}.xlsx"),
        os.path.join("processed_data1", f"{prov}.xlsx"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"未找到 {prov} 的旬度 processed 文件，请确认 data/processed_data1/{prov}.xlsx 存在。"
    )


def load_gas_series(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {DATE_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"文件缺少必要列: {missing}")

    out = df[[DATE_COL, TARGET_COL]].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out.dropna(subset=[DATE_COL, TARGET_COL])
    out[DATE_COL] = out[DATE_COL].map(_normalize_tenday_start).map(_date_str)
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    return out


def split_asof_bridge(
    df: pd.DataFrame,
    as_of_month: str = AS_OF_DATE,
    val_start: str = VAL_START,
    val_end: str = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """截至 as_of 为训练集；返回验证段、桥接旬列表（as_of 下一旬至验证前一旬）。"""
    as_of_date = _date_str(_normalize_tenday_start(as_of_month))
    val_start_date = _date_str(_normalize_tenday_start(val_start))
    val_end_date = _date_str(_normalize_tenday_start(val_end))

    train = df[df[DATE_COL] <= as_of_date].copy()
    val = df[(df[DATE_COL] >= val_start_date) & (df[DATE_COL] <= val_end_date)].copy()

    bridge_start = _add_one_tenday(_normalize_tenday_start(as_of_date))
    bridge_end = _normalize_tenday_start(val_start_date) - pd.Timedelta(days=1)
    bridge_dates = tenday_range(_date_str(bridge_start), _date_str(bridge_end))
    if train.empty:
        raise ValueError("训练集为空，请检查数据。")
    if val.empty:
        raise ValueError("验证集为空，请检查旬度日期范围。")
    return train, val, bridge_dates


def split_asof_forecast9(
    df: pd.DataFrame,
    as_of_month: str = AS_OF_DATE,
    val_start: str = VAL_START,
    val_end: str = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """两阶段里的「一次性预测」：forecast_dates = 桥接旬 + 验证旬。"""
    train, val, bridge_dates = split_asof_bridge(df, as_of_month, val_start, val_end)
    forecast_dates = bridge_dates + list(val[DATE_COL].astype(str).values)
    return train, val, forecast_dates


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
    """合并旬度预测与真实值，打 phase / error；返回 (result, y_true, y_pred) 仅评估段。"""
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
    val_start_date = _date_str(_normalize_tenday_start(val_start))
    val_end_date = _date_str(_normalize_tenday_start(val_end))
    result["phase"] = np.where(
        (result[month_col] >= val_start_date) & (result[month_col] <= val_end_date),
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
    """两阶段旬度预测：桥接段 + 评估段拼成一张表。"""
    bridge_result = pd.DataFrame(
        {
            DATE_COL: bridge_months,
            "actual_gas_sales": np.nan,
            "predicted_gas_sales": bridge_pred[: len(bridge_months)],
            "phase": "bridge(unknown_at_asof)",
        }
    )
    eval_result = pd.DataFrame(
        {
            DATE_COL: val_months,
            "actual_gas_sales": y_true,
            "predicted_gas_sales": val_pred,
            "error": val_pred - y_true,
            "abs_error": np.abs(val_pred - y_true),
            "phase": "evaluation",
        }
    )
    return pd.concat([bridge_result, eval_result], ignore_index=True)
