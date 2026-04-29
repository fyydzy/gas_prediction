import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


MONTH_START = "2016-04"
MONTH_END = "2026-03"
HDD_BASE_TEMP = 15.0
EXTREME_COLD_THRESHOLD = 0.0

# Province -> representative city used in temperature sheets (usually capital).
# If a province is not listed, we will try matching by province name directly.
PROVINCE_TO_TEMP_REGION: Dict[str, str] = {
    "北京": "北京",
    "天津": "天津",
    "河北": "石家庄",
    "山西": "太原",
    "内蒙古": "呼和浩特",
    "辽宁": "沈阳",
    "陕西": "西安",
    "甘肃": "兰州",
    "青海": "西宁",
    "宁夏": "银川",
    "新疆": "乌鲁木齐",
}


@dataclass(frozen=True)
class Paths:
    workspace: str
    original_data: str
    processed_data: str
    temperature_dir: str


def _find_temperature_dir(original_data_dir: str) -> str:
    # The folder name may appear garbled in some terminals due to encoding.
    # We simply pick the first subdirectory under original_data.
    subs = [
        os.path.join(original_data_dir, name)
        for name in os.listdir(original_data_dir)
        if os.path.isdir(os.path.join(original_data_dir, name))
    ]
    if not subs:
        raise FileNotFoundError(f"No subdirectory (temperature folder) found under {original_data_dir!r}")
    if len(subs) > 1:
        # If multiple subfolders exist later, prefer ones containing "温度".
        for p in subs:
            if "温度" in os.path.basename(p):
                return p
    return subs[0]


def _month_range(start_ym: str, end_ym: str) -> pd.Series:
    start = pd.Period(start_ym, freq="M")
    end = pd.Period(end_ym, freq="M")
    months = pd.period_range(start, end, freq="M").astype(str)
    return pd.Series(months, name="month")


def _province_from_filename(path: str) -> str:
    base = os.path.basename(path)
    if base.lower().endswith(".xlsx"):
        base = base[:-5]
    if base.lower().endswith(".xls"):
        base = base[:-4]
    return base.strip()


def _looks_like_month(x) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    s = str(x).strip()
    return bool(re.fullmatch(r"\d{4}-\d{2}", s))


def read_gas_sales_excel(path: str) -> pd.DataFrame:
    """
    Parse a province gas sales excel into columns: month, gas_sales.
    Robust to the layout where months are in one row across many columns.
    """
    df_raw = pd.read_excel(path, sheet_name=0, header=None)
    if df_raw.empty:
        raise ValueError(f"Empty gas sales file: {path}")

    # Find the row that contains many YYYY-MM values.
    best_row = None
    best_score = -1
    for i in range(min(len(df_raw), 30)):  # first 30 rows is enough here
        row = df_raw.iloc[i].tolist()
        score = sum(_looks_like_month(v) for v in row)
        if score > best_score:
            best_score = score
            best_row = i

    if best_row is None or best_score < 10:
        raise ValueError(f"Could not locate month header row in gas sales file: {path}")

    months_row = df_raw.iloc[best_row]
    month_cols = [j for j, v in enumerate(months_row.tolist()) if _looks_like_month(v)]
    months = [str(months_row.iloc[j]).strip() for j in month_cols]

    # Find the first subsequent row with mostly numeric values at these month columns.
    value_row = None
    for i in range(best_row + 1, min(best_row + 10, len(df_raw))):
        vals = pd.to_numeric(df_raw.iloc[i, month_cols], errors="coerce")
        numeric_cnt = int(vals.notna().sum())
        if numeric_cnt >= max(10, int(0.7 * len(month_cols))):
            value_row = i
            break
    if value_row is None:
        # fallback: pick the row with max numeric count in the next 10 rows
        best_i, best_cnt = None, -1
        for i in range(best_row + 1, min(best_row + 10, len(df_raw))):
            vals = pd.to_numeric(df_raw.iloc[i, month_cols], errors="coerce")
            numeric_cnt = int(vals.notna().sum())
            if numeric_cnt > best_cnt:
                best_cnt, best_i = numeric_cnt, i
        value_row = best_i

    values = pd.to_numeric(df_raw.iloc[value_row, month_cols], errors="coerce")
    out = pd.DataFrame({"month": months, "gas_sales": values.values})
    out["month"] = out["month"].astype(str).str.strip()
    return out


def _pick_temperature_sheets(sheet_names: List[str]) -> Tuple[str, str, str]:
    """
    Pick (daily_avg, daily_max, daily_min) sheets.
    Prefer sheets containing '日' and corresponding keywords.
    Fallback to non-daily ones if needed.
    """
    def pick(keys: List[str], prefer_daily: bool) -> Optional[str]:
        # 1) daily preferred
        if prefer_daily:
            for s in sheet_names:
                if "日" in s and all(k in s for k in keys):
                    return s
        # 2) any match
        for s in sheet_names:
            if all(k in s for k in keys):
                return s
        return None

    avg = pick(["平均", "温"], prefer_daily=True) or pick(["平均"], prefer_daily=True)
    mx = pick(["最高", "温"], prefer_daily=True) or pick(["最高"], prefer_daily=True)
    mn = pick(["最低", "温"], prefer_daily=True) or pick(["最低"], prefer_daily=True)

    if not (avg and mx and mn):
        raise ValueError(f"Could not pick avg/max/min temperature sheets from: {sheet_names}")
    return avg, mx, mn


def _get_province_name_column(df: pd.DataFrame) -> str:
    """
    Temperature sheets in this dataset typically have:
    - an unnamed first column
    - a serial-number column
    - a region/city name column (we want this one)

    We pick the column that looks most like "name": mostly non-numeric strings and high uniqueness.
    """
    if df.empty:
        raise ValueError("Empty temperature sheet")

    # Consider only the first few non-date columns.
    non_date_cols = []
    for c in df.columns:
        if isinstance(c, (pd.Timestamp,)) or hasattr(c, "year"):
            continue
        non_date_cols.append(c)
        if len(non_date_cols) >= 6:
            break

    if not non_date_cols:
        raise ValueError("No non-date columns found in temperature sheet")

    best_col, best_score = None, float("-inf")
    sample = df.head(60)
    for c in non_date_cols:
        s = sample[c]
        s_str = s.astype(str).str.strip()
        # numeric ratio
        num = pd.to_numeric(s_str, errors="coerce")
        numeric_ratio = float(num.notna().mean())
        unique_ratio = float(s_str.nunique(dropna=True) / max(1, len(s_str)))
        non_empty_ratio = float((s_str != "").mean())
        # Prefer: low numeric_ratio, high unique_ratio, high non_empty
        score = (1 - numeric_ratio) * 2.0 + unique_ratio * 1.0 + non_empty_ratio * 0.5
        if score > best_score:
            best_score, best_col = score, c

    if best_col is None:
        raise ValueError("Cannot determine region-name column in temperature sheet")
    return best_col


def _extract_daily_series_for_region(df: pd.DataFrame, region: str) -> pd.Series:
    """
    From a temperature sheet (rows=stations/provinces, cols include many datetime columns),
    return a Series indexed by datetime with daily values for the given province row.
    """
    name_col = _get_province_name_column(df)
    sub = df[df[name_col].astype(str).str.strip() == region]
    if sub.empty:
        # Some sheets might use shortened names; try fuzzy contains as fallback.
        sub = df[df[name_col].astype(str).str.contains(region, na=False)]
    if sub.empty:
        raise KeyError(f"Region {region!r} not found in temperature sheet (col={name_col!r})")

    row = sub.iloc[0]
    # Date columns are datetime-like; keep only those.
    date_cols = [c for c in df.columns if isinstance(c, (pd.Timestamp,)) or hasattr(c, "year")]
    if not date_cols:
        # Try to parse columns into datetime
        parsed = pd.to_datetime([c for c in df.columns], errors="coerce")
        date_cols = [df.columns[i] for i, v in enumerate(parsed) if pd.notna(v)]

    values = pd.to_numeric(row[date_cols], errors="coerce")
    # Build datetime index from columns
    idx = pd.to_datetime(date_cols, errors="coerce")
    s = pd.Series(values.values, index=idx).sort_index()
    s = s[~s.index.isna()]
    return s


def build_monthly_temperature_table(temp_dir: str, region: str) -> pd.DataFrame:
    """
    Aggregate daily temperature across all yearly files into monthly avg/max/min.
    Output columns: month, avg_temp, max_temp, min_temp, HDD, extreme_cold_days.
    """
    records = []
    for fname in sorted(os.listdir(temp_dir)):
        if fname.startswith("~$"):
            continue
        if not fname.lower().endswith((".xlsx", ".xls")):
            continue
        fpath = os.path.join(temp_dir, fname)
        xls = pd.ExcelFile(fpath)
        avg_sheet, max_sheet, min_sheet = _pick_temperature_sheets(xls.sheet_names)

        df_avg = pd.read_excel(fpath, sheet_name=avg_sheet)
        df_max = pd.read_excel(fpath, sheet_name=max_sheet)
        df_min = pd.read_excel(fpath, sheet_name=min_sheet)

        try:
            s_avg = _extract_daily_series_for_region(df_avg, region)
            s_max = _extract_daily_series_for_region(df_max, region)
            s_min = _extract_daily_series_for_region(df_min, region)
        except KeyError:
            # Some yearly files may not include the requested region row.
            # Skip that file instead of failing the whole province.
            continue

        # Monthly aggregation (skip NaN by default)
        m_avg = s_avg.groupby(s_avg.index.to_period("M")).mean()
        m_max = s_max.groupby(s_max.index.to_period("M")).max()
        m_min = s_min.groupby(s_min.index.to_period("M")).min()
        # HDD: 先按天计算 max(0, 15 - 日平均气温)，再按月求和
        s_hdd = (HDD_BASE_TEMP - s_avg).clip(lower=0.0)
        m_hdd = s_hdd.groupby(s_hdd.index.to_period("M")).sum()
        # extreme_cold_days: 统计当月内最低气温低于阈值的天数
        s_extreme_cold = (s_min < EXTREME_COLD_THRESHOLD).astype(float)
        m_extreme_cold = s_extreme_cold.groupby(s_extreme_cold.index.to_period("M")).sum()

        m = (
            pd.DataFrame(
                {
                    "month": m_avg.index.astype(str),
                    "avg_temp": m_avg.values,
                    "max_temp": m_max.reindex(m_avg.index).values,
                    "min_temp": m_min.reindex(m_avg.index).values,
                    "HDD": m_hdd.reindex(m_avg.index).values,
                    "extreme_cold_days": m_extreme_cold.reindex(m_avg.index).values,
                }
            )
            .dropna(subset=["month"])
        )
        records.append(m)

    if not records:
        return pd.DataFrame(
            columns=["month", "avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days"]
        )

    out = pd.concat(records, ignore_index=True)
    # If duplicate months exist due to overlapping files, keep the first non-null values by month.
    out = out.sort_values("month")
    out = (
        out.groupby("month", as_index=False)
        .agg(
            {
                "avg_temp": "mean",
                "max_temp": "max",
                "min_temp": "min",
                "HDD": "mean",
                "extreme_cold_days": "sum",
            }
        )
        .reset_index(drop=True)
    )
    return out


def build_province_output(
    province: str,
    gas_path: str,
    temp_dir: str,
    month_start: str = MONTH_START,
    month_end: str = MONTH_END,
) -> pd.DataFrame:
    gas = read_gas_sales_excel(gas_path)
    gas = gas[gas["month"].between(month_start, "9999-99")].copy()

    region = PROVINCE_TO_TEMP_REGION.get(province, province)
    temp = build_monthly_temperature_table(temp_dir, region)

    axis = pd.DataFrame({"month": _month_range(month_start, month_end)})
    out = axis.merge(gas, on="month", how="left").merge(temp, on="month", how="left")

    # enforce range strictly
    out = out[out["month"].between(month_start, month_end)].copy()
    out = out[
        ["month", "gas_sales", "avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days"]
    ]
    return out


def main() -> None:
    # scripts/ 下的处理脚本：数据统一放在项目根目录的 data/ 里
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    original_data = os.path.join(workspace, "data", "original_data")
    processed_data = os.path.join(workspace, "data", "processed_data")
    os.makedirs(processed_data, exist_ok=True)

    temp_dir = _find_temperature_dir(original_data)

    gas_files = [
        os.path.join(original_data, n)
        for n in os.listdir(original_data)
        if os.path.isfile(os.path.join(original_data, n)) and n.lower().endswith((".xlsx", ".xls"))
    ]

    if not gas_files:
        raise FileNotFoundError(f"No province gas excel files found under {original_data!r}")

    def safe_write_excel(df: pd.DataFrame, out_path: str) -> str:
        # Try to overwrite; if file is locked (opened in Excel), write to a new file.
        try:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except PermissionError:
                    pass
            df.to_excel(out_path, index=False, sheet_name="data")
            return out_path
        except PermissionError:
            base, ext = os.path.splitext(out_path)
            alt = f"{base}_new{ext}"
            df.to_excel(alt, index=False, sheet_name="data")
            return alt

    for gas_path in sorted(gas_files):
        province = _province_from_filename(gas_path)
        try:
            out = build_province_output(province=province, gas_path=gas_path, temp_dir=temp_dir)
        except Exception as e:
            print(f"[FAIL] {province}: {e}")
            continue

        out_path = os.path.join(processed_data, f"{province}.xlsx")
        written = safe_write_excel(out, out_path)
        print(f"[OK] {province} -> {written}")


if __name__ == "__main__":
    main()

