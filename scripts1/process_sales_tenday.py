import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATE_MATCH_THRESHOLD = 3
NUMERIC_ROW_THRESHOLD = 0.5
HDD_BASE_TEMP = 18.0
EXTREME_COLD_THRESHOLD = 0.0

# Province -> representative city used in temperature sheets (usually capital).
# If a province is not listed, try matching by province name directly.
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
    "江苏": "南京",
    "浙江": "杭州",
    "安徽": "合肥",
    "福建": "福州",
    "江西": "南昌",
    "山东": "济南",
    "河南": "郑州",
    "湖北": "武汉",
    "湖南": "长沙",
    "广东": "广州",
    "广西": "南宁",
    "海南": "海口",
    "四川": "成都",
    "贵州": "贵阳",
    "云南": "昆明",
    "西藏": "拉萨",
    "陕西": "西安",
    "甘肃": "兰州",
    "青海": "西宁",
    "宁夏": "银川",
    "新疆": "乌鲁木齐",
    "台湾": "台北",
    "香港": "香港",
    "澳门": "澳门",
}


def _looks_like_date(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return pd.notna(pd.to_datetime(value, errors="coerce"))


def _find_date_row(df: pd.DataFrame) -> int:
    best_row = None
    best_score = -1
    for row_idx in range(min(len(df), 20)):
        score = sum(_looks_like_date(value) for value in df.iloc[row_idx].tolist())
        if score > best_score:
            best_score = score
            best_row = row_idx

    if best_row is None or best_score < DATE_MATCH_THRESHOLD:
        raise ValueError("Could not locate a date header row")
    return best_row


def _period_order(day: int) -> int:
    if day <= 10:
        return 1
    if day <= 20:
        return 2
    return 3


def _tenday_start_day(period_order: int) -> int:
    if period_order == 1:
        return 1
    if period_order == 2:
        return 11
    return 21


def _add_tenday_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(
        {
            "year": out["year"],
            "month": out["month"],
            "day": out["period_order"].map(_tenday_start_day),
        }
    )
    return out


def _natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def _pick_temperature_sheets(sheet_names: List[str]) -> Tuple[str, str, str]:
    """
    Pick (daily_avg, daily_max, daily_min) sheets.
    Prefer sheets containing '日' and corresponding keywords.
    """
    def pick(keys: List[str], prefer_daily: bool) -> Optional[str]:
        if prefer_daily:
            for sheet_name in sheet_names:
                if "日" in sheet_name and all(key in sheet_name for key in keys):
                    return sheet_name
        for sheet_name in sheet_names:
            if all(key in sheet_name for key in keys):
                return sheet_name
        return None

    avg = pick(["平均", "温"], prefer_daily=True) or pick(["平均"], prefer_daily=True)
    mx = pick(["最高", "温"], prefer_daily=True) or pick(["最高"], prefer_daily=True)
    mn = pick(["最低", "温"], prefer_daily=True) or pick(["最低"], prefer_daily=True)

    if not (avg and mx and mn):
        raise ValueError(f"Could not pick avg/max/min temperature sheets from: {sheet_names}")
    return avg, mx, mn


def _get_region_name_column(df: pd.DataFrame) -> str:
    if df.empty:
        raise ValueError("Empty temperature sheet")

    non_date_cols = []
    for col in df.columns:
        if isinstance(col, pd.Timestamp) or hasattr(col, "year"):
            continue
        non_date_cols.append(col)
        if len(non_date_cols) >= 6:
            break

    if not non_date_cols:
        raise ValueError("No non-date columns found in temperature sheet")

    best_col, best_score = None, float("-inf")
    sample = df.head(60)
    for col in non_date_cols:
        s_str = sample[col].astype(str).str.strip()
        numeric_ratio = float(pd.to_numeric(s_str, errors="coerce").notna().mean())
        unique_ratio = float(s_str.nunique(dropna=True) / max(1, len(s_str)))
        non_empty_ratio = float((s_str != "").mean())
        score = (1 - numeric_ratio) * 2.0 + unique_ratio + non_empty_ratio * 0.5
        if score > best_score:
            best_score, best_col = score, col

    if best_col is None:
        raise ValueError("Cannot determine region-name column in temperature sheet")
    return best_col


def _extract_daily_series_for_region(df: pd.DataFrame, region: str) -> pd.Series:
    name_col = _get_region_name_column(df)
    sub = df[df[name_col].astype(str).str.strip() == region]
    if sub.empty:
        sub = df[df[name_col].astype(str).str.contains(region, na=False)]
    if sub.empty:
        raise KeyError(f"Region {region!r} not found in temperature sheet")

    row = sub.iloc[0]
    date_cols = [col for col in df.columns if isinstance(col, pd.Timestamp) or hasattr(col, "year")]
    if not date_cols:
        parsed = pd.to_datetime(list(df.columns), errors="coerce")
        date_cols = [df.columns[idx] for idx, value in enumerate(parsed) if pd.notna(value)]

    values = pd.to_numeric(row[date_cols], errors="coerce")
    index = pd.to_datetime(date_cols, errors="coerce")
    series = pd.Series(values.values, index=index).sort_index()
    return series[~series.index.isna()]


def read_sales_file(path: Path) -> pd.DataFrame:
    """
    Read one exported sales workbook and return columns: date, gas_sales.

    The source layout is typically:
    - one row containing daily dates across columns
    - one row containing "销售\\n完成"
    - one or more numeric rows

    Exact duplicate numeric rows are removed before summing because these exports
    often repeat the same total row.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df_raw = pd.read_excel(path, sheet_name=0, header=None)

    if df_raw.empty:
        raise ValueError("Empty workbook")

    date_row = _find_date_row(df_raw)
    parsed_dates = pd.to_datetime(df_raw.iloc[date_row], errors="coerce")
    date_cols = [idx for idx, value in enumerate(parsed_dates) if pd.notna(value)]
    if not date_cols:
        raise ValueError("No date columns found")

    values = df_raw.iloc[date_row + 1 :, date_cols].apply(pd.to_numeric, errors="coerce")
    min_numeric = max(1, int(len(date_cols) * NUMERIC_ROW_THRESHOLD))
    values = values[values.notna().sum(axis=1) >= min_numeric]
    values = values.drop_duplicates()
    if values.empty:
        raise ValueError("No numeric sales rows found")

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(parsed_dates.iloc[date_cols]).dt.normalize(),
            "gas_sales": values.sum(axis=0, skipna=True).to_numpy(),
        }
    )
    daily = daily.dropna(subset=["date"]).sort_values("date")
    return daily


def aggregate_tenday(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["period_order"] = daily["date"].dt.day.map(_period_order)

    grouped = (
        daily.groupby(["year", "month", "period_order"], as_index=False)
        .agg(gas_sales=("gas_sales", "sum"))
        .sort_values(["year", "month", "period_order"])
        .reset_index(drop=True)
    )
    grouped = _add_tenday_date(grouped)
    return grouped[["date", "gas_sales"]]


def build_tenday_temperature_table(temp_dir: Path, region: str) -> pd.DataFrame:
    """
    Aggregate daily temperature across all temperature files into tenday features.
    Output columns: date, avg_temp, max_temp, min_temp, HDD, extreme_cold_days.
    """
    records = []
    temp_files = sorted(
        (
            p
            for p in temp_dir.iterdir()
            if p.is_file() and not p.name.startswith("~$") and p.suffix.lower() in {".xlsx", ".xls"}
        ),
        key=_natural_sort_key,
    )

    for file_path in temp_files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xls = pd.ExcelFile(file_path)
            avg_sheet, max_sheet, min_sheet = _pick_temperature_sheets(xls.sheet_names)
            df_avg = pd.read_excel(file_path, sheet_name=avg_sheet)
            df_max = pd.read_excel(file_path, sheet_name=max_sheet)
            df_min = pd.read_excel(file_path, sheet_name=min_sheet)

        try:
            s_avg = _extract_daily_series_for_region(df_avg, region)
            s_max = _extract_daily_series_for_region(df_max, region)
            s_min = _extract_daily_series_for_region(df_min, region)
        except KeyError:
            continue

        records.append(
            pd.DataFrame(
                {
                    "date": s_avg.index,
                    "avg_temp": s_avg.values,
                    "max_temp": s_max.reindex(s_avg.index).values,
                    "min_temp": s_min.reindex(s_avg.index).values,
                }
            )
        )

    columns = ["date", "avg_temp", "max_temp", "min_temp", "HDD", "extreme_cold_days"]
    if not records:
        return pd.DataFrame(columns=columns)

    daily = pd.concat(records, ignore_index=True).dropna(subset=["date"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.groupby("date", as_index=False).agg(
        avg_temp=("avg_temp", "mean"),
        max_temp=("max_temp", "max"),
        min_temp=("min_temp", "min"),
    )
    daily["HDD"] = (HDD_BASE_TEMP - daily["avg_temp"]).clip(lower=0.0)
    daily["extreme_cold_days"] = (daily["min_temp"] < EXTREME_COLD_THRESHOLD).astype(float)
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["period_order"] = daily["date"].dt.day.map(_period_order)

    grouped = (
        daily.groupby(["year", "month", "period_order"], as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            max_temp=("max_temp", "max"),
            min_temp=("min_temp", "min"),
            HDD=("HDD", "sum"),
            extreme_cold_days=("extreme_cold_days", "sum"),
        )
        .sort_values(["year", "month", "period_order"])
        .reset_index(drop=True)
    )
    grouped = _add_tenday_date(grouped)
    return grouped[columns]


def iter_province_dirs(sales_dir: Path) -> Iterable[Path]:
    return sorted((p for p in sales_dir.iterdir() if p.is_dir()), key=lambda p: p.name)


def process_province(province_dir: Path, temp_dir: Path) -> pd.DataFrame:
    files = sorted(
        (
            p
            for p in province_dir.rglob("*")
            if p.is_file() and not p.name.startswith("~$") and p.suffix.lower() in {".xlsx", ".xls"}
        ),
        key=_natural_sort_key,
    )
    if not files:
        raise FileNotFoundError(f"No Excel files found under {province_dir}")

    daily_parts = []
    for file_path in files:
        part = read_sales_file(file_path)
        part["source_file"] = file_path.name
        daily_parts.append(part)

    daily = pd.concat(daily_parts, ignore_index=True)
    daily = daily.groupby("date", as_index=False).agg(gas_sales=("gas_sales", "sum"))
    sales = aggregate_tenday(daily)

    region = PROVINCE_TO_TEMP_REGION.get(province_dir.name, province_dir.name)
    temp = build_tenday_temperature_table(temp_dir, region)
    out = sales.merge(temp, on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def safe_write_excel(df: pd.DataFrame, out_path: Path) -> Path:
    try:
        if out_path.exists():
            try:
                out_path.unlink()
            except PermissionError:
                pass
        df.to_excel(out_path, index=False, sheet_name="data")
        return out_path
    except PermissionError:
        alt_path = out_path.with_name(f"{out_path.stem}_new{out_path.suffix}")
        df.to_excel(alt_path, index=False, sheet_name="data")
        return alt_path


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    sales_dir = workspace / "data" / "original_data1" / "销量"
    temp_dir = workspace / "data" / "original_data1" / "温度"
    processed_dir = workspace / "data" / "processed_data1"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not sales_dir.exists():
        raise FileNotFoundError(f"Sales directory not found: {sales_dir}")
    if not temp_dir.exists():
        raise FileNotFoundError(f"Temperature directory not found: {temp_dir}")

    processed_count = 0
    for province_dir in iter_province_dirs(sales_dir):
        try:
            province_output = process_province(province_dir, temp_dir)
        except Exception as exc:
            print(f"[FAIL] {province_dir.name}: {exc}")
            continue

        out_path = safe_write_excel(province_output, processed_dir / f"{province_dir.name}.xlsx")
        processed_count += 1
        print(f"[OK] {province_dir.name} -> {out_path}")

    if processed_count == 0:
        raise RuntimeError("No province data was processed successfully")


if __name__ == "__main__":
    main()
