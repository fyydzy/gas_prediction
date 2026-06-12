"""特征工程工具函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd


SPRING_FESTIVAL_DATES = {
    2010: "2010-02-14",
    2011: "2011-02-03",
    2012: "2012-01-23",
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
    2025: "2025-01-29",
    2026: "2026-02-17",
    2027: "2027-02-06",
    2028: "2028-01-26",
    2029: "2029-02-13",
    2030: "2030-02-03",
    2031: "2031-01-23",
    2032: "2032-02-11",
    2033: "2033-01-31",
    2034: "2034-02-19",
    2035: "2035-02-08",
}


def add_weather_features(
    df: pd.DataFrame,
    *,
    hdd_col: str = "HDD",
    cold_days_col: str = "extreme_cold_days",  # 假设你的极端低温天数列名为此
    max_temp_col: str = "max_temp",
    min_temp_col: str = "min_temp",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    接收底层已经由日度数据聚合好的高质量天气特征：
    - HDD: 采暖度日数 (日度累加)
    - extreme_cold_days: 极端低温天数
    - temp_range: 旬内最高气温 - 旬内最低气温
    同时生成：
    - temp_range: 最高气温 - 最低气温
    """
    required = {hdd_col, cold_days_col, max_temp_col, min_temp_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"底层数据缺失必要的气象列: {sorted(missing)}。请检查输入数据！")

    out = df if inplace else df.copy()
    
    # 强制类型转换，确保后续矩阵运算安全
    out[hdd_col] = pd.to_numeric(out[hdd_col], errors="coerce")
    out[cold_days_col] = pd.to_numeric(out[cold_days_col], errors="coerce")
    max_temp = pd.to_numeric(out[max_temp_col], errors="coerce")
    min_temp = pd.to_numeric(out[min_temp_col], errors="coerce")

    # 计算旬内极限温差
    out["temp_range"] = max_temp - min_temp
    
    return out


def add_time_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    添加旬度时间维度特征：
    - time_index: 线性递增的趋势项 (1, 2, 3...)
    - month_sin, month_cos: 月份的周期性三角函数编码
    - tenday_in_month: 月内旬序号 (1=上旬, 2=中旬, 3=下旬)
    - is_heating_season: 是否为供暖季 (11月至次年3月)
    - spring_rework_peak: 春节后复工峰值强度
    """
    if date_col not in df.columns:
        raise ValueError(f"缺少时间列: {date_col}")

    out = df if inplace else df.copy()

    dates = pd.to_datetime(out[date_col])
    months = dates.dt.month
    days = dates.dt.day
    tenday_in_month = np.select([days <= 10, days <= 20], [1, 2], default=3).astype(int)

    # 1. 宏观线性趋势项
    out["time_index"] = np.arange(1, len(out) + 1)

    # 2. 周期编码 (映射到圆周上)
    out["month_sin"] = np.sin(2 * np.pi * months / 12)
    out["month_cos"] = np.cos(2 * np.pi * months / 12)

    # 3. 月内旬序号
    out["tenday_in_month"] = tenday_in_month

    # 4. 供暖季虚拟变量 (11, 12, 1, 2, 3 月)
    out["is_heating_season"] = months.isin([11, 12, 1, 2, 3]).astype(int)

    missing_years = sorted(set(dates.dt.year) - set(SPRING_FESTIVAL_DATES))
    if missing_years:
        raise ValueError(f"缺少这些年份的春节日期配置: {missing_years}，请更新 SPRING_FESTIVAL_DATES。")

    spring_dates = pd.to_datetime(dates.dt.year.map(SPRING_FESTIVAL_DATES))
    days_after_spring_festival = (dates - spring_dates).dt.days
    out["spring_rework_peak"] = np.select(
        [
            days_after_spring_festival.between(0, 9),
            days_after_spring_festival.between(10, 19),
            days_after_spring_festival.between(20, 29),
            days_after_spring_festival.between(30, 39),
        ],
        [0.5, 1.0, 0.6, 0.3],
        default=0.0,
    )

    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    *,
    target_col: str = "gas_sales",
    date_col: str = "date",
    lags: tuple[int, ...] = (36,),
    inplace: bool = False,
) -> pd.DataFrame:
    """
    添加历史记忆特征 (滞后项与滚动均值)。
    注意：此操作依赖于时间的严格排序！
    """
    if target_col not in df.columns or date_col not in df.columns:
        raise ValueError(f"缺少目标列或时间列")

    out = df if inplace else df.copy()

    # 极其关键：构造滞后项前，必须确保数据严格按时间正序排列
    out = out.sort_values(date_col)

    # 1. 滞后特征 (Lag)
    for lag in lags:
        out[f"Lag_{lag}"] = out[target_col].shift(lag)

    return out


def add_interaction_features(
    df: pd.DataFrame,
    *,
    cold_days_col: str = "extreme_cold_days",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    添加交互与非线性特征。
    依赖于前置生成的 HDD、极端天气特征和 Lag_36。
    """
    required = {"HDD", cold_days_col, "Lag_36", "is_heating_season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少前置依赖特征，请先运行天气和滞后特征生成: {sorted(missing)}")

    out = df if inplace else df.copy()

    # 1. 非线性特征：极寒放缩 (HDD的平方项，模拟寒潮下耗气量的非线性飙升)
    out["HDD_squared"] = out["HDD"] ** 2

    # 2. 交互特征：历史基数 x 当期气候 (大盘基数越大，同样的降温带来的增量越大)
    out["HDD_cross_Lag_36"] = out["HDD"] * out["Lag_36"]
    out["HDD_cross_HeatingSeason"] = out["HDD"] * out["is_heating_season"]

    # 3. 新增：极端天气杀手锏特征
    # 逻辑：如果去年同期体量很大，且本旬出现多次跌破冰点的极寒天气，管网保供压力会呈指数级放大
    out["ColdDays_cross_Lag_36"] = out[cold_days_col] * out["Lag_36"]

    return out


def build_features_pipeline(
    df: pd.DataFrame,
    target_col: str = "gas_sales",
    date_col: str = "date",
    dropna: bool = True
) -> pd.DataFrame:
    """
    旬度特征工程总流水线：一键完成所有特征构造并清洗。

    典型输入（各脚本 `model_input_cols`）含：
    ``date, gas_sales, avg_temp, max_temp, min_temp, HDD, extreme_cold_days`` 时，
    流水线结束后 **DataFrame 列**（共 19 列）依次为：

    1. ``date`` — 旬起始日
    2. ``gas_sales`` — 目标
    3. ``avg_temp``, ``max_temp``, ``min_temp`` — 原表温度
    4. ``HDD``, ``extreme_cold_days`` — 原表气象
    5. ``temp_range`` — ``add_weather_features``
    6. ``time_index``, ``month_sin``, ``month_cos``, ``tenday_in_month``,
       ``is_heating_season``, ``spring_rework_peak`` — ``add_time_features``
    7. ``Lag_36`` — ``add_lag_rolling_features``
    8. ``HDD_squared``, ``HDD_cross_Lag_36``, ``HDD_cross_HeatingSeason``,
       ``ColdDays_cross_Lag_36`` — ``add_interaction_features``

    **建模用数值特征**为除 ``date``、``gas_sales`` 外的 **17** 列；
    若输入列与上述典型不一致，列数会随之变化。
    """
    print(f"开始特征工程流水线，原始数据形状: {df.shape}")

    # 确保排序 (一切时序操作的基石)
    df = df.sort_values(date_col).reset_index(drop=True)

    # 请确保 df 中已经包含了列名为 'extreme_cold_days' 的日度统计数据，如果是别的名字请在传参时修改
    df = add_weather_features(df)
    df = add_time_features(df, date_col=date_col)
    df = add_lag_rolling_features(df, target_col=target_col, date_col=date_col)
    df = add_interaction_features(df)

    if dropna:
        before_drop = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = before_drop - len(df)
        print(f"因构造滞后特征产生 NaN，自动裁剪了头部 {dropped} 行数据。")

    feature_cols = [c for c in df.columns if c not in (date_col, target_col)]
    print(
        f"特征工程完成，有效样本数: {len(df)}；建模数值特征数: {len(feature_cols)}；"
        f"列名: {feature_cols}"
    )
    return df