"""特征工程工具函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    - temp_range: 最高气温 - 最低气温
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

    # 计算月度极限温差
    out["temp_range"] = max_temp - min_temp
    
    return out


def add_time_features(
    df: pd.DataFrame,
    *,
    month_col: str = "month",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    添加时间维度特征：
    - time_index: 线性递增的趋势项 (1, 2, 3...)
    - month_sin, month_cos: 月份的周期性三角函数编码
    - is_heating_season: 是否为供暖季 (11月至次年3月)
    """
    if month_col not in df.columns:
        raise ValueError(f"缺少时间列: {month_col}")

    out = df if inplace else df.copy()
    
    dates = pd.to_datetime(out[month_col])
    months = dates.dt.month
    
    # 1. 宏观线性趋势项
    out["time_index"] = np.arange(1, len(out) + 1)
    
    # 2. 周期编码 (映射到圆周上)
    out["month_sin"] = np.sin(2 * np.pi * months / 12)
    out["month_cos"] = np.cos(2 * np.pi * months / 12)
    
    # 3. 供暖季虚拟变量 (11, 12, 1, 2, 3 月)
    out["is_heating_season"] = months.isin([11, 12, 1, 2, 3]).astype(int)
    
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    *,
    target_col: str = "gas_sales",
    month_col: str = "month",
    lags: tuple[int, ...] = (12,),
    inplace: bool = False,
) -> pd.DataFrame:
    """
    添加历史记忆特征 (滞后项与滚动均值)。
    注意：此操作依赖于时间的严格排序！
    """
    if target_col not in df.columns or month_col not in df.columns:
        raise ValueError(f"缺少目标列或时间列")

    out = df if inplace else df.copy()
    
    # 极其关键：构造滞后项前，必须确保数据严格按时间正序排列
    out = out.sort_values(month_col)
    
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
    依赖于前置生成的 HDD、极端天气特征和 Lag_12。
    """
    required = {"HDD", cold_days_col, "Lag_12", "is_heating_season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少前置依赖特征，请先运行天气和滞后特征生成: {sorted(missing)}")

    out = df if inplace else df.copy()
    
    # 1. 非线性特征：极寒放缩 (HDD的平方项，模拟寒潮下耗气量的非线性飙升)
    out["HDD_squared"] = out["HDD"] ** 2
    
    # 2. 交互特征：历史基数 x 当期气候 (大盘基数越大，同样的降温带来的增量越大)
    out["HDD_cross_Lag_12"] = out["HDD"] * out["Lag_12"]
    out["HDD_cross_HeatingSeason"] = out["HDD"] * out["is_heating_season"]

    # 3. 新增：极端天气杀手锏特征
    # 逻辑：如果去年体量很大，且本月出现多次跌破冰点的极寒天气，管网保供压力会呈指数级放大
    out["ColdDays_cross_Lag_12"] = out[cold_days_col] * out["Lag_12"]
    
    return out


def build_features_pipeline(
    df: pd.DataFrame, 
    target_col: str = "gas_sales", 
    month_col: str = "month",
    dropna: bool = True
) -> pd.DataFrame:
    """
    特征工程总流水线：一键完成所有特征构造并清洗。
    """
    print(f"开始特征工程流水线，原始数据形状: {df.shape}")
    
    # 确保排序 (一切时序操作的基石)
    df = df.sort_values(month_col).reset_index(drop=True)
    
    # 请确保 df 中已经包含了列名为 'extreme_cold_days' 的日度统计数据，如果是别的名字请在传参时修改
    df = add_weather_features(df)
    df = add_time_features(df, month_col=month_col)
    df = add_lag_rolling_features(df, target_col=target_col, month_col=month_col)
    df = add_interaction_features(df)
    
    if dropna:
        before_drop = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = before_drop - len(df)
        print(f"因构造滞后特征产生 NaN，自动裁剪了头部 {dropped} 行数据。")
        
    print(f"特征工程完成，当前可用特征数: {len(df.columns) - 2}，有效样本数: {len(df)}")
    return df