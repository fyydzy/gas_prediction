import os  # 文件路径、创建目录
import time  # 记录训练耗时
import warnings  # 忽略无关告警
from itertools import product  # 生成超参笛卡尔积

import numpy as np  # 数值计算
import pandas as pd  # 表格数据
from neuralforecast import NeuralForecast  # 时间序列深度学习封装
from neuralforecast.models import TiDE  # TiDE 模型类

from gas_prediction.feature_engineering import build_features_pipeline  # 项目内特征工程
from gas_prediction.forecast_common import (  # 公共常量与数据路径
    MONTH_COL,  # 月份列名（字符串 YYYY-MM）
    PROCESSED_PROVINCE,  # 省份/序列标识，用于 unique_id 与输出文件名
    TARGET_COL,  # 目标列名（销量）
    find_processed_excel,  # 定位处理后的 Excel
)

warnings.filterwarnings("ignore")  # 减少控制台噪音

# 训练只到 TRAIN_END，测试只取 TEST 区间
# TiDE 的 predict 要求“未来月份”在 futr_df 里连续，所以会把桥接月也一起预测，但指标只算测试月
TRAIN_END = "2025-06"  # 训练集最后一条可含此月
TEST_START = "2025-11"  # 测试集起
TEST_END = "2026-03"  # 测试集止
SERIES_ID = PROCESSED_PROVINCE  # 单序列 id，与 NeuralForecast 的 unique_id 对应
OUTPUT_DIR = "output"  # 主输出目录
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tide")  # 网格结果等副输出

# 与 run_lstm_0 相同：只用三列外生；需由 build_features_pipeline 生成
SELECTED_FEATURES = [
    "Lag_12",  # 去年同月销量（滞后 12）
    "HDD",  # 采暖度日
    "is_heating_season",  # 是否供暖季
]

# future 已知外生列名，传给 TiDE 的 futr_exog_list
FUTR_EXOG_LIST = [
    *SELECTED_FEATURES,  # 与上面列表一致
]

# 超参网格：4 维，全组合 5*3*2*2=60 种，再随机抽 RANDOM_SEARCH_N_TRIALS 组
GRID_INPUT_SIZE = [6, 9, 12, 15, 18]  # 回看窗口长度（月）
GRID_HIDDEN_SIZE = [32, 64, 96]  # MLP 隐层宽度
GRID_DROPOUT = [0.1, 0.2]  # Dropout
GRID_LEARNING_RATE = [5e-4, 1e-3]  # 学习率
RANDOM_SEARCH_N_TRIALS = 20  # 最多尝试多少组（不超过全组合数）
RANDOM_SEARCH_SEED = 42  # 随机抽样种子，可复现

# 不随网格变化、各次训练共用的固定项
BASE_TIDE_PARAMS = {
    "decoder_output_dim": 16,  # 解码器中间维
    "temporal_decoder_dim": 32,  # 时序解码器维
    "layernorm": True,  # 层归一化
    "num_encoder_layers": 1,  # 编码 MLP 层数
    "num_decoder_layers": 1,  # 解码 MLP 层数
    "temporal_width": 4,  # TiDE temporal 分支宽度
    "max_steps": 500,  # 最大训练步数（仍受早停影响）
    "batch_size": 16,  # 批大小
    "windows_batch_size": 64,  # 滑窗批处理
    "scaler_type": "standard",  # 对目标做时序标准化
    "random_seed": 42,  # 模型内随机种子
    "accelerator": "cpu",  # 用 CPU
    "devices": 1,  # 单设备
}

VAL_CHECK_STEPS = 25  # 每隔多少步在验证上算一次（供早停观察）
EARLY_STOP_PATIENCE_STEPS = 5  # 验证无提升时提前停
ENABLE_TRAIN_LOG = True  # 是否打印 Lightning 训练日志/进度条


def _load_monthly_frame() -> pd.DataFrame:
    input_path = find_processed_excel()  # 找数据文件
    raw_df = pd.read_excel(input_path)  # 读 Excel
    required = {MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp"}  # 原始表至少这些列
    missing = required - set(raw_df.columns)  # 差集
    if missing:  # 缺列则直接报错
        raise ValueError(f"输入数据缺少必要列: {sorted(missing)}")

    raw_df = raw_df.copy()  # 避免改原表
    raw_df[MONTH_COL] = raw_df[MONTH_COL].astype(str).str.slice(0, 7)  # 统一成 YYYY-MM
    raw_df[TARGET_COL] = pd.to_numeric(raw_df[TARGET_COL], errors="coerce")  # 转数字
    raw_df["avg_temp"] = pd.to_numeric(raw_df["avg_temp"], errors="coerce")
    raw_df["max_temp"] = pd.to_numeric(raw_df["max_temp"], errors="coerce")
    raw_df["min_temp"] = pd.to_numeric(raw_df["min_temp"], errors="coerce")
    raw_df = raw_df.dropna(  # 基础列缺就丢行
        subset=[MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp"]
    ).reset_index(drop=True)
    df = build_features_pipeline(raw_df, target_col=TARGET_COL, month_col=MONTH_COL)  # 造特征
    df["ds"] = pd.PeriodIndex(df[MONTH_COL].astype(str), freq="M").to_timestamp()  # 月首时间戳
    df["unique_id"] = SERIES_ID  # 单序列
    df["y"] = df[TARGET_COL].astype(float)  # 目标，与 neuralforecast 约定列名
    return df  # 返回特征+时间+id 的表


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0  # 分母为 0 的不参加
    if np.any(mask):  # 至少有一个有效点
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)  # MAPE%
    return float("nan")  # 全 0 则无法算


def _build_tide_model(
    *,
    h: int,  # 预测步长：与当前阶段要预测的未来月数一致
    input_size: int,  # 历史回望长度
    hidden_size: int,  # 隐层宽
    dropout: float,  # 失活
    learning_rate: float,  # 学习率
    enable_train_log: bool,  # 是否显示训练过程
    alias: str,  # 输出列名
) -> TiDE:
    params = dict(BASE_TIDE_PARAMS)  # 先复制固定项
    params.update(  # 再覆盖本次要搜/要训的项
        {
            "h": h,  # 多步预测长度
            "input_size": int(input_size),
            "hidden_size": int(hidden_size),
            "dropout": float(dropout),
            "learning_rate": float(learning_rate),
            "futr_exog_list": FUTR_EXOG_LIST,  # 未来外生
            "early_stop_patience_steps": EARLY_STOP_PATIENCE_STEPS,  # 开早停
            "val_check_steps": VAL_CHECK_STEPS,  # 验证检查频率
            "enable_progress_bar": enable_train_log,  # 进度条
            "logger": enable_train_log,  # 日志
            "log_every_n_steps": 1,  # 每步可打 log（较细，可改大）
            "alias": alias,  # predict 里列名
        }
    )
    return TiDE(**params)  # 构造模型


def _grid_search_tide(train_df: pd.DataFrame, val_size: int) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []  # 存每种超参的验证 MAPE
    best_params: dict | None = None  # 当前最优参数字典
    best_mape = float("inf")  # 当前最优 MAPE，越小越好

    # 为和 xgboost 一样有“可解释”的选优指标：在 holdout 的 val 上算 MAPE
    # 注意：下面 fit 仍传 val_size，TiDE 自己也会用尾部当内部验证做早停
    train_inner = train_df.iloc[:-val_size].copy()  # 训练子集（搜参时真正 fit 的表）
    val_df = train_df.iloc[-val_size:].copy()  # 留出的验证月
    if train_inner.empty or val_df.empty:  # 切分失败
        raise ValueError("训练/验证切分失败，无法执行 TiDE 网格搜索。")

    train_nf_df = train_inner[["unique_id", "ds", "y", *FUTR_EXOG_LIST]].copy()  # 给 fit
    val_futr_df = val_df[["unique_id", "ds", *FUTR_EXOG_LIST]].copy()  # 给 predict 的未来外生
    val_y = val_df["y"].to_numpy(dtype=float)  # 验证真值

    all_combos = list(product(GRID_INPUT_SIZE, GRID_HIDDEN_SIZE, GRID_DROPOUT, GRID_LEARNING_RATE))  # 全组合
    total_combos = len(all_combos)  # 组合数
    n_trials = min(RANDOM_SEARCH_N_TRIALS, total_combos)  # 实际试几组
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)  # 固定种子
    chosen_idx = rng.choice(total_combos, size=n_trials, replace=False)  # 不重复抽样下标
    sampled_combos = [all_combos[i] for i in chosen_idx]  # 取到具体超参元组
    print(f"TiDE 参数组合总数={total_combos}，随机抽样评估={n_trials}（seed={RANDOM_SEARCH_SEED}）")  # 说明

    for input_size, hidden_size, dropout, learning_rate in sampled_combos:  # 遍历抽中的组合
        model = _build_tide_model(  # 建一个候选模型
            h=val_size,  # 与 val 月数相同，满足 neuralforecast：val_size 为 0 或 >= h
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
            enable_train_log=False,  # 搜参时静默，快
            alias="TiDE",
        )
        nf = NeuralForecast(models=[model], freq="MS")  # 包一层，月频
        nf.fit(df=train_nf_df, val_size=val_size)  # 在 train_inner 上训，尾部 val_size 给内部早停
        pred_df = nf.predict(futr_df=val_futr_df).reset_index()  # 对验证月做逐步预测
        val_pred = pred_df["TiDE"].to_numpy(dtype=float)  # 取预测列
        val_mape = _mape(val_y, val_pred)  # 用你熟悉的 MAPE 选优

        rows.append(  # 记一条结果
            {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "val_mape_pct": val_mape,
            }
        )
        if val_mape < best_mape:  # 更优则更新
            best_mape = val_mape
            best_params = {
                "input_size": int(input_size),
                "hidden_size": int(hidden_size),
                "dropout": float(dropout),
                "learning_rate": float(learning_rate),
            }

    if best_params is None:  # 一条没成功
        raise ValueError("TiDE 网格搜索失败，未找到有效参数。")
    grid_df = pd.DataFrame(rows).sort_values("val_mape_pct").reset_index(drop=True)  # 按 MAPE 升序
    return best_params, grid_df  # 最优参 + 全表


def main() -> None:
    print("正在读取真实月度数据并构造 TiDE 外生特征...")  # 提示
    df = _load_monthly_frame()  # 读数+特征
    missing_features = [f for f in FUTR_EXOG_LIST if f not in df.columns]  # 检查外生列
    if missing_features:  # 缺则停
        raise ValueError(f"特征工程后缺少所需特征: {missing_features}")

    train_df = df[df[MONTH_COL] <= TRAIN_END].copy()  # 训练段
    test_df = df[(df[MONTH_COL] >= TEST_START) & (df[MONTH_COL] <= TEST_END)].copy()  # 测试段
    if train_df.empty:  # 无训练
        raise ValueError("训练集为空，请检查时间范围和数据。")
    if test_df.empty:  # 无测试
        raise ValueError("测试集为空，请检查时间范围和数据。")

    train_end_ts = pd.Period(TRAIN_END, freq="M").to_timestamp()  # 训练最后一月的时间戳
    test_end_ts = pd.Period(TEST_END, freq="M").to_timestamp()  # 测试最后月
    future_df = df[(df["ds"] > train_end_ts) & (df["ds"] <= test_end_ts)].copy()  # 桥+测 的连续未来
    if future_df.empty:  # 无未来可预测
        raise ValueError("训练结束后无可预测月份，请检查 TRAIN_END/TEST_END。")
    forecast_horizon = len(future_df)  # 要连续预测多少个月
    n_train = len(train_df)  # 训练行数
    val_size = min(12, max(6, n_train // 5))  # 约 20% 作验证，6~12
    if val_size >= n_train:  # 验证不能占满
        val_size = max(1, n_train - 1)
    if val_size <= 0:  # 不可为 0
        raise ValueError("训练样本过少，无法切分验证集。")

    nf_train_df = train_df[["unique_id", "ds", "y", *FUTR_EXOG_LIST]].copy()  # 最终重训用全训练
    futr_df = future_df[["unique_id", "ds", *FUTR_EXOG_LIST]].copy()  # 未来外生

    bridge_size = max(len(future_df) - len(test_df), 0)  # 桥接月数 = 未来总 - 测试
    print(  # 打一下各段规模
        f"训练样本={len(train_df)}，验证窗口={val_size}，"
        f"桥接样本={bridge_size}，测试样本={len(test_df)}，预测步长={forecast_horizon}"
    )

    # 搜参：在 train_df 上切出与 xgboost 类似的尾段验证，用 MAPE 排个名
    print("开始 TiDE 随机网格搜索（训练子集 + 内部早停 + 外部 MAPE 选优）...")  # 与 print 内容一致
    best_params, grid_df = _grid_search_tide(train_df, val_size=val_size)  # 跑网格
    print(f"网格搜索完成，共 {len(grid_df)} 组。最优验证 MAPE={grid_df.iloc[0]['val_mape_pct']:.4f}%")
    print(f"最优参数: {best_params}")

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)  # 建子目录
    grid_path = os.path.join(PLOT_OUTPUT_DIR, "tide_grid_search_results.csv")  # 网格结果路径
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")  # 带 BOM 的 utf-8
    print(f"网格搜索结果已保存: {grid_path}")

    model = _build_tide_model(  # 用最优超参 + 真实要预测的未来步长 h
        h=forecast_horizon,  # 这里 h=桥+测 的总月数
        input_size=best_params["input_size"],
        hidden_size=best_params["hidden_size"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        enable_train_log=ENABLE_TRAIN_LOG,  # 最终可开日志
        alias="TiDE",
    )
    nf = NeuralForecast(models=[model], freq="MS")  # 再包一层

    print("开始用最优参数重训 TiDE...")  # 全训练集
    train_start = time.time()  # 记时
    nf.fit(df=nf_train_df, val_size=val_size)  # 在完整 train 上训，仍用尾部 val 做早停
    train_seconds = time.time() - train_start
    print(f"TiDE 训练完成，用时: {train_seconds:.2f}s")

    print("开始预测桥接期 + 测试期（最终仅评估测试期）...")  # 说明
    forecast_df = nf.predict(futr_df=futr_df).reset_index()  # 多步预测
    forecast_df = forecast_df.rename(columns={"TiDE": "predicted_gas_sales"})  # 列名统一

    eval_df = future_df[[MONTH_COL, "ds", TARGET_COL]].merge(  # 真值与预测按时间对齐
        forecast_df[["ds", "predicted_gas_sales"]],
        on="ds",
        how="left",
    )
    if eval_df["predicted_gas_sales"].isna().any():  # 缺预测就错
        raise ValueError("TiDE 预测结果存在缺失，请检查 futr_df 与预测步长。")

    result_df = eval_df[  # 只保留正式测试月
        (eval_df[MONTH_COL] >= TEST_START) & (eval_df[MONTH_COL] <= TEST_END)
    ].copy()
    result_df = result_df[[MONTH_COL, TARGET_COL, "predicted_gas_sales"]]  # 三列
    result_df = result_df.rename(columns={TARGET_COL: "actual_gas_sales"})  # 列名
    result_df["error"] = result_df["predicted_gas_sales"] - result_df["actual_gas_sales"]  # 误差
    result_df["abs_error"] = np.abs(result_df["error"])  # 绝对误差
    result_df["mape_pct"] = np.where(  # 行内 MAPE
        result_df["actual_gas_sales"] != 0,
        np.abs(result_df["error"] / result_df["actual_gas_sales"]) * 100.0,
        np.nan,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 保证主输出目录存在
    excel_path = os.path.join(  # Excel 路径
        OUTPUT_DIR,
        f"{PROCESSED_PROVINCE}_tide_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    result_df.to_excel(excel_path, index=False, sheet_name="test_forecast")  # 写出
    print(f"测试集预测结果已保存: {excel_path}")

    y_true = result_df["actual_gas_sales"].to_numpy(dtype=float)  # 真值数组
    y_pred = result_df["predicted_gas_sales"].to_numpy(dtype=float)  # 预测数组
    mape = _mape(y_true, y_pred)  # 整段测试 MAPE
    if np.isfinite(mape):  # 能算
        print(f"\nTiDE 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
        print(f"Test MAPE: {mape:.4f}%")
    else:  # 有 0 分母等
        print("测试集包含 0 或为空，跳过 MAPE 输出。")


if __name__ == "__main__":  # 作为脚本直接运行
    main()  # 入口
