import os
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from gas_prediction.feature_engineering import build_features_pipeline
from gas_prediction.forecast_common import (
    MONTH_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
)

warnings.filterwarnings("ignore")

# 与 TiDE/LSTM 脚本一致：训练到 2025-06，测试 2025-11~2026-03。
# NHITS 预测也要求 future exog 覆盖连续未来月份，因此预测输入包含桥接期，但只评估测试期。
TRAIN_END = "2025-06"
TEST_START = "2025-11"
TEST_END = "2026-03"
SERIES_ID = PROCESSED_PROVINCE
OUTPUT_DIR = "output"
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "nhits")

# 与 run_lstm_0.py / run_tide_0.py 对齐的核心外生特征。
SELECTED_FEATURES = ["Lag_12", "HDD", "is_heating_season"]
FUTR_EXOG_LIST = [*SELECTED_FEATURES]

# 随机网格搜索：全组合后随机抽样，按验证集 MAPE 选优。
GRID_INPUT_SIZE = [6, 9, 12, 15, 18]
GRID_MLP_UNITS = [
    [[32, 32], [32, 32], [32, 32]],
    [[64, 64], [64, 64], [64, 64]],
    [[96, 96], [96, 96], [96, 96]],
]
GRID_DROPOUT = [0.0, 0.1]
GRID_LEARNING_RATE = [5e-4, 1e-3]
RANDOM_SEARCH_N_TRIALS = 20
RANDOM_SEARCH_SEED = 42

# 固定参数说明：
# stack_types/n_blocks：三层 identity 堆栈，每层 1 个 block，控制模型结构复杂度。
# n_pool_kernel_size/n_freq_downsample：N-HiTS 多尺度池化/下采样设置，适配月度序列的粗到细分解。
# scaler_type：内部对序列做 standard 标准化，预测输出自动回到原尺度。
BASE_NHITS_PARAMS = {
    "stack_types": ["identity", "identity", "identity"],
    "n_blocks": [1, 1, 1],
    "n_pool_kernel_size": [2, 2, 1],
    "n_freq_downsample": [4, 2, 1],
    "pooling_mode": "MaxPool1d",
    "interpolation_mode": "linear",
    "activation": "ReLU",
    "max_steps": 500,
    "batch_size": 16,
    "windows_batch_size": 64,
    "scaler_type": "standard",
    "random_seed": 42,
    "accelerator": "cpu",
    "devices": 1,
}

VAL_CHECK_STEPS = 25
EARLY_STOP_PATIENCE_STEPS = 5
ENABLE_TRAIN_LOG = True


def _load_monthly_frame() -> pd.DataFrame:
    input_path = find_processed_excel()
    raw_df = pd.read_excel(input_path)
    required = {MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {sorted(missing)}")

    raw_df = raw_df.copy()
    raw_df[MONTH_COL] = raw_df[MONTH_COL].astype(str).str.slice(0, 7)
    raw_df[TARGET_COL] = pd.to_numeric(raw_df[TARGET_COL], errors="coerce")
    raw_df["avg_temp"] = pd.to_numeric(raw_df["avg_temp"], errors="coerce")
    raw_df["max_temp"] = pd.to_numeric(raw_df["max_temp"], errors="coerce")
    raw_df["min_temp"] = pd.to_numeric(raw_df["min_temp"], errors="coerce")
    raw_df = raw_df.dropna(
        subset=[MONTH_COL, TARGET_COL, "avg_temp", "max_temp", "min_temp"]
    ).reset_index(drop=True)

    df = build_features_pipeline(raw_df, target_col=TARGET_COL, month_col=MONTH_COL)
    df["ds"] = pd.PeriodIndex(df[MONTH_COL].astype(str), freq="M").to_timestamp()
    df["unique_id"] = SERIES_ID
    df["y"] = df[TARGET_COL].astype(float)
    return df


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if np.any(mask):
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return float("nan")


def _build_nhits_model(
    *,
    h: int,
    input_size: int,
    mlp_units: list[list[int]],
    dropout: float,
    learning_rate: float,
    enable_train_log: bool,
    alias: str,
) -> NHITS:
    params = dict(BASE_NHITS_PARAMS)
    params.update(
        {
            "h": int(h),
            "input_size": int(input_size),
            "mlp_units": mlp_units,
            "dropout_prob_theta": float(dropout),
            "learning_rate": float(learning_rate),
            "futr_exog_list": FUTR_EXOG_LIST,
            "early_stop_patience_steps": EARLY_STOP_PATIENCE_STEPS,
            "val_check_steps": VAL_CHECK_STEPS,
            "enable_progress_bar": enable_train_log,
            "logger": enable_train_log,
            "log_every_n_steps": 1,
            "alias": alias,
        }
    )
    return NHITS(**params)


def _grid_search_nhits(train_df: pd.DataFrame, val_size: int) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    best_params: dict | None = None
    best_mape = float("inf")

    # 外部验证段用于和其他脚本一致地计算 MAPE；fit 内部仍使用 val_size 触发早停。
    train_inner = train_df.iloc[:-val_size].copy()
    val_df = train_df.iloc[-val_size:].copy()
    if train_inner.empty or val_df.empty:
        raise ValueError("训练/验证切分失败，无法执行 NHITS 网格搜索。")

    train_nf_df = train_inner[["unique_id", "ds", "y", *FUTR_EXOG_LIST]].copy()
    val_futr_df = val_df[["unique_id", "ds", *FUTR_EXOG_LIST]].copy()
    val_y = val_df["y"].to_numpy(dtype=float)

    all_combos = list(product(GRID_INPUT_SIZE, GRID_MLP_UNITS, GRID_DROPOUT, GRID_LEARNING_RATE))
    total_combos = len(all_combos)
    n_trials = min(RANDOM_SEARCH_N_TRIALS, total_combos)
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)
    chosen_idx = rng.choice(total_combos, size=n_trials, replace=False)
    sampled_combos = [all_combos[i] for i in chosen_idx]
    print(f"NHITS 参数组合总数={total_combos}，随机抽样评估={n_trials}（seed={RANDOM_SEARCH_SEED}）")

    for input_size, mlp_units, dropout, learning_rate in sampled_combos:
        hidden_width = int(mlp_units[0][0])
        model = _build_nhits_model(
            h=val_size,
            input_size=input_size,
            mlp_units=mlp_units,
            dropout=dropout,
            learning_rate=learning_rate,
            enable_train_log=False,
            alias="NHITS",
        )
        nf = NeuralForecast(models=[model], freq="MS")
        nf.fit(df=train_nf_df, val_size=val_size)
        pred_df = nf.predict(futr_df=val_futr_df).reset_index()
        val_pred = pred_df["NHITS"].to_numpy(dtype=float)
        val_mape = _mape(val_y, val_pred)

        rows.append(
            {
                "input_size": input_size,
                "hidden_width": hidden_width,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "val_mape_pct": val_mape,
            }
        )
        if val_mape < best_mape:
            best_mape = val_mape
            best_params = {
                "input_size": int(input_size),
                "mlp_units": mlp_units,
                "hidden_width": hidden_width,
                "dropout": float(dropout),
                "learning_rate": float(learning_rate),
            }

    if best_params is None:
        raise ValueError("NHITS 网格搜索失败，未找到有效参数。")
    grid_df = pd.DataFrame(rows).sort_values("val_mape_pct").reset_index(drop=True)
    return best_params, grid_df


def main() -> None:
    print("正在读取真实月度数据并构造 NHITS 外生特征...")
    df = _load_monthly_frame()
    missing_features = [f for f in FUTR_EXOG_LIST if f not in df.columns]
    if missing_features:
        raise ValueError(f"特征工程后缺少所需特征: {missing_features}")

    train_df = df[df[MONTH_COL] <= TRAIN_END].copy()
    test_df = df[(df[MONTH_COL] >= TEST_START) & (df[MONTH_COL] <= TEST_END)].copy()
    if train_df.empty:
        raise ValueError("训练集为空，请检查时间范围和数据。")
    if test_df.empty:
        raise ValueError("测试集为空，请检查时间范围和数据。")

    train_end_ts = pd.Period(TRAIN_END, freq="M").to_timestamp()
    test_end_ts = pd.Period(TEST_END, freq="M").to_timestamp()
    future_df = df[(df["ds"] > train_end_ts) & (df["ds"] <= test_end_ts)].copy()
    if future_df.empty:
        raise ValueError("训练结束后无可预测月份，请检查 TRAIN_END/TEST_END。")

    forecast_horizon = len(future_df)
    n_train = len(train_df)
    val_size = min(12, max(6, n_train // 5))
    if val_size >= n_train:
        val_size = max(1, n_train - 1)
    if val_size <= 0:
        raise ValueError("训练样本过少，无法切分验证集。")

    nf_train_df = train_df[["unique_id", "ds", "y", *FUTR_EXOG_LIST]].copy()
    futr_df = future_df[["unique_id", "ds", *FUTR_EXOG_LIST]].copy()

    bridge_size = max(len(future_df) - len(test_df), 0)
    print(
        f"训练样本={len(train_df)}，验证窗口={val_size}，"
        f"桥接样本={bridge_size}，测试样本={len(test_df)}，预测步长={forecast_horizon}"
    )

    print("开始 NHITS 随机网格搜索（训练子集 + 内部早停 + 外部 MAPE 选优）...")
    best_params, grid_df = _grid_search_nhits(train_df, val_size=val_size)
    print(f"网格搜索完成，共 {len(grid_df)} 组。最优验证 MAPE={grid_df.iloc[0]['val_mape_pct']:.4f}%")
    print(f"最优参数: {best_params}")

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    grid_path = os.path.join(PLOT_OUTPUT_DIR, "nhits_grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print(f"网格搜索结果已保存: {grid_path}")

    model = _build_nhits_model(
        h=forecast_horizon,
        input_size=best_params["input_size"],
        mlp_units=best_params["mlp_units"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        enable_train_log=ENABLE_TRAIN_LOG,
        alias="NHITS",
    )
    nf = NeuralForecast(models=[model], freq="MS")

    print("开始用最优参数重训 NHITS...")
    train_start = time.time()
    nf.fit(df=nf_train_df, val_size=val_size)
    train_seconds = time.time() - train_start
    print(f"NHITS 训练完成，用时: {train_seconds:.2f}s")

    print("开始预测桥接期 + 测试期（最终仅评估测试期）...")
    forecast_df = nf.predict(futr_df=futr_df).reset_index()
    forecast_df = forecast_df.rename(columns={"NHITS": "predicted_gas_sales"})

    eval_df = future_df[[MONTH_COL, "ds", TARGET_COL]].merge(
        forecast_df[["ds", "predicted_gas_sales"]],
        on="ds",
        how="left",
    )
    if eval_df["predicted_gas_sales"].isna().any():
        raise ValueError("NHITS 预测结果存在缺失，请检查 futr_df 与预测步长。")

    result_df = eval_df[
        (eval_df[MONTH_COL] >= TEST_START) & (eval_df[MONTH_COL] <= TEST_END)
    ].copy()
    result_df = result_df[[MONTH_COL, TARGET_COL, "predicted_gas_sales"]]
    result_df = result_df.rename(columns={TARGET_COL: "actual_gas_sales"})
    result_df["error"] = result_df["predicted_gas_sales"] - result_df["actual_gas_sales"]
    result_df["abs_error"] = np.abs(result_df["error"])
    result_df["mape_pct"] = np.where(
        result_df["actual_gas_sales"] != 0,
        np.abs(result_df["error"] / result_df["actual_gas_sales"]) * 100.0,
        np.nan,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    excel_path = os.path.join(
        OUTPUT_DIR,
        f"{PROCESSED_PROVINCE}_nhits_test_{TEST_START}_to_{TEST_END}.xlsx",
    )
    result_df.to_excel(excel_path, index=False, sheet_name="test_forecast")
    print(f"测试集预测结果已保存: {excel_path}")

    y_true = result_df["actual_gas_sales"].to_numpy(dtype=float)
    y_pred = result_df["predicted_gas_sales"].to_numpy(dtype=float)
    mape = _mape(y_true, y_pred)
    if np.isfinite(mape):
        print(f"\nNHITS 最终测试集 ({TEST_START} ~ {TEST_END}) 成绩:")
        print(f"Test MAPE: {mape:.4f}%")
    else:
        print("测试集包含 0 或为空，跳过 MAPE 输出。")


if __name__ == "__main__":
    main()
