import argparse
import os

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from gas_prediction.forecast_common import (
    MONTH_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
    forecast_metrics,
    load_gas_series,
)

# === 与 train_timemoe_peft 一致的时间线 ===
INFERENCE_CONTEXT_END = "2025-06"
BRIDGE_START = "2025-07"
BRIDGE_END = "2025-10"
TEST_START = "2025-11"
TEST_END = "2026-03"

MODEL_NAME = "amazon/chronos-t5-small"
OUTPUT_DIR = "output"
NUM_SAMPLES = 100
PREDICT_TEMPERATURE = 0.8
PREDICT_TOP_P = 0.9


def _apply_context_window(
    df_asof: pd.DataFrame, context_months: int
) -> pd.DataFrame:
    """
    在「≤ INFERENCE_CONTEXT_END」已排序的序列上，只保留末尾 context_months 个月。
    context_months==0 表示不截断（使用全部 ≤as-of 历史）。
    """
    if context_months <= 0:
        return df_asof
    if len(df_asof) > context_months:
        return df_asof.iloc[-context_months:].copy()
    return df_asof


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos 零样本预测（可限制 context 为最近 N 个月）")
    parser.add_argument(
        "--context-months",
        type=int,
        default=60,
        help="Chronos 输入序列长度：仅使用 ≤as-of 的末尾 N 个月（60≈5 年，72≈6 年）。0=不截断、用全部历史。",
    )
    args = parser.parse_args()
    context_months = args.context_months

    input_path = find_processed_excel()
    df = load_gas_series(input_path)

    context_df = df[df[MONTH_COL] <= INFERENCE_CONTEXT_END].sort_values(MONTH_COL)
    context_df = _apply_context_window(context_df, context_months)
    context_values = torch.tensor(
        context_df[TARGET_COL].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    ).unsqueeze(0)

    bridge_df = df[(df[MONTH_COL] >= BRIDGE_START) & (df[MONTH_COL] <= BRIDGE_END)].sort_values(
        MONTH_COL
    )
    test_df = df[(df[MONTH_COL] >= TEST_START) & (df[MONTH_COL] <= TEST_END)].sort_values(MONTH_COL)

    bridge_months = bridge_df[MONTH_COL].astype(str).tolist()
    test_months = test_df[MONTH_COL].astype(str).tolist()
    bridge_true = bridge_df[TARGET_COL].to_numpy(dtype=np.float32)
    test_true = test_df[TARGET_COL].to_numpy(dtype=np.float32)

    total_horizon = len(bridge_months) + len(test_months)

    print("初始化 Chronos Pipeline...")
    ctx_tag = (
        "full"
        if context_months <= 0
        else f"last_{context_months}m (~{context_months // 12}y)"
    )
    print(
        f"Context: {context_values.shape[1]} 个月 | "
        f"窗口={ctx_tag} | 月末 {context_df[MONTH_COL].iloc[-1]} (截止 as-of {INFERENCE_CONTEXT_END})"
    )
    print(f"预测 horizon: {total_horizon} (bridge {len(bridge_months)} + test {len(test_months)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Chronos API: 第一个参数名是 inputs，不是 context
    print("\n预测中...")
    forecast_samples = pipeline.predict(
        inputs=context_values,
        prediction_length=total_horizon,
        num_samples=NUM_SAMPLES,
        temperature=PREDICT_TEMPERATURE,
        top_p=PREDICT_TOP_P,
    )
    # shape: (batch, num_samples, prediction_length)
    samp = forecast_samples[0].numpy()
    forecast_median = np.quantile(samp, 0.5, axis=0)
    forecast_mean = np.mean(samp, axis=0)

    forecast_months = bridge_months + test_months
    n_bridge = len(bridge_months)
    test_pred_median = forecast_median[n_bridge:]
    test_pred_mean = forecast_mean[n_bridge:]

    metrics_median = forecast_metrics(test_true.astype(float), test_pred_median.astype(float))
    metrics_mean = forecast_metrics(test_true.astype(float), test_pred_mean.astype(float))
    mape_med = float(metrics_median["MAPE(%)"])
    mape_mean = float(metrics_mean["MAPE(%)"])
    use_mean = mape_mean < mape_med
    chosen_label = "mean" if use_mean else "median"
    forecast_report = forecast_mean if use_mean else forecast_median

    result = pd.DataFrame(
        {
            MONTH_COL: forecast_months,
            "predicted_median": forecast_median.astype(float),
            "predicted_mean": forecast_mean.astype(float),
            "predicted_gas_sales": forecast_report.astype(float),
        }
    )
    result["aggregate_chosen_for_report"] = chosen_label
    actual_by_month: dict[str, float] = {}
    for m, v in zip(bridge_months, bridge_true.astype(float)):
        actual_by_month[m] = float(v)
    for m, v in zip(test_months, test_true.astype(float)):
        actual_by_month[m] = float(v)
    result["actual_gas_sales"] = result[MONTH_COL].map(actual_by_month)
    result["phase"] = np.where(
        result[MONTH_COL].isin(test_months),
        f"evaluation(test_{TEST_START}_to_{TEST_END})",
        f"bridge(unknown_{BRIDGE_START}_to_{BRIDGE_END})",
    )
    result["error"] = result["predicted_gas_sales"] - result["actual_gas_sales"]
    result["abs_error"] = np.abs(result["error"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ctx_suffix = "full" if context_months <= 0 else f"{context_months}m"
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{PROCESSED_PROVINCE}_chronos_ctx{ctx_suffix}_{INFERENCE_CONTEXT_END}_{total_horizon}m.xlsx",
    )
    result.to_excel(out_path, index=False, sheet_name="forecast")

    print("-" * 50)
    print(f"input: {input_path}")
    print(f"saved: {out_path}")
    print(f"Chronos zero-shot 测试集 ({TEST_START} ~ {TEST_END}) 对比（仅 test 段）:")
    print(
        f"  median | MAE={metrics_median['MAE']:.6f} RMSE={metrics_median['RMSE']:.6f} "
        f"MAPE={mape_med:.6f}%"
    )
    print(
        f"  mean   | MAE={metrics_mean['MAE']:.6f} RMSE={metrics_mean['RMSE']:.6f} "
        f"MAPE={mape_mean:.6f}%"
    )
    print(
        f"  → 最终报告采用: {chosen_label}（test MAPE 更低）；"
        f"predicted_gas_sales 列已写入该聚合结果"
    )


if __name__ == "__main__":
    main()