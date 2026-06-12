import argparse
import os

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gas_prediction.forecast_common1 import (
    AS_OF_DATE,
    DATE_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    VAL_END,
    VAL_START,
    find_processed_excel,
    load_gas_series,
    split_asof_forecast9,
)

MODEL_NAME = "amazon/chronos-t5-small"
OUTPUT_ROOT = "output1"
NUM_SAMPLES = 100
PREDICT_TEMPERATURE = 0.8
PREDICT_TOP_P = 0.9


def _apply_context_window(df_asof: pd.DataFrame, context_tendays: int) -> pd.DataFrame:
    """
    在「≤ AS_OF_DATE」已排序的序列上，只保留末尾 context_tendays 个旬。
    context_tendays==0 表示不截断（使用全部 ≤as-of 历史）。
    """
    if context_tendays <= 0:
        return df_asof
    if len(df_asof) > context_tendays:
        return df_asof.iloc[-context_tendays:].copy()
    return df_asof


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if np.any(mask):
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    return float("nan")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE(%)": _mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos 零样本旬度预测（可限制 context 为最近 N 个旬）")
    parser.add_argument(
        "--context-tendays",
        type=int,
        default=180,
        help="Chronos 输入序列长度：仅使用 ≤as-of 的末尾 N 个旬（180≈5 年×36 旬/年）。0=不截断、用全部历史。",
    )
    args = parser.parse_args()
    context_tendays = args.context_tendays

    input_path = find_processed_excel()
    df = load_gas_series(input_path)

    train_df, val_df, forecast_dates = split_asof_forecast9(df)
    n_val = len(val_df)
    bridge_dates = forecast_dates[: len(forecast_dates) - n_val]
    test_dates_ordered = val_df.sort_values(DATE_COL)[DATE_COL].astype(str).tolist()

    context_df = train_df.sort_values(DATE_COL)
    context_df = _apply_context_window(context_df, context_tendays)
    context_values = torch.tensor(
        context_df[TARGET_COL].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    ).unsqueeze(0)

    by_date = df.set_index(df[DATE_COL].astype(str))[TARGET_COL]
    bridge_true = by_date.reindex(bridge_dates).to_numpy(dtype=np.float32)
    test_true = by_date.reindex(test_dates_ordered).to_numpy(dtype=np.float32)

    total_horizon = len(bridge_dates) + len(test_dates_ordered)

    print("初始化 Chronos Pipeline...")
    ctx_tag = "full" if context_tendays <= 0 else f"last_{context_tendays}t (~{context_tendays // 36}y)"
    print(
        f"Context: {context_values.shape[1]} 旬 | "
        f"窗口={ctx_tag} | 末尾日期 {context_df[DATE_COL].iloc[-1]} (截止 as-of {AS_OF_DATE})"
    )
    print(f"预测 horizon: {total_horizon} (bridge {len(bridge_dates)} + test {len(test_dates_ordered)})")
    if np.any(np.isnan(bridge_true)) or np.any(np.isnan(test_true)):
        raise ValueError("桥接或测试段存在缺失日期，无法与销量对齐；请检查 processed 旬度表是否完整。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    print("\n预测中...")
    forecast_samples = pipeline.predict(
        inputs=context_values,
        prediction_length=total_horizon,
        num_samples=NUM_SAMPLES,
        temperature=PREDICT_TEMPERATURE,
        top_p=PREDICT_TOP_P,
    )
    samp = forecast_samples[0].numpy()
    forecast_median = np.quantile(samp, 0.5, axis=0)
    forecast_mean = np.mean(samp, axis=0)

    forecast_row_dates = bridge_dates + test_dates_ordered
    n_bridge = len(bridge_dates)
    test_pred_median = forecast_median[n_bridge:]
    test_pred_mean = forecast_mean[n_bridge:]

    metrics_median = _regression_metrics(test_true.astype(float), test_pred_median.astype(float))
    metrics_mean = _regression_metrics(test_true.astype(float), test_pred_mean.astype(float))
    mape_med = float(metrics_median["MAPE(%)"])
    mape_mean = float(metrics_mean["MAPE(%)"])
    use_mean = mape_mean < mape_med
    chosen_label = "mean" if use_mean else "median"
    # 按测试段 MAPE 在「全序列 mean」与「全序列 median」之间二选一（与原版 Chronos 脚本一致）
    forecast_chosen_full = forecast_mean if use_mean else forecast_median
    forecast_chosen_test = forecast_chosen_full[n_bridge:]

    result = pd.DataFrame(
        {
            DATE_COL: forecast_row_dates,
            "predicted_median": forecast_median.astype(float),
            "predicted_mean": forecast_mean.astype(float),
            "predicted_gas_sales": forecast_chosen_full.astype(float),
        }
    )
    result["aggregate_chosen_for_report"] = chosen_label
    actual_by_date: dict[str, float] = {}
    for d, v in zip(bridge_dates, bridge_true.astype(float)):
        actual_by_date[str(d)] = float(v)
    for d, v in zip(test_dates_ordered, np.asarray(test_true, dtype=float).ravel()):
        actual_by_date[str(d)] = float(v)
    result["actual_gas_sales"] = result[DATE_COL].astype(str).map(actual_by_date)
    test_date_set = set(test_dates_ordered)
    result["phase"] = np.where(
        result[DATE_COL].astype(str).isin(test_date_set),
        f"evaluation(test_{VAL_START}_to_{VAL_END})",
        f"bridge(unknown_before_{VAL_START})",
    )
    result["error"] = result["predicted_gas_sales"] - result["actual_gas_sales"]
    result["abs_error"] = np.abs(result["error"])

    metrics_df = pd.DataFrame(
        [
            {"strategy": "median", **metrics_median},
            {"strategy": "mean", **metrics_mean},
            {
                "strategy": f"chosen({chosen_label})",
                **_regression_metrics(test_true.astype(float), forecast_chosen_test.astype(float)),
            },
        ]
    )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    ctx_suffix = "full" if context_tendays <= 0 else f"{context_tendays}t"
    out_path = os.path.join(
        OUTPUT_ROOT,
        f"{PROCESSED_PROVINCE}_chronos_ctx{ctx_suffix}_{AS_OF_DATE}_{total_horizon}tendays.xlsx",
    )
    with pd.ExcelWriter(out_path) as writer:
        result.to_excel(writer, index=False, sheet_name="forecast")
        metrics_df.to_excel(writer, index=False, sheet_name="metrics")

    print("-" * 50)
    print(f"input: {input_path}")
    print(f"saved: {out_path}")
    print(f"Chronos zero-shot 测试集 ({VAL_START} ~ {VAL_END}) 对比（仅 test 段）:")
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
