import os

import numpy as np
import pandas as pd
import timesfm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gas_prediction.forecast_common1 import (
    AS_OF_DATE,
    PROCESSED_PROVINCE,
    VAL_END,
    VAL_START,
    find_processed_excel,
    finalize_validation_table_9m,
    load_gas_series,
    split_asof_forecast9,
    TARGET_COL,
)


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


def _forecast_series(train_values: np.ndarray, horizon_len: int, *, freq: int) -> np.ndarray:
    """TimesFM：freq 0=高频 1=中频 2=低频；旬度序列用 1 较接近原月度脚本用 2 的相对「粒度」。"""
    train_values = np.asarray(train_values, dtype=np.float32)
    patch_len = 32
    pad = (-len(train_values)) % patch_len
    if pad:
        train_values = np.concatenate([np.full(pad, np.nan, dtype=np.float32), train_values])

    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=1,
            horizon_len=horizon_len,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
        ),
    )
    pred = model.forecast([train_values], freq=[freq])
    pred_arr = np.asarray(pred[0] if isinstance(pred, tuple) else pred)
    return pred_arr[0][:horizon_len] if pred_arr.ndim == 2 else pred_arr[:horizon_len]


def main() -> None:
    input_path = find_processed_excel()
    output_root = "output1"
    output_path = os.path.join(output_root, f"{PROCESSED_PROVINCE}_timesfm_validation.xlsx")

    df = load_gas_series(input_path)
    train_df, val_df, forecast_dates = split_asof_forecast9(df)

    horizon_len = len(forecast_dates)
    full_pred = _forecast_series(train_df[TARGET_COL].values, horizon_len=horizon_len, freq=2)
    full_pred = full_pred[:horizon_len]

    result, y_true, y_pred = finalize_validation_table_9m(forecast_dates, full_pred, val_df)

    metrics = _regression_metrics(y_true.astype(float), y_pred.astype(float))
    metrics_df = pd.DataFrame([metrics])

    os.makedirs(output_root, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        result.to_excel(writer, index=False, sheet_name="validation")
        metrics_df.to_excel(writer, index=False, sheet_name="metrics")

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"as-of date: {AS_OF_DATE}")
    print(f"forecast range: {forecast_dates[0]} ~ {forecast_dates[-1]} ({len(forecast_dates)} tendays)")
    print(f"validation range: {VAL_START} ~ {VAL_END}")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
