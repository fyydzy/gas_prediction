import os

import numpy as np
import timesfm

from gas_prediction.forecast_common import (
    AS_OF_MONTH,
    PROCESSED_PROVINCE,
    VAL_END,
    VAL_START,
    find_processed_excel,
    finalize_validation_table_9m,
    forecast_metrics,
    load_gas_series,
    split_asof_forecast9,
    TARGET_COL,
)


def _forecast_monthly(train_values: np.ndarray, horizon_len: int) -> np.ndarray:
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
    pred = model.forecast([train_values], freq=[2])
    pred_arr = np.asarray(pred[0] if isinstance(pred, tuple) else pred)
    return pred_arr[0][:horizon_len] if pred_arr.ndim == 2 else pred_arr[:horizon_len]


def main() -> None:
    input_path = find_processed_excel()
    output_path = os.path.join("output", f"{PROCESSED_PROVINCE}_timesfm_validation.xlsx")

    df = load_gas_series(input_path)
    train_df, val_df, forecast_months = split_asof_forecast9(df)

    full_pred = _forecast_monthly(train_df[TARGET_COL].values, horizon_len=len(forecast_months))
    full_pred = full_pred[: len(forecast_months)]

    result, y_true, y_pred = finalize_validation_table_9m(
        forecast_months, full_pred, val_df
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_excel(output_path, index=False, sheet_name="validation")

    metrics = forecast_metrics(y_true, y_pred)
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"as-of month: {AS_OF_MONTH}")
    print(f"forecast range: {forecast_months[0]} ~ {forecast_months[-1]}")
    print(f"validation range: {VAL_START} ~ {VAL_END}")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
