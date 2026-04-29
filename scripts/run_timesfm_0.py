import os

import numpy as np
import pandas as pd
import timesfm

from gas_prediction.forecast_common import (
    AS_OF_MONTH,
    MONTH_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    VAL_END,
    VAL_START,
    build_result_two_stage,
    find_processed_excel,
    forecast_metrics,
    load_gas_series,
    split_asof_bridge,
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
    train_df, val_df, bridge_months = split_asof_bridge(df)

    bridge_pred = _forecast_monthly(train_df[TARGET_COL].values, horizon_len=len(bridge_months))
    bridge_series = pd.Series(bridge_pred[: len(bridge_months)], index=bridge_months, name=TARGET_COL)

    context_values = np.concatenate(
        [train_df[TARGET_COL].to_numpy(dtype=float), bridge_series.to_numpy(dtype=float)]
    )
    y_pred = _forecast_monthly(context_values, horizon_len=len(val_df))
    y_true = val_df[TARGET_COL].to_numpy(dtype=float)

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    val_months = val_df[MONTH_COL].values[:min_len]

    result = build_result_two_stage(
        bridge_months,
        bridge_pred[: len(bridge_months)],
        val_months,
        y_pred,
        y_true,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_excel(output_path, index=False, sheet_name="validation")

    metrics = forecast_metrics(y_true, y_pred)
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"as-of month: {AS_OF_MONTH}")
    print(f"bridge range: {bridge_months[0]} ~ {bridge_months[-1]}")
    print(f"validation range: {VAL_START} ~ {VAL_END}")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
