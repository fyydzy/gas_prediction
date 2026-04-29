import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

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
from gas_prediction.timemoe_generation_compat import patch_timemoe_generation


def _forecast_monthly(train_values: np.ndarray, horizon_len: int) -> np.ndarray:
    seq = torch.tensor(np.asarray(train_values, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
    mean = seq.mean(dim=-1, keepdim=True)
    std = seq.std(dim=-1, keepdim=True).clamp_min(1e-6)
    normed_seq = (seq - mean) / std

    model = AutoModelForCausalLM.from_pretrained(
        "Maple728/TimeMoE-50M",
        device_map="cpu",
        trust_remote_code=True,
    )
    patch_timemoe_generation(model)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    normed_seq = normed_seq.to(dtype=model_dtype)
    with torch.no_grad():
        output = model.generate(
            normed_seq,
            max_new_tokens=horizon_len,
            do_sample=False,
            use_cache=False,
        )

    normed_pred = output[:, -horizon_len:]
    pred = (normed_pred * std + mean).squeeze(0).cpu().numpy()
    return np.clip(pred, 0, None).astype(float)


def main() -> None:
    input_path = find_processed_excel()
    output_path = os.path.join("output", f"{PROCESSED_PROVINCE}_timemoe_validation.xlsx")

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
