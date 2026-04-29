"""
使用已保存的 LoRA adapter（默认 `output/timemoe50m_lora/best_adapter`）做推理，
数据切分、归一化、context、9 个月预测与 `train_timemoe_peft.py` 一致（仅不训练）。

在项目根目录执行：
  uv run python scripts/run_timemoe_1.py
  uv run python scripts/run_timemoe_1.py --adapter output/timemoe50m_lora/adapters/某次时间戳
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# 保证能 import 同目录下的 train_timemoe_peft
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train_timemoe_peft as tm  # noqa: E402


def _default_adapter_path() -> str:
    return os.path.join(tm.OUTPUT_DIR, "best_adapter")


def main() -> None:
    parser = argparse.ArgumentParser(description="TimeMoE LoRA adapter 推理（与 train_timemoe_peft 口径一致）")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help=f"adapter 目录（含 adapter_config.json）。默认: {_default_adapter_path()}",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="结果 xlsx 路径；默认 output/timemoe50m_lora/{{省}}_context_{{asof}}_9m_infer.xlsx",
    )
    args = parser.parse_args()

    adapter_path = os.path.abspath(args.adapter or _default_adapter_path())
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"adapter 目录不存在: {adapter_path}")

    (
        train_batch,
        eval_context,
        _val_true,
        test_true,
        val_months,
        test_months,
        input_path,
    ) = tm.prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"input: {input_path}")
    print(f"province: {tm.PROCESSED_PROVINCE}")
    print(f"adapter: {adapter_path}")
    print(f"device: {device}")
    print(
        f"推理口径: context ≤ {tm.INFERENCE_CONTEXT_END}; "
        f"bridge {tm.VAL_START}~{tm.VAL_END}; test {tm.TEST_START}~{tm.TEST_END}"
    )
    print(f"eval context shape: {tuple(eval_context.shape)}")
    print(f"normalization(mean/std): {train_batch.norm_mean:.6f} / {train_batch.norm_std:.6f}")

    model = tm.load_best_adapter_model(device=device, adapter_path=adapter_path)
    result_forecast, test_metrics = tm.infer_9m_then_eval_last5(
        model=model,
        eval_context=eval_context.to(device),
        val_months=val_months,
        test_months=test_months,
        test_true=test_true,
        norm_mean=train_batch.norm_mean,
        norm_std=train_batch.norm_std,
    )

    out_dir = tm.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    if args.out:
        result_path = os.path.abspath(args.out)
    else:
        result_path = os.path.join(
            out_dir,
            f"{tm.PROCESSED_PROVINCE}_context_{tm.INFERENCE_CONTEXT_END}_9m_infer.xlsx",
        )
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    result_forecast.to_excel(result_path, index=False, sheet_name="forecast_9m")

    print("-" * 50)
    print(
        f"forecast: {val_months[0]} ~ {test_months[-1]} "
        f"(bridge={len(val_months)}, test={len(test_months)})"
    )
    print(f"saved: {result_path}")
    for k, v in test_metrics.items():
        print(f"test_{k}: {v:.6f}")


if __name__ == "__main__":
    main()
