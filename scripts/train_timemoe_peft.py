import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from gas_prediction.forecast_common import (
    MONTH_COL,
    PROCESSED_PROVINCE,
    TARGET_COL,
    find_processed_excel,
    forecast_metrics,
    load_gas_series,
)
from gas_prediction.timemoe_generation_compat import patch_timemoe_generation


# =============================================================================
# 时间线（YYYY-MM，与 Excel 的 month 列一致；本脚本自洽）
# =============================================================================
#
# 【训练集】
#   从数据「最早月份」起至 TRAIN_END（含），与 as-of 一致：默认 = INFERENCE_CONTEXT_END（2025-06）。
#   LoRA 滑窗与归一化都使用该段。
#
# 【验证集 = 推理 bridge】
#   VAL_START～VAL_END（默认 2025-07～2025-10）：早停看 bridge_*，与最终一次性推理的 bridge 段一致。
#
# 【测试集】
#   TEST_START～TEST_END（2025-11～2026-03）：最终 test_* 仅看该段。
#
# 【推理】
#   - context：全部 ≤ INFERENCE_CONTEXT_END（2025-06）
#   - 一次生成 len(VAL)+len(TEST)=9 个月；仅 TEST 段计入最终 test_*。
#
TRAIN_END = "2024-12"
VAL_START = "2025-07"
VAL_END = "2025-10"
TEST_START = "2025-11"
TEST_END = "2026-03"
INFERENCE_CONTEXT_END = "2025-06"

# 每个训练样本使用的连续月份长度（只在训练拟合段内部滑窗）。
TRAIN_SEQ_LEN = 48
# 滑窗步长。1 表示最大样本量；更大步长可减少样本相关性。
TRAIN_STRIDE = 1
EPS = 1e-6
MODEL_NAME = "Maple728/TimeMoE-50M"
OUTPUT_DIR = os.path.join("output", "timemoe50m_lora")
# LoRA 权重保存目录：None = 每次运行自动生成 `adapters/YYYYMMDD_HHMMSS`（不覆盖历史）；
# 设为固定字符串如 "my_run1" 则保存到 `adapters/my_run1`（同名会覆盖）。
ADAPTER_RUN_NAME: str | None = None


def _resolve_adapter_save_dir() -> str:
    run = ADAPTER_RUN_NAME if ADAPTER_RUN_NAME else datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, "adapters", run)


# 验证时用哪种推理上下文长度（序列均截止到 INFERENCE_CONTEXT_END）：
# - "full": 用 ≤INFERENCE_CONTEXT_END 的全部历史
# - "last_window": 只用最后 TRAIN_SEQ_LEN 个月
EVAL_CONTEXT_MODE = "full"

# 训练超参数
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPOCHS = 30
PATIENCE = 5
MIN_DELTA = 1e-4

# LoRA 参数
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


@dataclass
class TimeMoeTrainBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_masks: torch.Tensor
    norm_mean: float
    norm_std: float


def _normalize_1d(x: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < eps:
        std = eps
    x_norm = (x - mean) / std
    return x_norm.astype(np.float32), mean, std


def build_train_batch(
    train_values: np.ndarray,
    seq_len: int = TRAIN_SEQ_LEN,
    stride: int = TRAIN_STRIDE,
    norm_mean: float | None = None,
    norm_std: float | None = None,
) -> TimeMoeTrainBatch:
    """
    构造 TimeMoE 训练输入：
    - input_ids: [N, L, 1]
    - labels:    [N, L, 1]
    - loss_masks:[N, L, 1]

    序列来自训练集：≤TRAIN_END。
    """
    values = np.asarray(train_values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"train_values 需要一维序列，收到 shape={values.shape}")
    if len(values) < seq_len:
        raise ValueError(
            f"训练序列长度不足，当前 {len(values)}，至少需要 seq_len={seq_len}"
        )
    if stride <= 0:
        raise ValueError(f"stride 必须 > 0，当前为 {stride}")

    if norm_mean is None or norm_std is None:
        # 默认使用训练段全局归一化，保证训练/推理尺度一致。
        _, norm_mean, norm_std = _normalize_1d(values)
    norm_std = max(float(norm_std), EPS)
    values_norm = ((values - float(norm_mean)) / norm_std).astype(np.float32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ms: list[np.ndarray] = []

    for start in range(0, len(values) - seq_len + 1, stride):
        window_norm = values_norm[start : start + seq_len]

        # TimeMoE 训练时 labels 与 input_ids 对齐，由模型内部计算不同 horizon 的 AR loss。
        x = window_norm[:, None]  # [L, 1]
        y = window_norm[:, None]  # [L, 1]
        m = np.ones_like(y, dtype=np.float32)  # [L, 1]

        xs.append(x)
        ys.append(y)
        ms.append(m)

    input_ids = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
    labels = torch.tensor(np.stack(ys, axis=0), dtype=torch.float32)
    loss_masks = torch.tensor(np.stack(ms, axis=0), dtype=torch.float32)
    return TimeMoeTrainBatch(
        input_ids=input_ids,
        labels=labels,
        loss_masks=loss_masks,
        norm_mean=float(norm_mean),
        norm_std=float(norm_std),
    )


def build_eval_context(
    train_values: np.ndarray,
    norm_mean: float,
    norm_std: float,
) -> torch.Tensor:
    """
    评估/推理上下文
    返回 shape: [1, L]。
    说明：TimeMoE 的 TSGenerationMixin._greedy_search 只接受二维输入。
    """
    seq = np.asarray(train_values, dtype=np.float32)
    seq_norm = (seq - float(norm_mean)) / max(float(norm_std), EPS)
    return torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)


def denormalize(values: np.ndarray, norm_mean: float, norm_std: float) -> np.ndarray:
    """把归一化值恢复到真实销量尺度。"""
    return np.asarray(values, dtype=np.float32) * float(norm_std) + float(norm_mean)


def _split_train_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """训练 ≤TRAIN_END；验证(=bridge) VAL_*；测试 TEST_*。"""
    train_df = df[df[MONTH_COL] <= TRAIN_END].copy()
    val_df = df[(df[MONTH_COL] >= VAL_START) & (df[MONTH_COL] <= VAL_END)].copy()
    test_df = df[(df[MONTH_COL] >= TEST_START) & (df[MONTH_COL] <= TEST_END)].copy()
    if train_df.empty:
        raise ValueError(f"训练集为空（需要月份 ≤ {TRAIN_END}）。")
    if val_df.empty:
        raise ValueError(f"验证集(bridge)为空，请检查数据中是否存在 {VAL_START} ~ {VAL_END}。")
    if test_df.empty:
        raise ValueError(f"测试集为空，请检查数据中是否存在 {TEST_START} ~ {TEST_END}。")
    return train_df, val_df, test_df


def prepare_data(
    seq_len: int = TRAIN_SEQ_LEN,
    stride: int = TRAIN_STRIDE,
) -> tuple[
    TimeMoeTrainBatch,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    str,
]:
    """
    训练 ≤TRAIN_END；验证(bridge) VAL_*；测试 TEST_*。
    推理 context：≤INFERENCE_CONTEXT_END（与训练/验证/测试同一套 VAL+TEST 口径）。
    """
    input_path = find_processed_excel()
    df = load_gas_series(input_path)
    train_df, val_df, test_df = _split_train_val_test(df)

    train_df = train_df.sort_values(MONTH_COL).reset_index(drop=True)
    train_values = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    if len(train_values) < seq_len:
        raise ValueError(
            f"训练集长度不足：len={len(train_values)}，需要 >= seq_len={seq_len}。"
        )

    infer_context_df = (
        df[df[MONTH_COL] <= INFERENCE_CONTEXT_END]
        .copy()
        .sort_values(MONTH_COL)
        .reset_index(drop=True)
    )
    if infer_context_df.empty:
        raise ValueError(f"推理 context 为空（需要月份 ≤ {INFERENCE_CONTEXT_END}）。")
    infer_context_values = infer_context_df[TARGET_COL].to_numpy(dtype=np.float32)

    val_true = val_df[TARGET_COL].to_numpy(dtype=np.float32)
    test_true = test_df[TARGET_COL].to_numpy(dtype=np.float32)
    val_months = val_df[MONTH_COL].astype(str).tolist()
    test_months = test_df[MONTH_COL].astype(str).tolist()

    _, norm_mean, norm_std = _normalize_1d(train_values)
    train_batch = build_train_batch(
        train_values,
        seq_len=seq_len,
        stride=stride,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    if EVAL_CONTEXT_MODE == "last_window":
        ctx_values = infer_context_values[-seq_len:]
    else:
        ctx_values = infer_context_values
    eval_context = build_eval_context(
        ctx_values,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    return (
        train_batch,
        eval_context,
        val_true,
        test_true,
        val_months,
        test_months,
        input_path,
    )


def _build_lora_model(device: torch.device) -> PeftModel:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(base_model, peft_config)
    # transformers 新版会把 loss_function 映射到 CausalLM 词表损失，
    # 而 TimeMoE 训练期需要回归损失（Huber），这里显式覆盖。
    huber = torch.nn.HuberLoss(reduction="none", delta=2.0)
    # 注意：在 PEFT 包装链下直接写 `loss_function` 可能不会落到真实 TimeMoE 实例，
    # 这里直接注入 `_loss_function`，确保 PreTrainedModel.loss_function getter 命中。
    if hasattr(model, "_loss_function"):
        model._loss_function = huber
    if hasattr(model, "base_model"):
        if hasattr(model.base_model, "_loss_function"):
            model.base_model._loss_function = huber
        if hasattr(model.base_model, "model"):
            model.base_model.model._loss_function = huber
    patch_timemoe_generation(model)
    model.print_trainable_parameters()
    return model


def _iter_minibatches(
    batch: TimeMoeTrainBatch,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    n = batch.input_ids.shape[0]
    idx = torch.randperm(n)
    mini_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        bidx = idx[s:e]
        mini_batches.append(
            (
                batch.input_ids[bidx],
                batch.labels[bidx],
                batch.loss_masks[bidx],
            )
        )
    return mini_batches


def _forecast_with_model(
    model: PeftModel,
    eval_context: torch.Tensor,
    horizon_len: int,
    norm_mean: float,
    norm_std: float,
) -> np.ndarray:
    model.eval()
    gen_input = eval_context
    if gen_input.ndim == 3 and gen_input.shape[-1] == 1:
        # 兼容旧缓存：若传入 [B, L, 1]，生成时降为 [B, L]
        gen_input = gen_input.squeeze(-1)
    with torch.no_grad():
        output = model.generate(
            gen_input,
            max_new_tokens=horizon_len,
            do_sample=False,
            use_cache=False,
        )
    # 统一压成一维 [H]，避免与 y_true=[H] 计算指标时发生广播歧义。
    pred_chunk = output[:, -horizon_len:]
    pred_norm = pred_chunk.reshape(-1).detach().cpu().numpy()
    pred = denormalize(pred_norm, norm_mean=norm_mean, norm_std=norm_std)
    return np.clip(pred.astype(float), 0, None)


def train_with_early_stopping(
    model: PeftModel,
    train_batch: TimeMoeTrainBatch,
    eval_context: torch.Tensor,
    val_true: np.ndarray,
    test_true: np.ndarray,
    device: torch.device,
    adapter_save_dir: str,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_mape = float("inf")
    best_epoch = -1
    best_adapter_path = adapter_save_dir
    os.makedirs(best_adapter_path, exist_ok=True)
    wait = 0

    eval_context = eval_context.to(device)
    last_train_loss = float("nan")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses: list[float] = []
        for x, y, m in _iter_minibatches(train_batch, BATCH_SIZE):
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            outputs = model(input_ids=x, labels=y, loss_masks=m, use_cache=False)
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        last_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        # 与最终推理一致：同一 eval_context，一次生成 bridge(VAL)+test
        full_pred = _forecast_with_model(
            model,
            eval_context=eval_context,
            horizon_len=len(val_true) + len(test_true),
            norm_mean=train_batch.norm_mean,
            norm_std=train_batch.norm_std,
        )
        val_pred = full_pred[: len(val_true)]
        test_pred = full_pred[len(val_true) : len(val_true) + len(test_true)]
        val_metrics = forecast_metrics(val_true.astype(float), val_pred.astype(float))
        test_metrics = forecast_metrics(test_true.astype(float), test_pred.astype(float))
        val_mape = float(val_metrics["MAPE(%)"])

        print(
            f"[epoch {epoch:03d}] train_loss={last_train_loss:.6f} "
            f"bridge_MAE={val_metrics['MAE']:.6f} bridge_RMSE={val_metrics['RMSE']:.6f} "
            f"bridge_MAPE={val_mape:.6f} | "
            f"test_MAE={test_metrics['MAE']:.6f} test_RMSE={test_metrics['RMSE']:.6f} "
            f"test_MAPE={test_metrics['MAPE(%)']:.6f}"
        )

        if val_mape + MIN_DELTA < best_val_mape:
            best_val_mape = val_mape
            best_epoch = epoch
            wait = 0
            model.save_pretrained(best_adapter_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"early stopping at epoch={epoch}, best_epoch={best_epoch}")
                break

    return {
        "best_epoch": float(best_epoch),
        "best_val_mape": float(best_val_mape),
        "last_train_loss": float(last_train_loss),
        "best_adapter_path": best_adapter_path,
    }


def load_best_adapter_model(device: torch.device, adapter_path: str) -> PeftModel:
    """加载最佳 LoRA adapter，并应用 TimeMoE 兼容补丁。"""
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    patch_timemoe_generation(model)
    return model


def infer_9m_then_eval_last5(
    model: PeftModel,
    eval_context: torch.Tensor,
    val_months: list[str],
    test_months: list[str],
    test_true: np.ndarray,
    norm_mean: float,
    norm_std: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    业务口径：context 截止 INFERENCE_CONTEXT_END；一次生成 len(VAL)+len(TEST)=9 个月；
    仅 TEST_START～TEST_END 段计入最终 test_* 指标。
    """
    forecast_months = val_months + test_months
    full_pred = _forecast_with_model(
        model=model,
        eval_context=eval_context,
        horizon_len=len(forecast_months),
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    full_pred = full_pred[: len(forecast_months)]

    bridge_len = len(val_months)
    test_pred = full_pred[bridge_len : bridge_len + len(test_months)]
    test_metrics = forecast_metrics(test_true.astype(float), test_pred.astype(float))

    result = pd.DataFrame(
        {
            MONTH_COL: forecast_months,
            "predicted_gas_sales": full_pred.astype(float),
        }
    )
    result["phase"] = np.where(
        result[MONTH_COL].isin(test_months),
        f"evaluation(test_{TEST_START}_to_{TEST_END})",
        f"bridge(unknown_{VAL_START}_to_{VAL_END})",
    )
    # 按月份对齐真值：test_true 仅覆盖 TEST 段，不能和整表 forecast 行数直接广播
    actual_by_month = dict(zip(test_months, test_true.astype(float)))
    result["actual_gas_sales"] = result[MONTH_COL].map(actual_by_month)
    result["error"] = result["predicted_gas_sales"] - result["actual_gas_sales"]
    result["abs_error"] = np.abs(result["error"])
    return result, test_metrics


def main() -> None:
    (
        train_batch,
        eval_context,
        val_true,
        test_true,
        val_months,
        test_months,
        input_path,
    ) = prepare_data()

    os.makedirs("output", exist_ok=True)
    print(f"input: {input_path}")
    print(f"province: {PROCESSED_PROVINCE}")
    print(f"训练集: 数据首月起至 {TRAIN_END}（含）")
    print(f"验证集(=bridge，早停看 bridge_MAPE): {VAL_START} ~ {VAL_END}")
    print(f"测试集(最终 test_*): {TEST_START} ~ {TEST_END}")
    print(
        f"推理: context 截止 {INFERENCE_CONTEXT_END}；"
        f"bridge {VAL_START}~{VAL_END}；test {TEST_START}~{TEST_END}"
    )
    print("-" * 50)
    print(f"train input_ids shape: {tuple(train_batch.input_ids.shape)}")
    print(f"train labels shape: {tuple(train_batch.labels.shape)}")
    print(f"train loss_masks shape: {tuple(train_batch.loss_masks.shape)}")
    print(f"eval context shape: {tuple(eval_context.shape)}")
    print(f"eval context mode: {EVAL_CONTEXT_MODE}")
    print(f"normalization(mean/std): {train_batch.norm_mean:.6f} / {train_batch.norm_std:.6f}")
    print(
        f"val(bridge) months: {val_months[0]} ~ {val_months[-1]}, "
        f"y_true shape={tuple(val_true.shape)}"
    )
    print(f"test months: {test_months[0]} ~ {test_months[-1]}, y_true shape={tuple(test_true.shape)}")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = _build_lora_model(device=device)

    adapter_save_dir = _resolve_adapter_save_dir()
    print(f"adapter save dir: {adapter_save_dir}")

    stats = train_with_early_stopping(
        model=model,
        train_batch=train_batch,
        eval_context=eval_context,
        val_true=val_true,
        test_true=test_true,
        device=device,
        adapter_save_dir=adapter_save_dir,
    )
    print("-" * 50)
    print(f"best_epoch: {int(stats['best_epoch'])}")
    print(f"best_val_mape: {stats['best_val_mape']:.6f}")
    print(f"last_train_loss: {stats['last_train_loss']:.6f}")
    best_adapter_path = str(stats["best_adapter_path"])
    print(f"best adapter saved to: {best_adapter_path}")

    # 先加载最优参数，再做业务口径推理：一次性生成 VAL+TEST 全 horizon，仅 TEST 段算最终 test_*。
    best_model = load_best_adapter_model(device=device, adapter_path=best_adapter_path)
    result_forecast, test_metrics = infer_9m_then_eval_last5(
        model=best_model,
        eval_context=eval_context.to(device),
        val_months=val_months,
        test_months=test_months,
        test_true=test_true,
        norm_mean=train_batch.norm_mean,
        norm_std=train_batch.norm_std,
    )

    result_path = os.path.join(
        OUTPUT_DIR,
        f"{PROCESSED_PROVINCE}_context_{INFERENCE_CONTEXT_END}_9m.xlsx",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_forecast.to_excel(result_path, index=False, sheet_name="forecast_9m")

    print("-" * 50)
    print(
        f"forecast months: {val_months[0]} ~ {test_months[-1]} "
        f"(bridge={len(val_months)}, test={len(test_months)}, total={len(val_months)+len(test_months)})"
    )
    print(f"saved forecast: {result_path}")
    for k, v in test_metrics.items():
        print(f"test_{k}: {v:.6f}")


if __name__ == "__main__":
    main()
