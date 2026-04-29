"""TimeMoE 与新版 transformers 的兼容补丁。

1) `_extract_past_from_model_output`：远程 `ts_generation_mixin` 仍调用旧 API。
2) `_sample`：transformers>=4.46 将贪心解码也路由到 `GenerationMixin._sample`（2D token 拼接），
   不再走 `TSGenerationMixin._greedy_search`（时间序列 3D 逻辑），会导致维度错误。
   因此在 `do_sample=False` 且非 beam 时改回调用 `TSGenerationMixin._greedy_search`。
"""

from __future__ import annotations

import types
from typing import Any, Optional

from transformers.generation.utils import GenerationMixin


def _extract_past_from_model_output(
    self: Any,
    outputs: Any,
    standardize_cache_format: bool = False,
) -> Any:
    _ = standardize_cache_format
    return getattr(outputs, "past_key_values", None)


def _find_ts_greedy_search(model_cls: type) -> Optional[Any]:
    for base in model_cls.__mro__:
        if base.__name__ == "TSGenerationMixin" and "_greedy_search" in base.__dict__:
            return base.__dict__["_greedy_search"]
    return None


def _patched_sample(
    self: Any,
    input_ids,
    logits_processor,
    stopping_criteria,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    greedy_fn = _find_ts_greedy_search(type(self))
    if greedy_fn is not None and not getattr(generation_config, "do_sample", False):
        if getattr(generation_config, "num_beams", 1) == 1:
            return greedy_fn(
                self,
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=stopping_criteria.max_length,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_attentions=generation_config.output_attentions,
                output_hidden_states=generation_config.output_hidden_states,
                output_scores=generation_config.output_scores,
                output_logits=getattr(generation_config, "output_logits", None),
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
    return GenerationMixin._sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        **model_kwargs,
    )


def _collect_model_candidates(model: Any) -> list[Any]:
    out: list[Any] = [model]
    base = getattr(model, "base_model", None)
    if base is not None:
        out.append(base)
        inner = getattr(base, "model", None)
        if inner is not None:
            out.append(inner)
    if callable(getattr(model, "get_base_model", None)):
        try:
            bm = model.get_base_model()
            if bm is not None:
                out.append(bm)
        except Exception:
            pass
    return out


def patch_timemoe_generation(model: Any) -> None:
    """对 TimeMoE（及 PEFT 底层）绑定缺失方法，并修正贪心解码路由。"""
    candidates = _collect_model_candidates(model)
    seen: set[int] = set()
    for m in candidates:
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        if _find_ts_greedy_search(type(m)) is None:
            continue
        if getattr(m, "_extract_past_from_model_output", None) is None:
            m._extract_past_from_model_output = types.MethodType(
                _extract_past_from_model_output, m
            )
        if not getattr(m, "__timemoe_compat_patched__", False):
            # transformers>=4.46 在 generate() 中通过 `getattr(type(self), "_sample")`
            # 获取解码方法，因此必须覆写到类级别，实例级别覆写不会生效。
            setattr(type(m), "_sample", _patched_sample)
            m.__timemoe_compat_patched__ = True
