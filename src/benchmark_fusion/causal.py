"""Shared causal-observer inference helpers."""

from __future__ import annotations

import math
from typing import Any

import torch


@torch.inference_mode()
def bounded_context_codelength(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    word_count: int,
    utf8_bytes: int,
    context_tokens: int = 128,
) -> torch.Tensor:
    """Score each transition once in overlapping bounded-context windows."""

    if input_ids.ndim != 1 or input_ids.numel() < 2 or context_tokens < 2:
        raise ValueError("bounded context requires a one-dimensional sequence with a transition")
    if word_count < 1 or utf8_bytes < 1:
        raise ValueError("codelength document normalizers must be positive")
    surprisal_sum = torch.zeros((), dtype=torch.float64, device=input_ids.device)
    transition_count = 0
    start = 0
    while start < input_ids.numel() - 1:
        end = min(input_ids.numel(), start + context_tokens)
        window = input_ids[start:end].unsqueeze(0)
        output: Any = model(
            input_ids=window, attention_mask=torch.ones_like(window), use_cache=False
        )
        logits = output.logits
        targets = window[:, 1:]
        log_probability = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        surprisal_sum += (-log_probability).sum().double()
        transition_count += targets.numel()
        start = end - 1
    expected = input_ids.numel() - 1
    if transition_count != expected:
        raise AssertionError(f"bounded-context transition coverage drift: {transition_count} != {expected}")
    bits = surprisal_sum / math.log(2.0)
    output = torch.stack(
        (
            bits / transition_count,
            bits / int(word_count),
            bits / int(utf8_bytes),
        )
    ).float()
    if not torch.isfinite(output).all():
        raise FloatingPointError("non-finite bounded-context codelength")
    return output
