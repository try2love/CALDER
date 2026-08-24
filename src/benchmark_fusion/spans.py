"""Tokenizer-independent mapping from token evidence to common character spans."""

from __future__ import annotations

import numpy as np


def common_span_boundaries(text_length: int, num_spans: int) -> np.ndarray:
    if text_length < 0 or num_spans < 1:
        raise ValueError("text_length must be non-negative and num_spans positive")
    return np.asarray(
        [(index * text_length) // num_spans for index in range(num_spans + 1)],
        dtype=np.int64,
    )


def aggregate_to_common_spans(
    offsets: np.ndarray,
    values: np.ndarray,
    text_length: int,
    num_spans: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlap-weighted mean of token values in fixed character spans.

    Special/padding tokens with empty offsets are ignored. A token crossing a
    span boundary contributes proportionally to both spans.
    """

    offsets = np.asarray(offsets, dtype=np.int64)
    values = np.asarray(values)
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("offsets must have shape [tokens,2]")
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != offsets.shape[0]:
        raise ValueError("values must have shape [tokens,channels]")
    if not np.isfinite(values).all():
        raise ValueError("token values must be finite")
    boundaries = common_span_boundaries(text_length, num_spans)
    sums = np.zeros((num_spans, values.shape[1]), dtype=np.float64)
    weights = np.zeros(num_spans, dtype=np.float64)
    for token_index, (raw_start, raw_end) in enumerate(offsets):
        start = max(0, min(text_length, int(raw_start)))
        end = max(0, min(text_length, int(raw_end)))
        if end <= start:
            continue
        first = min(
            num_spans - 1,
            int(np.searchsorted(boundaries, start, side="right")) - 1,
        )
        last = min(
            num_spans - 1,
            int(np.searchsorted(boundaries, end - 1, side="right")) - 1,
        )
        for span_index in range(first, last + 1):
            overlap = max(
                0,
                min(end, int(boundaries[span_index + 1]))
                - max(start, int(boundaries[span_index])),
            )
            if overlap:
                sums[span_index] += values[token_index].astype(np.float64) * overlap
                weights[span_index] += overlap
    mask = weights > 0
    output = np.zeros_like(sums, dtype=np.float32)
    output[mask] = (sums[mask] / weights[mask, None]).astype(np.float32)
    return output, mask
