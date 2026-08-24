"""Numerically stable observer statistics shared by feature views and baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TokenStatistics:
    log_probability: torch.Tensor
    surprisal: torch.Tensor
    entropy: torch.Tensor
    log_rank: torch.Tensor


@dataclass
class AlignmentStatistics:
    aligned_minus_base: torch.Tensor
    information_weighted_imprint: torch.Tensor
    rai: torch.Tensor
    observed_imprint: torch.Tensor


@dataclass
class ExactLAPDStatistics:
    score: torch.Tensor
    observed_statistic: torch.Tensor
    null_mean: torch.Tensor
    null_std: torch.Tensor
    token_count: torch.Tensor


@dataclass
class TokenNullMoments:
    mean: torch.Tensor
    variance: torch.Tensor


def token_statistics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 32,
) -> TokenStatistics:
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or mask.shape != targets.shape:
        raise ValueError("logits/targets/mask shapes do not align")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    log_probabilities = []
    entropies = []
    log_ranks = []
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        chunk = logits[:, start:end].float()
        chunk_targets = targets[:, start:end].long()
        target_logits = chunk.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)
        log_partition = torch.logsumexp(chunk, dim=-1)
        log_probability = target_logits - log_partition
        probabilities = torch.softmax(chunk, dim=-1)
        entropy = log_partition - (probabilities * chunk).sum(dim=-1)
        rank = (chunk > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
        log_probabilities.append(log_probability)
        entropies.append(entropy)
        log_ranks.append(torch.log(rank.float()))
    valid = mask.bool()
    log_probability = torch.cat(log_probabilities, dim=1).masked_fill(~valid, 0.0)
    entropy = torch.cat(entropies, dim=1).masked_fill(~valid, 0.0)
    log_rank = torch.cat(log_ranks, dim=1).masked_fill(~valid, 0.0)
    if not all(torch.isfinite(value).all() for value in (log_probability, entropy, log_rank)):
        raise FloatingPointError("non-finite observer statistic")
    return TokenStatistics(
        log_probability=log_probability,
        surprisal=-log_probability,
        entropy=entropy,
        log_rank=log_rank,
    )


def summarize_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return mean/std/q10/q25/q50/q75/q90/min/max for each row."""

    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("values and mask must share [batch,tokens] shape")
    summaries = []
    quantiles = torch.tensor(
        [0.10, 0.25, 0.50, 0.75, 0.90], device=values.device, dtype=torch.float32
    )
    for row, row_mask in zip(values.float(), mask.bool()):
        selected = row[row_mask]
        if selected.numel() == 0 or not torch.isfinite(selected).all():
            raise ValueError("masked statistic row must be finite and non-empty")
        summaries.append(
            torch.cat(
                (
                    selected.mean().view(1),
                    selected.std(correction=0).view(1),
                    torch.quantile(selected, quantiles),
                    selected.min().view(1),
                    selected.max().view(1),
                )
            )
        )
    return torch.stack(summaries)


def alignment_statistics(
    base_log_probability: torch.Tensor,
    aligned_log_probability: torch.Tensor,
    mask: torch.Tensor,
) -> AlignmentStatistics:
    if (
        base_log_probability.shape != aligned_log_probability.shape
        or mask.shape != base_log_probability.shape
    ):
        raise ValueError("base/aligned/mask shapes must match")
    valid = mask.bool()
    counts = valid.sum(dim=1).clamp_min(1).float()
    aligned_minus_base = (aligned_log_probability - base_log_probability).masked_fill(
        ~valid, 0.0
    )
    imprint = (
        aligned_log_probability * (base_log_probability - aligned_log_probability)
    ).masked_fill(~valid, 0.0)
    rai = aligned_minus_base.sum(dim=1) / counts
    observed_imprint = imprint.sum(dim=1) / counts
    return AlignmentStatistics(
        aligned_minus_base=aligned_minus_base,
        information_weighted_imprint=imprint,
        rai=rai,
        observed_imprint=observed_imprint,
    )


def exact_lapd_statistics_from_logits(
    base_logits: torch.Tensor,
    aligned_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 16,
    min_std: float = 1e-8,
) -> ExactLAPDStatistics:
    """Analytic conditional moments for the released LAPD perturbation null.

    Each perturbation independently samples one vocabulary item per position
    from the base distribution.  The mean of the document statistic is the
    mean of the token expectations; independence makes its variance the sum
    of token variances divided by the squared valid-token count.
    """

    if (
        base_logits.ndim != 3
        or base_logits.shape != aligned_logits.shape
        or targets.shape != base_logits.shape[:2]
        or mask.shape != targets.shape
    ):
        raise ValueError("base/aligned logits, targets, and mask do not align")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    valid = mask.bool()
    token_count = valid.sum(dim=1)
    if torch.any(token_count < 1):
        raise ValueError("every document must contain a valid transition")

    observed_sum = torch.zeros(base_logits.shape[0], device=base_logits.device, dtype=torch.float64)
    mean_sum = torch.zeros_like(observed_sum)
    variance_sum = torch.zeros_like(observed_sum)
    for start in range(0, base_logits.shape[1], chunk_size):
        end = min(start + chunk_size, base_logits.shape[1])
        base_log_probability = torch.log_softmax(base_logits[:, start:end].float(), dim=-1)
        aligned_log_probability = torch.log_softmax(
            aligned_logits[:, start:end].float(), dim=-1
        )
        reference_probability = base_log_probability.exp()
        imprint = aligned_log_probability * (
            base_log_probability - aligned_log_probability
        )
        expectation = (reference_probability * imprint).sum(dim=-1)
        second_moment = (reference_probability * imprint.square()).sum(dim=-1)
        variance = torch.clamp(second_moment - expectation.square(), min=0.0)

        chunk_targets = targets[:, start:end].long().unsqueeze(-1)
        observed_base = base_log_probability.gather(-1, chunk_targets).squeeze(-1)
        observed_aligned = aligned_log_probability.gather(-1, chunk_targets).squeeze(-1)
        observed = observed_aligned * (observed_base - observed_aligned)
        chunk_mask = valid[:, start:end]
        observed_sum += observed.masked_fill(~chunk_mask, 0.0).sum(dim=1).double()
        mean_sum += expectation.masked_fill(~chunk_mask, 0.0).sum(dim=1).double()
        variance_sum += variance.masked_fill(~chunk_mask, 0.0).sum(dim=1).double()

    count = token_count.double()
    observed_statistic = observed_sum / count
    null_mean = mean_sum / count
    null_std = torch.sqrt(variance_sum) / count
    if not torch.isfinite(null_std).all() or torch.any(null_std < min_std):
        raise ValueError("LAPD conditional null standard deviation is too small")
    score = (observed_statistic - null_mean) / null_std
    if not all(
        torch.isfinite(value).all()
        for value in (score, observed_statistic, null_mean, null_std)
    ):
        raise FloatingPointError("non-finite exact LAPD statistic")
    return ExactLAPDStatistics(
        score=score,
        observed_statistic=observed_statistic,
        null_mean=null_mean,
        null_std=null_std,
        token_count=token_count,
    )


def token_lapd_null_moments_from_logits(
    base_logits: torch.Tensor,
    aligned_logits: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 16,
) -> TokenNullMoments:
    """Return token-local conditional moments of the LAPD imprint null.

    The returned mean and variance are the analytic moments of
    ``log p_aligned(y) * (log p_base(y) - log p_aligned(y))`` when ``y`` is
    sampled from the base distribution at each position. Invalid transitions
    are zero-filled and must remain excluded by ``mask`` downstream.
    """

    if (
        base_logits.ndim != 3
        or base_logits.shape != aligned_logits.shape
        or mask.shape != base_logits.shape[:2]
    ):
        raise ValueError("base/aligned logits and mask do not align")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    valid = mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every document must contain a valid transition")
    means = []
    variances = []
    for start in range(0, base_logits.shape[1], chunk_size):
        end = min(start + chunk_size, base_logits.shape[1])
        base_log_probability = torch.log_softmax(
            base_logits[:, start:end].float(), dim=-1
        )
        aligned_log_probability = torch.log_softmax(
            aligned_logits[:, start:end].float(), dim=-1
        )
        base_probability = base_log_probability.exp()
        imprint = aligned_log_probability * (
            base_log_probability - aligned_log_probability
        )
        mean = (base_probability * imprint).sum(dim=-1)
        second_moment = (base_probability * imprint.square()).sum(dim=-1)
        variance = torch.clamp(second_moment - mean.square(), min=0.0)
        chunk_mask = valid[:, start:end]
        means.append(mean.masked_fill(~chunk_mask, 0.0))
        variances.append(variance.masked_fill(~chunk_mask, 0.0))
    mean = torch.cat(means, dim=1)
    variance = torch.cat(variances, dim=1)
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise FloatingPointError("non-finite token LAPD null moment")
    return TokenNullMoments(mean=mean, variance=variance)


def fast_discrepancy_analytic(
    reference_logits: torch.Tensor,
    scoring_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 16,
    min_std: float = 1e-8,
) -> torch.Tensor:
    """Canonical AI-high analytic Fast-DetectGPT sampling discrepancy."""

    if (
        reference_logits.ndim != 3
        or reference_logits.shape != scoring_logits.shape
        or targets.shape != reference_logits.shape[:2]
        or mask.shape != targets.shape
    ):
        raise ValueError("reference/scoring logits, targets, and mask do not align")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    valid = mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("every document must contain a valid transition")
    observed_sum = torch.zeros(reference_logits.shape[0], device=reference_logits.device, dtype=torch.float64)
    mean_sum = torch.zeros_like(observed_sum)
    variance_sum = torch.zeros_like(observed_sum)
    for start in range(0, reference_logits.shape[1], chunk_size):
        end = min(start + chunk_size, reference_logits.shape[1])
        reference_probability = torch.softmax(reference_logits[:, start:end].float(), dim=-1)
        scoring_log_probability = torch.log_softmax(scoring_logits[:, start:end].float(), dim=-1)
        mean = (reference_probability * scoring_log_probability).sum(dim=-1)
        variance = torch.clamp(
            (reference_probability * scoring_log_probability.square()).sum(dim=-1)
            - mean.square(),
            min=0.0,
        )
        observed = scoring_log_probability.gather(
            -1, targets[:, start:end].long().unsqueeze(-1)
        ).squeeze(-1)
        chunk_mask = valid[:, start:end]
        observed_sum += observed.masked_fill(~chunk_mask, 0.0).sum(dim=1).double()
        mean_sum += mean.masked_fill(~chunk_mask, 0.0).sum(dim=1).double()
        variance_sum += variance.masked_fill(~chunk_mask, 0.0).sum(dim=1).double()
    standard_deviation = torch.sqrt(variance_sum)
    if torch.any(standard_deviation < min_std) or not torch.isfinite(standard_deviation).all():
        raise ValueError("Fast-DetectGPT conditional null standard deviation is too small")
    score = (observed_sum - mean_sum) / standard_deviation
    if not torch.isfinite(score).all():
        raise FloatingPointError("non-finite Fast-DetectGPT score")
    return score


def binoculars_ai_score(
    observer_logits: torch.Tensor,
    performer_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Faithful upstream Binoculars ratio with sign canonicalized to AI-high."""

    if (
        observer_logits.ndim != 3
        or observer_logits.shape != performer_logits.shape
        or input_ids.shape != observer_logits.shape[:2]
        or attention_mask.shape != input_ids.shape
    ):
        raise ValueError("Binoculars logits and encoding shapes do not align")
    valid_tokens = attention_mask.bool()
    transition_mask = valid_tokens[:, :-1] & valid_tokens[:, 1:]
    if not transition_mask.any(dim=1).all():
        raise ValueError("every document must contain a valid transition")
    performer_nll_sum = torch.zeros(observer_logits.shape[0], device=observer_logits.device, dtype=torch.float64)
    cross_entropy_sum = torch.zeros_like(performer_nll_sum)
    for start in range(0, observer_logits.shape[1], chunk_size):
        end = min(start + chunk_size, observer_logits.shape[1])
        observer_probability = torch.softmax(observer_logits[:, start:end].float(), dim=-1)
        performer_log_probability = torch.log_softmax(
            performer_logits[:, start:end].float(), dim=-1
        )
        cross_entropy = -(observer_probability * performer_log_probability).sum(dim=-1)
        cross_entropy_sum += cross_entropy.masked_fill(
            ~valid_tokens[:, start:end], 0.0
        ).sum(dim=1).double()
        transition_end = min(end, observer_logits.shape[1] - 1)
        if transition_end > start:
            log_probability = performer_log_probability[
                :, : transition_end - start
            ].gather(
                -1, input_ids[:, start + 1 : transition_end + 1].long().unsqueeze(-1)
            ).squeeze(-1)
            performer_nll_sum += (-log_probability).masked_fill(
                ~transition_mask[:, start:transition_end], 0.0
            ).sum(dim=1).double()
    performer_mean_nll = performer_nll_sum / transition_mask.sum(dim=1).double()
    cross_entropy_mean = cross_entropy_sum / valid_tokens.sum(dim=1).double()
    if torch.any(cross_entropy_mean <= 0):
        raise ValueError("Binoculars cross-model entropy must be positive")
    score = -(performer_mean_nll / cross_entropy_mean)
    if not torch.isfinite(score).all():
        raise FloatingPointError("non-finite Binoculars score")
    return score


def ideal_codelength_features(
    surprisal: torch.Tensor,
    mask: torch.Tensor,
    word_counts: torch.Tensor,
    utf8_byte_counts: torch.Tensor,
) -> torch.Tensor:
    if surprisal.ndim != 2 or mask.shape != surprisal.shape:
        raise ValueError("surprisal and mask must share [batch,tokens] shape")
    if word_counts.shape != surprisal.shape[:1] or utf8_byte_counts.shape != word_counts.shape:
        raise ValueError("document counts must match batch")
    token_counts = mask.bool().sum(dim=1)
    if torch.any(token_counts < 1) or torch.any(word_counts < 1) or torch.any(utf8_byte_counts < 1):
        raise ValueError("codelength normalizers must be positive")
    bits = surprisal.masked_fill(~mask.bool(), 0.0).sum(dim=1).float() / math.log(2.0)
    features = torch.stack(
        (
            bits / token_counts.float(),
            bits / word_counts.float(),
            bits / utf8_byte_counts.float(),
        ),
        dim=1,
    )
    if not torch.isfinite(features).all():
        raise FloatingPointError("non-finite codelength feature")
    return features
