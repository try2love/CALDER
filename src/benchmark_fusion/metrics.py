"""Frozen binary metrics, low-FPR tie handling, and paired bootstrap inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RankingMetrics:
    auroc: float
    auprc: float
    tpr_at_fpr_005_inclusive: float
    tpr_at_fpr_005_strict: float
    tpr_first_at_or_above_fpr_005: float
    raw_false_positives_at_inclusive_point: int
    negatives: int
    positives: int


def _validated(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be equal-length one-dimensional arrays")
    if labels.size == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be non-empty and binary")
    if np.unique(labels).size != 2:
        raise ValueError("both classes are required")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    return labels, scores


def roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return thresholds, FPR, TPR, FP and TP after complete tied-score groups."""

    labels, scores = _validated(labels, scores)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    group_ends = np.flatnonzero(
        np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    )
    tp = np.cumsum(ordered_labels, dtype=np.int64)[group_ends]
    fp = np.cumsum(1 - ordered_labels, dtype=np.int64)[group_ends]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    thresholds = np.r_[np.inf, ordered_scores[group_ends]]
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    return thresholds, fp / negatives, tp / positives, fp, tp


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> RankingMetrics:
    labels, scores = _validated(labels, scores)
    _, fpr, tpr, fp, tp = roc_points(labels, scores)
    auroc = float(np.trapz(tpr, fpr))

    positives = int(labels.sum())
    recall_increments = np.diff(tpr)
    precision = tp[1:] / np.maximum(1, tp[1:] + fp[1:])
    auprc = float(np.sum(recall_increments * precision))

    inclusive = fpr <= 0.005
    strict = fpr < 0.005
    inclusive_index = int(np.flatnonzero(inclusive)[-1])
    above = np.flatnonzero(fpr >= 0.005)
    above_index = int(above[0]) if above.size else len(fpr) - 1
    return RankingMetrics(
        auroc=auroc,
        auprc=auprc,
        tpr_at_fpr_005_inclusive=float(tpr[inclusive].max()),
        tpr_at_fpr_005_strict=float(tpr[strict].max()),
        tpr_first_at_or_above_fpr_005=float(tpr[above_index]),
        raw_false_positives_at_inclusive_point=int(fp[inclusive_index]),
        negatives=int(labels.size - positives),
        positives=positives,
    )


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select the largest score threshold among thresholds maximizing dev F1."""

    labels, scores = _validated(labels, scores)
    thresholds, _, _, fp, tp = roc_points(labels, scores)
    false_negatives = int(labels.sum()) - tp
    denominator = 2 * tp + fp + false_negatives
    f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )
    maximum = float(f1.max())
    tied = np.flatnonzero(np.isclose(f1, maximum, rtol=0.0, atol=1e-15))
    return float(np.max(thresholds[tied]))


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    labels, scores = _validated(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("classification threshold must be finite")
    predictions = scores >= threshold
    positives = labels == 1
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & ~positives))
    tn = int(np.sum(~predictions & ~positives))
    fn = int(np.sum(~predictions & positives))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / labels.size,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(np.finfo(float).eps, precision + recall),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def stratified_bootstrap_indices(
    labels: np.ndarray, *, resamples: int, seed: int
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a one-dimensional binary array")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    human = np.flatnonzero(labels == 0)
    ai = np.flatnonzero(labels == 1)
    if not human.size or not ai.size:
        raise ValueError("both classes are required")
    rng = np.random.default_rng(seed)
    human_draws = rng.choice(human, size=(resamples, human.size), replace=True)
    ai_draws = rng.choice(ai, size=(resamples, ai.size), replace=True)
    return np.concatenate((human_draws, ai_draws), axis=1)


def paired_bootstrap(
    labels: np.ndarray,
    method_scores: Mapping[str, np.ndarray],
    *,
    resamples: int = 10_000,
    seed: int = 20_260_815,
) -> dict[str, object]:
    """Compute paired label-stratified AUROC/AUPRC/low-FPR distributions."""

    labels = np.asarray(labels, dtype=np.int8)
    checked = {
        name: _validated(labels, np.asarray(scores))[1]
        for name, scores in method_scores.items()
    }
    if not checked:
        raise ValueError("at least one method is required")
    indices = stratified_bootstrap_indices(labels, resamples=resamples, seed=seed)
    metric_names = ("auroc", "auprc", "tpr_at_fpr_005_inclusive")
    samples = {
        name: {metric: np.empty(resamples, dtype=np.float64) for metric in metric_names}
        for name in checked
    }
    for index, draw in enumerate(indices):
        draw_labels = labels[draw]
        for name, scores in checked.items():
            result = ranking_metrics(draw_labels, scores[draw])
            for metric in metric_names:
                samples[name][metric][index] = getattr(result, metric)

    summaries: dict[str, object] = {}
    for name, values in samples.items():
        summaries[name] = {
            metric: {
                "estimate": getattr(ranking_metrics(labels, checked[name]), metric),
                "ci95": np.quantile(distribution, (0.025, 0.975)).tolist(),
            }
            for metric, distribution in values.items()
        }
    deltas: dict[str, object] = {}
    names = sorted(checked)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}_minus_{right}"
            deltas[key] = {}
            for metric in metric_names:
                distribution = samples[left][metric] - samples[right][metric]
                deltas[key][metric] = {
                    "estimate": getattr(ranking_metrics(labels, checked[left]), metric)
                    - getattr(ranking_metrics(labels, checked[right]), metric),
                    "ci95": np.quantile(distribution, (0.025, 0.975)).tolist(),
                }
    return {
        "schema_version": 1,
        "resamples": resamples,
        "seed": seed,
        "sampling": "paired_label_stratified_percentile",
        "methods": summaries,
        "paired_deltas": deltas,
    }


def multiseed_paired_bootstrap(
    labels: np.ndarray,
    method_seed_scores: Mapping[str, list[np.ndarray]],
    *,
    expected_seeds: int = 3,
    resamples: int = 10_000,
    seed: int = 20_260_815,
) -> dict[str, object]:
    """Paired bootstrap of each method's mean metric over all prescribed seeds."""

    labels = np.asarray(labels, dtype=np.int8)
    checked: dict[str, list[np.ndarray]] = {}
    for method, scores_by_seed in method_seed_scores.items():
        if len(scores_by_seed) != expected_seeds:
            raise ValueError(f"{method} has {len(scores_by_seed)} seeds, expected {expected_seeds}")
        checked[method] = [_validated(labels, np.asarray(scores))[1] for scores in scores_by_seed]
    if not checked:
        raise ValueError("at least one complete method is required")
    indices = stratified_bootstrap_indices(labels, resamples=resamples, seed=seed)
    metric_names = ("auroc", "auprc", "tpr_at_fpr_005_inclusive")
    point: dict[str, dict[str, float]] = {}
    distributions = {
        method: {metric: np.empty(resamples, dtype=np.float64) for metric in metric_names}
        for method in checked
    }
    for method, seed_scores in checked.items():
        seed_metrics = [ranking_metrics(labels, scores) for scores in seed_scores]
        point[method] = {
            metric: float(np.mean([getattr(result, metric) for result in seed_metrics]))
            for metric in metric_names
        }
    for draw_index, draw in enumerate(indices):
        draw_labels = labels[draw]
        for method, seed_scores in checked.items():
            results = [ranking_metrics(draw_labels, scores[draw]) for scores in seed_scores]
            for metric in metric_names:
                distributions[method][metric][draw_index] = np.mean(
                    [getattr(result, metric) for result in results]
                )
    summaries = {
        method: {
            metric: {
                "estimate": point[method][metric],
                "ci95": np.quantile(distribution, (0.025, 0.975)).tolist(),
            }
            for metric, distribution in values.items()
        }
        for method, values in distributions.items()
    }
    deltas: dict[str, object] = {}
    names = sorted(checked)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}_minus_{right}"
            deltas[key] = {
                metric: {
                    "estimate": point[left][metric] - point[right][metric],
                    "ci95": np.quantile(
                        distributions[left][metric] - distributions[right][metric],
                        (0.025, 0.975),
                    ).tolist(),
                }
                for metric in metric_names
            }
    return {
        "schema_version": 1,
        "resamples": resamples,
        "seed": seed,
        "expected_seeds": expected_seeds,
        "sampling": "paired_label_stratified_percentile_with_seed_mean_inside_draw",
        "methods": summaries,
        "paired_deltas": deltas,
    }
