from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_fusion.metrics import (
    paired_bootstrap,
    multiseed_paired_bootstrap,
    ranking_metrics,
    select_f1_threshold,
    stratified_bootstrap_indices,
    threshold_metrics,
)


class MetricProtocolTest(unittest.TestCase):
    def test_perfect_ranking_and_threshold_metrics(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        ranking = ranking_metrics(labels, scores)
        self.assertEqual(ranking.auroc, 1.0)
        self.assertEqual(ranking.auprc, 1.0)
        threshold = select_f1_threshold(labels, scores)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(
            threshold_metrics(labels, scores, threshold)["confusion_matrix"],
            {"tn": 2, "fp": 0, "fn": 0, "tp": 2},
        )

    def test_low_fpr_uses_complete_tied_score_groups(self) -> None:
        labels = np.asarray([0] * 1000 + [1] * 4)
        scores = np.asarray([0.9] * 5 + [0.1] * 995 + [0.9] * 2 + [0.8] * 2)
        result = ranking_metrics(labels, scores)
        self.assertEqual(result.raw_false_positives_at_inclusive_point, 5)
        self.assertEqual(result.tpr_at_fpr_005_inclusive, 1.0)
        self.assertEqual(result.tpr_at_fpr_005_strict, 0.0)
        self.assertEqual(result.tpr_first_at_or_above_fpr_005, 0.5)

    def test_bootstrap_is_stratified_paired_and_deterministic(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1])
        first = stratified_bootstrap_indices(labels, resamples=8, seed=7)
        second = stratified_bootstrap_indices(labels, resamples=8, seed=7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(labels[first].sum(axis=1), np.full(8, 2))
        result = paired_bootstrap(
            labels,
            {"good": np.asarray([0.1, 0.2, 0.3, 0.8, 0.9]), "flat": np.zeros(5)},
            resamples=16,
            seed=7,
        )
        self.assertEqual(result["resamples"], 16)
        self.assertIn("flat_minus_good", result["paired_deltas"])

    def test_nonfinite_scores_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ranking_metrics(np.asarray([0, 1]), np.asarray([0.0, np.nan]))

    def test_multiseed_bootstrap_requires_all_seeds_and_pairs_methods(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        good = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        flat = np.zeros(6)
        with self.assertRaisesRegex(ValueError, "expected 3"):
            multiseed_paired_bootstrap(labels, {"good": [good]}, resamples=4)
        result = multiseed_paired_bootstrap(
            labels,
            {"good": [good, good, good], "flat": [flat, flat, flat]},
            resamples=16,
            seed=7,
        )
        delta = result["paired_deltas"]["flat_minus_good"]["auroc"]
        self.assertEqual(delta["estimate"], -0.5)
        self.assertLessEqual(delta["ci95"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
