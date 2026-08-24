from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_fusion.calder_model import CalderCore


class CalderCoreTest(unittest.TestCase):
    def moments(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {
            "document_probability": (torch.zeros(108), torch.ones(108)),
            "document_alignment": (torch.zeros(56), torch.ones(56)),
            "compression": (torch.zeros(51), torch.ones(51)),
        }

    def model(self, **overrides: object) -> CalderCore:
        arguments = {
            "semantic_dim": 12,
            "hidden_dim": 16,
            "convolution_channels": 4,
            "scalar_moments": self.moments(),
        } | overrides
        return CalderCore(**arguments)

    def inputs(self) -> dict[str, torch.Tensor]:
        return {
            "semantic_spans": torch.randn(3, 16, 12),
            "semantic_mask": torch.tensor([
                [True] * 16,
                [True] * 12 + [False] * 4,
                [True] * 4 + [False] * 12,
            ]),
            "token_probability": torch.randn(3, 64, 6),
            "alignment_evidence": torch.randn(3, 64, 8),
            "token_mask": torch.tensor([
                [True] * 64,
                [True] * 48 + [False] * 16,
                [True] * 16 + [False] * 48,
            ]),
            "document_probability": torch.randn(3, 108),
            "document_alignment": torch.randn(3, 56),
            "compression": torch.randn(3, 51),
        }

    def test_adaptive_gate_has_five_normalized_branches(self) -> None:
        output = self.model()(**self.inputs())
        self.assertEqual(tuple(output.logit.shape), (3,))
        self.assertEqual(tuple(output.branch_embeddings.shape), (3, 5, 16))
        torch.testing.assert_close(output.gate_weights.sum(dim=1), torch.ones(3))

    def test_concat_mlp_has_same_public_output_without_detector_score(self) -> None:
        output = self.model(fusion="concat_mlp")(**self.inputs())
        self.assertEqual(tuple(output.logit.shape), (3,))
        self.assertEqual(tuple(output.gate_weights.shape), (3, 5))

    def test_sample_layernorm_requires_no_training_moments(self) -> None:
        model = CalderCore(
            semantic_dim=12,
            hidden_dim=16,
            convolution_channels=4,
            scalar_normalization="sample_layernorm",
        )
        self.assertEqual(tuple(model(**self.inputs()).logit.shape), (3,))

    def test_nonfinite_input_fails_closed(self) -> None:
        values = self.inputs()
        values["compression"][0, 0] = torch.nan
        with self.assertRaises(FloatingPointError):
            self.model()(**values)

    def test_branch_dropout_never_removes_every_branch(self) -> None:
        model = self.model(branch_dropout_probability=0.99).train()
        torch.manual_seed(7)
        output = model(**self.inputs())
        torch.testing.assert_close(output.gate_weights.sum(dim=1), torch.ones(3))
        self.assertTrue((output.gate_weights == 0).any())

    def test_explicit_full_branch_set_is_identical_to_default(self) -> None:
        torch.manual_seed(17)
        default = self.model()
        torch.manual_seed(17)
        explicit = self.model(enabled_branches=CalderCore.branch_names)
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)

    def test_explicit_full_alignment_is_identical_to_default(self) -> None:
        torch.manual_seed(19)
        default = self.model()
        torch.manual_seed(19)
        explicit = self.model(alignment_feature_mode="full")
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)

    def test_leave_one_out_has_four_branches(self) -> None:
        enabled = tuple(name for name in CalderCore.branch_names if name != "compression")
        output = self.model(enabled_branches=enabled)(**self.inputs())
        self.assertEqual(tuple(output.branch_embeddings.shape), (3, 4, 16))
        self.assertEqual(tuple(output.gate_weights.shape), (3, 4))

    def test_removed_token_probability_cannot_leak_through_semantic_guidance(self) -> None:
        enabled = tuple(
            name for name in CalderCore.branch_names if name != "token_probability"
        )
        model = self.model(enabled_branches=enabled).eval()
        first = self.inputs()
        second = {name: value.clone() for name, value in first.items()}
        second["token_probability"] += 1000
        torch.testing.assert_close(model(**first).logit, model(**second).logit)

    def test_removed_alignment_cannot_leak_through_semantic_guidance(self) -> None:
        enabled = tuple(
            name for name in CalderCore.branch_names if name != "alignment_imprint"
        )
        model = self.model(enabled_branches=enabled).eval()
        first = self.inputs()
        second = {name: value.clone() for name, value in first.items()}
        second["alignment_evidence"] += 1000
        second["document_alignment"] += 1000
        torch.testing.assert_close(model(**first).logit, model(**second).logit)

    def test_no_null_alignment_cannot_use_null_channels(self) -> None:
        model = self.model(alignment_feature_mode="no_null_moments").eval()
        first = self.inputs()
        second = {name: value.clone() for name, value in first.items()}
        second["alignment_evidence"][..., [2, 3, 6, 7]] += 1000
        second["document_alignment"][..., [*range(14, 28), *range(42, 56)]] += 1000
        torch.testing.assert_close(model(**first).logit, model(**second).logit)

    def test_gap_only_alignment_cannot_use_imprint_or_null_channels(self) -> None:
        model = self.model(alignment_feature_mode="gap_only").eval()
        first = self.inputs()
        second = {name: value.clone() for name, value in first.items()}
        second["alignment_evidence"][..., [1, 2, 3, 5, 6, 7]] += 1000
        second["document_alignment"][..., [
            *range(7, 28), *range(35, 56)
        ]] += 1000
        torch.testing.assert_close(model(**first).logit, model(**second).logit)

    def test_alignment_internal_ablations_reduce_parameters(self) -> None:
        full = sum(parameter.numel() for parameter in self.model().parameters())
        no_null = sum(
            parameter.numel()
            for parameter in self.model(alignment_feature_mode="no_null_moments").parameters()
        )
        gap_only = sum(
            parameter.numel()
            for parameter in self.model(alignment_feature_mode="gap_only").parameters()
        )
        self.assertGreater(full, no_null)
        self.assertGreater(no_null, gap_only)

    def test_invalid_branch_set_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.model(enabled_branches=("compression", "semantic"))
        with self.assertRaises(ValueError):
            self.model(enabled_branches=("semantic", "unknown"))
        with self.assertRaises(ValueError):
            self.model(alignment_feature_mode="unknown")


if __name__ == "__main__":
    unittest.main()
