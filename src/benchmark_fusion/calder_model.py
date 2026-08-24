"""CALDER Core five-view fusion model over frozen observer features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .fusion_model import SemanticSpanEncoder, TokenEvidenceEncoder


@dataclass
class CalderOutput:
    logit: torch.Tensor
    branch_embeddings: torch.Tensor
    gate_weights: torch.Tensor


class FixedZScore(nn.Module):
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        if mean.ndim != 1 or scale.shape != mean.shape or torch.any(scale <= 0):
            raise ValueError("z-score mean/scale must be positive aligned vectors")
        self.register_buffer("mean", mean.float())
        self.register_buffer("scale", scale.float())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale


class ScalarEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        normalization: str,
        mean: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if normalization == "train_zscore":
            if mean is None or scale is None:
                raise ValueError("train_zscore requires frozen training moments")
            normalizer: nn.Module = FixedZScore(mean, scale)
        elif normalization == "sample_layernorm":
            normalizer = nn.LayerNorm(input_dim)
        else:
            raise ValueError(f"unknown scalar normalization: {normalization}")
        self.network = nn.Sequential(
            normalizer,
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class AlignmentEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        convolution_channels: int,
        normalization: str,
        document_mean: torch.Tensor | None,
        document_scale: torch.Tensor | None,
        local_channels: int = 8,
        document_dim: int = 56,
    ) -> None:
        super().__init__()
        self.local = TokenEvidenceEncoder(
            local_channels, hidden_dim, convolution_channels, kernels=(3, 5, 7)
        )
        self.document = ScalarEncoder(
            document_dim,
            hidden_dim,
            normalization,
            document_mean,
            document_scale,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        local: torch.Tensor,
        mask: torch.Tensor,
        document: torch.Tensor,
    ) -> torch.Tensor:
        return self.output(torch.cat((self.local(local, mask), self.document(document)), dim=1))


class CalderCore(nn.Module):
    """Five CALDER branches without training-free detector score inputs."""

    branch_names = (
        "semantic",
        "token_probability",
        "alignment_imprint",
        "document_probability",
        "compression",
    )
    alignment_feature_indices = {
        "full": (
            tuple(range(8)),
            tuple(range(56)),
        ),
        "no_null_moments": (
            (0, 1, 4, 5),
            tuple((*range(0, 14), *range(28, 42))),
        ),
        "gap_only": (
            (0, 4),
            tuple((*range(0, 7), *range(28, 35))),
        ),
    }

    def __init__(
        self,
        *,
        semantic_dim: int,
        hidden_dim: int,
        convolution_channels: int = 64,
        branch_dropout_probability: float = 0.1,
        scalar_normalization: str = "train_zscore",
        fusion: str = "adaptive_gate",
        scalar_moments: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        classifier_dropout_probability: float = 0.2,
        gate_temperature: float = 1.0,
        enabled_branches: tuple[str, ...] | None = None,
        alignment_feature_mode: str = "full",
    ) -> None:
        super().__init__()
        if not 0.0 <= branch_dropout_probability < 1.0:
            raise ValueError("branch dropout must lie in [0,1)")
        if gate_temperature <= 0:
            raise ValueError("gate temperature must be positive")
        enabled = self.branch_names if enabled_branches is None else tuple(enabled_branches)
        if not enabled or len(set(enabled)) != len(enabled):
            raise ValueError("CALDER enabled branches must be a non-empty unique sequence")
        if any(name not in self.branch_names for name in enabled):
            raise ValueError("CALDER enabled branches contain an unknown branch")
        canonical_enabled = tuple(name for name in self.branch_names if name in enabled)
        if enabled != canonical_enabled:
            raise ValueError("CALDER enabled branches must retain canonical order")
        self.enabled_branches = enabled
        if alignment_feature_mode not in self.alignment_feature_indices:
            raise ValueError(f"unknown CALDER alignment feature mode: {alignment_feature_mode}")
        self.alignment_feature_mode = alignment_feature_mode
        self.alignment_local_indices, self.alignment_document_indices = (
            self.alignment_feature_indices[alignment_feature_mode]
        )
        moments = scalar_moments or {}

        def moment(name: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            return moments.get(name, (None, None))

        combined_channels = 14
        self.semantic = (
            SemanticSpanEncoder(semantic_dim, hidden_dim)
            if "semantic" in enabled
            else None
        )
        self.semantic_guidance = (
            nn.Sequential(
                nn.LayerNorm(combined_channels),
                nn.Linear(combined_channels, hidden_dim),
                nn.Tanh(),
            )
            if "semantic" in enabled
            else None
        )
        self.token_probability = (
            TokenEvidenceEncoder(6, hidden_dim, convolution_channels, kernels=(3, 5, 7))
            if "token_probability" in enabled
            else None
        )
        alignment_mean, alignment_scale = moment("document_alignment")
        if alignment_mean is not None:
            alignment_mean = alignment_mean[list(self.alignment_document_indices)]
        if alignment_scale is not None:
            alignment_scale = alignment_scale[list(self.alignment_document_indices)]
        self.alignment_imprint = (
            AlignmentEncoder(
                hidden_dim,
                convolution_channels,
                scalar_normalization,
                alignment_mean,
                alignment_scale,
                local_channels=len(self.alignment_local_indices),
                document_dim=len(self.alignment_document_indices),
            )
            if "alignment_imprint" in enabled
            else None
        )
        probability_mean, probability_scale = moment("document_probability")
        compression_mean, compression_scale = moment("compression")
        self.document_probability = (
            ScalarEncoder(
                108,
                hidden_dim,
                scalar_normalization,
                probability_mean,
                probability_scale,
            )
            if "document_probability" in enabled
            else None
        )
        self.compression = (
            ScalarEncoder(
                51,
                hidden_dim,
                scalar_normalization,
                compression_mean,
                compression_scale,
            )
            if "compression" in enabled
            else None
        )
        self.branch_dropout_probability = branch_dropout_probability
        self.fusion = fusion
        self.gate_temperature = gate_temperature
        if fusion == "adaptive_gate":
            self.branch_gate = nn.ModuleList(
                [nn.Linear(hidden_dim, 1) for _ in self.enabled_branches]
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout_probability),
                nn.Linear(hidden_dim, 1),
            )
            self.concat_classifier = None
        elif fusion == "concat_mlp":
            branch_count = len(self.enabled_branches)
            bottleneck = max(1, round((hidden_dim + branch_count) / branch_count))
            self.branch_gate = None
            self.classifier = None
            self.concat_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * branch_count),
                nn.Linear(hidden_dim * branch_count, bottleneck),
                nn.GELU(),
                nn.Dropout(classifier_dropout_probability),
                nn.Linear(bottleneck, 1),
            )
        else:
            raise ValueError(f"unknown CALDER fusion: {fusion}")

    @staticmethod
    def _to_semantic_spans(
        evidence: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if evidence.shape[1:] != (64, 14) or mask.shape[1:] != (64,):
            raise ValueError("CALDER semantic guidance expects 64 spans and 14 channels")
        batch = evidence.shape[0]
        grouped = evidence.reshape(batch, 16, 4, 14)
        grouped_mask = mask.bool().reshape(batch, 16, 4)
        denominator = grouped_mask.sum(dim=2).clamp_min(1).unsqueeze(-1).to(evidence.dtype)
        pooled = (grouped * grouped_mask.unsqueeze(-1)).sum(dim=2) / denominator
        return pooled, grouped_mask.any(dim=2)

    def _drop_branches(self, stacked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keep = torch.ones(stacked.shape[:2], dtype=torch.bool, device=stacked.device)
        if self.training and self.branch_dropout_probability > 0:
            keep = torch.rand(stacked.shape[:2], device=stacked.device) >= self.branch_dropout_probability
            empty = ~keep.any(dim=1)
            keep[empty, 0] = True
            stacked = stacked * keep.unsqueeze(-1)
        return stacked, keep

    def forward(
        self,
        *,
        semantic_spans: torch.Tensor,
        semantic_mask: torch.Tensor,
        token_probability: torch.Tensor,
        alignment_evidence: torch.Tensor,
        token_mask: torch.Tensor,
        document_probability: torch.Tensor,
        document_alignment: torch.Tensor,
        compression: torch.Tensor,
    ) -> CalderOutput:
        floating = (
            semantic_spans,
            token_probability,
            alignment_evidence,
            document_probability,
            document_alignment,
            compression,
        )
        if not all(torch.isfinite(value).all() for value in floating):
            raise FloatingPointError("non-finite CALDER input")
        embeddings_by_name: dict[str, torch.Tensor] = {}
        if self.semantic is not None:
            semantic_token_probability = (
                token_probability
                if "token_probability" in self.enabled_branches
                else torch.zeros_like(token_probability)
            )
            semantic_alignment = (
                self._alignment_guidance(alignment_evidence)
                if "alignment_imprint" in self.enabled_branches
                else torch.zeros_like(alignment_evidence)
            )
            combined = torch.cat((semantic_token_probability, semantic_alignment), dim=2)
            guidance, guidance_mask = self._to_semantic_spans(combined, token_mask)
            joint_semantic_mask = semantic_mask.bool() & guidance_mask
            embeddings_by_name["semantic"] = self.semantic(
                semantic_spans,
                joint_semantic_mask,
                self.semantic_guidance(guidance),
            )
        if self.token_probability is not None:
            embeddings_by_name["token_probability"] = self.token_probability(
                token_probability, token_mask
            )
        if self.alignment_imprint is not None:
            embeddings_by_name["alignment_imprint"] = self.alignment_imprint(
                alignment_evidence[..., list(self.alignment_local_indices)],
                token_mask,
                document_alignment[..., list(self.alignment_document_indices)],
            )
        if self.document_probability is not None:
            embeddings_by_name["document_probability"] = self.document_probability(
                document_probability
            )
        if self.compression is not None:
            embeddings_by_name["compression"] = self.compression(compression)
        embeddings = [embeddings_by_name[name] for name in self.enabled_branches]
        stacked, keep = self._drop_branches(torch.stack(embeddings, dim=1))
        if self.fusion == "adaptive_gate":
            gate_scores = torch.cat(
                [head(value) for head, value in zip(self.branch_gate, embeddings)], dim=1
            ) / self.gate_temperature
            gate_scores = gate_scores.masked_fill(~keep, -torch.inf)
            gate_weights = torch.softmax(gate_scores, dim=1)
            fused = torch.einsum("bv,bvd->bd", gate_weights, stacked)
            logit = self.classifier(fused).squeeze(-1)
        else:
            gate_weights = keep.float() / keep.sum(dim=1, keepdim=True)
            logit = self.concat_classifier(stacked.flatten(1)).squeeze(-1)
        return CalderOutput(
            logit=logit,
            branch_embeddings=stacked,
            gate_weights=gate_weights,
        )

    def _alignment_guidance(self, evidence: torch.Tensor) -> torch.Tensor:
        if self.alignment_feature_mode == "full":
            return evidence
        selected = torch.zeros_like(evidence)
        indices = list(self.alignment_local_indices)
        selected[..., indices] = evidence[..., indices]
        return selected
