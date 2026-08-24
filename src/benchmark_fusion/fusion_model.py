"""Full gated multi-view fusion model over frozen, auditable feature shards."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FusionOutput:
    logit: torch.Tensor
    branch_logits: torch.Tensor
    branch_uncertainty: torch.Tensor
    gate_weights: torch.Tensor
    fused_embedding: torch.Tensor


class ScalarViewProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class SemanticSpanEncoder(nn.Module):
    """Trainable attention pooling over frozen ModernBERT common-span states."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Parameter(torch.empty(hidden_dim))
        self.evidence_query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.query, std=hidden_dim**-0.5)
        nn.init.normal_(self.evidence_query, std=hidden_dim**-0.5)
        self.output = nn.Sequential(nn.GELU(), nn.Dropout(0.1))

    def forward(
        self, spans: torch.Tensor, mask: torch.Tensor, evidence: torch.Tensor
    ) -> torch.Tensor:
        if not mask.bool().any(dim=1).all():
            raise ValueError("every semantic document must contain a valid span")
        projected = self.projection(self.normalization(spans))
        if evidence.shape != projected.shape:
            raise ValueError("semantic evidence guidance must match projected spans")
        scores = torch.einsum("bsd,d->bs", projected, self.query)
        scores = scores + torch.einsum("bsd,d->bs", evidence, self.evidence_query)
        scores = scores / projected.shape[-1] ** 0.5
        scores = scores.masked_fill(~mask.bool(), -torch.inf)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bs,bsd->bd", weights, projected)
        return self.output(pooled)


class TokenEvidenceEncoder(nn.Module):
    """Multi-kernel temporal encoder over common-span evidence channels."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        convolution_channels: int = 64,
        kernels: tuple[int, ...] = (3, 5, 7),
    ) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_channels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        convolution_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    ),
                    nn.GELU(),
                    nn.Conv1d(
                        convolution_channels,
                        convolution_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GELU(),
                )
                for kernel in kernels
            ]
        )
        pooled_dim = len(kernels) * convolution_channels * 2
        self.output = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, evidence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask.bool().any(dim=1).all():
            raise ValueError("every token-evidence document must contain a valid span")
        normalized = self.normalization(evidence)
        inputs = normalized.transpose(1, 2)
        expanded_mask = mask[:, None, :].bool()
        denominator = expanded_mask.sum(dim=-1).clamp_min(1).to(inputs.dtype)
        pooled = []
        for branch in self.branches:
            encoded = branch(inputs).masked_fill(~expanded_mask, 0.0)
            pooled.append(encoded.sum(dim=-1) / denominator)
            maximum = encoded.masked_fill(~expanded_mask, -torch.inf).amax(dim=-1)
            pooled.append(torch.nan_to_num(maximum, neginf=0.0))
        return self.output(torch.cat(pooled, dim=-1))


class GatedEvidenceFusion(nn.Module):
    """Fuse semantic, temporal, and four document-scalar views."""

    branch_names = (
        "semantic",
        "token_evidence",
        "probability",
        "alignment",
        "compression",
        "detector_score",
    )

    def __init__(
        self,
        semantic_dim: int,
        token_channels: int,
        scalar_dims: dict[str, int],
        hidden_dim: int = 256,
        convolution_channels: int = 64,
        branch_dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()
        expected = {"probability", "alignment", "compression", "detector_score"}
        if set(scalar_dims) != expected or any(scalar_dims[name] < 1 for name in expected):
            raise ValueError(f"scalar_dims must contain positive dimensions for {expected}")
        if not 0.0 <= branch_dropout_probability < 1.0:
            raise ValueError("branch_dropout_probability must lie in [0,1)")
        self.semantic = SemanticSpanEncoder(semantic_dim, hidden_dim)
        self.evidence_guidance = nn.Sequential(
            nn.LayerNorm(token_channels),
            nn.Linear(token_channels, hidden_dim),
            nn.Tanh(),
        )
        self.token_evidence = TokenEvidenceEncoder(
            token_channels, hidden_dim, convolution_channels
        )
        self.scalar_views = nn.ModuleDict(
            {
                name: ScalarViewProjection(scalar_dims[name], hidden_dim)
                for name in sorted(expected)
            }
        )
        branch_count = len(self.branch_names)
        self.branch_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(branch_count)]
        )
        self.branch_log_variance = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(branch_count)]
        )
        self.branch_gate = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(branch_count)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.branch_dropout_probability = branch_dropout_probability

    def _semantic_evidence(
        self, token_evidence: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_evidence.shape[1] != 64:
            raise ValueError("frozen evidence guidance expects 64 token spans")
        batch, _, channels = token_evidence.shape
        evidence = token_evidence.reshape(batch, 16, 4, channels)
        mask = token_mask.reshape(batch, 16, 4).bool()
        denominator = mask.sum(dim=2).clamp_min(1).unsqueeze(-1).to(evidence.dtype)
        pooled = (evidence * mask.unsqueeze(-1)).sum(dim=2) / denominator
        return self.evidence_guidance(pooled), mask.any(dim=2)

    def forward(
        self,
        semantic_spans: torch.Tensor,
        semantic_mask: torch.Tensor,
        token_evidence: torch.Tensor,
        token_mask: torch.Tensor,
        probability: torch.Tensor,
        alignment: torch.Tensor,
        compression: torch.Tensor,
        detector_score: torch.Tensor,
    ) -> FusionOutput:
        tensors = (
            semantic_spans,
            token_evidence,
            probability,
            alignment,
            compression,
            detector_score,
        )
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise FloatingPointError("non-finite model input")
        semantic_evidence, semantic_evidence_mask = self._semantic_evidence(
            token_evidence, token_mask
        )
        joint_semantic_mask = semantic_mask.bool() & semantic_evidence_mask
        embeddings = [
            self.semantic(semantic_spans, joint_semantic_mask, semantic_evidence),
            self.token_evidence(token_evidence, token_mask),
            self.scalar_views["probability"](probability),
            self.scalar_views["alignment"](alignment),
            self.scalar_views["compression"](compression),
            self.scalar_views["detector_score"](detector_score),
        ]
        stacked = torch.stack(embeddings, dim=1)
        branch_logits = torch.cat(
            [head(value) for head, value in zip(self.branch_classifiers, embeddings)],
            dim=1,
        )
        log_variance = torch.cat(
            [head(value) for head, value in zip(self.branch_log_variance, embeddings)],
            dim=1,
        ).clamp(-8.0, 8.0)
        branch_uncertainty = torch.exp(log_variance)
        gate_scores = torch.cat(
            [head(value) for head, value in zip(self.branch_gate, embeddings)], dim=1
        ) - 0.5 * log_variance
        if self.training and self.branch_dropout_probability > 0:
            dropped = torch.rand_like(gate_scores) < self.branch_dropout_probability
            all_dropped = dropped.all(dim=1)
            dropped[all_dropped, 0] = False
            gate_scores = gate_scores.masked_fill(dropped, -torch.inf)
        gate_weights = torch.softmax(gate_scores, dim=1)
        fused = torch.einsum("bv,bvd->bd", gate_weights, stacked)
        logit = self.classifier(fused).squeeze(-1)
        return FusionOutput(
            logit=logit,
            branch_logits=branch_logits,
            branch_uncertainty=branch_uncertainty,
            gate_weights=gate_weights,
            fused_embedding=fused,
        )
