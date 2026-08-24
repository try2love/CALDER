#!/usr/bin/env python3
"""Evaluate one frozen CALDER checkpoint on a complete feature manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PACKAGE_ROOT))

from benchmark_fusion.calder_data import CalderDataset
from benchmark_fusion.calder_model import CalderCore
from benchmark_fusion.identity import sha256_file
from benchmark_fusion.metrics import ranking_metrics


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def model_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "semantic_spans": batch["semantic_spans"].to(device).float(),
        "semantic_mask": batch["semantic_mask"].to(device),
        "token_probability": batch["token_probability"].to(device).float(),
        "alignment_evidence": batch["alignment_evidence"].to(device).float(),
        "token_mask": batch["token_mask"].to(device),
        "document_probability": batch["document_probability"].to(device).float(),
        "document_alignment": batch["document_alignment"].to(device).float(),
        "compression": batch["compression"].to(device).float(),
    }


def require_exact_dataset_identity(
    identities: list[dict[str, object]], test_dataset: str
) -> None:
    """Require the frozen feature identities to retain the registered test name."""

    observed = {str(identity["dataset"]) for identity in identities}
    if observed != {test_dataset}:
        raise ValueError(
            "CALDER frozen-test dataset identity differs: "
            f"expected={test_dataset!r}, observed={sorted(observed)!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--test-feature-manifest", required=True, type=Path)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("method") != "calder_core":
        raise ValueError("CALDER training summary is not complete")
    for key, hash_key in (
        ("run_spec", "run_spec_sha256"),
        ("best_model", "best_model_sha256"),
        ("train_scalar_moments", "train_scalar_moments_sha256"),
    ):
        if sha256_file(Path(summary[key])) != summary[hash_key]:
            raise ValueError(f"CALDER evaluation input hash differs: {key}")
    dataset = CalderDataset(args.test_feature_manifest, validate=True)
    require_exact_dataset_identity(dataset.identities, args.test_dataset)
    archive = np.load(summary["train_scalar_moments"], allow_pickle=False)
    moments = {
        name: (
            torch.from_numpy(archive[f"{name}_mean"]),
            torch.from_numpy(archive[f"{name}_scale"]),
        )
        for name in ("document_probability", "document_alignment", "compression")
    }
    configuration = summary["configuration"]
    enabled_branches = tuple(summary.get("enabled_branches", CalderCore.branch_names))
    alignment_feature_mode = str(summary.get("alignment_feature_mode", "full"))
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("formal CALDER frozen-test evaluation requires CUDA")
    model = CalderCore(
        semantic_dim=int(dataset.shards[0]["semantic_spans"].shape[2]),
        hidden_dim=int(configuration["hidden_dimension"]),
        convolution_channels=64,
        branch_dropout_probability=float(configuration["branch_dropout"]),
        scalar_normalization=str(configuration["scalar_normalization"]),
        fusion=str(configuration["fusion"]),
        scalar_moments=moments,
        classifier_dropout_probability=0.2,
        gate_temperature=1.0,
        enabled_branches=enabled_branches,
        alignment_feature_mode=alignment_feature_mode,
    ).to(device)
    checkpoint = torch.load(summary["best_model"], map_location=device, weights_only=False)
    checkpoint_branches = tuple(checkpoint.get("enabled_branches", CalderCore.branch_names))
    if checkpoint_branches != enabled_branches:
        raise ValueError("CALDER checkpoint branch identity differs from training summary")
    if str(checkpoint.get("alignment_feature_mode", "full")) != alignment_feature_mode:
        raise ValueError("CALDER checkpoint alignment identity differs from training summary")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    losses: list[float] = []
    sample_ids: list[str] = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    autocast = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    with torch.inference_mode():
        for batch in loader:
            with autocast:
                logits = model(**model_inputs(batch, device)).logit
            target = batch["labels"].to(device).float()
            labels.extend(target.cpu().numpy().tolist())
            scores.extend(logits.float().cpu().numpy().tolist())
            losses.extend(
                F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
                .cpu()
                .numpy()
                .tolist()
            )
            sample_ids.extend(
                str(dataset.identities[int(index)]["sample_id"])
                for index in batch["index"].numpy().tolist()
            )
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    predictions = args.output.with_suffix(".npz")
    atomic_npz(
        predictions,
        labels=labels_array,
        scores=scores_array,
        sample_ids=np.asarray(sample_ids, dtype="U96"),
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "method": "calder_core",
        "ablation_id": summary.get("ablation_id", "full"),
        "enabled_branches": list(enabled_branches),
        "alignment_feature_mode": alignment_feature_mode,
        "cell_id": args.cell_id,
        "test_dataset": args.test_dataset,
        "records": len(dataset),
        "training_summary": str(args.training_summary.resolve()),
        "training_summary_sha256": sha256_file(args.training_summary),
        "test_feature_manifest": str(args.test_feature_manifest.resolve()),
        "test_feature_manifest_sha256": sha256_file(args.test_feature_manifest),
        "predictions": str(predictions.resolve()),
        "predictions_sha256": sha256_file(predictions),
        "metrics": asdict(ranking_metrics(labels_array, scores_array))
        | {"binary_cross_entropy": float(np.mean(losses)), "records": len(dataset)},
    }
    if args.output.exists() and json.loads(args.output.read_text(encoding="utf-8")) != payload:
        raise ValueError("existing CALDER evaluation identity differs")
    atomic_json(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
