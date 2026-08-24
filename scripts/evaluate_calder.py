#!/usr/bin/env python3
"""Evaluate one frozen CALDER checkpoint on a named dev partition."""

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
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
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


def inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device).float()
        for name in (
            "semantic_spans", "token_probability", "alignment_evidence",
            "document_probability", "document_alignment", "compression",
        )
    } | {
        "semantic_mask": batch["semantic_mask"].to(device),
        "token_mask": batch["token_mask"].to(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--partition", choices=("tune_dev", "confirm_dev"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--engineering-cpu-only", action="store_true")
    args = parser.parse_args()
    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("method") != "calder_core":
        raise ValueError("CALDER training summary is not complete")
    for key, hash_key in (
        ("run_spec", "run_spec_sha256"),
        ("best_model", "best_model_sha256"),
        ("train_scalar_moments", "train_scalar_moments_sha256"),
        ("dev_feature_manifest", "dev_feature_manifest_sha256"),
    ):
        if sha256_file(Path(summary[key])) != summary[hash_key]:
            raise ValueError(f"CALDER evaluation input hash differs: {key}")
    dataset = CalderDataset(
        Path(summary["dev_feature_manifest"]), partition=args.partition, validate=True
    )
    if any(str(identity["dataset"]) != summary["source"] for identity in dataset.identities):
        raise ValueError("CALDER evaluation dataset is not source-specific")
    archive = np.load(summary["train_scalar_moments"], allow_pickle=False)
    moments = {
        name: (
            torch.from_numpy(archive[f"{name}_mean"]),
            torch.from_numpy(archive[f"{name}_scale"]),
        )
        for name in ("document_probability", "document_alignment", "compression")
    }
    configuration = summary["configuration"]
    device = torch.device("cpu" if args.engineering_cpu_only else "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("formal CALDER evaluation requires CUDA")
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
    ).to(device)
    checkpoint = torch.load(summary["best_model"], map_location=device, weights_only=False)
    if checkpoint["run_spec_sha256"] != summary["run_spec_sha256"]:
        raise ValueError("CALDER checkpoint run identity differs")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels = []
    scores = []
    losses = []
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )
    with torch.inference_mode():
        for batch in loader:
            with autocast:
                logits = model(**inputs(batch, device)).logit
            target = batch["labels"].to(device).float()
            labels.extend(target.cpu().numpy().tolist())
            scores.extend(logits.float().cpu().numpy().tolist())
            losses.extend(
                F.binary_cross_entropy_with_logits(
                    logits.float(), target, reduction="none"
                ).cpu().numpy().tolist()
            )
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    metrics = asdict(ranking_metrics(labels_array, scores_array)) | {
        "binary_cross_entropy": float(np.mean(losses)),
        "records": len(dataset),
    }
    predictions_path = args.output.with_suffix(".npz")
    atomic_npz(
        predictions_path,
        labels=labels_array,
        scores=scores_array,
        sample_ids=np.asarray(
            [str(identity["sample_id"]) for identity in dataset.identities], dtype="U96"
        ),
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "method": "calder_core",
        "configuration_id": summary["configuration_id"],
        "source": summary["source"],
        "partition": args.partition,
        "training_summary": str(args.training_summary.resolve()),
        "training_summary_sha256": sha256_file(args.training_summary),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics": metrics,
    }
    atomic_json(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
