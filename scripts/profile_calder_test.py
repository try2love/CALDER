#!/usr/bin/env python3
"""Re-run a completed CALDER cell to add measured inference-cost evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PACKAGE_ROOT))

from benchmark_fusion.calder_data import CalderDataset  # noqa: E402
from benchmark_fusion.calder_model import CalderCore  # noqa: E402
from benchmark_fusion.identity import sha256_file  # noqa: E402


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


def validate_existing(payload: dict[str, object], evaluation: dict[str, object], output: Path) -> None:
    required = {
        "status": "complete",
        "cell_id": evaluation["cell_id"],
        "records": evaluation["records"],
        "predictions_sha256": evaluation["predictions_sha256"],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError(f"existing CALDER runtime profile differs: {output}")
    for name in ("elapsed_seconds", "records_per_second", "peak_gpu_memory_bytes"):
        if payload.get(name) is None:
            raise ValueError(f"existing CALDER runtime profile lacks {name}: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation_summary.read_text(encoding="utf-8"))
    if evaluation.get("status") != "complete" or evaluation.get("method") != "calder_core":
        raise ValueError("CALDER evaluation summary is incomplete")
    for key, hash_key in (
        ("predictions", "predictions_sha256"),
        ("training_summary", "training_summary_sha256"),
        ("test_feature_manifest", "test_feature_manifest_sha256"),
    ):
        if sha256_file(Path(evaluation[key])) != evaluation[hash_key]:
            raise ValueError(f"CALDER profile input hash differs: {key}")
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        validate_existing(payload, evaluation, args.output)
        print(json.dumps(payload, sort_keys=True))
        return

    training = json.loads(Path(evaluation["training_summary"]).read_text(encoding="utf-8"))
    for key, hash_key in (
        ("best_model", "best_model_sha256"),
        ("train_scalar_moments", "train_scalar_moments_sha256"),
    ):
        if sha256_file(Path(training[key])) != training[hash_key]:
            raise ValueError(f"CALDER profile training input hash differs: {key}")
    dataset = CalderDataset(Path(evaluation["test_feature_manifest"]), validate=True)
    if len(dataset) != int(evaluation["records"]):
        raise ValueError("CALDER profile frozen-test coverage differs")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal CALDER runtime profiling requires CUDA")
    with np.load(training["train_scalar_moments"], allow_pickle=False) as archive:
        moments = {
            name: (
                torch.from_numpy(archive[f"{name}_mean"]),
                torch.from_numpy(archive[f"{name}_scale"]),
            )
            for name in ("document_probability", "document_alignment", "compression")
        }
    configuration = training["configuration"]
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
    checkpoint = torch.load(training["best_model"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    sample_ids: list[str] = []
    autocast = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            with autocast:
                logits = model(**model_inputs(batch, device)).logit
            labels.append(batch["labels"].numpy().astype(np.int8))
            scores.append(logits.float().cpu().numpy().astype(np.float64))
            sample_ids.extend(
                str(dataset.identities[int(index)]["sample_id"])
                for index in batch["index"].numpy().tolist()
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    labels_array = np.concatenate(labels)
    scores_array = np.concatenate(scores)
    with np.load(evaluation["predictions"], allow_pickle=False) as frozen:
        if (
            not np.array_equal(labels_array, np.asarray(frozen["labels"], dtype=np.int8))
            or sample_ids != [str(value) for value in frozen["sample_ids"].tolist()]
            or not np.array_equal(scores_array, np.asarray(frozen["scores"], dtype=np.float64))
        ):
            raise ValueError("profile rerun predictions differ from frozen CALDER evaluation")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "cell_id": evaluation["cell_id"],
        "test_dataset": evaluation["test_dataset"],
        "records": len(dataset),
        "evaluation_summary": str(args.evaluation_summary.resolve()),
        "evaluation_summary_sha256": sha256_file(args.evaluation_summary),
        "predictions": str(Path(evaluation["predictions"]).resolve()),
        "predictions_sha256": evaluation["predictions_sha256"],
        "device": str(device),
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "records_per_second": len(dataset) / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "prediction_identity": "exactly_equal_to_frozen_evaluation",
    }
    atomic_json(payload, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
