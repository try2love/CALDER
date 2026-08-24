#!/usr/bin/env python3
"""Single-GPU source-specific CALDER Core trainer."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PACKAGE_ROOT))

from benchmark_fusion.calder_data import CalderDataset, scalar_training_moments
from benchmark_fusion.calder_model import CalderCore
from benchmark_fusion.identity import sha256_file
from benchmark_fusion.metrics import ranking_metrics, select_f1_threshold
from benchmark_fusion.training import (
    CheckpointScore,
    capture_rng_state,
    checkpoint_is_better,
    restore_rng_state,
    validate_implementation_manifest,
)
from benchmark_fusion.training_data import build_balanced_training_index


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


def atomic_torch(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
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


def evaluate(
    model: CalderCore,
    dataset: CalderDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    indices = []
    labels = []
    scores = []
    losses = []
    model.eval()
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )
    with torch.inference_mode():
        for batch in loader:
            with autocast:
                logits = model(**model_inputs(batch, device)).logit
            target = batch["labels"].to(device).float()
            loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
            indices.extend(batch["index"].numpy().tolist())
            labels.extend(target.cpu().numpy().tolist())
            scores.extend(logits.float().cpu().numpy().tolist())
            losses.extend(loss.cpu().numpy().tolist())
    order = np.argsort(np.asarray(indices), kind="stable")
    labels_array = np.asarray(labels, dtype=np.int8)[order]
    scores_array = np.asarray(scores, dtype=np.float64)[order]
    if not np.array_equal(np.asarray(indices)[order], np.arange(len(dataset))):
        raise RuntimeError("CALDER evaluation did not cover every record exactly once")
    result = ranking_metrics(labels_array, scores_array)
    metrics = asdict(result) | {
        "binary_cross_entropy": float(np.asarray(losses)[order].mean()),
        "classification_threshold": select_f1_threshold(labels_array, scores_array),
        "records": len(dataset),
    }
    return metrics, {
        "indices": np.arange(len(dataset), dtype=np.int64),
        "labels": labels_array,
        "scores": scores_array,
        "sample_ids": np.asarray(
            [str(identity["sample_id"]) for identity in dataset.identities], dtype="U96"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--engineering-cpu-only", action="store_true")
    args = parser.parse_args()
    run_spec = json.loads(args.run_spec.read_text(encoding="utf-8"))
    validate_implementation_manifest(run_spec)
    if run_spec.get("method") != "calder_core" or int(run_spec.get("seed", -1)) != 42:
        raise ValueError("formal CALDER selection requires method=calder_core and seed=42")
    legacy_source = run_spec.get("source")
    train_sources = list(run_spec.get("train_sources", [legacy_source]))
    dev_sources = list(run_spec.get("dev_sources", train_sources))
    qualified_sources = {"m4", "detectrl", "raid"}
    if (
        not train_sources
        or not dev_sources
        or None in train_sources
        or None in dev_sources
        or not set(train_sources) <= qualified_sources
        or not set(dev_sources) <= qualified_sources
        or not set(dev_sources) <= set(train_sources)
    ):
        raise ValueError("CALDER train/dev source identity is not qualified")
    if legacy_source is not None and (
        train_sources != [legacy_source]
        or dev_sources != [legacy_source]
        or legacy_source not in ("m4", "detectrl")
    ):
        raise ValueError("legacy CALDER source-specific identity differs")
    device = torch.device("cpu" if args.engineering_cpu_only else "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("formal CALDER training requires CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("formal deterministic CALDER training requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    run_slug = str(run_spec.get("run_slug", legacy_source))
    if not run_slug or run_slug == "None" or any(value in run_slug for value in ("/", "\\", "..")):
        raise ValueError("CALDER run slug is unsafe")
    output_dir = args.output_root / str(run_spec["configuration_id"]) / run_slug
    complete_path = output_dir / "training_summary.json"
    if complete_path.exists():
        summary = json.loads(complete_path.read_text(encoding="utf-8"))
        for key, hash_key in (
            ("run_spec", "run_spec_sha256"),
            ("best_model", "best_model_sha256"),
            ("best_tune_predictions", "best_tune_predictions_sha256"),
            ("history", "history_sha256"),
        ):
            if sha256_file(Path(summary[key])) != summary[hash_key]:
                raise ValueError(f"completed CALDER artifact hash differs: {key}")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_checkpoint.pt"
    if latest_path.exists() and not args.resume:
        raise FileExistsError("incomplete CALDER run exists; pass --resume")
    train_manifest = Path(run_spec["train_feature_manifest"])
    dev_manifest = Path(run_spec["dev_feature_manifest"])
    if sha256_file(train_manifest) != run_spec["train_feature_manifest_sha256"]:
        raise ValueError("CALDER train feature manifest hash differs")
    if sha256_file(dev_manifest) != run_spec["dev_feature_manifest_sha256"]:
        raise ValueError("CALDER dev feature manifest hash differs")
    train_dataset = CalderDataset(train_manifest, validate=True)
    tune_dataset = CalderDataset(dev_manifest, partition="tune_dev", validate=True)
    observed_train_sources = {str(identity["dataset"]) for identity in train_dataset.identities}
    observed_dev_sources = {str(identity["dataset"]) for identity in tune_dataset.identities}
    if observed_train_sources != set(train_sources):
        raise ValueError("CALDER train dataset sources differ from frozen run spec")
    if observed_dev_sources != set(dev_sources):
        raise ValueError("CALDER tune dataset sources differ from frozen run spec")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    moments_np = scalar_training_moments(train_dataset)
    normalization_path = output_dir / "train_scalar_moments.npz"
    if not normalization_path.exists():
        atomic_npz(
            normalization_path,
            **{
                f"{name}_{suffix}": value
                for name, pair in moments_np.items()
                for suffix, value in zip(("mean", "scale"), pair)
            },
        )
    moments = {
        name: (torch.from_numpy(mean), torch.from_numpy(scale))
        for name, (mean, scale) in moments_np.items()
    }
    configuration = run_spec["configuration"]
    enabled_branches = tuple(run_spec.get("enabled_branches", CalderCore.branch_names))
    alignment_feature_mode = str(run_spec.get("alignment_feature_mode", "full"))
    model = CalderCore(
        semantic_dim=int(train_dataset.shards[0]["semantic_spans"].shape[2]),
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    batch_size = int(configuration["global_batch_size"])
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    protocol = json.loads(Path(run_spec["selection_protocol"]).read_text(encoding="utf-8"))
    fixed = protocol["fixed"]
    max_epochs = max(
        int(fixed["maximum_epochs"]),
        math.ceil(int(fixed["minimum_optimizer_steps_before_early_stop"]) / steps_per_epoch),
    )
    start_epoch = 0
    global_step = 0
    best_score = None
    epochs_without_improvement = 0
    elapsed_before = 0.0
    history = []
    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["run_spec_sha256"] != sha256_file(args.run_spec):
            raise ValueError("CALDER resume run-spec identity differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint["global_step"])
        best_score = CheckpointScore(**checkpoint["best_score"]) if checkpoint["best_score"] else None
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
        history = checkpoint["history"]
        restore_rng_state(checkpoint["rng_state"])
    started = time.perf_counter()
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )
    for epoch in range(start_epoch, max_epochs):
        balanced = build_balanced_training_index(train_dataset.identities, seed, epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=[int(index) for index in balanced.order],
            num_workers=0,
        )
        model.train()
        epoch_loss_sum = 0.0
        epoch_weight_sum = 0.0
        for batch in loader:
            indices = batch["index"].numpy()
            weights = torch.from_numpy(balanced.loss_weights[indices]).to(device)
            labels = batch["labels"].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                logits = model(**model_inputs(batch, device)).logit
                per_record = F.binary_cross_entropy_with_logits(
                    logits.float(), labels, reduction="none"
                )
                loss = (per_record * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(fixed["gradient_clip_norm"]))
            optimizer.step()
            global_step += 1
            epoch_loss_sum += float((per_record.detach() * weights).sum())
            epoch_weight_sum += float(weights.sum())
        metrics, predictions = evaluate(model, tune_dataset, device, batch_size)
        score = CheckpointScore(
            auroc=float(metrics["auroc"]),
            auprc=float(metrics["auprc"]),
            binary_cross_entropy=float(metrics["binary_cross_entropy"]),
            epoch=epoch,
        )
        improved = checkpoint_is_better(score, best_score)
        if improved:
            best_score = score
            epochs_without_improvement = 0
            atomic_torch(
                {
                    "schema_version": 1,
                    "run_spec_sha256": sha256_file(args.run_spec),
                    "configuration": configuration,
                    "source": legacy_source,
                    "train_sources": train_sources,
                    "dev_sources": dev_sources,
                    "enabled_branches": enabled_branches,
                    "alignment_feature_mode": alignment_feature_mode,
                    "score": asdict(score),
                    "model": model.state_dict(),
                },
                output_dir / "best_model.pt",
            )
            atomic_npz(output_dir / "best_tune_predictions.npz", **predictions)
        else:
            epochs_without_improvement += 1
        history.append({
            "epoch": epoch,
            "global_step": global_step,
            "train_weighted_bce": epoch_loss_sum / epoch_weight_sum,
            "tune_metrics": metrics,
            "improved": improved,
        })
        atomic_json({"schema_version": 1, "epochs": history}, output_dir / "history.json")
        elapsed = elapsed_before + time.perf_counter() - started
        atomic_torch(
            {
                "schema_version": 1,
                "run_spec_sha256": sha256_file(args.run_spec),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "global_step": global_step,
                "best_score": asdict(best_score),
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": elapsed,
                "history": history,
                "rng_state": capture_rng_state(),
            },
            latest_path,
        )
        minimum_steps = int(fixed["minimum_optimizer_steps_before_early_stop"])
        if (
            global_step >= minimum_steps
            and epochs_without_improvement >= int(fixed["early_stopping_patience_complete_epochs"])
        ):
            break
    elapsed = elapsed_before + time.perf_counter() - started
    best_path = output_dir / "best_model.pt"
    prediction_path = output_dir / "best_tune_predictions.npz"
    history_path = output_dir / "history.json"
    summary = {
        "schema_version": 1,
        "status": "complete",
        "method": "calder_core",
        "configuration_id": run_spec["configuration_id"],
        "configuration": configuration,
        "ablation_id": run_spec.get("ablation_id", "full"),
        "enabled_branches": list(enabled_branches),
        "alignment_feature_mode": alignment_feature_mode,
        "source": legacy_source,
        "train_sources": train_sources,
        "dev_sources": dev_sources,
        "training_job_id": run_spec.get("training_job_id"),
        "track": run_spec.get("track"),
        "target_group": run_spec.get("target_group"),
        "seed": seed,
        "run_spec": str(args.run_spec.resolve()),
        "run_spec_sha256": sha256_file(args.run_spec),
        "train_feature_manifest": str(train_manifest),
        "train_feature_manifest_sha256": sha256_file(train_manifest),
        "dev_feature_manifest": str(dev_manifest),
        "dev_feature_manifest_sha256": sha256_file(dev_manifest),
        "train_records": len(train_dataset),
        "tune_records": len(tune_dataset),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "global_batch_size": batch_size,
        "global_steps": global_step,
        "best_score": asdict(best_score),
        "best_model": str(best_path.resolve()),
        "best_model_sha256": sha256_file(best_path),
        "best_tune_predictions": str(prediction_path.resolve()),
        "best_tune_predictions_sha256": sha256_file(prediction_path),
        "history": str(history_path.resolve()),
        "history_sha256": sha256_file(history_path),
        "train_scalar_moments": str(normalization_path.resolve()),
        "train_scalar_moments_sha256": sha256_file(normalization_path),
        "elapsed_seconds": elapsed,
        "checkpoint_retention": "completed_best_model_only",
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    atomic_json(summary, complete_path)
    latest_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
