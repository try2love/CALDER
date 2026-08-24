"""Beginner-friendly CALDER training and prediction over portable NPZ features."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .calder_model import CalderCore
from .metrics import ranking_metrics, select_f1_threshold


FLOAT_INPUTS = {
    "semantic_spans": (16, None),
    "token_probability": (64, 6),
    "alignment_evidence": (64, 8),
    "document_probability": (108,),
    "document_alignment": (56,),
    "compression": (51,),
}
MASK_INPUTS = {"semantic_mask": (16,), "token_mask": (64,)}
MODEL_INPUTS = tuple((*FLOAT_INPUTS, *MASK_INPUTS))


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


class PortableFeatureDataset(Dataset[dict[str, torch.Tensor]]):
    """Validated in-memory dataset for the public single-file NPZ format."""

    def __init__(self, path: Path, *, require_labels: bool) -> None:
        self.path = path
        with np.load(path, allow_pickle=False) as archive:
            missing = set(MODEL_INPUTS) - set(archive.files)
            if missing:
                raise ValueError(f"portable feature file lacks arrays: {sorted(missing)}")
            self.arrays = {name: np.array(archive[name], copy=True) for name in MODEL_INPUTS}
            self.labels = (
                np.array(archive["labels"], dtype=np.int8, copy=True)
                if "labels" in archive.files
                else None
            )
            if "sample_ids" in archive.files:
                self.sample_ids = [str(value) for value in archive["sample_ids"].tolist()]
            else:
                records = int(self.arrays["semantic_spans"].shape[0])
                self.sample_ids = [f"sample-{index:08d}" for index in range(records)]
        self._validate(require_labels=require_labels)

    def _validate(self, *, require_labels: bool) -> None:
        records = int(self.arrays["semantic_spans"].shape[0])
        if records < 1:
            raise ValueError("portable feature file is empty")
        semantic_dim = int(self.arrays["semantic_spans"].shape[-1])
        if semantic_dim < 1:
            raise ValueError("semantic feature dimension must be positive")
        expected = dict(FLOAT_INPUTS)
        expected["semantic_spans"] = (16, semantic_dim)
        expected.update(MASK_INPUTS)
        for name, trailing_shape in expected.items():
            array = self.arrays[name]
            if array.shape != (records, *trailing_shape):
                raise ValueError(
                    f"{name} shape differs: {array.shape} != {(records, *trailing_shape)}"
                )
            if name in FLOAT_INPUTS and not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in MASK_INPUTS:
            self.arrays[name] = self.arrays[name].astype(np.bool_, copy=False)
            if not self.arrays[name].any(axis=1).all():
                raise ValueError(f"{name} contains a record without a valid span")
        for name in FLOAT_INPUTS:
            self.arrays[name] = self.arrays[name].astype(np.float32, copy=False)
        if len(self.sample_ids) != records or len(set(self.sample_ids)) != records:
            raise ValueError("sample IDs must be aligned and unique")
        if require_labels and self.labels is None:
            raise ValueError("training/evaluation feature file requires labels")
        if self.labels is not None:
            if self.labels.shape != (records,) or not np.isin(self.labels, (0, 1)).all():
                raise ValueError("labels must be an aligned binary vector")
            if require_labels and np.unique(self.labels).size != 2:
                raise ValueError("training/evaluation requires both label classes")

    @property
    def semantic_dim(self) -> int:
        return int(self.arrays["semantic_spans"].shape[-1])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = {
            name: torch.from_numpy(np.array(self.arrays[name][index], copy=True))
            for name in MODEL_INPUTS
        }
        row["index"] = torch.tensor(index, dtype=torch.int64)
        if self.labels is not None:
            row["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.int8)
        return row


def scalar_moments(dataset: PortableFeatureDataset) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result = {}
    for name in ("document_probability", "document_alignment", "compression"):
        values = dataset.arrays[name].astype(np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        result[name] = (
            torch.from_numpy(mean.astype(np.float32)),
            torch.from_numpy(scale.astype(np.float32)),
        )
    return result


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True).float() for name in FLOAT_INPUTS
    } | {
        name: batch[name].to(device, non_blocking=True).bool() for name in MASK_INPUTS
    }


def _evaluate(
    model: CalderCore,
    dataset: PortableFeatureDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    labels: list[int] = []
    scores: list[float] = []
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(**_inputs(batch, device)).logit.float()
            target = batch["labels"].to(device).float()
            labels.extend(target.cpu().numpy().astype(np.int8).tolist())
            scores.extend(logits.cpu().numpy().astype(np.float64).tolist())
            losses.extend(
                F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                .cpu()
                .numpy()
                .tolist()
            )
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=np.float64)
    metrics = asdict(ranking_metrics(labels_array, scores_array))
    metrics["binary_cross_entropy"] = float(np.mean(losses))
    return metrics, scores_array


def train(args: argparse.Namespace) -> dict[str, object]:
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim, args.convolution_channels) < 1:
        raise ValueError("epochs, patience, batch size, and model dimensions must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip_norm <= 0:
        raise ValueError("optimizer values are outside their valid ranges")
    output_dir = args.output.resolve()
    existing = [] if not output_dir.exists() else [path for path in output_dir.iterdir() if path.name != ".cache"]
    if existing:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = PortableFeatureDataset(args.train, require_labels=True)
    dev_dataset = PortableFeatureDataset(args.dev, require_labels=True)
    if train_dataset.semantic_dim != dev_dataset.semantic_dim:
        raise ValueError("train/dev semantic dimensions differ")
    device = _device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    moments = scalar_moments(train_dataset)
    model = CalderCore(
        semantic_dim=train_dataset.semantic_dim,
        hidden_dim=args.hidden_dim,
        convolution_channels=args.convolution_channels,
        branch_dropout_probability=args.branch_dropout,
        scalar_normalization="train_zscore",
        fusion=args.fusion,
        scalar_moments=moments,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    labels = train_dataset.labels
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weights = len(labels) / (2.0 * class_counts)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_auroc = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    best_path = output_dir / "calder_model.pt"
    for epoch in range(args.epochs):
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for batch in loader:
            target = batch["labels"].to(device).long()
            weights = torch.as_tensor(class_weights, device=device, dtype=torch.float32)[target]
            optimizer.zero_grad(set_to_none=True)
            logits = model(**_inputs(batch, device)).logit.float()
            per_record = F.binary_cross_entropy_with_logits(
                logits, target.float(), reduction="none"
            )
            loss = (per_record * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            weighted_loss_sum += float((per_record.detach() * weights).sum())
            weight_sum += float(weights.sum())
        dev_metrics, dev_scores = _evaluate(model, dev_dataset, device, args.batch_size)
        improved = float(dev_metrics["auroc"]) > best_auroc
        if improved:
            best_auroc = float(dev_metrics["auroc"])
            best_epoch = epoch
            epochs_without_improvement = 0
            threshold = select_f1_threshold(dev_dataset.labels, dev_scores)
            torch.save(
                {
                    "schema_version": 1,
                    "format": "calder_portable_model",
                    "label_convention": {"human": 0, "ai": 1},
                    "semantic_dim": train_dataset.semantic_dim,
                    "configuration": {
                        "hidden_dim": args.hidden_dim,
                        "convolution_channels": args.convolution_channels,
                        "branch_dropout": args.branch_dropout,
                        "fusion": args.fusion,
                    },
                    "scalar_moments": {
                        name: {"mean": mean.cpu(), "scale": scale.cpu()}
                        for name, (mean, scale) in moments.items()
                    },
                    "classification_threshold": threshold,
                    "best_epoch": best_epoch,
                    "dev_metrics": dev_metrics,
                    "model_state": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": weighted_loss_sum / weight_sum,
                "dev_metrics": dev_metrics,
                "improved": improved,
            }
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_bce={weighted_loss_sum / weight_sum:.6f} "
            f"dev_auroc={float(dev_metrics['auroc']):.6f}"
        )
        if epochs_without_improvement >= args.patience:
            break
    detailed_summary = {
        "status": "complete",
        "model": str(best_path),
        "best_epoch": best_epoch,
        "best_dev_auroc": best_auroc,
        "train_records": len(train_dataset),
        "dev_records": len(dev_dataset),
        "device": str(device),
        "history": history,
    }
    summary_path = output_dir / "training_summary.json"
    _atomic_json(detailed_summary, summary_path)
    return {key: value for key, value in detailed_summary.items() if key != "history"} | {
        "training_summary": str(summary_path)
    }


def _load_model(path: Path, device: torch.device) -> tuple[CalderCore, dict[str, Any]]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("format") != "calder_portable_model" or bundle.get("schema_version") != 1:
        raise ValueError("model is not a CALDER portable model bundle")
    moments = {
        name: (value["mean"], value["scale"])
        for name, value in bundle["scalar_moments"].items()
    }
    configuration = bundle["configuration"]
    model = CalderCore(
        semantic_dim=int(bundle["semantic_dim"]),
        hidden_dim=int(configuration["hidden_dim"]),
        convolution_channels=int(configuration["convolution_channels"]),
        branch_dropout_probability=float(configuration["branch_dropout"]),
        scalar_normalization="train_zscore",
        fusion=str(configuration["fusion"]),
        scalar_moments=moments,
    ).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, bundle


def predict(args: argparse.Namespace) -> dict[str, object]:
    dataset = PortableFeatureDataset(args.input, require_labels=False)
    device = _device(args.device)
    model, bundle = _load_model(args.model, device)
    if dataset.semantic_dim != int(bundle["semantic_dim"]):
        raise ValueError("prediction semantic dimension differs from the trained model")
    threshold = float(bundle["classification_threshold"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows: list[dict[str, object]] = []
    all_scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = model(**_inputs(batch, device)).logit.float().cpu().numpy()
            for local_index, score in zip(batch["index"].numpy().tolist(), logits.tolist()):
                exponential = np.exp(-abs(score))
                probability = 1.0 / (1.0 + exponential) if score >= 0 else exponential / (1.0 + exponential)
                row: dict[str, object] = {
                    "sample_id": dataset.sample_ids[int(local_index)],
                    "ai_score": float(score),
                    "ai_probability": float(probability),
                    "prediction": "AI" if score >= threshold else "human",
                }
                if dataset.labels is not None:
                    row["label"] = int(dataset.labels[int(local_index)])
                rows.append(row)
                all_scores.append(float(score))
    _atomic_jsonl(rows, args.output)
    summary: dict[str, object] = {
        "status": "complete",
        "records": len(dataset),
        "model": str(args.model),
        "input": str(args.input),
        "output": str(args.output),
        "classification_threshold": threshold,
        "device": str(device),
    }
    if dataset.labels is not None and np.unique(dataset.labels).size == 2:
        summary["metrics"] = asdict(
            ranking_metrics(dataset.labels, np.asarray(all_scores, dtype=np.float64))
        )
    _atomic_json(summary, args.output.with_suffix(args.output.suffix + ".summary.json"))
    return summary


def make_demo(args: argparse.Namespace) -> dict[str, object]:
    if args.semantic_dim < 1:
        raise ValueError("demo semantic dimension must be positive")
    output_dir = args.output.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty demo directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def generate(path: Path, records: int, seed: int) -> None:
        if records < 4 or records % 2:
            raise ValueError("demo split size must be an even integer of at least four")
        rng = np.random.default_rng(seed)
        labels = np.arange(records, dtype=np.int8) % 2
        rng.shuffle(labels)
        shift = labels.astype(np.float32)[:, None]
        arrays = {
            "semantic_spans": rng.normal(size=(records, 16, args.semantic_dim)).astype(np.float32),
            "semantic_mask": np.ones((records, 16), dtype=np.bool_),
            "token_probability": rng.normal(size=(records, 64, 6)).astype(np.float32),
            "alignment_evidence": rng.normal(size=(records, 64, 8)).astype(np.float32),
            "token_mask": np.ones((records, 64), dtype=np.bool_),
            "document_probability": rng.normal(size=(records, 108)).astype(np.float32),
            "document_alignment": rng.normal(size=(records, 56)).astype(np.float32),
            "compression": rng.normal(size=(records, 51)).astype(np.float32),
            "labels": labels,
            "sample_ids": np.asarray([f"{path.stem}-{index:05d}" for index in range(records)]),
        }
        arrays["semantic_spans"] += shift[:, None, :] * 0.80
        arrays["token_probability"] += shift[:, None, :] * 0.80
        arrays["document_probability"] += shift * 1.00
        arrays["compression"] += shift * 1.00
        np.savez_compressed(path, **arrays)

    paths = {}
    for name, records, offset in (
        ("train", args.train_records, 0),
        ("dev", args.dev_records, 1),
        ("test", args.test_records, 2),
    ):
        path = output_dir / f"{name}.npz"
        generate(path, records, args.seed + offset)
        paths[name] = str(path)
    payload = {
        "status": "complete",
        "purpose": "synthetic_engineering_demo_only",
        "files": paths,
    }
    _atomic_json(payload, output_dir / "demo_manifest.json")
    return payload
