"""Python-module CLI for raw-text CALDER training and inference."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from .quickstart import predict as predict_features
from .quickstart import train as train_features
from .text_features import (
    CalderTextFeatureExtractor,
    TextRecord,
    load_observer_config,
    load_records,
    resolve_devices,
)


def _atomic_torch(payload: object, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _extractor(args: argparse.Namespace, config: dict[str, object]) -> CalderTextFeatureExtractor:
    return CalderTextFeatureExtractor(
        config,
        devices=resolve_devices(args.devices),
        batch_size=args.feature_batch_size,
        local_files_only=args.local_files_only,
    )


def train_text(args: argparse.Namespace) -> dict[str, object]:
    records_train = load_records(args.train, require_labels=True)
    records_dev = load_records(args.dev, require_labels=True)
    config = load_observer_config(args.observer_config)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    cache = output / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    extractor = _extractor(args, config)
    combined_npz = extractor.extract(
        [*records_train, *records_dev], cache / "combined_features.npz"
    )
    train_npz = cache / "train_features.npz"
    dev_npz = cache / "dev_features.npz"
    with np.load(combined_npz, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    split = len(records_train)
    np.savez_compressed(train_npz, **{name: value[:split] for name, value in arrays.items()})
    np.savez_compressed(dev_npz, **{name: value[split:] for name, value in arrays.items()})
    combined_npz.unlink()
    summary = train_features(
        SimpleNamespace(
            train=train_npz,
            dev=dev_npz,
            output=output,
            device=args.train_device,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            convolution_channels=args.convolution_channels,
            branch_dropout=args.branch_dropout,
            fusion=args.fusion,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip_norm,
        )
    )
    model_path = Path(summary["model"])
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    bundle["text_pipeline"] = {
        "schema_version": 1,
        "observer_config": config,
        "input_fields": {"text": "text or content", "label": "human=0, AI=1"},
    }
    _atomic_torch(bundle, model_path)
    (output / "observer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["raw_text_training"] = True
    summary["feature_cache"] = str(cache)
    return summary


def _prediction_records(args: argparse.Namespace) -> list[TextRecord]:
    provided = sum(value is not None for value in (args.text, args.text_file, args.input))
    if provided != 1:
        raise ValueError("provide exactly one of --text, --text-file, or --input")
    if args.text is not None:
        if not args.text.strip():
            raise ValueError("--text must be non-empty")
        return [TextRecord(sample_id="input-0001", text=args.text)]
    path = args.text_file if args.text_file is not None else args.input
    return load_records(path, require_labels=False)


def predict_text(args: argparse.Namespace) -> dict[str, object]:
    bundle = torch.load(args.model, map_location="cpu", weights_only=False)
    text_pipeline = bundle.get("text_pipeline")
    if not isinstance(text_pipeline, dict) or text_pipeline.get("schema_version") != 1:
        raise ValueError("model bundle lacks raw-text observer configuration")
    config = (
        load_observer_config(args.observer_config)
        if args.observer_config is not None
        else text_pipeline["observer_config"]
    )
    records = _prediction_records(args)
    output = args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="calder-predict-") as temporary:
        feature_path = Path(temporary) / "features.npz"
        _extractor(args, config).extract(records, feature_path)
        summary = predict_features(
            SimpleNamespace(
                model=args.model,
                input=feature_path,
                output=output,
                device=args.train_device,
                batch_size=args.batch_size,
            )
        )
    if len(records) == 1:
        summary["prediction"] = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmark_fusion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--observer-config", type=Path)
    common.add_argument("--devices", default="auto", help="comma-separated feature-extraction devices")
    common.add_argument("--feature-batch-size", type=int, default=1)
    common.add_argument("--local-files-only", action="store_true")
    common.add_argument("--train-device", default="auto", help="CALDER fusion-model device")

    training = subparsers.add_parser("train", parents=[common], help="train from labeled text/content")
    training.add_argument("--train", required=True, type=Path)
    training.add_argument("--dev", required=True, type=Path)
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--patience", type=int, default=4)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--hidden-dim", type=int, default=128)
    training.add_argument("--convolution-channels", type=int, default=64)
    training.add_argument("--branch-dropout", type=float, default=0.0)
    training.add_argument("--fusion", choices=("adaptive_gate", "concat_mlp"), default="adaptive_gate")
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.0)
    training.add_argument("--gradient-clip-norm", type=float, default=1.0)
    training.set_defaults(handler=train_text)

    prediction = subparsers.add_parser("predict", parents=[common], help="classify raw text")
    prediction.add_argument("--model", required=True, type=Path)
    prediction.add_argument("--text")
    prediction.add_argument("--text-file", type=Path)
    prediction.add_argument("--input", type=Path, help="batch JSONL/CSV")
    prediction.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    prediction.add_argument("--batch-size", type=int, default=256)
    prediction.set_defaults(handler=predict_text)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = args.handler(args)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
