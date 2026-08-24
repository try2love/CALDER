"""Atomic, fail-closed storage for formal frozen-feature shards."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .identity import sha256_file


class FeatureShardError(ValueError):
    """Raised when a feature shard is partial, malformed, or identity-drifted."""


ARRAY_DTYPES = {
    "semantic_spans": np.dtype("float16"),
    "semantic_mask": np.dtype("bool"),
    "token_evidence": np.dtype("float16"),
    "token_mask": np.dtype("bool"),
    "probability": np.dtype("float32"),
    "alignment": np.dtype("float32"),
    "compression": np.dtype("float32"),
    "detector_score": np.dtype("float32"),
    "labels": np.dtype("int8"),
}


@dataclass(frozen=True)
class FeatureShardPayload:
    arrays: dict[str, np.ndarray]
    identities: list[dict[str, Any]]


def _validate_payload(payload: FeatureShardPayload) -> int:
    if set(payload.arrays) != set(ARRAY_DTYPES):
        raise FeatureShardError(f"feature arrays must be exactly {sorted(ARRAY_DTYPES)}")
    counts = {int(array.shape[0]) for array in payload.arrays.values()}
    counts.add(len(payload.identities))
    if len(counts) != 1 or not counts or next(iter(counts)) < 1:
        raise FeatureShardError("all feature arrays and identities must have one positive count")
    records = next(iter(counts))
    expected_shapes = {
        "semantic_spans": (records, 16, None),
        "semantic_mask": (records, 16),
        "token_evidence": (records, 64, 10),
        "token_mask": (records, 64),
        "probability": (records, 108),
        "alignment": (records, 32),
        "compression": (records, 51),
        "detector_score": (records, 7),
        "labels": (records,),
    }
    for name, array in payload.arrays.items():
        if array.dtype != ARRAY_DTYPES[name]:
            raise FeatureShardError(f"{name} dtype {array.dtype} != {ARRAY_DTYPES[name]}")
        expected = expected_shapes[name]
        if array.ndim != len(expected) or any(
            value is not None and array.shape[index] != value
            for index, value in enumerate(expected)
        ):
            raise FeatureShardError(f"{name} shape {array.shape} does not match {expected}")
        if name == "semantic_spans" and array.shape[2] < 1:
            raise FeatureShardError("semantic hidden dimension must be positive")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise FeatureShardError(f"{name} contains non-finite values")
    if not payload.arrays["semantic_mask"].any(axis=1).all():
        raise FeatureShardError("every record must have at least one semantic span")
    if not payload.arrays["token_mask"].any(axis=1).all():
        raise FeatureShardError("every record must have at least one token-evidence span")
    labels = payload.arrays["labels"].astype(np.int64, copy=False)
    if not np.isin(labels, (0, 1)).all():
        raise FeatureShardError("labels must be canonical human=0/ai=1")
    required_identity = {
        "sample_id",
        "label",
        "dataset",
        "split",
        "source_component_id",
        "input_sha256",
        "feature_schema_sha256",
        "model_protocol_sha256",
    }
    sample_ids: set[str] = set()
    for index, identity in enumerate(payload.identities):
        if not required_identity.issubset(identity):
            raise FeatureShardError(f"identity {index} is missing required fields")
        sample_id = identity["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise FeatureShardError(f"invalid or duplicate sample_id at identity {index}")
        sample_ids.add(sample_id)
        if int(identity["label"]) != int(labels[index]):
            raise FeatureShardError(f"identity/array label mismatch at record {index}")
        if not isinstance(identity["source_component_id"], str):
            raise FeatureShardError(f"invalid source_component_id at record {index}")
    return records


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_array(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_identities(path: Path, identities: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for identity in identities:
            handle.write(json.dumps(identity, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_summary(target: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    embedded = target / "manifest.json"
    return {
        "schema_version": 1,
        "status": "complete",
        "artifact_dir": str(target),
        "artifact_manifest_sha256": sha256_file(embedded),
        "records": manifest["records"],
        "files": manifest["files"],
        "provenance": manifest["provenance"],
    }


def validate_feature_shard(target: Path) -> dict[str, Any]:
    manifest_path = target / "manifest.json"
    if not target.is_dir() or not manifest_path.is_file():
        raise FeatureShardError(f"incomplete feature shard: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise FeatureShardError("feature shard manifest is not complete schema version 1")
    records = int(manifest["records"])
    files = manifest["files"]
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_DTYPES:
        spec = files[f"{name}.npy"]
        path = target / f"{name}.npy"
        if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
            raise FeatureShardError(f"feature file identity mismatch: {path}")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
            raise FeatureShardError(f"feature array metadata mismatch: {path}")
        arrays[name] = array
    identity_spec = files["identities.jsonl"]
    identity_path = target / "identities.jsonl"
    if (
        identity_path.stat().st_size != int(identity_spec["bytes"])
        or sha256_file(identity_path) != identity_spec["sha256"]
    ):
        raise FeatureShardError("feature identity sidecar mismatch")
    identities = [json.loads(line) for line in identity_path.read_text(encoding="utf-8").splitlines()]
    _validate_payload(FeatureShardPayload(arrays=arrays, identities=identities))
    if len(identities) != records:
        raise FeatureShardError("feature shard count differs from manifest")
    return manifest


def write_feature_shard(
    payload: FeatureShardPayload,
    target: Path,
    durable_manifest: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Publish one shard transactionally; complete reruns validate and reuse it."""

    records = _validate_payload(payload)
    if target.exists():
        manifest = validate_feature_shard(target)
        if manifest["provenance"] != provenance:
            raise FeatureShardError("complete artifact provenance differs from invocation")
        if int(manifest["records"]) != records:
            raise FeatureShardError("complete artifact count differs from invocation")
        summary = _artifact_summary(target, manifest)
        if durable_manifest.exists():
            if json.loads(durable_manifest.read_text(encoding="utf-8")) != summary:
                raise FeatureShardError("durable manifest differs from complete artifact")
        else:
            _atomic_json(summary, durable_manifest)
        return summary
    if durable_manifest.exists():
        raise FeatureShardError("durable manifest exists without its feature artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.incomplete.", dir=target.parent))
    try:
        files: dict[str, Any] = {}
        for name, array in payload.arrays.items():
            path = temporary / f"{name}.npy"
            _write_array(path, array)
            files[path.name] = {
                "bytes": path.stat().st_size,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": sha256_file(path),
            }
        identity_path = temporary / "identities.jsonl"
        _write_identities(identity_path, payload.identities)
        files[identity_path.name] = {
            "bytes": identity_path.stat().st_size,
            "records": records,
            "sha256": sha256_file(identity_path),
        }
        embedded = {
            "schema_version": 1,
            "status": "complete",
            "records": records,
            "files": files,
            "provenance": provenance,
        }
        _atomic_json(embedded, temporary / "manifest.json")
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, target)
        manifest = validate_feature_shard(target)
        summary = _artifact_summary(target, manifest)
        _atomic_json(summary, durable_manifest)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
