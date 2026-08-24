"""Atomic intermediate view caches with exact sample-order provenance."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .identity import sha256_file


class ViewShardError(ValueError):
    """Raised when an intermediate view shard is incomplete or identity-drifted."""


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_arrays(arrays: Mapping[str, np.ndarray], sample_ids: list[str]) -> int:
    if not arrays or not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ViewShardError("arrays and unique sample IDs must be non-empty")
    if any(not isinstance(name, str) or not name or "/" in name for name in arrays):
        raise ViewShardError("array names must be non-empty path-safe strings")
    records = len(sample_ids)
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.ndim < 1 or array.shape[0] != records:
            raise ViewShardError(f"{name} must be an ndarray with the sample count first")
        if array.dtype.hasobject:
            raise ViewShardError(f"{name} may not use object dtype")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ViewShardError(f"{name} contains non-finite values")
    return records


def validate_view_shard(
    target: Path,
    expected_arrays: Mapping[str, tuple[str, tuple[int | None, ...]]] | None = None,
) -> dict[str, Any]:
    manifest_path = target / "manifest.json"
    if not target.is_dir() or not manifest_path.is_file():
        raise ViewShardError(f"incomplete view shard: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ViewShardError("view shard manifest is not complete schema version 1")
    records = int(manifest["records"])
    arrays: dict[str, np.ndarray] = {}
    file_specs = manifest["files"]
    names = set(manifest["arrays"])
    if expected_arrays is not None and names != set(expected_arrays):
        raise ViewShardError("view shard array names differ from expected schema")
    for name in sorted(names):
        path = target / f"{name}.npy"
        spec = file_specs[path.name]
        if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
            raise ViewShardError(f"view array identity mismatch: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
            raise ViewShardError(f"view array metadata mismatch: {path}")
        if array.shape[0] != records:
            raise ViewShardError(f"view array record count mismatch: {path}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ViewShardError(f"view array contains non-finite values: {path}")
        if expected_arrays is not None:
            expected_dtype, expected_tail = expected_arrays[name]
            if str(array.dtype) != expected_dtype or array.ndim != len(expected_tail) + 1:
                raise ViewShardError(f"view array schema mismatch: {name}")
            if any(value is not None and array.shape[index + 1] != value for index, value in enumerate(expected_tail)):
                raise ViewShardError(f"view array trailing shape mismatch: {name}")
        arrays[name] = array
    sample_path = target / "sample_ids.jsonl"
    sample_spec = file_specs[sample_path.name]
    if sample_path.stat().st_size != int(sample_spec["bytes"]) or sha256_file(sample_path) != sample_spec["sha256"]:
        raise ViewShardError("view sample identity sidecar mismatch")
    sample_ids = [json.loads(line)["sample_id"] for line in sample_path.read_text(encoding="utf-8").splitlines()]
    _validate_arrays(arrays, sample_ids)
    if len(sample_ids) != records:
        raise ViewShardError("view sample count differs from manifest")
    return manifest


def write_view_shard(
    arrays: Mapping[str, np.ndarray],
    sample_ids: list[str],
    target: Path,
    durable_manifest: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write a complete intermediate cache or validate and reuse an exact rerun."""

    records = _validate_arrays(arrays, sample_ids)
    if target.exists():
        manifest = validate_view_shard(target)
        if manifest["provenance"] != provenance or int(manifest["records"]) != records:
            raise ViewShardError("existing view shard provenance or count differs")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "artifact_dir": str(target),
            "artifact_manifest_sha256": sha256_file(target / "manifest.json"),
            "records": records,
            "files": manifest["files"],
            "provenance": provenance,
        }
        if durable_manifest.exists():
            if json.loads(durable_manifest.read_text(encoding="utf-8")) != summary:
                raise ViewShardError("durable view manifest differs from artifact")
        else:
            _atomic_json(summary, durable_manifest)
        return summary
    if durable_manifest.exists():
        raise ViewShardError("durable view manifest exists without artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.incomplete.", dir=target.parent))
    try:
        files: dict[str, Any] = {}
        for name, array in sorted(arrays.items()):
            path = temporary / f"{name}.npy"
            with path.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            files[path.name] = {
                "bytes": path.stat().st_size,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": sha256_file(path),
            }
        sample_path = temporary / "sample_ids.jsonl"
        with sample_path.open("w", encoding="utf-8") as handle:
            for sample_id in sample_ids:
                handle.write(json.dumps({"sample_id": sample_id}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        files[sample_path.name] = {
            "bytes": sample_path.stat().st_size,
            "records": records,
            "sha256": sha256_file(sample_path),
        }
        embedded = {
            "schema_version": 1,
            "status": "complete",
            "records": records,
            "arrays": sorted(arrays),
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
        manifest = validate_view_shard(target)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "artifact_dir": str(target),
            "artifact_manifest_sha256": sha256_file(target / "manifest.json"),
            "records": records,
            "files": manifest["files"],
            "provenance": provenance,
        }
        _atomic_json(summary, durable_manifest)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
