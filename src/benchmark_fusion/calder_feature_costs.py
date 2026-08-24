"""Audit and aggregate the one-time shared CALDER feature-extraction cost."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .identity import sha256_file


VIEW_NAMES = {
    "semantic": "semantic",
    "llama": "llama2_base_chat",
    "falcon": "falcon_base_instruct",
    "gpt2": "gpt2",
    "gpt2_large": "gpt2_large",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_devices(worker: object) -> int:
    devices = [item.strip() for item in str(worker).split(",") if item.strip()]
    if not devices:
        raise ValueError("feature runtime has no worker devices")
    return len(devices)


def _add_counter(target: Counter[str], value: object) -> None:
    if isinstance(value, Mapping):
        for key, count in value.items():
            target[str(key)] += int(count)
    else:
        target["model"] += int(value)


def _runtime_path(
    view_manifest: Path,
    runtime_maps: Iterable[tuple[Path, Path]],
) -> Path:
    matches = []
    for artifact_prefix, runtime_root in runtime_maps:
        try:
            relative = view_manifest.parent.relative_to(artifact_prefix.resolve())
        except ValueError:
            continue
        matches.append((len(artifact_prefix.resolve().parts), runtime_root.resolve(), relative))
    if not matches:
        raise ValueError(f"no runtime mapping covers CALDER view: {view_manifest}")
    _, root, relative = max(matches, key=lambda item: item[0])
    return root / relative.parent / f"{relative.name}.json"


def _empty_view() -> dict[str, Any]:
    return {
        "records": 0,
        "shards": 0,
        "gpu_hours": 0.0,
        "elapsed_seconds_sum": 0.0,
        "peak_gpu_memory_bytes": 0,
        "observer_forward_calls": Counter(),
        "observer_record_forwards": Counter(),
        "runtime_manifests": [],
    }


def build_calder_feature_cost_registry(
    *,
    feature_scopes: Iterable[tuple[str, Path]],
    runtime_maps: Iterable[tuple[Path, Path]],
    null_manifests: Iterable[tuple[str, Path]],
    expected_records: int,
) -> dict[str, Any]:
    """Validate exact feature lineage and count each shared extraction once."""

    mappings = list(runtime_maps)
    if not mappings:
        raise ValueError("at least one CALDER runtime mapping is required")
    base_views = {name: _empty_view() for name in VIEW_NAMES.values()}
    coverage = {name: set() for name in VIEW_NAMES.values()}
    feature_scope_identities = []
    seen_scopes: set[Path] = set()
    seen_views: set[Path] = set()
    unique_inputs: set[tuple[str, int]] = set()

    for scope_name, scope_path in feature_scopes:
        resolved_scope = scope_path.resolve()
        if resolved_scope in seen_scopes:
            raise ValueError(f"duplicate CALDER feature scope: {scope_path}")
        seen_scopes.add(resolved_scope)
        scope = _json(scope_path)
        if scope.get("status") != "complete" or int(scope.get("records", -1)) < 1:
            raise ValueError(f"CALDER feature scope is incomplete: {scope_path}")
        observed = 0
        for shard in scope.get("shards", []):
            records = int(shard["records"])
            input_identity = (str(shard["input_sha256"]), records)
            unique_inputs.add(input_identity)
            artifact_dir = Path(str(shard.get("artifact_dir") or shard.get("feature_dir")))
            artifact_manifest = artifact_dir / "manifest.json"
            expected_hash = str(
                shard.get("artifact_manifest_sha256") or shard.get("feature_manifest_sha256")
            )
            if sha256_file(artifact_manifest) != expected_hash:
                raise ValueError(f"assembled CALDER feature hash differs: {artifact_manifest}")
            assembled = _json(artifact_manifest)
            if assembled.get("status") != "complete" or int(assembled.get("records", -1)) != records:
                raise ValueError(f"assembled CALDER feature identity differs: {artifact_manifest}")
            view_manifests = assembled.get("provenance", {}).get("view_manifests", {})
            if set(view_manifests) != set(VIEW_NAMES):
                raise ValueError(f"CALDER base-view set differs: {artifact_manifest}")
            for raw_name, registered_name in VIEW_NAMES.items():
                identity = view_manifests[raw_name]
                view_path = Path(str(identity["path"])).resolve()
                if sha256_file(view_path) != identity["sha256"]:
                    raise ValueError(f"CALDER view hash differs: {view_path}")
                view = _json(view_path)
                if (
                    view.get("status") != "complete"
                    or int(view.get("records", -1)) != records
                    or view.get("provenance", {}).get("input_sha256") != input_identity[0]
                ):
                    raise ValueError(f"CALDER view identity differs: {view_path}")
                if view_path in seen_views:
                    if input_identity not in coverage[registered_name]:
                        raise ValueError(f"CALDER view path reused for a different input: {view_path}")
                    continue
                seen_views.add(view_path)
                runtime_path = _runtime_path(view_path, mappings)
                runtime = _json(runtime_path)
                if (
                    runtime.get("status") != "complete"
                    or int(runtime.get("records", -1)) != records
                    or runtime.get("input_sha256") != input_identity[0]
                ):
                    raise ValueError(f"CALDER feature runtime identity differs: {runtime_path}")
                values = runtime.get("runtime", {})
                elapsed = float(values["elapsed_seconds"])
                devices = _parse_devices(runtime.get("worker"))
                summary = base_views[registered_name]
                summary["records"] += records
                summary["shards"] += 1
                summary["elapsed_seconds_sum"] += elapsed
                summary["gpu_hours"] += elapsed * devices / 3600.0
                summary["peak_gpu_memory_bytes"] = max(
                    summary["peak_gpu_memory_bytes"],
                    int(values.get("peak_gpu_memory_bytes", 0)),
                )
                _add_counter(summary["observer_forward_calls"], values["observer_forward_calls"])
                _add_counter(summary["observer_record_forwards"], values["observer_record_forwards"])
                summary["runtime_manifests"].append({
                    "path": str(runtime_path.resolve()),
                    "sha256": sha256_file(runtime_path),
                })
                coverage[registered_name].add(input_identity)
            observed += records
        if observed != int(scope["records"]):
            raise ValueError(f"CALDER feature scope record count differs: {scope_path}")
        feature_scope_identities.append({
            "name": scope_name,
            "path": str(resolved_scope),
            "sha256": sha256_file(scope_path),
            "records": int(scope["records"]),
        })

    if len({frozenset(items) for items in coverage.values()}) != 1:
        raise ValueError("CALDER base views do not cover the same frozen inputs")
    if {sum(records for _, records in items) for items in coverage.values()} != {expected_records}:
        raise ValueError("CALDER base views do not cover the expected record total")
    if sum(records for _, records in unique_inputs) != expected_records:
        raise ValueError("CALDER feature scopes contain duplicated frozen inputs")

    null_views = {"llama2_base_chat": _empty_view(), "falcon_base_instruct": _empty_view()}
    null_identities = []
    seen_null: set[Path] = set()
    for scope_name, manifest_path in null_manifests:
        resolved = manifest_path.resolve()
        if resolved in seen_null:
            raise ValueError(f"duplicate CALDER null manifest: {manifest_path}")
        seen_null.add(resolved)
        payload = _json(manifest_path)
        pair = str(payload.get("provenance", {}).get("pair"))
        if pair not in null_views or payload.get("status") != "complete":
            raise ValueError(f"CALDER null manifest identity differs: {manifest_path}")
        records = int(payload.get("records", -1))
        artifact_manifest = Path(str(payload["artifact_dir"])) / "manifest.json"
        if sha256_file(artifact_manifest) != payload["artifact_manifest_sha256"]:
            raise ValueError(f"CALDER null artifact hash differs: {artifact_manifest}")
        runtime = payload.get("provenance", {}).get("runtime", {})
        elapsed = float(runtime["elapsed_seconds"])
        memory_by_device = runtime["peak_gpu_memory_bytes_by_device"]
        devices = len(memory_by_device)
        if devices != 2:
            raise ValueError(f"CALDER null extraction must record two observer devices: {manifest_path}")
        calls_per_model = int(runtime["observer_forward_calls_per_model"])
        summary = null_views[pair]
        summary["records"] += records
        summary["shards"] += 1
        summary["elapsed_seconds_sum"] += elapsed
        summary["gpu_hours"] += elapsed * devices / 3600.0
        summary["peak_gpu_memory_bytes"] = max(
            summary["peak_gpu_memory_bytes"], *(int(value) for value in memory_by_device.values())
        )
        summary["observer_forward_calls"]["base"] += calls_per_model
        summary["observer_forward_calls"]["aligned"] += calls_per_model
        summary["observer_record_forwards"]["base"] += records
        summary["observer_record_forwards"]["aligned"] += records
        summary["runtime_manifests"].append({
            "path": str(resolved),
            "sha256": sha256_file(manifest_path),
        })
        null_identities.append({
            "name": scope_name,
            "pair": pair,
            "path": str(resolved),
            "sha256": sha256_file(manifest_path),
            "records": records,
        })
    if {int(summary["records"]) for summary in null_views.values()} != {expected_records}:
        raise ValueError("CALDER null views do not cover the expected record total")

    all_views = {**base_views, **{f"null_{key}": value for key, value in null_views.items()}}
    for summary in all_views.values():
        summary["observer_forward_calls"] = dict(sorted(summary["observer_forward_calls"].items()))
        summary["observer_record_forwards"] = dict(sorted(summary["observer_record_forwards"].items()))
        summary["runtime_manifests"].sort(key=lambda item: item["path"])
    total_calls: Counter[str] = Counter()
    total_record_forwards: Counter[str] = Counter()
    for name, summary in all_views.items():
        for observer, count in summary["observer_forward_calls"].items():
            total_calls[f"{name}:{observer}"] += int(count)
        for observer, count in summary["observer_record_forwards"].items():
            total_record_forwards[f"{name}:{observer}"] += int(count)
    return {
        "schema_version": 1,
        "status": "complete",
        "definition": "one-time hash-deduplicated CALDER mainline feature extraction; hyperparameter-search-only runs excluded",
        "records_per_view": expected_records,
        "unique_input_shards": len(unique_inputs),
        "feature_scopes": sorted(feature_scope_identities, key=lambda item: item["name"]),
        "null_manifests": sorted(null_identities, key=lambda item: item["name"]),
        "base_views": dict(sorted(base_views.items())),
        "null_views": dict(sorted(null_views.items())),
        "total_gpu_hours": sum(float(summary["gpu_hours"]) for summary in all_views.values()),
        "maximum_peak_gpu_memory_bytes": max(
            int(summary["peak_gpu_memory_bytes"]) for summary in all_views.values()
        ),
        "observer_forward_calls": dict(sorted(total_calls.items())),
        "observer_forward_calls_total": sum(total_calls.values()),
        "observer_record_forwards": dict(sorted(total_record_forwards.items())),
        "observer_record_forwards_total": sum(total_record_forwards.values()),
    }
