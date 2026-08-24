"""Training run identity, checkpoint selection, and RNG state helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_fusion.identity import sha256_file


@dataclass(frozen=True)
class EffectiveEpochSchedule:
    configured_epochs: int
    effective_epochs: int
    steps_per_epoch: int
    minimum_optimizer_steps: int | None


def effective_epoch_schedule(
    run_spec: dict[str, Any],
    *,
    train_records: int,
    global_batch_size: int,
    configured_epochs: int,
) -> EffectiveEpochSchedule:
    """Apply the frozen few-shot update floor without altering full-data fidelity."""

    if train_records < 1 or global_batch_size < 1 or configured_epochs < 1:
        raise ValueError("training schedule inputs must be positive")
    steps_per_epoch = math.ceil(train_records / global_batch_size)
    training_budget = run_spec.get("training_budget")
    if training_budget is None:
        return EffectiveEpochSchedule(
            configured_epochs=configured_epochs,
            effective_epochs=configured_epochs,
            steps_per_epoch=steps_per_epoch,
            minimum_optimizer_steps=None,
        )
    identity = run_spec.get("fewshot_protocol")
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ValueError("few-shot run spec lacks a valid fewshot_protocol identity")
    path = Path(identity["path"])
    if sha256_file(path) != identity["sha256"]:
        raise ValueError("few-shot protocol hash differs")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_formal_sampling":
        raise ValueError("few-shot protocol is not frozen before formal sampling")
    if training_budget == "full":
        return EffectiveEpochSchedule(
            configured_epochs=configured_epochs,
            effective_epochs=configured_epochs,
            steps_per_epoch=steps_per_epoch,
            minimum_optimizer_steps=None,
        )
    if not isinstance(training_budget, int) or isinstance(training_budget, bool) or training_budget < 1:
        raise ValueError("training_budget must be a positive integer or full")
    optimization = protocol.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("few-shot protocol lacks optimization policy")
    minimum_steps = int(optimization.get("minimum_optimizer_steps_per_trainable_stage", 0))
    if minimum_steps < 1:
        raise ValueError("few-shot minimum optimizer steps must be positive")
    effective_epochs = max(configured_epochs, math.ceil(minimum_steps / steps_per_epoch))
    return EffectiveEpochSchedule(
        configured_epochs=configured_epochs,
        effective_epochs=effective_epochs,
        steps_per_epoch=steps_per_epoch,
        minimum_optimizer_steps=minimum_steps,
    )


def canonical_run_id(run_spec: dict[str, Any]) -> str:
    payload = json.dumps(
        run_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return f"bfv1:{hashlib.sha256(payload).hexdigest()[:24]}"


def validate_implementation_manifest(run_spec: dict[str, Any]) -> dict[str, Any] | None:
    """Fail closed on code drift for formal runs; diagnostics may remain unpinned."""

    identity = run_spec.get("implementation_manifest")
    if identity is None:
        if run_spec.get("status") == "engineering_only":
            return None
        raise ValueError("formal run spec lacks implementation_manifest")
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ValueError("implementation_manifest identity must contain only path and sha256")
    manifest_path = Path(identity["path"])
    if sha256_file(manifest_path) != identity["sha256"]:
        raise ValueError("implementation manifest hash differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or not manifest.get("git_commit"):
        raise ValueError("implementation manifest is not frozen to a Git commit")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("implementation manifest has no files")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("implementation manifest file entry schema differs")
        path = Path(entry["path"])
        if str(path) in seen:
            raise ValueError("implementation manifest contains duplicate paths")
        seen.add(str(path))
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"implementation file differs: {path}")
    return identity


@dataclass(frozen=True)
class CheckpointScore:
    auroc: float
    auprc: float
    binary_cross_entropy: float
    epoch: int


def checkpoint_is_better(candidate: CheckpointScore, incumbent: CheckpointScore | None) -> bool:
    if incumbent is None:
        return True
    if candidate.auroc != incumbent.auroc:
        return candidate.auroc > incumbent.auroc
    if candidate.auprc != incumbent.auprc:
        return candidate.auprc > incumbent.auprc
    if candidate.binary_cross_entropy != incumbent.binary_cross_entropy:
        return candidate.binary_cross_entropy < incumbent.binary_cross_entropy
    return candidate.epoch < incumbent.epoch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(state["torch_cuda"])


def prune_complete_checkpoints(
    checkpoint_directory: Path, current: Path, *, retain: int = 2
) -> list[Path]:
    """Keep a bounded rolling set after the durable latest pointer is committed."""

    if retain < 1 or not current.is_file() or current.parent != checkpoint_directory:
        raise ValueError("invalid checkpoint retention request")
    candidates = sorted(
        checkpoint_directory.glob("*.pt"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    protected = {current}
    for path in candidates:
        if len(protected) >= retain:
            break
        protected.add(path)
    removed = []
    for path in candidates:
        if path not in protected:
            path.unlink()
            removed.append(path)
    return removed
