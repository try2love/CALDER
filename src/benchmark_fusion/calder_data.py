"""Validated zero-copy CALDER views over legacy frozen feature shards."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_shard import validate_feature_shard
from .identity import sha256_file
from .view_shard import validate_view_shard


NULL_VIEW_SCHEMA = {
    "null_evidence": ("float16", (64, 2)),
    "token_mask": ("bool", (64,)),
    "null_summary": ("float32", (14,)),
}


def load_calder_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ValueError(f"CALDER feature manifest is not complete: {path}")
    entries = payload.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("CALDER feature manifest has no shards")
    records = sum(int(entry["records"]) for entry in entries)
    if records != int(payload["records"]):
        raise ValueError("CALDER feature manifest record count differs")
    return payload


class CalderDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        partition: str | None = None,
        validate: bool = True,
    ) -> None:
        payload = load_calder_manifest(manifest_path)
        partition_map: dict[str, str] = {}
        partition_identity = payload.get("partition_membership")
        if partition is not None:
            if not isinstance(partition_identity, dict):
                raise ValueError("partitioned CALDER dataset lacks membership identity")
            membership_path = Path(partition_identity["path"])
            if sha256_file(membership_path) != partition_identity["sha256"]:
                raise ValueError("CALDER partition membership hash differs")
            with membership_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    partition_map[str(row["sample_id"])] = str(row["partition"])
            if partition not in set(partition_map.values()):
                raise ValueError(f"unknown CALDER partition: {partition}")

        self.shards: list[dict[str, np.ndarray]] = []
        self.ends: list[int] = []
        self.identities: list[dict[str, object]] = []
        self.lookup: list[tuple[int, int]] = []
        total = 0
        for entry in payload["shards"]:
            feature_dir = Path(entry["feature_dir"])
            llama_dir = Path(entry["llama_null_dir"])
            falcon_dir = Path(entry["falcon_null_dir"])
            if validate:
                feature_manifest = validate_feature_shard(feature_dir)
                llama_manifest = validate_view_shard(llama_dir, NULL_VIEW_SCHEMA)
                falcon_manifest = validate_view_shard(falcon_dir, NULL_VIEW_SCHEMA)
                expected = int(entry["records"])
                if any(int(item["records"]) != expected for item in (
                    feature_manifest, llama_manifest, falcon_manifest
                )):
                    raise ValueError("CALDER source view record count differs")
            old = {
                name: np.load(feature_dir / f"{name}.npy", allow_pickle=False, mmap_mode="r")
                for name in (
                    "semantic_spans", "semantic_mask", "token_evidence", "token_mask",
                    "probability", "alignment", "compression", "labels",
                )
            }
            llama = {
                name: np.load(llama_dir / f"{name}.npy", allow_pickle=False, mmap_mode="r")
                for name in NULL_VIEW_SCHEMA
            }
            falcon = {
                name: np.load(falcon_dir / f"{name}.npy", allow_pickle=False, mmap_mode="r")
                for name in NULL_VIEW_SCHEMA
            }
            identities = []
            with (feature_dir / "identities.jsonl").open("r", encoding="utf-8") as handle:
                identities.extend(json.loads(line) for line in handle)
            feature_ids = [str(item["sample_id"]) for item in identities]
            for null_dir in (llama_dir, falcon_dir):
                with (null_dir / "sample_ids.jsonl").open("r", encoding="utf-8") as handle:
                    null_ids = [str(json.loads(line)["sample_id"]) for line in handle]
                if null_ids != feature_ids:
                    raise ValueError("CALDER null-view sample order differs from feature shard")
            shard_index = len(self.shards)
            self.shards.append(old | {
                "llama_null_evidence": llama["null_evidence"],
                "llama_null_mask": llama["token_mask"],
                "llama_null_summary": llama["null_summary"],
                "falcon_null_evidence": falcon["null_evidence"],
                "falcon_null_mask": falcon["token_mask"],
                "falcon_null_summary": falcon["null_summary"],
            })
            for local_index, identity in enumerate(identities):
                sample_id = str(identity["sample_id"])
                if partition is None or partition_map.get(sample_id) == partition:
                    self.lookup.append((shard_index, local_index))
                    self.identities.append(identity)
            total += len(identities)
            self.ends.append(total)
        sample_ids = [str(identity["sample_id"]) for identity in self.identities]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("duplicate sample ID in CALDER dataset")
        if partition is not None and not self.lookup:
            raise ValueError("CALDER partition is empty")

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, index: int | tuple[int, float]) -> dict[str, torch.Tensor]:
        multiplier = 1.0
        if isinstance(index, tuple):
            index, multiplier = index
        if not 0 <= index < len(self.lookup):
            raise IndexError(index)
        shard_index, local_index = self.lookup[index]
        arrays = self.shards[shard_index]
        token = np.asarray(arrays["token_evidence"][local_index], dtype=np.float32)
        llama_null = np.asarray(arrays["llama_null_evidence"][local_index], dtype=np.float32)
        falcon_null = np.asarray(arrays["falcon_null_evidence"][local_index], dtype=np.float32)
        token_probability = token[:, (0, 1, 2, 5, 6, 7)]
        alignment_evidence = np.concatenate(
            (
                token[:, 3:5], llama_null,
                token[:, 8:10], falcon_null,
            ),
            axis=1,
        )
        old_alignment = np.asarray(arrays["alignment"][local_index], dtype=np.float32)
        document_alignment = np.concatenate(
            (
                old_alignment[0:14],
                np.asarray(arrays["llama_null_summary"][local_index], dtype=np.float32),
                old_alignment[16:30],
                np.asarray(arrays["falcon_null_summary"][local_index], dtype=np.float32),
            )
        )
        mask = (
            np.asarray(arrays["token_mask"][local_index], dtype=np.bool_)
            & np.asarray(arrays["llama_null_mask"][local_index], dtype=np.bool_)
            & np.asarray(arrays["falcon_null_mask"][local_index], dtype=np.bool_)
        )
        if not mask.any():
            raise ValueError("CALDER record has no shared valid token span")
        return {
            "semantic_spans": torch.from_numpy(
                np.array(arrays["semantic_spans"][local_index], copy=True)
            ),
            "semantic_mask": torch.from_numpy(
                np.array(arrays["semantic_mask"][local_index], copy=True)
            ),
            "token_probability": torch.from_numpy(np.array(token_probability, copy=True)),
            "alignment_evidence": torch.from_numpy(np.array(alignment_evidence, copy=True)),
            "token_mask": torch.from_numpy(np.array(mask, copy=True)),
            "document_probability": torch.from_numpy(
                np.array(arrays["probability"][local_index], dtype=np.float32, copy=True)
            ),
            "document_alignment": torch.from_numpy(
                np.array(document_alignment, dtype=np.float32, copy=True)
            ),
            "compression": torch.from_numpy(
                np.array(arrays["compression"][local_index], dtype=np.float32, copy=True)
            ),
            "labels": torch.tensor(int(arrays["labels"][local_index]), dtype=torch.int8),
            "index": torch.tensor(index, dtype=torch.int64),
            "loss_multiplier": torch.tensor(multiplier, dtype=torch.float32),
        }


def scalar_training_moments(dataset: CalderDataset) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values: dict[str, list[np.ndarray]] = {
        "document_probability": [],
        "document_alignment": [],
        "compression": [],
    }
    for index in range(len(dataset)):
        row = dataset[index]
        for name in values:
            values[name].append(row[name].numpy())
    result = {}
    for name, rows in values.items():
        matrix = np.stack(rows).astype(np.float64)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        result[name] = (mean.astype(np.float32), scale.astype(np.float32))
    return result
