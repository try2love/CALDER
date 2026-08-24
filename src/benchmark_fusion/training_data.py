"""Memory-mapped feature dataset and deterministic full-coverage balancing."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset, Sampler

from .feature_shard import ARRAY_DTYPES, validate_feature_shard


@dataclass(frozen=True)
class BalancedTrainingIndex:
    order: np.ndarray
    loss_weights: np.ndarray
    length_quartiles: dict[str, tuple[float, float, float]]
    stratum_counts: dict[str, int]


def _stratum_seed(seed: int, epoch: int, key: tuple[object, ...]) -> int:
    serialized = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(f"{seed}\0{epoch}\0{serialized}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_balanced_training_index(
    identities: Sequence[dict[str, object]], seed: int, epoch: int
) -> BalancedTrainingIndex:
    if not identities:
        raise ValueError("training identities must be non-empty")
    lengths_by_dataset: dict[str, list[int]] = defaultdict(list)
    for identity in identities:
        if identity.get("split") != "train":
            raise ValueError("balanced training index may only use train identities")
        dataset = str(identity["dataset"])
        length = int(identity["text_length"])
        if length < 1:
            raise ValueError("training text length must be positive")
        lengths_by_dataset[dataset].append(length)
    quartiles = {
        dataset: tuple(
            float(value)
            for value in np.quantile(values, [0.25, 0.50, 0.75], method="linear")
        )
        for dataset, values in lengths_by_dataset.items()
    }
    strata: dict[tuple[object, ...], list[int]] = defaultdict(list)
    sample_strata = []
    for index, identity in enumerate(identities):
        dataset = str(identity["dataset"])
        length_bin = int(np.searchsorted(quartiles[dataset], int(identity["text_length"]), side="right"))
        key = (
            dataset,
            int(identity["label"]),
            str(identity.get("domain", "unknown")),
            str(identity.get("generator_family", "unknown")),
            length_bin,
        )
        strata[key].append(index)
        sample_strata.append(key)
    keys = sorted(strata, key=lambda item: json.dumps(item, ensure_ascii=False))
    shuffled: dict[tuple[object, ...], np.ndarray] = {}
    for key in keys:
        indices = np.asarray(strata[key], dtype=np.int64)
        np.random.default_rng(_stratum_seed(seed, epoch, key)).shuffle(indices)
        shuffled[key] = indices
    order = np.empty(len(identities), dtype=np.int64)
    pointers = {key: 0 for key in keys}
    cursor = 0
    while cursor < len(order):
        for key in keys:
            pointer = pointers[key]
            values = shuffled[key]
            if pointer < len(values):
                order[cursor] = values[pointer]
                pointers[key] = pointer + 1
                cursor += 1
    counts = {key: len(values) for key, values in strata.items()}
    raw = np.asarray([1.0 / math.sqrt(counts[key]) for key in sample_strata], dtype=np.float64)
    weights = np.clip(raw / raw.mean(), 0.25, 4.0).astype(np.float32)
    return BalancedTrainingIndex(
        order=order,
        loss_weights=weights,
        length_quartiles=quartiles,
        stratum_counts={json.dumps(key, ensure_ascii=False): count for key, count in counts.items()},
    )


class DistributedFullCoverageSampler(Sampler[int]):
    def __init__(self, order: np.ndarray, rank: int = 0, world_size: int = 1) -> None:
        if order.ndim != 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid global order or distributed rank")
        if len(order) != len(np.unique(order)):
            raise ValueError("global order must contain unique full-coverage indices")
        self.local_order = order[rank::world_size]

    def __iter__(self) -> Iterator[int]:
        return (int(index) for index in self.local_order)

    def __len__(self) -> int:
        return len(self.local_order)


class PaddedDistributedBatchSampler(BatchSampler):
    """Equal-step DDP batches; padding records carry zero training weight."""

    def __init__(
        self,
        order: np.ndarray,
        micro_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        global_cursor: int = 0,
    ) -> None:
        if (
            order.ndim != 1
            or len(order) < 1
            or micro_batch_size < 1
            or world_size < 1
            or not 0 <= rank < world_size
            or not 0 <= global_cursor <= len(order)
        ):
            raise ValueError("invalid padded distributed batch sampler arguments")
        self.order = order
        self.micro_batch_size = micro_batch_size
        self.rank = rank
        self.world_size = world_size
        self.global_cursor = global_cursor
        self.global_batch_size = micro_batch_size * world_size

    def __iter__(self) -> Iterator[list[tuple[int, float]]]:
        fallback = int(self.order[0])
        for start in range(self.global_cursor, len(self.order), self.global_batch_size):
            chunk = self.order[start : start + self.global_batch_size]
            local = [int(value) for value in chunk[self.rank :: self.world_size]]
            batch = [(value, 1.0) for value in local]
            batch.extend((fallback, 0.0) for _ in range(self.micro_batch_size - len(batch)))
            yield batch

    def __len__(self) -> int:
        remaining = len(self.order) - self.global_cursor
        return math.ceil(remaining / self.global_batch_size)


class FeatureShardDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, shard_dirs: Sequence[Path], validate: bool = True) -> None:
        if not shard_dirs:
            raise ValueError("at least one feature shard is required")
        self.shards = []
        self.ends = []
        self.identities: list[dict[str, object]] = []
        total = 0
        for shard_dir in shard_dirs:
            manifest = validate_feature_shard(shard_dir) if validate else json.loads(
                (shard_dir / "manifest.json").read_text(encoding="utf-8")
            )
            arrays = {
                name: np.load(shard_dir / f"{name}.npy", allow_pickle=False, mmap_mode="r")
                for name in ARRAY_DTYPES
            }
            identities = [
                json.loads(line)
                for line in (shard_dir / "identities.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            if len(identities) != int(manifest["records"]):
                raise ValueError(f"identity count mismatch in {shard_dir}")
            self.shards.append(arrays)
            self.identities.extend(identities)
            total += int(manifest["records"])
            self.ends.append(total)
        sample_ids = [str(identity["sample_id"]) for identity in self.identities]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("duplicate sample ID across feature shards")

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int | tuple[int, float]) -> dict[str, torch.Tensor]:
        multiplier = 1.0
        if isinstance(index, tuple):
            index, multiplier = index
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        local_index = index - start
        return {
            name: torch.from_numpy(np.array(array[local_index], copy=True))
            for name, array in self.shards[shard_index].items()
        } | {
            "index": torch.tensor(index, dtype=torch.int64),
            "loss_multiplier": torch.tensor(multiplier, dtype=torch.float32),
        }
