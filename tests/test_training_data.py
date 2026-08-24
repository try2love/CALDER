from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_fusion.feature_shard import FeatureShardPayload, write_feature_shard
from benchmark_fusion.training_data import (
    DistributedFullCoverageSampler,
    FeatureShardDataset,
    PaddedDistributedBatchSampler,
    build_balanced_training_index,
)


class TrainingDataTest(unittest.TestCase):
    def identities(self) -> list[dict[str, object]]:
        return [
            {
                "sample_id": f"s{index}",
                "label": index % 2,
                "dataset": "a" if index < 6 else "b",
                "split": "train",
                "source_component_id": f"c{index}",
                "input_sha256": "x",
                "feature_schema_sha256": "f",
                "model_protocol_sha256": "m",
                "domain": "d",
                "generator_family": "human" if index % 2 == 0 else "g",
                "text_length": index + 1,
            }
            for index in range(10)
        ]

    def test_balanced_order_is_deterministic_unique_and_distributed_without_loss(self) -> None:
        identities = self.identities()
        first = build_balanced_training_index(identities, seed=42, epoch=0)
        second = build_balanced_training_index(identities, seed=42, epoch=0)
        np.testing.assert_array_equal(first.order, second.order)
        np.testing.assert_array_equal(np.sort(first.order), np.arange(len(identities)))
        self.assertTrue(((first.loss_weights >= 0.25) & (first.loss_weights <= 4.0)).all())
        distributed = []
        for rank in range(3):
            distributed.extend(DistributedFullCoverageSampler(first.order, rank, 3))
        self.assertEqual(sorted(distributed), list(range(len(identities))))
        actual = []
        batch_counts = []
        for rank in range(3):
            batches = list(PaddedDistributedBatchSampler(first.order, 2, rank, 3))
            batch_counts.append(len(batches))
            actual.extend(index for batch in batches for index, weight in batch if weight == 1.0)
            self.assertTrue(all(len(batch) == 2 for batch in batches))
        self.assertEqual(batch_counts, [2, 2, 2])
        self.assertEqual(sorted(actual), list(range(len(identities))))

    def test_dataset_reads_multiple_validated_memory_mapped_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_dirs = []
            identities = self.identities()[:4]
            for shard_index in range(2):
                selected = identities[shard_index * 2 : (shard_index + 1) * 2]
                records = len(selected)
                arrays = {
                    "semantic_spans": np.ones((records, 16, 3), dtype=np.float16),
                    "semantic_mask": np.ones((records, 16), dtype=np.bool_),
                    "token_evidence": np.ones((records, 64, 10), dtype=np.float16),
                    "token_mask": np.ones((records, 64), dtype=np.bool_),
                    "probability": np.ones((records, 108), dtype=np.float32),
                    "alignment": np.ones((records, 32), dtype=np.float32),
                    "compression": np.ones((records, 51), dtype=np.float32),
                    "detector_score": np.ones((records, 7), dtype=np.float32),
                    "labels": np.asarray([item["label"] for item in selected], dtype=np.int8),
                }
                target = root / f"shard{shard_index}"
                write_feature_shard(
                    FeatureShardPayload(arrays=arrays, identities=selected),
                    target,
                    root / f"shard{shard_index}.json",
                    {"shard": shard_index},
                )
                shard_dirs.append(target)
            dataset = FeatureShardDataset(shard_dirs)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(int(dataset[3]["labels"]), 1)
            self.assertEqual(tuple(dataset[0]["probability"].shape), (108,))
            self.assertEqual(float(dataset[(0, 0.0)]["loss_multiplier"]), 0.0)


if __name__ == "__main__":
    unittest.main()
