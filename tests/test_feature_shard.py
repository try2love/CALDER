from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_fusion.feature_shard import (
    FeatureShardError,
    FeatureShardPayload,
    validate_feature_shard,
    write_feature_shard,
)


def fixture_payload(records: int = 2) -> FeatureShardPayload:
    arrays = {
        "semantic_spans": np.ones((records, 16, 4), dtype=np.float16),
        "semantic_mask": np.ones((records, 16), dtype=np.bool_),
        "token_evidence": np.ones((records, 64, 10), dtype=np.float16),
        "token_mask": np.ones((records, 64), dtype=np.bool_),
        "probability": np.ones((records, 108), dtype=np.float32),
        "alignment": np.ones((records, 32), dtype=np.float32),
        "compression": np.ones((records, 51), dtype=np.float32),
        "detector_score": np.ones((records, 7), dtype=np.float32),
        "labels": np.asarray([index % 2 for index in range(records)], dtype=np.int8),
    }
    identities = [
        {
            "sample_id": f"sample-{index}",
            "label": int(arrays["labels"][index]),
            "dataset": "fixture",
            "split": "train",
            "source_component_id": f"component-{index}",
            "input_sha256": "a" * 64,
            "feature_schema_sha256": "b" * 64,
            "model_protocol_sha256": "c" * 64,
        }
        for index in range(records)
    ]
    return FeatureShardPayload(arrays=arrays, identities=identities)


class FeatureShardTest(unittest.TestCase):
    def test_atomic_roundtrip_and_complete_rerun_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "shard-00000"
            durable = root / "durable.json"
            first = write_feature_shard(fixture_payload(), target, durable, {"run_id": "x"})
            second = write_feature_shard(fixture_payload(), target, durable, {"run_id": "x"})
            self.assertEqual(first, second)
            self.assertEqual(validate_feature_shard(target)["records"], 2)
            self.assertEqual(json.loads(durable.read_text())["status"], "complete")

    def test_nonfinite_feature_fails_before_artifact_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = fixture_payload()
            payload.arrays["probability"][0, 0] = np.nan
            with self.assertRaisesRegex(FeatureShardError, "non-finite"):
                write_feature_shard(payload, root / "shard", root / "durable.json", {})
            self.assertFalse((root / "shard").exists())

    def test_partial_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "shard"
            target.mkdir()
            with self.assertRaisesRegex(FeatureShardError, "incomplete"):
                write_feature_shard(fixture_payload(), target, root / "durable.json", {})

    def test_complete_target_rejects_provenance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "shard"
            durable = root / "durable.json"
            write_feature_shard(fixture_payload(), target, durable, {"run_id": "one"})
            with self.assertRaisesRegex(FeatureShardError, "provenance differs"):
                write_feature_shard(
                    fixture_payload(), target, durable, {"run_id": "different"}
                )


if __name__ == "__main__":
    unittest.main()
