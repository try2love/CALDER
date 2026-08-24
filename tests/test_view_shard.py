from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_fusion.view_shard import ViewShardError, validate_view_shard, write_view_shard


class ViewShardTest(unittest.TestCase):
    def payload(self) -> tuple[dict[str, np.ndarray], list[str]]:
        return {
            "states": np.arange(12, dtype=np.float16).reshape(2, 2, 3),
            "mask": np.ones((2, 2), dtype=np.bool_),
        }, ["a", "b"]

    def test_atomic_round_trip_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays, sample_ids = self.payload()
            summary = write_view_shard(arrays, sample_ids, root / "view", root / "view.json", {"run": "x"})
            self.assertEqual(summary["records"], 2)
            manifest = validate_view_shard(
                root / "view",
                {"states": ("float16", (2, 3)), "mask": ("bool", (2,))},
            )
            self.assertEqual(manifest["provenance"], {"run": "x"})
            self.assertEqual(json.loads((root / "view.json").read_text())["status"], "complete")

    def test_complete_rerun_is_idempotent_but_provenance_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays, sample_ids = self.payload()
            first = write_view_shard(arrays, sample_ids, root / "view", root / "view.json", {"run": "x"})
            second = write_view_shard(arrays, sample_ids, root / "view", root / "view.json", {"run": "x"})
            self.assertEqual(first, second)
            with self.assertRaises(ViewShardError):
                write_view_shard(arrays, sample_ids, root / "view", root / "view.json", {"run": "y"})

    def test_corruption_and_nonfinite_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays, sample_ids = self.payload()
            write_view_shard(arrays, sample_ids, root / "view", root / "view.json", {"run": "x"})
            with (root / "view" / "states.npy").open("ab") as handle:
                handle.write(b"bad")
            with self.assertRaises(ViewShardError):
                validate_view_shard(root / "view")
            arrays["states"][0, 0, 0] = np.nan
            with self.assertRaises(ViewShardError):
                write_view_shard(arrays, sample_ids, root / "other", root / "other.json", {"run": "x"})


if __name__ == "__main__":
    unittest.main()
