from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_fusion.identity import (
    IdentityError,
    build_records,
    load_declared_texts,
    normalize_text,
    sha256_file,
    write_jsonl_atomic,
)


class IdentityTest(unittest.TestCase):
    def test_normalization_is_nfkc_and_whitespace_only(self) -> None:
        self.assertEqual(normalize_text("Ａ\n  B\tC"), "A B C")

    def test_declared_source_fails_closed_on_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(json.dumps(["one"]), encoding="utf-8")
            spec = {
                "path": str(path),
                "sha256": "0" * 64,
                "container": "list",
                "count": 1,
            }
            with self.assertRaisesRegex(IdentityError, "hash mismatch"):
                load_declared_texts(spec)

    def test_build_records_has_stable_labels_and_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(json.dumps(["same", "same"]), encoding="utf-8")
            spec = {
                "path": str(path),
                "sha256": sha256_file(path),
                "container": "list",
                "count": 2,
                "dataset": "fixture",
                "label": 1,
                "source_uri": "fixture://source",
                "source_version": "v1",
            }
            records = build_records([spec], lambda text: len(text), {"id": "fixture"})
            self.assertEqual([item["label"] for item in records], [1, 1])
            self.assertEqual(len({item["sample_id"] for item in records}), 2)
            self.assertEqual(records[0]["text_sha256"], records[1]["text_sha256"])

    def test_atomic_jsonl_has_deterministic_sorted_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.jsonl"
            write_jsonl_atomic([{"z": 1, "a": 2}], output)
            self.assertEqual(output.read_text(encoding="utf-8"), '{"a": 2, "z": 1}\n')
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(), sha256_file(output)
            )


if __name__ == "__main__":
    unittest.main()
