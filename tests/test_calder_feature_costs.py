from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmark_fusion.calder_feature_costs import build_calder_feature_cost_registry


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CalderFeatureCostsTest(unittest.TestCase):
    def test_hash_deduplicated_base_and_null_costs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_prefix = root / "artifacts/views"
            runtime_root = root / "runtimes"
            input_sha = "a" * 64
            view_identities = {}
            workers = {
                "semantic": "cuda:0",
                "llama": "cuda:0,cuda:1",
                "falcon": "cuda:2,cuda:3",
                "gpt2": "cuda:4",
                "gpt2_large": "cuda:5",
            }
            for index, (view, worker) in enumerate(workers.items()):
                view_path = artifact_prefix / view / "shard" / "manifest.json"
                write_json(view_path, {
                    "status": "complete",
                    "records": 3,
                    "provenance": {"input_sha256": input_sha},
                })
                runtime_path = runtime_root / view / "shard.json"
                write_json(runtime_path, {
                    "status": "complete",
                    "records": 3,
                    "input_sha256": input_sha,
                    "worker": worker,
                    "runtime": {
                        "elapsed_seconds": 3600,
                        "peak_gpu_memory_bytes": 100 + index,
                        "observer_forward_calls": {"model": 1},
                        "observer_record_forwards": {"model": 3},
                    },
                })
                view_identities[view] = {"path": str(view_path), "sha256": digest(view_path)}
            assembled_dir = root / "assembled"
            assembled = assembled_dir / "manifest.json"
            write_json(assembled, {
                "status": "complete",
                "records": 3,
                "provenance": {"view_manifests": view_identities},
            })
            scope = root / "scope.json"
            write_json(scope, {
                "status": "complete",
                "records": 3,
                "shards": [{
                    "records": 3,
                    "input_sha256": input_sha,
                    "artifact_dir": str(assembled_dir),
                    "artifact_manifest_sha256": digest(assembled),
                }],
            })
            null_manifests = []
            for name, pair in (("llama", "llama2_base_chat"), ("falcon", "falcon_base_instruct")):
                null_dir = root / f"null-{name}"
                null_artifact = null_dir / "manifest.json"
                write_json(null_artifact, {"status": "complete", "records": 3})
                manifest = root / f"null-{name}.json"
                write_json(manifest, {
                    "status": "complete",
                    "records": 3,
                    "artifact_dir": str(null_dir),
                    "artifact_manifest_sha256": digest(null_artifact),
                    "provenance": {
                        "pair": pair,
                        "runtime": {
                            "elapsed_seconds": 3600,
                            "observer_forward_calls_per_model": 3,
                            "peak_gpu_memory_bytes_by_device": {"cuda:0": 200, "cuda:1": 201},
                        },
                    },
                })
                null_manifests.append((name, manifest))
            payload = build_calder_feature_cost_registry(
                feature_scopes=[("formal", scope)],
                runtime_maps=[(artifact_prefix, runtime_root)],
                null_manifests=null_manifests,
                expected_records=3,
            )
            self.assertEqual(payload["records_per_view"], 3)
            self.assertEqual(payload["unique_input_shards"], 1)
            self.assertEqual(payload["total_gpu_hours"], 11.0)
            self.assertEqual(payload["maximum_peak_gpu_memory_bytes"], 201)
            self.assertEqual(payload["observer_forward_calls_total"], 17)
            self.assertEqual(payload["observer_record_forwards_total"], 27)

    def test_rejects_null_coverage_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            build_calder_feature_cost_registry(
                feature_scopes=[], runtime_maps=[(Path("a"), Path("b"))],
                null_manifests=[], expected_records=1,
            )


if __name__ == "__main__":
    unittest.main()
