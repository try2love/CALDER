from __future__ import annotations

import sys
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_fusion.training import (
    CheckpointScore,
    canonical_run_id,
    checkpoint_is_better,
    effective_epoch_schedule,
    prune_complete_checkpoints,
    validate_implementation_manifest,
)


class TrainingProtocolTest(unittest.TestCase):
    def test_run_id_is_canonical_and_spec_sensitive(self) -> None:
        self.assertEqual(canonical_run_id({"b": 2, "a": 1}), canonical_run_id({"a": 1, "b": 2}))
        self.assertNotEqual(canonical_run_id({"a": 1}), canonical_run_id({"a": 2}))

    def test_checkpoint_tie_order_is_frozen(self) -> None:
        incumbent = CheckpointScore(0.9, 0.8, 0.4, 2)
        self.assertTrue(checkpoint_is_better(CheckpointScore(0.91, 0.1, 1.0, 8), incumbent))
        self.assertTrue(checkpoint_is_better(CheckpointScore(0.9, 0.81, 1.0, 8), incumbent))
        self.assertTrue(checkpoint_is_better(CheckpointScore(0.9, 0.8, 0.39, 8), incumbent))
        self.assertFalse(checkpoint_is_better(CheckpointScore(0.9, 0.8, 0.4, 3), incumbent))

    def test_fewshot_epoch_floor_is_protocol_bound_and_full_schedule_is_unchanged(self) -> None:
        unchanged = effective_epoch_schedule(
            {}, train_records=64, global_batch_size=256, configured_epochs=3
        )
        self.assertEqual(unchanged.effective_epochs, 3)
        self.assertIsNone(unchanged.minimum_optimizer_steps)
        with tempfile.TemporaryDirectory() as directory:
            protocol = Path(directory) / "fewshot.json"
            protocol.write_text(
                json.dumps(
                    {
                        "status": "frozen_before_formal_sampling",
                        "optimization": {
                            "minimum_optimizer_steps_per_trainable_stage": 50
                        },
                    }
                ),
                encoding="utf-8",
            )
            spec = {
                "training_budget": 32,
                "fewshot_protocol": {
                    "path": str(protocol),
                    "sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                },
            }
            schedule = effective_epoch_schedule(
                spec, train_records=64, global_batch_size=256, configured_epochs=3
            )
            self.assertEqual(schedule.steps_per_epoch, 1)
            self.assertEqual(schedule.effective_epochs, 50)
            self.assertEqual(schedule.minimum_optimizer_steps, 50)
            full_schedule = effective_epoch_schedule(
                {**spec, "training_budget": "full"},
                train_records=64,
                global_batch_size=256,
                configured_epochs=3,
            )
            self.assertEqual(full_schedule.effective_epochs, 3)
            self.assertIsNone(full_schedule.minimum_optimizer_steps)

    def test_formal_run_requires_and_validates_frozen_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            implementation = root / "trainer.py"
            implementation.write_text("frozen\n", encoding="utf-8")
            digest = hashlib.sha256(implementation.read_bytes()).hexdigest()
            manifest = root / "implementation.json"
            payload = {
                "schema_version": 1,
                "status": "frozen",
                "git_commit": "0123456789abcdef",
                "files": [
                    {
                        "path": str(implementation),
                        "bytes": implementation.stat().st_size,
                        "sha256": digest,
                    }
                ],
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            identity = {"path": str(manifest), "sha256": manifest_hash}
            self.assertEqual(
                validate_implementation_manifest({"status": "frozen", "implementation_manifest": identity}),
                identity,
            )
            implementation.write_text("drifted\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_implementation_manifest({"status": "frozen", "implementation_manifest": identity})

    def test_only_engineering_runs_may_omit_implementation_manifest(self) -> None:
        self.assertIsNone(validate_implementation_manifest({"status": "engineering_only"}))
        with self.assertRaises(ValueError):
            validate_implementation_manifest({"status": "frozen"})

    def test_checkpoint_retention_keeps_current_and_previous_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"checkpoint-{index}.pt" for index in range(3)]
            for index, path in enumerate(paths):
                path.write_bytes(str(index).encode())
                os.utime(path, ns=(index + 1, index + 1))
            removed = prune_complete_checkpoints(root, paths[-1], retain=2)
            self.assertEqual(removed, [paths[0]])
            self.assertFalse(paths[0].exists())
            self.assertTrue(paths[1].is_file())
            self.assertTrue(paths[2].is_file())


if __name__ == "__main__":
    unittest.main()
