from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_fusion.quickstart import PortableFeatureDataset, make_demo, predict, train


class QuickstartTest(unittest.TestCase):
    def test_demo_train_predict_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            demo = make_demo(
                SimpleNamespace(
                    output=root / "demo",
                    train_records=8,
                    dev_records=8,
                    test_records=8,
                    semantic_dim=8,
                    seed=42,
                )
            )
            self.assertEqual(demo["purpose"], "synthetic_engineering_demo_only")
            dataset = PortableFeatureDataset(root / "demo/train.npz", require_labels=True)
            self.assertEqual(len(dataset), 8)
            summary = train(
                SimpleNamespace(
                    train=root / "demo/train.npz",
                    dev=root / "demo/dev.npz",
                    output=root / "model",
                    device="cpu",
                    seed=42,
                    epochs=1,
                    patience=1,
                    batch_size=8,
                    hidden_dim=8,
                    convolution_channels=2,
                    branch_dropout=0.0,
                    fusion="adaptive_gate",
                    learning_rate=3e-4,
                    weight_decay=0.0,
                    gradient_clip_norm=1.0,
                )
            )
            self.assertTrue(Path(summary["model"]).is_file())
            prediction = predict(
                SimpleNamespace(
                    model=Path(summary["model"]),
                    input=root / "demo/test.npz",
                    output=root / "predictions.jsonl",
                    device="cpu",
                    batch_size=8,
                )
            )
            self.assertEqual(prediction["records"], 8)
            self.assertEqual(len((root / "predictions.jsonl").read_text().splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
