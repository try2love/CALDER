from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from benchmark_fusion.text_cli import predict_text, train_text
from benchmark_fusion.text_features import (
    TextRecord,
    assemble_compression,
    load_observer_config,
    load_records,
)


class FakeExtractor:
    def extract(self, records: list[TextRecord], output: Path) -> Path:
        rng = np.random.default_rng(42 + len(records))
        labels = np.asarray(
            [record.label if record.label is not None else 0 for record in records],
            dtype=np.int8,
        )
        shift = labels.astype(np.float32)[:, None]
        arrays = {
            "semantic_spans": rng.normal(size=(len(records), 16, 8)).astype(np.float32),
            "semantic_mask": np.ones((len(records), 16), dtype=np.bool_),
            "token_probability": rng.normal(size=(len(records), 64, 6)).astype(np.float32),
            "alignment_evidence": rng.normal(size=(len(records), 64, 8)).astype(np.float32),
            "token_mask": np.ones((len(records), 64), dtype=np.bool_),
            "document_probability": rng.normal(size=(len(records), 108)).astype(np.float32),
            "document_alignment": rng.normal(size=(len(records), 56)).astype(np.float32),
            "compression": rng.normal(size=(len(records), 51)).astype(np.float32),
            "sample_ids": np.asarray([record.sample_id for record in records]),
        }
        arrays["document_probability"] += shift
        arrays["compression"] += shift
        if all(record.label is not None for record in records):
            arrays["labels"] = labels
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
        return output


class TextPipelineTest(unittest.TestCase):
    def test_jsonl_accepts_text_or_content_and_human_ai_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            path.write_text(
                "\n".join(
                    (
                        json.dumps({"id": "a", "text": "human text", "label": "human"}),
                        json.dumps({"id": "b", "content": "machine text", "label": "AI"}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_records(path, require_labels=True)
            self.assertEqual([record.label for record in records], [0, 1])
            self.assertEqual([record.sample_id for record in records], ["a", "b"])

    def test_compression_layout_has_51_finite_columns(self) -> None:
        roles = (
            "llama2_base", "llama2_chat", "falcon_base",
            "falcon_instruct", "gpt2", "gpt2_large",
        )
        long = {role: np.full((2, 3), index + 1.0) for index, role in enumerate(roles)}
        short = {role: long[role] + 0.5 for role in ("llama2_base", "falcon_base", "gpt2")}
        output = assemble_compression(long, short)
        self.assertEqual(output.shape, (2, 51))
        self.assertTrue(np.isfinite(output).all())

    @patch("benchmark_fusion.text_cli._extractor", return_value=FakeExtractor())
    def test_raw_text_train_and_predict_round_trip(self, _extractor_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"sample_id": f"sample-{index}", "text": f"document {index} with enough words", "label": index % 2}
                for index in range(8)
            ]
            for split in ("train", "dev"):
                (root / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            shared = {
                "observer_config": None,
                "devices": "cpu",
                "feature_batch_size": 1,
                "local_files_only": True,
                "train_device": "cpu",
            }
            summary = train_text(
                SimpleNamespace(
                    **shared,
                    train=root / "train.jsonl",
                    dev=root / "dev.jsonl",
                    output=root / "model",
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
            bundle = torch.load(summary["model"], map_location="cpu", weights_only=False)
            self.assertEqual(bundle["text_pipeline"]["schema_version"], 1)
            prediction = predict_text(
                SimpleNamespace(
                    **shared,
                    model=Path(summary["model"]),
                    text="A plain text document supplied directly by the user.",
                    text_file=None,
                    input=None,
                    output=root / "prediction.jsonl",
                    batch_size=8,
                )
            )
            self.assertEqual(prediction["records"], 1)
            self.assertIn("prediction", prediction)

    def test_default_observer_profile_is_paper_length(self) -> None:
        config = load_observer_config(None)
        self.assertEqual(config["max_length"], 1024)
        self.assertEqual([pair["name"] for pair in config["pairs"]], ["llama", "falcon"])


if __name__ == "__main__":
    unittest.main()
