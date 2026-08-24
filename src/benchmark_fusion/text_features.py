"""Raw-text feature extraction for the public CALDER training and inference API."""

from __future__ import annotations

import csv
import gc
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .causal import bounded_context_codelength
from .observer import (
    alignment_statistics,
    summarize_masked,
    token_lapd_null_moments_from_logits,
    token_statistics_from_logits,
)
from .spans import aggregate_to_common_spans


DEFAULT_OBSERVER_CONFIG: dict[str, object] = {
    "schema_version": 1,
    "max_length": 1024,
    "short_context": 128,
    "semantic": {
        "role": "modernbert",
        "model": "answerdotai/ModernBERT-base",
        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "dtype": "float16",
    },
    "pairs": [
        {
            "name": "llama",
            "base": {
                "role": "llama2_base",
                "model": "meta-llama/Llama-2-7b-hf",
                "revision": "8a0442e81540efaeb1a0fe3e95477b5e0edfd423",
            },
            "aligned": {
                "role": "llama2_chat",
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "revision": "92011f62d7604e261f748ec0cfe6329f31193e33",
            },
            "dtype": "bfloat16",
        },
        {
            "name": "falcon",
            "base": {
                "role": "falcon_base",
                "model": "tiiuae/falcon-7b",
                "revision": "ec89142b67d748a1865ea4451372db8313ada0d8",
            },
            "aligned": {
                "role": "falcon_instruct",
                "model": "tiiuae/falcon-7b-instruct",
                "revision": "8782b5c5d8c9290412416618f36a133653e85285",
            },
            "dtype": "bfloat16",
        },
    ],
    "singles": [
        {
            "role": "gpt2",
            "model": "openai-community/gpt2",
            "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            "dtype": "float16",
        },
        {
            "role": "gpt2_large",
            "model": "openai-community/gpt2-large",
            "revision": "main",
            "dtype": "float16",
        },
    ],
}


@dataclass(frozen=True)
class TextRecord:
    sample_id: str
    text: str
    label: int | None = None


def _label(value: object, *, row: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip().lower() in {
        "0", "1", "human", "ai", "machine", "generated",
    }:
        normalized = value.strip().lower()
        return 0 if normalized in {"0", "human"} else 1
    raise ValueError(f"row {row} has an invalid label; use human=0 and AI=1")


def _record(payload: dict[str, object], *, row: int, require_labels: bool) -> TextRecord:
    text = payload.get("text", payload.get("content"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"row {row} lacks non-empty text/content")
    raw_id = payload.get("sample_id", payload.get("id", f"sample-{row:08d}"))
    if not isinstance(raw_id, (str, int)) or not str(raw_id):
        raise ValueError(f"row {row} has an invalid sample_id")
    raw_label = payload.get("label")
    label = _label(raw_label, row=row) if raw_label is not None else None
    if require_labels and label is None:
        raise ValueError(f"row {row} lacks label")
    return TextRecord(sample_id=str(raw_id), text=text, label=label)


def load_records(path: Path, *, require_labels: bool) -> list[TextRecord]:
    suffix = path.suffix.lower()
    records: list[TextRecord] = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for row, line in enumerate(handle, 1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{row}") from error
                if not isinstance(payload, dict):
                    raise ValueError(f"row {row} must be a JSON object")
                records.append(_record(payload, row=row, require_labels=require_labels))
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row, payload in enumerate(csv.DictReader(handle), 2):
                records.append(_record(dict(payload), row=row, require_labels=require_labels))
    elif suffix == ".txt" and not require_labels:
        records.append(TextRecord(sample_id=path.stem, text=path.read_text(encoding="utf-8")))
    else:
        allowed = "JSONL or CSV" if require_labels else "JSONL, CSV, or TXT"
        raise ValueError(f"unsupported input format; expected {allowed}")
    if not records:
        raise ValueError("input contains no records")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be unique")
    if require_labels and {record.label for record in records} != {0, 1}:
        raise ValueError("training and development files must contain both classes")
    return records


def load_observer_config(path: Path | None) -> dict[str, object]:
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path is not None
        else json.loads(json.dumps(DEFAULT_OBSERVER_CONFIG))
    )
    if payload.get("schema_version") != 1:
        raise ValueError("observer config schema_version must be 1")
    if int(payload.get("max_length", 0)) != 1024 or int(payload.get("short_context", 0)) != 128:
        raise ValueError("CALDER public paper profile requires max_length=1024 and short_context=128")
    pairs = payload.get("pairs")
    singles = payload.get("singles")
    if not isinstance(pairs, list) or [item.get("name") for item in pairs] != ["llama", "falcon"]:
        raise ValueError("observer config requires ordered llama and falcon pairs")
    if not isinstance(singles, list) or [item.get("role") for item in singles] != ["gpt2", "gpt2_large"]:
        raise ValueError("observer config requires ordered gpt2 and gpt2_large observers")
    return payload


def resolve_devices(value: str) -> tuple[str, ...]:
    if value == "auto":
        if torch.cuda.is_available():
            if torch.cuda.device_count() >= 2:
                return tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))
            return ("cuda:0", "cpu")
        return ("cpu",)
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices:
        raise ValueError("--devices must contain at least one PyTorch device")
    for item in devices:
        device = torch.device(item)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {item}")
    return devices


def _dtype(name: str, device: str) -> torch.dtype:
    if torch.device(device).type == "cpu":
        return torch.float32
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unknown model dtype: {name}")
    return mapping[name]


def _model_kwargs(spec: dict[str, object], *, device: str, local_files_only: bool) -> dict[str, object]:
    return {
        "revision": str(spec.get("revision", "main")),
        "local_files_only": local_files_only,
        "trust_remote_code": False,
        "dtype": _dtype(str(spec.get("dtype", "bfloat16")), device),
    }


def _tokenizer_kwargs(spec: dict[str, object], *, local_files_only: bool) -> dict[str, object]:
    return {
        "revision": str(spec.get("revision", "main")),
        "local_files_only": local_files_only,
        "use_fast": True,
        "trust_remote_code": False,
    }


def _prepare_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("observer tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def _summary7(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return summarize_masked(values, mask)[:, (0, 1, 2, 4, 6, 7, 8)]


def _codelength(log_probability: torch.Tensor, mask: torch.Tensor, texts: list[str]) -> np.ndarray:
    rows = []
    for values, valid, text in zip(log_probability.double(), mask.bool(), texts):
        transitions = int(valid.sum())
        if transitions < 1:
            raise ValueError("text is too short to contain a causal transition")
        bits = float((-values[valid]).sum() / math.log(2.0))
        rows.append(
            (
                bits / transitions,
                bits / max(1, len(text.split())),
                bits / max(1, len(text.encode("utf-8"))),
            )
        )
    return np.asarray(rows, dtype=np.float32)


def assemble_compression(codelength: dict[str, np.ndarray], short: dict[str, np.ndarray]) -> np.ndarray:
    roles = ("llama2_base", "llama2_chat", "falcon_base", "falcon_instruct", "gpt2", "gpt2_large")
    columns: list[np.ndarray] = [codelength[role] for role in roles]
    columns.extend(
        codelength[base] - codelength[aligned]
        for base, aligned in (("llama2_base", "llama2_chat"), ("falcon_base", "falcon_instruct"))
    )
    columns.extend(
        (codelength[left][:, 2] - codelength[right][:, 2])[:, None]
        for left, right in combinations(roles, 2)
    )
    columns.append(codelength["gpt2"] - codelength["gpt2_large"])
    for role in ("llama2_base", "falcon_base", "gpt2"):
        columns.append(short[role] - codelength[role])
    output = np.concatenate(columns, axis=1).astype(np.float32)
    if output.shape[1] != 51 or not np.isfinite(output).all():
        raise ValueError("compression feature assembly differs from the CALDER schema")
    return output


def _release_models() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CalderTextFeatureExtractor:
    """Sequentially load frozen observers and build CALDER tensors from text."""

    def __init__(
        self,
        config: dict[str, object],
        *,
        devices: tuple[str, ...],
        batch_size: int = 1,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("feature batch size must be positive")
        self.config = config
        self.devices = devices
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.max_length = int(config["max_length"])
        self.short_context = int(config["short_context"])

    def _semantic(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        from transformers import AutoModel, AutoTokenizer

        spec = dict(self.config["semantic"])
        device = self.devices[0]
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec["model"]), **_tokenizer_kwargs(spec, local_files_only=self.local_files_only)
        )
        if not tokenizer.is_fast:
            raise RuntimeError("semantic observer requires a fast tokenizer")
        model = AutoModel.from_pretrained(
            str(spec["model"]), **_model_kwargs(spec, device=device, local_files_only=self.local_files_only)
        ).eval().to(device)
        hidden_size = int(model.config.hidden_size)
        spans = np.zeros((len(texts), 16, hidden_size), dtype=np.float32)
        masks = np.zeros((len(texts), 16), dtype=np.bool_)
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                offsets = encoded.pop("offset_mapping").cpu().numpy()
                inputs = {name: value.to(device) for name, value in encoded.items()}
                hidden = model(**inputs).last_hidden_state.float().cpu().numpy()
                for local, text in enumerate(batch):
                    values, valid = aggregate_to_common_spans(
                        offsets[local], hidden[local], len(text), 16
                    )
                    if not valid.any():
                        raise ValueError("semantic observer produced no valid span")
                    spans[start + local] = values
                    masks[start + local] = valid
        del model, tokenizer
        _release_models()
        return spans, masks

    def _pair(
        self,
        pair: dict[str, object],
        texts: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_spec = dict(pair["base"])
        aligned_spec = dict(pair["aligned"])
        base_spec.setdefault("dtype", pair.get("dtype", "bfloat16"))
        aligned_spec.setdefault("dtype", pair.get("dtype", "bfloat16"))
        base_device = self.devices[0]
        aligned_device = self.devices[1] if len(self.devices) > 1 else self.devices[0]
        base_tokenizer = AutoTokenizer.from_pretrained(
            str(base_spec["model"]),
            **_tokenizer_kwargs(base_spec, local_files_only=self.local_files_only),
        )
        aligned_tokenizer = AutoTokenizer.from_pretrained(
            str(aligned_spec["model"]),
            **_tokenizer_kwargs(aligned_spec, local_files_only=self.local_files_only),
        )
        if not base_tokenizer.is_fast or not aligned_tokenizer.is_fast:
            raise RuntimeError("paired observers require fast tokenizers")
        _prepare_tokenizer(base_tokenizer)
        _prepare_tokenizer(aligned_tokenizer)
        if (
            base_tokenizer.get_vocab() != aligned_tokenizer.get_vocab()
            or base_tokenizer.special_tokens_map != aligned_tokenizer.special_tokens_map
        ):
            raise RuntimeError("base and aligned tokenizer identities differ")
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_spec["model"]),
            **_model_kwargs(base_spec, device=base_device, local_files_only=self.local_files_only),
        ).eval().to(base_device)
        aligned_model = AutoModelForCausalLM.from_pretrained(
            str(aligned_spec["model"]),
            **_model_kwargs(aligned_spec, device=aligned_device, local_files_only=self.local_files_only),
        ).eval().to(aligned_device)
        probability = np.zeros((len(texts), 64, 3), dtype=np.float32)
        alignment = np.zeros((len(texts), 64, 4), dtype=np.float32)
        masks = np.zeros((len(texts), 64), dtype=np.bool_)
        probability_document = np.zeros((len(texts), 27), dtype=np.float32)
        alignment_document = np.zeros((len(texts), 28), dtype=np.float32)
        base_role = str(base_spec["role"])
        aligned_role = str(aligned_spec["role"])
        long = {
            base_role: np.zeros((len(texts), 3), dtype=np.float32),
            aligned_role: np.zeros((len(texts), 3), dtype=np.float32),
        }
        short = {base_role: np.zeros((len(texts), 3), dtype=np.float32)}
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                base_encoding = base_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_offsets_mapping=True,
                    return_token_type_ids=False,
                    return_tensors="pt",
                )
                aligned_encoding = aligned_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_token_type_ids=False,
                    return_tensors="pt",
                )
                offsets = base_encoding.pop("offset_mapping").cpu().numpy()
                if not torch.equal(base_encoding["input_ids"], aligned_encoding["input_ids"]):
                    raise RuntimeError("base and aligned tokenization differ")
                base_inputs = {name: value.to(base_device) for name, value in base_encoding.items()}
                aligned_inputs = {name: value.to(aligned_device) for name, value in aligned_encoding.items()}
                base_logits = base_model(**base_inputs, use_cache=False).logits[:, :-1]
                aligned_logits = aligned_model(**aligned_inputs, use_cache=False).logits[:, :-1].to(base_device)
                targets = base_inputs["input_ids"][:, 1:]
                valid = base_inputs["attention_mask"][:, :-1].bool() & base_inputs["attention_mask"][:, 1:].bool()
                base_stats = token_statistics_from_logits(base_logits, targets, valid)
                aligned_stats = token_statistics_from_logits(aligned_logits, targets, valid)
                aligned = alignment_statistics(
                    base_stats.log_probability, aligned_stats.log_probability, valid
                )
                null = token_lapd_null_moments_from_logits(base_logits, aligned_logits, valid)
                probability_values = torch.stack(
                    (base_stats.surprisal, base_stats.entropy, base_stats.log_rank), dim=-1
                ).cpu().numpy()
                alignment_values = torch.stack(
                    (
                        aligned.aligned_minus_base,
                        aligned.information_weighted_imprint,
                        null.mean,
                        null.variance,
                    ),
                    dim=-1,
                ).cpu().numpy()
                probability_summary = torch.cat(
                    [
                        summarize_masked(value, valid)
                        for value in (base_stats.surprisal, base_stats.entropy, base_stats.log_rank)
                    ],
                    dim=1,
                ).cpu().numpy()
                alignment_summary = torch.cat(
                    [
                        _summary7(value, valid)
                        for value in (
                            aligned.aligned_minus_base,
                            aligned.information_weighted_imprint,
                            null.mean,
                            null.variance,
                        )
                    ],
                    dim=1,
                ).cpu().numpy()
                batch_end = start + len(batch)
                probability_document[start:batch_end] = probability_summary
                alignment_document[start:batch_end] = alignment_summary
                long[base_role][start:batch_end] = _codelength(
                    base_stats.log_probability, valid, batch
                )
                long[aligned_role][start:batch_end] = _codelength(
                    aligned_stats.log_probability, valid, batch
                )
                for local, text in enumerate(batch):
                    mapped_probability, probability_mask = aggregate_to_common_spans(
                        offsets[local, 1:], probability_values[local], len(text), 64
                    )
                    mapped_alignment, alignment_mask = aggregate_to_common_spans(
                        offsets[local, 1:], alignment_values[local], len(text), 64
                    )
                    mapped_mask = probability_mask & alignment_mask
                    if not mapped_mask.any():
                        raise ValueError("paired observer produced no valid common span")
                    probability[start + local] = mapped_probability
                    alignment[start + local] = mapped_alignment
                    masks[start + local] = mapped_mask
                    ids = base_inputs["input_ids"][local, base_inputs["attention_mask"][local].bool()]
                    short[base_role][start + local] = (
                        bounded_context_codelength(
                            base_model,
                            ids,
                            max(1, len(text.split())),
                            max(1, len(text.encode("utf-8"))),
                            self.short_context,
                        )
                        .cpu()
                        .numpy()
                    )
                del base_logits, aligned_logits
        del base_model, aligned_model, base_tokenizer, aligned_tokenizer
        _release_models()
        return probability, alignment, masks, probability_document, alignment_document, long, short

    def _single(
        self, spec: dict[str, object], texts: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.devices[0]
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec["model"]), **_tokenizer_kwargs(spec, local_files_only=self.local_files_only)
        )
        _prepare_tokenizer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            str(spec["model"]), **_model_kwargs(spec, device=device, local_files_only=self.local_files_only)
        ).eval().to(device)
        document = np.zeros((len(texts), 27), dtype=np.float32)
        long = np.zeros((len(texts), 3), dtype=np.float32)
        short = np.zeros((len(texts), 3), dtype=np.float32) if spec["role"] == "gpt2" else None
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_token_type_ids=False,
                    return_tensors="pt",
                )
                inputs = {name: value.to(device) for name, value in encoded.items()}
                logits = model(**inputs, use_cache=False).logits[:, :-1]
                targets = inputs["input_ids"][:, 1:]
                valid = inputs["attention_mask"][:, :-1].bool() & inputs["attention_mask"][:, 1:].bool()
                stats = token_statistics_from_logits(logits, targets, valid)
                batch_end = start + len(batch)
                document[start:batch_end] = (
                    torch.cat(
                        [
                            summarize_masked(value, valid)
                            for value in (stats.surprisal, stats.entropy, stats.log_rank)
                        ],
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
                long[start:batch_end] = _codelength(stats.log_probability, valid, batch)
                if short is not None:
                    for local, text in enumerate(batch):
                        ids = inputs["input_ids"][local, inputs["attention_mask"][local].bool()]
                        short[start + local] = (
                            bounded_context_codelength(
                                model,
                                ids,
                                max(1, len(text.split())),
                                max(1, len(text.encode("utf-8"))),
                                self.short_context,
                            )
                            .cpu()
                            .numpy()
                        )
                del logits
        del model, tokenizer
        _release_models()
        return document, long, short

    def extract(self, records: list[TextRecord], output: Path) -> Path:
        texts = [record.text for record in records]
        print(f"[CALDER] extracting semantic features for {len(records)} texts", flush=True)
        semantic_spans, semantic_mask = self._semantic(texts)
        token_parts = []
        alignment_parts = []
        token_masks = []
        probability_documents = []
        alignment_documents = []
        codelength: dict[str, np.ndarray] = {}
        short: dict[str, np.ndarray] = {}
        for pair_value in self.config["pairs"]:
            pair = dict(pair_value)
            print(f"[CALDER] extracting {pair['name']} probability/alignment features", flush=True)
            token, alignment, mask, probability_doc, alignment_doc, pair_long, pair_short = (
                self._pair(pair, texts)
            )
            token_parts.append(token)
            alignment_parts.append(alignment)
            token_masks.append(mask)
            probability_documents.append(probability_doc)
            alignment_documents.append(alignment_doc)
            codelength.update(pair_long)
            short.update(pair_short)
        for spec_value in self.config["singles"]:
            spec = dict(spec_value)
            print(f"[CALDER] extracting {spec['role']} document features", flush=True)
            document, values, short_values = self._single(spec, texts)
            probability_documents.append(document)
            codelength[str(spec["role"])] = values
            if short_values is not None:
                short[str(spec["role"])] = short_values
        arrays: dict[str, np.ndarray] = {
            "semantic_spans": semantic_spans,
            "semantic_mask": semantic_mask,
            "token_probability": np.concatenate(token_parts, axis=2),
            "alignment_evidence": np.concatenate(alignment_parts, axis=2),
            "token_mask": np.logical_and.reduce(token_masks),
            "document_probability": np.concatenate(probability_documents, axis=1),
            "document_alignment": np.concatenate(alignment_documents, axis=1),
            "compression": assemble_compression(codelength, short),
            "sample_ids": np.asarray([record.sample_id for record in records]),
        }
        if all(record.label is not None for record in records):
            arrays["labels"] = np.asarray([record.label for record in records], dtype=np.int8)
        expected_shapes = {
            "token_probability": (len(records), 64, 6),
            "alignment_evidence": (len(records), 64, 8),
            "document_probability": (len(records), 108),
            "document_alignment": (len(records), 56),
            "compression": (len(records), 51),
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"extracted {name} differs from CALDER schema")
        if not arrays["token_mask"].any(axis=1).all():
            raise ValueError("one or more texts have no common valid token span")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
        print(f"[CALDER] feature cache ready: {output}", flush=True)
        return output
