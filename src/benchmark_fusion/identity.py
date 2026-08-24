"""Fail-closed identities for frozen benchmark samples."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


class IdentityError(ValueError):
    """Raised when frozen source data violate the declared protocol."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize only for contamination detection; never replace raw input."""

    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_identity(text: str) -> dict[str, int | str]:
    if not isinstance(text, str):
        raise IdentityError(f"expected text string, got {type(text).__name__}")
    raw = text.encode("utf-8")
    normalized = normalize_text(text).encode("utf-8")
    return {
        "text_sha256": sha256_bytes(raw),
        "normalized_text_sha256": sha256_bytes(normalized),
        "text_length": len(text),
        "utf8_bytes": len(raw),
        "word_length": len(WORD_RE.findall(text)),
    }


def load_declared_texts(spec: Mapping[str, Any]) -> tuple[Path, list[str]]:
    path = Path(str(spec["path"]))
    if not path.is_file():
        raise IdentityError(f"missing frozen source: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != spec["sha256"]:
        raise IdentityError(
            f"source hash mismatch for {path}: {actual_hash} != {spec['sha256']}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    container = spec["container"]
    if container == "list":
        texts = payload
    elif container == "dict":
        if not isinstance(payload, dict) or spec["key"] not in payload:
            raise IdentityError(f"missing key {spec['key']!r} in {path}")
        texts = payload[spec["key"]]
    else:
        raise IdentityError(f"unsupported container {container!r}")
    if not isinstance(texts, list):
        raise IdentityError(f"declared texts are not a list in {path}")
    expected_count = int(spec["count"])
    if len(texts) != expected_count:
        raise IdentityError(
            f"record count mismatch for {path}: {len(texts)} != {expected_count}"
        )
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise IdentityError(f"non-string record at {path}:{index}")
    return path, texts


def build_records(
    specs: Iterable[Mapping[str, Any]],
    token_length: Callable[[str], int],
    tokenizer_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for spec in specs:
        path, texts = load_declared_texts(spec)
        label = int(spec["label"])
        if label not in (0, 1):
            raise IdentityError(f"invalid label {label} for {path}")
        for source_index, text in enumerate(texts):
            identity = text_identity(text)
            sample_id = (
                f"bfv1:test:{spec['dataset']}:{'ai' if label else 'human'}:"
                f"{source_index:04d}:{str(identity['text_sha256'])[:16]}"
            )
            if sample_id in sample_ids:
                raise IdentityError(f"duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "dataset": spec["dataset"],
                    "split": "test",
                    "domain": "unknown",
                    "generator": "human" if label == 0 else "unknown",
                    "generator_family": "human" if label == 0 else "unknown",
                    "source_group_id": None,
                    "attack": "unknown",
                    "language": "unknown",
                    **identity,
                    "token_length": int(token_length(text)),
                    "tokenizer_identity": dict(tokenizer_identity),
                    "source_uri": str(spec["source_uri"]),
                    "source_version": str(spec["source_version"]),
                    "source_path": str(path),
                    "source_file_sha256": str(spec["sha256"]),
                    "source_index": source_index,
                    "provenance_status": "frozen_file_and_index_verified",
                }
            )
    return records


def write_jsonl_atomic(records: Iterable[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
