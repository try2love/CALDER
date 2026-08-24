# CALDER input schema

The model consumes eight aligned tensors. The batch dimension is omitted below.

| Input | Shape | Meaning |
|---|---:|---|
| `semantic_spans` | `16 x D` | frozen semantic span embeddings |
| `semantic_mask` | `16` | valid semantic spans |
| `token_probability` | `64 x 6` | surprisal, entropy, and log-rank evidence from two base observers |
| `alignment_evidence` | `64 x 8` | base/aligned imprint and null-reference evidence |
| `token_mask` | `64` | spans valid across the aligned token views |
| `document_probability` | `108` | document summaries of probability evidence |
| `document_alignment` | `56` | document summaries of alignment and null-reference evidence |
| `compression` | `51` | codelength, observer-geometry, model-scale, and context-gain evidence |

`D` is the hidden size of the frozen semantic observer. All floating-point
inputs must be finite. Every record must have at least one valid semantic span
and one valid token-evidence span.

The beginner `calder train/predict` interface stores the batch dimension in one
portable NPZ file. Training and evaluation files additionally contain binary
`labels` with human=0/AI=1. An aligned unique `sample_ids` string array is
recommended; deterministic positional IDs are generated when it is absent.

The formal paper-protocol loader instead uses a manifest that references three aligned shard families:
the base feature shard and two null-reference shards. Each family must contain
the same ordered, unique sample IDs. The loader validates record counts, array
shapes, dtypes, file hashes, and sample alignment before training or inference.

The CALDER model does not accept a detector-score tensor. In particular, no
final LAPD, LastDE, DetectGPT, or other training-free detector score is passed
to the five-branch fusion model.
