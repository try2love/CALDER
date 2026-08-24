# CALDER

[Chinese](README.md) | [English](README_EN.md)

CALDER is a lightweight multi-feature fusion model for AI-generated text detection. It does not feed the final score of a training-free detector such as LAPD, LastDE, or DetectGPT into the classifier. Instead, it learns from five lower-level evidence branches:

1. semantic span representations;
2. token-probability evidence;
3. alignment-imprint evidence;
4. document-level probability statistics; and
5. compression and codelength geometry.

The branches are projected into a shared hidden space. The default head learns document-specific adaptive-gate weights and returns one AI-positive logit. A frozen `concat_mlp` alternative is also included.

## Quick start: train and classify in three commands

Install the package:

```bash
git clone https://github.com/try2love/CALDER.git
cd CALDER
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the complete workflow with synthetic features:

```bash
calder make-demo --output demo_data
calder train --train demo_data/train.npz --dev demo_data/dev.npz --output outputs/demo
calder predict --model outputs/demo/calder_model.pt --input demo_data/test.npz --output outputs/demo/predictions.jsonl
```

`make-demo` is an engineering self-check only. Its synthetic data and scores are not research results.

## Train with your own features

The beginner interface consumes one portable `.npz` file per split. Training and development files contain:

| Array | Shape | Meaning |
|---|---:|---|
| `semantic_spans` | `N x 16 x D` | frozen semantic span representations |
| `semantic_mask` | `N x 16` | valid semantic spans |
| `token_probability` | `N x 64 x 6` | surprisal, entropy, and log-rank evidence from two observers |
| `alignment_evidence` | `N x 64 x 8` | alignment-difference and null-reference evidence |
| `token_mask` | `N x 64` | valid token-evidence spans |
| `document_probability` | `N x 108` | document probability summaries |
| `document_alignment` | `N x 56` | document alignment summaries |
| `compression` | `N x 51` | compression, codelength, and observer geometry |
| `labels` | `N` | human=`0`, AI=`1` |
| `sample_ids` | `N` | unique sample IDs; recommended |

`D` is the hidden size of the frozen semantic observer and must match across splits.

Start training:

```bash
calder train \
  --train /path/to/train.npz \
  --dev /path/to/dev.npz \
  --output outputs/my_calder
```

Defaults are device auto-selection, seed 42, hidden size 128, learning rate `3e-4`, batch size 128, adaptive-gate fusion, at most 50 epochs, and early stopping after five epochs without dev-AUROC improvement. Only the best `calder_model.pt` is retained.

Select a GPU explicitly:

```bash
calder train --train train.npz --dev dev.npz --output outputs/run1 --device cuda:1
```

Run a small CPU engineering check:

```bash
calder train \
  --train train.npz --dev dev.npz --output outputs/debug \
  --device cpu --epochs 3 --hidden-dim 32 --convolution-channels 8
```

## Classify with a trained model

```bash
calder predict \
  --model outputs/my_calder/calder_model.pt \
  --input /path/to/test.npz \
  --output outputs/my_calder/predictions.jsonl
```

Each line contains one prediction:

```json
{"sample_id":"example-0001","ai_score":1.27,"ai_probability":0.7807,"prediction":"AI"}
```

`ai_score` is the raw AI-positive logit. `ai_probability` is its sigmoid value for readability. `prediction` uses the F1 threshold frozen on the development split during training. When the input NPZ contains labels, the summary also reports AUROC and AUPRC.

## Use the architecture from Python

```python
import torch
from benchmark_fusion.calder_model import CalderCore

model = CalderCore(
    semantic_dim=768,
    hidden_dim=128,
    convolution_channels=64,
    branch_dropout_probability=0.0,
    scalar_normalization="sample_layernorm",
    fusion="adaptive_gate",
)

output = model(
    semantic_spans=torch.randn(2, 16, 768),
    semantic_mask=torch.ones(2, 16, dtype=torch.bool),
    token_probability=torch.randn(2, 64, 6),
    alignment_evidence=torch.randn(2, 64, 8),
    token_mask=torch.ones(2, 64, dtype=torch.bool),
    document_probability=torch.randn(2, 108),
    document_alignment=torch.randn(2, 56),
    compression=torch.randn(2, 51),
)

print(output.logit.shape)         # torch.Size([2])
print(output.gate_weights.shape)  # torch.Size([2, 5])
```

Formal training uses `train_zscore`, with moments computed only from the training split. The standalone forward example uses `sample_layernorm` and therefore needs no dataset statistics.

## Frozen paper-protocol entry points

`calder train/predict` is the beginner interface. For strict replay of the frozen paper protocol, use:

- `scripts/train_calder.py`
- `scripts/evaluate_calder.py`
- `scripts/evaluate_calder_test.py`
- `scripts/profile_calder_test.py`

These entry points consume hash-bound sharded feature manifests and never download data or models automatically.

## Release boundary

This repository contains the CALDER architecture, training, inference, evaluation, and tests. It does not contain datasets, pretrained observer weights, trained CALDER checkpoints, predictions, experimental results, dashboards, paper sources, or the raw-text-to-feature observer pipeline.

Consequently, `calder predict` currently accepts precomputed feature NPZ files, not raw text. This boundary is intentional: the code does not silently download data or observer weights.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

A final open-source license has not been selected. Until `LICENSE_REQUIRED.md` is replaced by an approved `LICENSE`, the publicly visible code grants no permission to copy, modify, or redistribute.
