# CALDER

[中文](README.md) | [English](README_EN.md)

CALDER 是一个用于 AI 生成文本检测的轻量多特征融合模型。它不直接把 LAPD、LastDE、DetectGPT 等 training-free detector 的最终分数作为输入，而是学习融合五类底层证据：

1. 语义 span 表示；
2. token 概率证据；
3. 对齐印记证据；
4. 文档级概率统计；
5. 压缩率与码长几何。

每个分支先映射到共同隐空间，默认使用样本级自适应门控融合，最终输出一个 AI-positive logit。代码中同时保留了 `concat_mlp` 对照融合方式。

## 最快上手：3 条命令跑通训练和判别

先安装：

```bash
git clone https://github.com/try2love/CALDER.git
cd CALDER
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

然后用合成特征跑通完整流程：

```bash
calder make-demo --output demo_data
calder train --train demo_data/train.npz --dev demo_data/dev.npz --output outputs/demo
calder predict --model outputs/demo/calder_model.pt --input demo_data/test.npz --output outputs/demo/predictions.jsonl
```

如果三条命令都成功，就说明安装、模型训练、checkpoint 加载和批量判别流程均正常。`make-demo` 只生成工程自检用的合成数据，不是学术实验结果。

## 用自己的特征训练

### 1. 准备 NPZ

为了让普通用户无需理解内部 shard manifest，快速入口使用单个 `.npz` 文件。训练集和开发集都需包含：

| 数组 | 形状 | 含义 |
|---|---:|---|
| `semantic_spans` | `N x 16 x D` | 冻结语义 span 表示 |
| `semantic_mask` | `N x 16` | 有效语义 span |
| `token_probability` | `N x 64 x 6` | 两组 observer 的 surprisal/entropy/log-rank |
| `alignment_evidence` | `N x 64 x 8` | 对齐差异与 null-reference 证据 |
| `token_mask` | `N x 64` | 有效 token 证据 span |
| `document_probability` | `N x 108` | 文档级概率统计 |
| `document_alignment` | `N x 56` | 文档级对齐统计 |
| `compression` | `N x 51` | 压缩、码长与 observer 几何 |
| `labels` | `N` | human=`0`，AI=`1` |
| `sample_ids` | `N` | 唯一样本 ID，推荐提供 |

`D` 是语义 observer 的隐层维度。训练集和开发集的 `D` 必须相同。

### 2. 一条命令训练

```bash
calder train \
  --train /path/to/train.npz \
  --dev /path/to/dev.npz \
  --output outputs/my_calder
```

默认行为：

- 自动选择 `cuda:0` 或 CPU；
- seed=`42`；
- hidden dimension=`128`；
- learning rate=`3e-4`；
- batch size=`128`；
- adaptive-gate 融合；
- 最多 50 epochs，开发集 AUROC 连续 5 轮不提升则早停；
- 只保留最佳模型 `calder_model.pt`。

指定 GPU：

```bash
calder train --train train.npz --dev dev.npz --output outputs/run1 --device cuda:1
```

CPU 训练：

```bash
calder train --train train.npz --dev dev.npz --output outputs/run1 --device cpu
```

快速调试小模型：

```bash
calder train \
  --train train.npz --dev dev.npz --output outputs/debug \
  --device cpu --epochs 3 --hidden-dim 32 --convolution-channels 8
```

## 使用训练好的模型判别

```bash
calder predict \
  --model outputs/my_calder/calder_model.pt \
  --input /path/to/test.npz \
  --output outputs/my_calder/predictions.jsonl
```

每行输出一条样本：

```json
{"sample_id":"example-0001","ai_score":1.27,"ai_probability":0.7807,"prediction":"AI"}
```

- `ai_score` 是原始 logit，越大越像 AI 文本；
- `ai_probability` 是 sigmoid 后的便于阅读值；
- `prediction` 使用训练时从开发集固定的 F1 阈值；
- 如果输入 NPZ 带 `labels`，还会在 summary 中计算 AUROC/AUPRC。

## Python 中直接使用架构

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

正式训练使用 `train_zscore`，且均值和方差只能由训练集计算。上面的纯前向例子使用 `sample_layernorm`，因此不需要数据统计。

## 论文冻结协议入口

`calder train/predict` 是面向一般用户的简化界面。需要严格重放冻结论文协议时，使用：

- `scripts/train_calder.py`
- `scripts/evaluate_calder.py`
- `scripts/evaluate_calder_test.py`
- `scripts/profile_calder_test.py`

这些入口使用带 SHA-256 身份的分片 feature manifest，不会自动下载数据或模型。

## 当前开源边界

本仓库只包含 CALDER 架构、训练、推理、评估和测试代码，不包含：

- 数据集或论文数据划分；
- 预训练 observer 权重；
- CALDER 训练后 checkpoint；
- 预测、实验结果、看板或论文源文件；
- 原始文本到全部 CALDER 特征的端到端 observer 提取管线。

因此，当前 `calder predict` 的输入是预计算特征 NPZ，不是原始文本。这是有意的代码开源边界，不会隐式下载数据或 observer 权重。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 许可证

正式开源许可证尚未选定。在 `LICENSE_REQUIRED.md` 被正式 `LICENSE` 替换前，本仓库代码虽公开可见，但不授予复制、修改或再分发权利。
