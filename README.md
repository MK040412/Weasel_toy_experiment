# Weasel Toy Experiment

Toy experiments on TPU v4-8: Qwen VLM inference/training, VLA (Vision-Language-Action) with flow matching, OGBench, CALVIN.

## Setup

```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment
uv sync
```

## Architecture

```
Images(top) + Language → Qwen3-VL 2B (frozen, JAX) → hidden(2048)
  → obs_proj(2048→1536) → GemmaActionExpert(12L, ~311M) → actions(50, 7)

Training: Flow Matching (openpi0.5) with optional RTC (arXiv 2512.05964)
  Stage 1: VLM embedding cache (one-time, saved as parquet)
  Stage 2: Action expert training (batched, HBM-resident)
```

## Structure

| Directory | Role |
|-----------|------|
| `src/qwen/qwen3vl/` | Qwen3-VL 2B JAX implementation |
| `src/qwen/qwen35/` | Qwen3.5-0.8B JAX implementation (GDN + full attn) |
| `src/qwen/vla/` | VLA pipeline: models, training, data, inference (JAX/Flax NNX) |
| `src/qwen/vla/_pytorch_ref/` | Archived PyTorch reference implementation |
| `data/download/` | Large dataset download scripts (RAM-based) |
| `bench/` | Benchmark wrappers (ogbench, calvin, vla) |
| `compare/` | Numerical validation (HF vs JAX, baseline vs RTC) |
| `result/` | Training outputs: checkpoints, VLM cache, videos |

## Quick Start

```bash
# VLA: train + evaluate + visualize (debug dataset, auto-downloads)
PYTHONPATH=src python src/qwen/vla/eval_and_viz.py

# VLA: CLI training with RTC
PYTHONPATH=src python src/qwen/vla/train.py --simulated-delay 15

# JAX inference
python src/qwen/inference.py --model qwen3vl
```

## VLA Train CLI

```
python src/qwen/vla/train.py
  --epochs N              Training epochs (default: 100)
  --lr FLOAT              Learning rate (default: 5e-5)
  --batch-size N          Batch size (default: 32)
  --simulated-delay N     RTC delay, 0=off (default: 0)
  --output-dir PATH       Checkpoint/cache directory
```

VLM cache at `{output-dir}/vlm_cache/` is auto-detected. If present, VLM loading is skipped entirely.

## Model Weights

```bash
export HF_TOKEN=<your_token>
mkdir -p ../models/qwen3-vl-2b ../models/qwen35-0.8b
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --include "*.safetensors" --local-dir ../models/qwen3-vl-2b
huggingface-cli download Qwen/Qwen3.5-0.8B --include "*.safetensors" --local-dir ../models/qwen35-0.8b
```

Override: `QWEN3VL_MODEL_PATH`, `QWEN35_MODEL_PATH`

## Requirements

- TPU v4-8 (4 chips, single host) for all experiments
- Python >= 3.10, uv
- No CUDA required (JAX/TPU only)
