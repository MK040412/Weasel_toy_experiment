# Qwen3-VL Serve

Qwen3-VL 2B / Qwen3.5-0.8B vision-language models in JAX/Flax for TPU v4-8.
Inference + batch-optimized training with 4-chip data parallelism.

## Quick Start (TPU v4-8)

```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment
uv sync

# Download weights (requires HF_TOKEN env var)
export HF_TOKEN=<your_huggingface_token>
mkdir -p ../models/qwen3-vl-2b
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --include "*.safetensors" --local-dir ../models/qwen3-vl-2b

# Optional: Qwen3.5-0.8B
mkdir -p ../models/qwen35-0.8b
huggingface-cli download Qwen/Qwen3.5-0.8B \
  --include "*.safetensors" --local-dir ../models/qwen35-0.8b
```

## Model Paths

Default: `../models/qwen3-vl-2b` and `../models/qwen35-0.8b` (relative to repo root).

Override with environment variables:
```bash
export QWEN3VL_MODEL_PATH=/path/to/qwen3-vl-2b
export QWEN35_MODEL_PATH=/path/to/qwen35-0.8b
```

## Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `run_qwen3vl.py` | Qwen3-VL 2B image+text inference | `python run_qwen3vl.py` |
| `run_qwen35.py` | Qwen3.5-0.8B text inference | `python run_qwen35.py` |
| `bench_qwen3vl.py` | Inference throughput benchmark | `python bench_qwen3vl.py` |
| `train_qwen3vl.py` | Training benchmark (single/multi) | `python train_qwen3vl.py --mode both` |
| `compare_*.py` | HF vs JAX numerical validation | `python compare_blocks.py` |

### Training modes

```bash
python train_qwen3vl.py --mode single   # 1-device baseline
python train_qwen3vl.py --mode multi    # 4-device data parallel
python train_qwen3vl.py --mode both     # comparison benchmark
```

## Architecture

| | Qwen3-VL 2B | Qwen3.5-0.8B |
|--|-------------|--------------|
| Vision | 24L, h=1024, 16 heads, DeepStack(5,11,17) | 12L, h=768, 12 heads |
| Text | 28L, h=2048, GQA 16Q/8KV | 24L, h=1024, GDN+Full mixed |
| Vocab | 151,936 | 248,320 |

## Training Optimizations

- **Gradient checkpointing**: `jax.checkpoint` per decoder layer
- **bf16 optimizer**: `optax.scale_by_adam(mu_dtype=jnp.bfloat16)`
- **Data parallel**: `Mesh + NamedSharding` across 4 TPU chips
- **TPU flags**: enhanced launch barrier, DP all-reduce opt, VMEM limit

## Benchmark (TPU v4-8, seq_len=256)

| Config | tok/s | Speedup |
|--------|-------|---------|
| single_bs1 | 1,783 | 1.0x |
| single_bs4 | 4,461 | 2.5x |
| multi_bs8 | 5,684 | 3.2x |
| multi_bs16 | 13,215 | 7.4x |

## Project Structure

```
model/                  # Qwen3-VL 2B
  modeling.py           # Vision + Text + forward_train
  params.py             # safetensors -> JAX weight loading
model35/                # Qwen3.5-0.8B
  modeling.py           # GDN + full attention hybrid
  params.py
  gated_delta_net.py    # Gated DeltaNet linear attention
```

## Requirements

- TPU v4-8 (4 chips, single host)
- JAX >= 0.6.0 with TPU backend
- Python >= 3.10, `uv`
