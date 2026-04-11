# Qwen3-VL Serve

Qwen3-VL 2B and Qwen3.5-0.8B vision-language models implemented in JAX/Flax for TPU v4-8.

Supports both inference and batch-optimized training with 4-chip data parallelism.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment
uv sync
```

### 2. Download model weights

```bash
# Qwen3-VL 2B
mkdir -p ../models/qwen3-vl-2b
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --include "*.safetensors" \
  --local-dir ../models/qwen3-vl-2b

# Qwen3.5-0.8B (optional)
mkdir -p ../models/qwen35-0.8b
huggingface-cli download Qwen/Qwen3.5-0.8B \
  --include "*.safetensors" \
  --local-dir ../models/qwen35-0.8b
```

Or set custom paths via environment variables:

```bash
export QWEN3VL_MODEL_PATH=/path/to/qwen3-vl-2b
export QWEN35_MODEL_PATH=/path/to/qwen35-0.8b
```

## Usage

### Inference

```bash
# Qwen3-VL 2B: image -> text generation
python run_qwen3vl.py

# Qwen3.5-0.8B: text generation
python run_qwen35.py

# Batch inference benchmark (batch 1/2/4)
python bench_qwen3vl.py
```

### Training

```bash
# Single-device baseline
python train_qwen3vl.py --mode single

# 4-device data parallel
python train_qwen3vl.py --mode multi

# Both (comparison benchmark)
python train_qwen3vl.py --mode both
```

### Validation (HF vs JAX comparison)

```bash
python compare_blocks.py     # Block-by-block
python compare_e2e.py        # End-to-end
python compare_gdn_exact.py  # GDN linear attention
```

## Architecture

### Qwen3-VL 2B
- Vision: 24-layer ViT (h=1024, 16 heads) + DeepStack (layers 5, 11, 17)
- Text: 28-layer decoder (h=2048, GQA 16Q/8KV heads)
- Vocab: 151,936

### Qwen3.5-0.8B
- Vision: 12-layer ViT (h=768, 12 heads)
- Text: 24-layer decoder with mixed attention (3 GDN linear + 1 full, repeated)
- Vocab: 248,320

## Training optimizations

- **Gradient checkpointing** (`jax.checkpoint`): recomputes activations per decoder layer
- **bf16 optimizer moments** (`optax.scale_by_adam(mu_dtype=jnp.bfloat16)`)
- **Mesh + NamedSharding**: data parallel across 4 TPU chips
- **Static shapes**: fixed seq_len to avoid JIT recompilation

### TPU flags (set automatically in `train_qwen3vl.py`)

```
--xla_tpu_use_enhanced_launch_barrier=true
--xla_tpu_enable_data_parallel_all_reduce_opt=true
--xla_tpu_scoped_vmem_limit_kib=98304
```

## Benchmark results (TPU v4-8, seq_len=256)

| Config | Throughput | Speedup |
|--------|-----------|---------|
| single_bs1 (baseline) | 1,783 tok/s | 1.00x |
| single_bs2 | 2,883 tok/s | 1.62x |
| single_bs4 | 4,461 tok/s | 2.50x |
| multi_bs4 (4 TPU) | 4,003 tok/s | 2.24x |
| multi_bs8 (4 TPU) | 5,684 tok/s | 3.19x |
| multi_bs16 (4 TPU) | 13,215 tok/s | 7.41x |

## Requirements

- TPU v4-8 (4 chips, single host)
- JAX >= 0.6.0 with TPU backend
- Python >= 3.10
- `uv` for package management

## Project structure

```
model/                  # Qwen3-VL 2B implementation
  modeling.py           # Full model (Vision + Text + forward_train)
  params.py             # safetensors weight loading
model35/                # Qwen3.5-0.8B implementation
  modeling.py           # GDN + full attention hybrid
  params.py             # Weight loading
  gated_delta_net.py    # Gated DeltaNet linear attention
run_qwen3vl.py          # VL inference demo
run_qwen35.py           # Text inference demo
bench_qwen3vl.py        # Inference benchmark
train_qwen3vl.py        # Training benchmark (single/multi-device)
compare_*.py            # HF vs JAX validation
```
