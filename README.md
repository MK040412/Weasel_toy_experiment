# Weasel Toy Experiment

Toy experiments on TPU v4-8: Qwen VLM inference/training, OGBench, CALVIN.
VLA (Vision-Language-Action) flow matching on CUDA GPU.

## Setup

```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment
uv sync                    # JAX (TPU) dependencies
uv sync --extra vla        # PyTorch (CUDA) dependencies for VLA
```

## Structure

| Directory | Role |
|-----------|------|
| `src/qwen/qwen3vl/` | Qwen3-VL 2B JAX model code |
| `src/qwen/qwen35/` | Qwen3.5-0.8B JAX model code (GDN + full attn) |
| `src/qwen/vla/` | VLA: Qwen3-VL + pi0-style action expert (PyTorch) |
| `src/qwen/inference.py` | JAX unified inference script |
| `src/qwen/train.py` | JAX training benchmark (single/multi-device) |
| `bench/` | Benchmark scripts (ogbench, calvin, vla) |
| `compare/` | Numerical validation (HF vs JAX, baseline vs RTC) |
| `result/` | Benchmark output storage |

## Usage

```bash
# --- JAX (TPU) ---
python src/qwen/inference.py --model qwen3vl    # VL inference
python src/qwen/train.py --mode both             # training benchmark

# --- VLA (CUDA) ---
python src/qwen/vla/train.py                             # standard training
python src/qwen/vla/train.py --simulated-delay 15         # with RTC
python src/qwen/vla/inference.py --checkpoint ckpt.pt     # inference
python compare/compare_rtc.py                             # baseline vs RTC ablation

# --- Benchmarks ---
bash bench/ogbench/run.sh                        # OGBench
bash bench/calvin/run.sh                         # CALVIN
bash bench/vla/run.sh train                      # VLA training
bash bench/vla/run.sh inference ckpt.pt          # VLA inference

# --- Validation ---
python compare/compare_blocks.py                 # HF vs JAX blocks
```

## Model Path Override

```bash
export QWEN3VL_MODEL_PATH=/path/to/qwen3-vl-2b
export QWEN35_MODEL_PATH=/path/to/qwen35-0.8b
```

Default: `../models/qwen3-vl-2b`, `../models/qwen35-0.8b` (relative to repo root)

## Requirements

- TPU v4-8 (4 chips, single host) for JAX experiments
- CUDA GPU (8GB+) for VLA experiments
- Python >= 3.10, uv
