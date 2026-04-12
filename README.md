# Weasel Toy Experiment

TPU v4-8 기반 toy experiments: Qwen VLM inference/training, **VLA (Vision-Language-Action) flow matching**, **CALVIN manipulation benchmark**, **OGBench offline RL**.

> 📖 **Full guide**: [CLAUDE.md](./CLAUDE.md) (setup, design principles, commands, troubleshooting)

## Quickstart

### 1. Setup

```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment
uv sync

# Qwen weights
export HF_TOKEN=<your_hf_token>
mkdir -p ../models/qwen3-vl-2b
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --include "*.safetensors" --local-dir ../models/qwen3-vl-2b
```

### 2. CALVIN Benchmark (End-to-End)

```bash
# CALVIN sim deps (pybullet + calvin_env)
uv pip install pybullet "hydra-core==1.1.1" gym omegaconf opencv-python \
  numpy-quaternion gitpython pytorch-lightning termcolor hydra-colorlog tacto
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# CALVIN repo (pybullet env + task oracle)
git clone --recurse-submodules https://github.com/mees/calvin.git ~/calvin
# (See CLAUDE.md for pyhash patch if install fails)
export CALVIN_DIR=~/calvin

# Run full pipeline (FLOWER recipe, ~3.5 hours total)
bash commands/download.sh calvin-abcd                               # 5 min
bash commands/preprocess.sh calvin-abcd-flower                      # 30 min
bash commands/train.sh calvin-abcd-flower --epochs 200              # 2.5 hours
bash commands/benchmark.sh calvin-abcd-flower --num-sequences 100   # 30 min
```

Result: `result/vla_abcd_flower/benchmark/results.json` with success rates.

### 3. OGBench (Offline GCRL)

```bash
# External repo
git clone https://github.com/seohongpark/ogbench.git ~/ogbench
cd ~/ogbench && uv venv && uv pip install -e ".[all]"
cd impls && uv pip install -r requirements.txt
export OGBENCH_DIR=~/ogbench

# Run
cd /path/to/Weasel_toy_experiment
bash commands/bench_ogbench.sh antmaze-large-navigate-v0 agents/gciql.py
```

## Architecture

```
Images(top+wrist) + Language → Qwen3-VL 2B (frozen, JAX) → hidden (2048)
  → obs_proj (2048→1536)
  → [proprio(1,8) + obs_tokens] → GemmaActionExpert (12L, ~311M)
  → actions (chunk_size, 7) via flow matching
```

**Pipeline**: 2-stage
1. **VLM cache** (Qwen3-VL forward, 1회): `commands/preprocess.sh`
2. **Action expert training** (flow matching, pmap 4-dev): `commands/train.sh`

**Recipes (`PipelineConfig` presets):**
| Preset | chunk | proprio | cams | 샘플 | 용도 |
|--------|-------|---------|------|-------|------|
| `calvin-debug` | 50 | 15 | top | 10 | Dev |
| `calvin-abcd` | 50 | 15 | top | 15k | Baseline |
| **`calvin-abcd-flower`** | **10** | **8** | **top+wrist** | **53k** | **Recommended** (FLOWER recipe) |

## Directory

| Path | 목적 |
|------|------|
| `src/qwen/` | Qwen3-VL, Qwen3.5, VLA (JAX/Flax NNX) |
| `src/qwen/vla/` | VLA pipeline: models, training, data, config |
| `commands/` | Shell wrappers ([README](commands/README.md)) |
| `scripts/` | Python entry points ([README](scripts/README.md)) |
| `bench/` | External benchmarks (calvin, ogbench) |
| `compare/` | Numerical validation (HF vs JAX) |
| `result/` | Training outputs (gitignored) |

## Key Features

- **pmap 4-device data parallel training** (1,200 samples/s)
- **Queue-based VLM cache pipeline** (CPU/TPU zero idle)
- **pi0 flow matching** (openpi-compatible, 7-dim action including gripper)
- **FLOWER recipe** (chunk=10, proprio 8-dim, 2-cam composite, 4-step denoise)
- **Host RAM numpy cache** (35 GB no HBM OOM)
- **Multi-process CALVIN sim** (N pybullet workers + batched TPU inference)
- **Official CALVIN benchmark** (1000 chain evaluation, matches `evaluate_policy.py`)

## Environment

- TPU v4-8 (4 chips, 132 GB HBM, 275 TFLOPS/chip bf16)
- 240 vCPU, 400 GB RAM
- Python 3.10, JAX 0.6.2, Flax 0.10.7
- No CUDA required (JAX/TPU only; torch CPU for CALVIN env)

## Documentation

- **[CLAUDE.md](./CLAUDE.md)** — Primary reference for setup, commands, design decisions
- **[commands/README.md](./commands/README.md)** — Shell script reference
- **[scripts/README.md](./scripts/README.md)** — Python entry points
- **[src/qwen/vla/README.md](./src/qwen/vla/README.md)** — VLA architecture
- **[bench/calvin/README.md](./bench/calvin/README.md)** — CALVIN install details
- **[bench/ogbench/README.md](./bench/ogbench/README.md)** — OGBench details
