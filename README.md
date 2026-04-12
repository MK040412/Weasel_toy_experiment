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
bash commands/train.sh calvin-abcd-flower --mode cached             # 2.5 hours
bash commands/benchmark.sh calvin-abcd-flower --num-sequences 100   # 30 min
```

Result: `result/vla_abcd_flower/benchmark/results.json` with success rates.

### 3. OGBench (Offline GCRL)

```bash
git clone https://github.com/seohongpark/ogbench.git ~/ogbench
cd ~/ogbench && uv venv && uv pip install -e ".[all]"
cd impls && uv pip install -r requirements.txt
export OGBENCH_DIR=~/ogbench

cd /path/to/Weasel_toy_experiment
bash commands/bench_ogbench.sh antmaze-large-navigate-v0 agents/gciql.py
```

## Training Modes (`--mode cached` vs `--mode online`)

### Overview

이 repo의 VLA trainer는 **두 가지 학습 모드**를 지원합니다. VLM (Qwen3-VL 2B)의 출력을 **미리 계산해서 저장** (`cached`)하거나, **학습 중에 매 step forward** (`online`)하는 차이입니다.

```bash
bash commands/train.sh <env> --mode cached    # pre-compute, fast per-step
bash commands/train.sh <env> --mode online    # compute on-the-fly, flexible
```

### Mode 1: `--mode cached` (default, 권장)

**동작:**
1. VLM cache parquet을 생성 (`commands/preprocess.sh` 또는 자동)
2. 학습 시 cache를 host RAM에 numpy로 올림
3. 매 step: batch 인덱싱 + HBM 전송 + action expert forward/backward

**장점:**
- **빠름**: VLM forward 제외 → per-step ~85ms (pmap 4-dev, bs=128)
- 여러 실험 재실행 시 cache 재사용
- `lr`, `epochs`, `simulated-delay` 등 다른 hyperparameter 테스트 시 cache 공유

**단점:**
- Cache 크기 제약: `N_samples × seq_len × d_model × dtype_bytes`
- 예: stride=1 ABCD-D (961k samples) → float32 632 GB, float16 316 GB (RAM 400 GB 초과)
- stride=25 flower (53k) → 35 GB ✓, stride=5 (193k) → 128 GB ✓

**워크플로우:**

```bash
# A. Split (preprocessing 1회, 여러 학습)
bash commands/preprocess.sh calvin-abcd-flower             # 30 min
bash commands/train.sh calvin-abcd-flower --mode cached    # ~2.5h
bash commands/train.sh calvin-abcd-flower --mode cached --lr 2e-4 --epochs 100  # 다른 lr로 재학습 (cache 재사용)

# B. All-in-one (cache 없으면 자동 생성)
bash commands/train.sh calvin-abcd-flower --mode cached    # preprocess + train 한 번에
```

### Mode 2: `--mode online` (FLOWER-style)

**동작:**
1. Dataset에서 batch 로드 (ThreadPool, PNG decode)
2. 매 step에서 VLM forward (vision pmap + batched language) — gradient 없음
3. Action expert forward/backward (pmap pmean)

**장점:**
- **Cache 불필요** → 메모리/디스크 제약 없음
- **stride=1 (full data) 가능** — 961k samples 학습
- 한 번에 여러 실험 시 cache 관리 안 해도 됨
- 진짜 random shuffle (cache의 shard 제약 없음)

**단점:**
- **느림**: VLM forward가 매 step 포함 → per-step ~1500ms
- Throughput ~80 samples/s (vs cached mode ~1500 samples/s)
- 여러 실험 실행 시 매번 VLM forward 반복 (낭비)

**워크플로우:**

```bash
# C. Online (cache skip, 바로 학습)
bash commands/train.sh calvin-abcd-flower-full --mode online --epochs 5
```

### 모드 선택 기준

| 상황 | 추천 모드 |
|------|----------|
| 작은 dataset (~50k samples) | **cached** |
| 중간 dataset (~200k) | **cached** |
| 큰 dataset (>500k) | **online** (cache OOM 위험) |
| stride=1 full data | **online** |
| 여러 hyperparameter 실험 | **cached** (cache 재사용) |
| 1회성 학습 + 최대 데이터 | **online** |
| 적은 GPU 메모리 (TPU HBM <30 GB) | **cached** (host RAM 활용) |

### Preset별 기본 권장 모드

| Preset | samples | 권장 모드 | 이유 |
|--------|---------|----------|------|
| `calvin-debug` | 10 | cached | 빠른 개발 |
| `calvin-abcd` | 15k | cached | 작음 |
| `calvin-abcd-flower` | 53k | cached | 35 GB cache fit |
| `calvin-abcd-flower-full` | **961k** (stride=1) | **online** | Cache 너무 큼 |

## Architecture

```
Images(top+wrist) + Language → Qwen3-VL 2B (frozen, JAX) → hidden (2048)
  → obs_proj (2048→1536)
  → [proprio(1,8) + obs_tokens] → GemmaActionExpert (12L, ~311M)
  → actions (chunk_size, 7) via flow matching
```

**Pipeline**: 2-stage (cached) or 1-stage (online)
- **Cached**: `preprocess.sh` → `train.sh --mode cached` (separate stages)
- **Online**: `train.sh --mode online` (VLM inline in training loop)

**Recipes (`PipelineConfig` presets):**
| Preset | chunk | proprio | cams | 샘플 | 기본 모드 |
|--------|-------|---------|------|-------|----------|
| `calvin-debug` | 50 | 15 | top | 10 | cached |
| `calvin-abcd` | 50 | 15 | top | 15k | cached |
| **`calvin-abcd-flower`** | **10** | **8** | **top+wrist** | **53k** | **cached** (권장) |
| `calvin-abcd-flower-full` | 10 | 8 | top+wrist | 961k (stride=1) | **online** |

## Directory

| Path | 목적 |
|------|------|
| `src/qwen/` | Qwen3-VL, Qwen3.5, VLA (JAX/Flax NNX) |
| `src/qwen/vla/` | VLA pipeline: models, training, data, config |
| `src/qwen/vla/training/trainer.py` | Cached trainer (fast) |
| `src/qwen/vla/training/online_trainer.py` | Online trainer (FLOWER-style) |
| `commands/` | Shell wrappers ([README](commands/README.md)) |
| `scripts/` | Python entry points ([README](scripts/README.md)) |
| `bench/` | External benchmarks (calvin, ogbench) |
| `compare/` | Numerical validation (HF vs JAX) |
| `result/` | Training outputs (gitignored) |

## Key Features

- **Two training modes** — cached (fast) / online (flexible)
- **pmap 4-device data parallel training** (1,200 samples/s cached, ~80 samples/s online)
- **Queue-based VLM cache pipeline** (CPU/TPU zero idle)
- **pi0 flow matching** (openpi-compatible, 7-dim action including gripper)
- **FLOWER recipe** (chunk=10, proprio 8-dim, 2-cam composite, 4-step denoise)
- **Host RAM numpy cache** with float16 support (avoids HBM OOM)
- **Async prefetch** (host→device transfer overlaps with TPU compute)
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
