# Weasel Toy Experiment

TPU v4-8에서 돌리는 toy project 모음. VLM, offline RL, robot manipulation 실험.

## Setup

```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment
uv sync

# Qwen 모델 가중치
export HF_TOKEN=<your_huggingface_token>
mkdir -p ../models/qwen3-vl-2b ../models/qwen35-0.8b
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --include "*.safetensors" --local-dir ../models/qwen3-vl-2b
huggingface-cli download Qwen/Qwen3.5-0.8B --include "*.safetensors" --local-dir ../models/qwen35-0.8b

# CALVIN benchmark 실행 시 추가 의존성 (pybullet + calvin_env)
uv pip install pybullet "hydra-core==1.1.1" gym omegaconf opencv-python \
  numpy-quaternion gitpython torch --index-url https://download.pytorch.org/whl/cpu \
  pytorch-lightning termcolor hydra-colorlog tacto

# CALVIN repo (sim env)
# ~/calvin 에 이미 설치되어 있다고 가정. 없으면 bench/calvin/README.md 참조.
```

## Quickstart (CALVIN ABCD-D Benchmark 전체 파이프라인)

```bash
# 1. 데이터셋 RAM 다운로드 (~5 min, 67 GB → /dev/shm)
bash commands/download.sh calvin-abcd

# 2. VLM 임베딩 전처리 (FLOWER recipe, ~30 min, 53k samples)
bash commands/preprocess.sh calvin-abcd-flower

# 3. Action expert 학습 (~2.5 hours, 200 epochs, pmap 4-dev)
bash commands/train.sh calvin-abcd-flower --epochs 200 --batch-size 128 --lr 1e-4

# 4. CALVIN sim benchmark (success rate + MP4)
bash commands/benchmark.sh calvin-abcd-flower --num-sequences 100 --num-workers 16

# 결과: result/vla_abcd_flower/{train_log.csv, checkpoint_train_final.npz, benchmark/}
```

## Design Principles

### 1. Config-Driven (Dataclass Presets)

모든 설정은 `PipelineConfig` dataclass 계층. 환경별 preset으로 한 줄 전환:

```python
cfg = PipelineConfig.calvin_debug()        # 10 chunks
cfg = PipelineConfig.calvin_abcd()         # 15k chunks, chunk=50, proprio=15
cfg = PipelineConfig.calvin_abcd_flower()  # 53k chunks, chunk=10, proprio=8, 2 cams (FLOWER recipe)
cfg.training.lr = 1e-4                     # override
```

`EnvConfig` 고유 필드: `action_dim, proprio_dim, cameras, image_size, chunk_size, stride, local_path`.
새 환경 추가 시 `EnvConfig.new_env()` classmethod + `@register_dataset("new-env")` 만.

### 2. Protocol + Registry (Dataset)

`VLADataset` Protocol. `@register_dataset("name")` 데코레이터로 자동 등록.
Trainer/VLMCacher는 protocol만 의존.

### 3. Separation of Concerns

```
VLMCacher     — VLM embedding 전처리 (compute/save/load). Queue-based pipeline.
VLATrainer    — Action expert 학습만 (VLMCache 받음). pmap data-parallel.
PipelineConfig — 모든 하이퍼파라미터 한 곳에.
```

### 4. pi0 Convention (Flow Matching)

openpi (`Physical-Intelligence/openpi`) 기준:
- **7-dim 통합 flow matching**: gripper 포함, 별도 head 없음
- **Proprio conditioning**: prefix tokens에 1개 추가 ([proprio, obs_tokens, actions])
- **Timestep**: `Beta(1.5, 1.0)` → [0.001, 1.0], velocity target = noise - actions
- **Euler denoising** (4 steps @ FLOWER, 10 steps @ baseline)
- **bf16 mixed precision**: forward bf16, loss f32
- **pmap 4-device**: data parallel with `jax.lax.pmean` gradient all-reduce

### 5. FLOWER Recipe (CALVIN 최적 설정)

`intuitive-robots/flower_vla_calvin` 기반:

| 항목 | FLOWER | 우리 구현 |
|------|--------|----------|
| chunk_size | 10 | 10 ✓ |
| proprio_dim | 8 | 8 ([0:7] + [14:15] from 15-dim state) ✓ |
| cameras | top + wrist | top + wrist (vstack composite) ✓ |
| denoising | 4 steps | 4 steps ✓ |
| batch_size | 8 × 4 GPU | 128 (32/dev × 4 TPU) |
| total steps | 40k | 83k (200 epochs × 415 steps) |
| precision | bf16 | bf16 ✓ |

## Directory Structure

```
src/qwen/vla/
  config.py               PipelineConfig + EnvConfig preset classmethods
  models/
    action_expert.py       GemmaActionExpert (prefix-LM, proprio projection)
    vla.py                 VLAPolicy (VLM frozen + obs_proj + action expert)
    layers.py              GQAttention, SwiGLU, RMSNorm
  data/
    protocol.py            VLADataset Protocol + DATASET_REGISTRY
    lerobot_calvin.py      CalvinDataset (supports all calvin-* envs)
  training/
    vlm_cache.py           VLMCacher + VLMCache (numpy cache, queue pipeline)
    trainer.py             VLATrainer (pmap 4-dev data parallel)
    flow_matching.py       Beta(1.5,1) timestep, make_noisy, velocity target, RTC
  train.py                 CLI: --env calvin-debug|calvin-abcd|calvin-abcd-flower
  eval_and_viz.py          debug dataset train + 3D trajectory viz

scripts/                   Standalone Python scripts
  preprocess_vlm_cache.py  Standalone VLM caching (used by commands/preprocess.sh)
  benchmark_calvin.py      CALVIN sim benchmark (single-process, sequential envs)
  benchmark_calvin_mp.py   CALVIN benchmark (multiprocessing parallel sim workers)
  eval_offline.py          Offline eval: pos_err/grip_acc on val split (no sim)

commands/                  Shell wrappers (auto-detect vCPU, set paths)
  download.sh              RAM download: HF → /dev/shm (ThreadPool 64 workers)
  preprocess.sh            VLM cache generation (auto-scale workers = 75% vCPU)
  train.sh                 Training with preset env
  benchmark.sh             CALVIN sim benchmark (TPU policy + parallel sim)
  eval.sh                  Offline eval

data/download/fywang/      Dataset download scripts (subset of commands/)

bench/                     Standalone benchmark wrappers (calvin, ogbench)
compare/                   Numerical validation scripts
result/                    Outputs (gitignored except train_log.csv)
  vla/                     calvin-debug
  vla_abcd/                calvin-abcd (baseline, chunk=50)
  vla_abcd_flower/         calvin-abcd-flower (FLOWER recipe, chunk=10)
```

## Command Reference

### `commands/download.sh ENV`

`/dev/shm`에 데이터셋 병렬 다운로드 (64 threads).
환경: `calvin-abcd` (fywang/calvin-task-ABCD-D-lerobot, 67 GB, ~5 min).

### `commands/preprocess.sh ENV [LOCAL_PATH]`

VLM cache 생성 (pmap 4-dev vision + batched language, queue pipeline).
자동 vCPU 감지 → 75% 사용 (`--workers N`).
출력: `result/vla_{env}/vlm_cache/embeddings.parquet` + `meta.json`.

| ENV | 샘플 수 | 시간 |
|-----|---------|------|
| calvin-debug | 10 | ~30 sec |
| calvin-abcd | 15,186 | ~8 min |
| calvin-abcd-flower | 53,093 | ~30 min |

### `commands/train.sh ENV [--epochs N --lr F --batch-size N ...]`

Action expert 학습. VLM cache 필수.
- pmap 4-device data parallel
- bf16 mixed precision
- 출력: `checkpoint_train_final.npz` + `train_log.csv`

### `commands/benchmark.sh ENV [--num-sequences N --num-workers N]`

CALVIN sim benchmark (공식 `evaluate_policy.py` 로직 동일).
- Main process: TPU JAX policy (batched inference)
- N worker processes: pybullet CALVIN envs (multiprocessing Queue IPC)
- 출력: `benchmark/results.json` + success/failure MP4s

### `commands/eval.sh ENV SPLIT`

Offline eval (no sim). val/test split에서 action prediction, pos_err/grip_acc 계산.

## VLA Pipeline

### 아키텍처

```
Images(top+wrist vstack) + Language → Qwen3-VL 2B (frozen)
  → hidden (B, seq, 2048) → obs_proj (2048→1536)
  → [proprio(1,8) → proprio_proj → 1 token] + [obs_tokens (~112)]
  → GemmaActionExpert (12L prefix-LM, ~311M)
  → actions (B, chunk_size, 7) via flow matching
```

### 2-Stage 파이프라인

```
Stage 1: VLM Embedding Cache (1회, ~30 min for 53k samples)
  CPU 180 workers → Queue → TPU pmap vision + batched lang
  결과: parquet 저장 (35 GB for FLOWER)

Stage 2: Action Expert Training (~2.5 hours for 200 epochs)
  Cache 호스트 RAM 유지 (35 GB numpy)
  Per-batch HBM transfer via jnp.array(cache[batch_idx])
  pmap 4-dev data parallel
  bf16 forward, f32 loss/grad
```

### 주요 최적화

1. **Queue-based preprocessing pipeline**: CPU/TPU overlap → idle zero
2. **pmap 4-dev training**: 571 → 1,200 samples/s (2.1x)
3. **Numpy cache (host RAM)**: HBM OOM 해결 (35 GB cache > 30 GB HBM/chip)
4. **VLM cache parquet**: 1회 전처리 후 영구 재사용

## CALVIN Benchmark 세부사항

**공식 evaluate_policy.py 동일 구성 요소:**
- `multistep_sequences.get_sequences(1000)` — 결정론적 task chain 생성 (seed=0)
- `new_playtable_tasks.yaml` — task oracle
- `new_playtable_validation.yaml` — language annotations
- `get_env_state_for_initial_condition` — initial scene state
- `EP_LEN=360`, 5-task chain, fail on first subtask failure

**Metrics:**
- 1/5, 2/5, 3/5, 4/5, 5/5 subtask success rates
- Avg chain length (0~5)

**구조:**
- Main process: TPU JAX policy, batched inference per pmap call
- N sim workers: `multiprocessing.Process` with pybullet env
- IPC: cmd_queue (reset/step/get_obs/get_info) + res_queue

## 알려진 실험 기록

| Config | VLM cache | Training | Loss (final) | CALVIN avg chain |
|--------|-----------|----------|--------------|------------------|
| baseline (chunk=50, proprio=15) | 8 min | 2.3 h, 1-dev | 0.243 | 0.06 / 5 ❌ |
| **flower (chunk=10, proprio=8, 2 cams, pmap)** | 30 min | 2.5 h, 4-dev | **0.135** | (재검증 필요) |

## Dataset

| | repo_id | chunks | 크기 |
|---|---------|--------|------|
| debug | `fywang/calvin-debug-lerobot` | 10 | 37 MB |
| ABCD→D | `fywang/calvin-task-ABCD-D-lerobot` | 15k/53k | 67 GB |

- Action (7,): delta `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`
- State (15,): FLOWER는 `[0:7] + [14:15]` = 8 dim 사용
- 정규화: quantile (q01, q99) → [-1, 1]

## 대용량 데이터 RAM 관리

디스크 66 GB로 ABCD-D 67 GB 수용 불가 → `/dev/shm` (tmpfs 201 GB) 사용.
`commands/download.sh`가 자동 처리.

## Model Path Override

```bash
export QWEN3VL_MODEL_PATH=/path/to/qwen3-vl-2b
export QWEN35_MODEL_PATH=/path/to/qwen35-0.8b
```

## 새 환경 추가 방법

1. `config.py`: `EnvConfig.new_env()` classmethod + `PipelineConfig.new_env()`
2. `data/new_env.py`: Dataset 구현 + `@register_dataset("new-env")`
3. `data/__init__.py`: import 추가
4. `commands/preprocess.sh` / `train.sh` case에 추가
5. Trainer/VLMCacher 수정 불필요 (protocol만 의존)

## Dependencies

**Core (uv sync):** `jax[tpu]>=0.6.0`, `flax`, `optax`, `transformers`, `pyarrow`, `huggingface-hub`, `imageio`

**CALVIN benchmark (uv pip install):** `pybullet`, `hydra-core==1.1.1`, `gym`, `omegaconf`, `opencv-python`, `numpy-quaternion`, `gitpython`, `torch` (CPU), `pytorch-lightning`, `termcolor`, `hydra-colorlog`, `tacto`

**Dev:** `ruff` (120 chars, E/F/I/W)

## Git Push

```bash
git remote set-url origin https://${GH_TOKEN}@github.com/MK040412/Weasel_toy_experiment.git
git push origin master
```
