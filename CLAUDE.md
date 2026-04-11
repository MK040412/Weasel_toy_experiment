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
```

## Design Principles

### 1. Config-Driven (Dataclass Presets)

모든 설정은 `PipelineConfig` dataclass로 관리. 환경별 preset으로 한 줄 전환:

```python
cfg = PipelineConfig.calvin_debug()   # 10 chunks, lr=5e-5
cfg = PipelineConfig.calvin_abcd()    # 19k chunks, lr=5e-5, RTC d=15
cfg.training.lr = 1e-4                # override
```

`EnvConfig`가 환경 고유 차원(action_dim, proprio_dim, cameras, image_size)을 캡슐화.
새 환경 추가 시 `EnvConfig.new_env()` classmethod만 추가.

### 2. Protocol + Registry (Dataset)

`VLADataset` Protocol이 데이터셋 인터페이스 정의:

```python
class VLADataset(Protocol):
    def __getitem__(self, i) -> dict:   # images, actions, proprio, language, episode
    @property
    def action_dim(self) -> int: ...
    @property
    def proprio_dim(self) -> int: ...
```

`@register_dataset("name")` 데코레이터로 DATASET_REGISTRY에 등록.
`create_dataset(env_config, split)` 으로 config에서 자동 생성.
Trainer/VLMCacher는 protocol만 의존 — 구체 클래스 모름.

### 3. Separation of Concerns

```
VLMCacher     — VLM embedding 전처리 (compute/save/load)
VLATrainer    — Action expert 학습만 (VLMCache 받음)
PipelineConfig — 모든 하이퍼파라미터 한 곳에
```

Trainer가 VLM 캐싱 책임을 지지 않음. VLMCacher는 독립적으로 사용 가능
(download 스크립트, 전처리 파이프라인 등에서 재사용).

### 4. pi0 Convention (Flow Matching)

openpi (`Physical-Intelligence/openpi`) 기준 구현:

- **7-dim 통합 flow matching**: gripper 포함, 별도 head 없음
- **Proprio conditioning**: `observation.state` (15-dim) → 1 토큰으로 prefix에 추가
- **Timestep**: `Beta(1.5, 1.0)` → [0.001, 1.0], velocity target = noise - actions
- **Inference**: Euler 10-step, KV cache로 obs 재계산 안 함
- **bf16 mixed precision**: forward bf16, loss f32

## Directory Structure

```
src/qwen/vla/
  config.py               PipelineConfig (EnvConfig, ModelConfig, TrainingConfig, ...)
  models/
    action_expert.py       GemmaActionExpert (~311M), prefix-LM, proprio projection
    vla.py                 VLAPolicy (VLM + obs_proj + action expert)
    layers.py              GQAttention, SwiGLU, RMSNorm
  data/
    protocol.py            VLADataset Protocol + DATASET_REGISTRY + create_dataset()
    lerobot_calvin.py      CalvinDataset (@register_dataset, PyArrow 직접 로드)
  training/
    vlm_cache.py           VLMCacher (compute/save/load), VLMCache dataclass
    trainer.py             VLATrainer (VLMCache 받아서 학습만)
    flow_matching.py       Flow matching scheduler + RTC
  train.py                 CLI (--env calvin-debug|calvin-abcd)
  eval_and_viz.py          학습 + 평가 + 시각화
  inference.py             추론 CLI
  _pytorch_ref/            PyTorch 아카이브 (미사용)

data/download/fywang/      대규모 데이터셋 RAM 다운로드 스크립트
bench/                     벤치마크 래퍼
compare/                   수치 검증
result/                    결과물
```

## VLA Pipeline

### 아키텍처

```
Images(top) + Language → Qwen3-VL 2B (frozen, JAX) → hidden (2048)
  → obs_proj (2048→1536)
  → [proprio(1, 15) → proprio_proj → 1 token] + [obs tokens (112)]
  → GemmaActionExpert (12L, prefix-LM, ~311M)
  → actions (50, 7) — all dims via flow matching
```

- Prefix: `[proprio_token, obs_tokens]` (113 tokens, bidirectional)
- Suffix: `[action_tokens]` (50 tokens, attend to all)
- Proprio: `observation.state` (TCP pos/orn, joints, gripper) — 첫 프레임만
- Gripper: action dim 6, flow matching으로 continuous 학습 (pi0 방식)

### 2-Stage 파이프라인

```
Stage 1: VLMCacher.compute(dataset, vlm) → VLMCache
  pmap vision (4-dev) + batched language (batch=128)
  결과를 parquet로 저장 → 이후 VLM 로드 불필요

Stage 2: VLATrainer(policy, cache, config).train()
  bf16 mixed precision, AdamW + warmup cosine decay
  HBM-resident 배치 학습
```

### RTC (Recurrent Time Chunking)

`--simulated-delay 15`:

```
chunk:    [a₁, ..., a_d, a_{d+1}, ..., a₅₀]
timestep: [0,  ...,  0,   t,      ...,  t  ]
          ├─ prefix (GT) ─┤ ├─ postfix (noisy) ─┤
```

### 데이터셋

| 데이터셋 | repo_id | chunks | 용도 |
|---------|---------|--------|------|
| Debug | `fywang/calvin-debug-lerobot` | 10 | 개발 |
| ABCD→D | `fywang/calvin-task-ABCD-D-lerobot` | 19,037 | Ablation |

- Action: (7,) delta = [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
- Proprio: (15,) = [TCP pos/orn, joints, gripper state]
- 정규화: quantile (q01, q99) → [-1, 1]

### 대규모 데이터셋 다운로드

디스크 66 GB로 ABCD→D 수용 불가 → `/dev/shm` (tmpfs 201 GB):

```bash
# 병렬 다운로드 (ThreadPoolExecutor, 64 workers) → VLM 캐시 생성 → RAM 정리
PYTHONPATH=src python data/download/fywang/calvin-task-ABCD-D-lerobot.py --cache-vlm --cleanup
```

## Quick Commands

```bash
# ─── VLA (config-driven) ───────────────────────
PYTHONPATH=src python src/qwen/vla/train.py                              # calvin-debug
PYTHONPATH=src python src/qwen/vla/train.py --env calvin-abcd            # ABCD-D
PYTHONPATH=src python src/qwen/vla/train.py --simulated-delay 15         # RTC
PYTHONPATH=src python src/qwen/vla/train.py --epochs 50 --lr 2e-4        # override
PYTHONPATH=src python src/qwen/vla/eval_and_viz.py                       # debug + viz

# ─── JAX ───────────────────────────────────────
python src/qwen/inference.py --model qwen3vl
python src/qwen/train.py --mode both

# ─── Benchmarks ────────────────────────────────
bash bench/ogbench/run.sh antmaze-large-navigate-v0 agents/gciql.py
bash bench/calvin/run.sh
```

## Adding a New Environment

1. `config.py`: `EnvConfig.new_env()` classmethod + `PipelineConfig.new_env()`
2. `data/new_env.py`: Dataset 구현 + `@register_dataset("new-env")`
3. `data/__init__.py`: import 추가 (registry 자동 등록)
4. 끝. Trainer/VLMCacher/모델 수정 불필요.

## Model Path Override

```bash
export QWEN3VL_MODEL_PATH=/path/to/qwen3-vl-2b
export QWEN35_MODEL_PATH=/path/to/qwen35-0.8b
```

## TPU Optimizations

- **bf16 training**: forward bf16, loss/grad f32 (2x matmul speedup)
- **pmap vision**: 4 TPU chip 병렬 vision encoding (`forward_static` — no int() tracing)
- **Batched language**: batch=128 language model forward
- **HBM-resident**: 전체 캐시 on HBM, pure JAX indexing (no Python int() sync)
- **VLM cache parquet**: 1회 전처리 후 영구 재사용

## Dependencies (pyproject.toml)

Core: `jax[tpu]>=0.6.0`, `flax>=0.10.0`, `optax>=0.2.0`, `transformers`, `safetensors`, `pillow`, `pyarrow`, `huggingface-hub`, `imageio`

Dev: `ruff>=0.15.10` (120 chars, E/F/I/W)

## Git Push

```bash
git remote set-url origin https://${GH_TOKEN}@github.com/MK040412/Weasel_toy_experiment.git
git push origin master
```
