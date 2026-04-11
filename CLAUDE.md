# Weasel Toy Experiment

TPU v4-8에서 돌리는 toy project 모음. VLM, offline RL, robot manipulation 실험.
VLA (Vision-Language-Action) flow matching은 CUDA GPU에서 실행.

## Setup

```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment
uv sync                    # JAX (TPU) 의존성
uv sync --extra vla        # PyTorch (CUDA) VLA 의존성

# Qwen 모델 가중치 (HF_TOKEN 환경변수 필요)
export HF_TOKEN=<your_huggingface_token>
mkdir -p ../models/qwen3-vl-2b ../models/qwen35-0.8b
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --include "*.safetensors" --local-dir ../models/qwen3-vl-2b
huggingface-cli download Qwen/Qwen3.5-0.8B --include "*.safetensors" --local-dir ../models/qwen35-0.8b
```

## Directory Structure

```
src/qwen/           모델 코드 (패키지: from qwen.* import)
  qwen3vl/           Qwen3-VL 2B (vision+text, 28L decoder, GQA) [JAX]
  qwen35/            Qwen3.5-0.8B (GDN linear attn + full attn) [JAX]
  vla/               VLA: Qwen3-VL + pi0-style action expert [PyTorch]
    config.py          설정 dataclass
    models/            GemmaActionExpert (~311M), VLAPolicy, transformer layers
    data/              LeRobot CALVIN v2.1, action quantile normalization
    training/          Flow matching scheduler (openpi0.5), 2-stage trainer
    train.py           VLA 학습 CLI
    inference.py       VLA 추론 CLI
  inference.py       JAX 통합 추론 스크립트
  train.py           JAX 학습 벤치마크 (single/multi-device)
bench/               벤치마크 실행 스크립트
  ogbench/           offline GCRL (외부 ogbench repo 래핑)
  calvin/            robot manipulation (외부 calvin repo 래핑)
  vla/               VLA 학습/추론/비교 wrapper
compare/             검증 및 비교
  compare_rtc.py     VLA: Baseline vs RTC ablation
result/              벤치마크 결과물 저장
  ogbench/           train.csv, eval.csv
  calvin/            rollout videos, metrics
  vla/               inference JSON, comparison plots
```

## Quick Commands

```bash
# JAX Inference
python src/qwen/inference.py --model qwen3vl
python src/qwen/inference.py --model qwen35

# JAX Training benchmark
python src/qwen/train.py --mode both

# VLA Training (CUDA)
python src/qwen/vla/train.py
python src/qwen/vla/train.py --simulated-delay 15

# VLA Inference (CUDA)
python src/qwen/vla/inference.py --checkpoint checkpoint_final.pt

# VLA Ablation
python compare/compare_rtc.py

# Benchmarks
bash bench/ogbench/run.sh antmaze-large-navigate-v0 agents/gciql.py
bash bench/calvin/run.sh
bash bench/vla/run.sh train

# Validation
python compare/compare_blocks.py
```

## VLA (Vision-Language-Action) 분석 및 실행 가이드

### 아키텍처
```
Images(top+wrist) + Language -> Qwen3-VL 2B (frozen, PyTorch) -> hidden(2048)
  -> obs_proj(2048->1536) -> GemmaActionExpert (12L, ~311M) -> actions(50,7)
학습: Flow Matching (openpi0.5), 2-stage (VLM cache -> expert train)
```

### 현재 환경 제약
- PyTorch CPU-only (CUDA 없음), TPU는 JAX 전용
- **해결**: VLM 임베딩 1회 캐싱 후 action expert만 CPU 학습 (trainer.py Stage 1이 이미 이 패턴)
- debug dataset 작으므로(~600 episodes) CPU 학습 실용적

### 데이터셋
- **HuggingFace**: `fywang/calvin-debug-lerobot` (LeRobot v2.1 CALVIN format)
- Action: (7,) = [x, y, z, rx, ry, rz, gripper] @ 30Hz
- Cameras: top (static), wrist (gripper)
- 정규화: quantile (q01, q99) -> [-1, 1]

### Gripper 처리 (구현 필요)
현재 gripper를 연속값으로 처리하지만, 실제로는 이산(open/close).
- **방안**: gripper를 별도 이진 분류 head로 분리
- continuous 6-dim (pos+orn) + discrete 1-dim (gripper BCE loss)
- flow matching은 6-dim에만 적용, gripper는 sigmoid + threshold

### CUDA 의존성 제거 필요 파일
| 파일 | 수정 내용 |
|------|----------|
| `models/vla.py:42-46` | `device_map="auto"` → `device_map=None`, CPU 명시 |
| `models/vla.py:96,117` | `torch.cuda.empty_cache()` → 조건부 or 제거 |
| `training/trainer.py:25-27` | device fallback 이미 있음, CPU 동작 확인 |
| `train.py:62-63` | CUDA seed 분기 → CPU 전용 |

### 실행 순서 (구현 후)
```bash
# 1. 의존성 설치
uv sync --extra vla

# 2. 데이터셋 다운로드 (자동, HuggingFace)
python -c "from qwen.vla.data.lerobot_calvin import LeRobotCalvinDataset; LeRobotCalvinDataset()"

# 3. 학습 (CPU, debug dataset)
python src/qwen/vla/train.py --batch-size 1 --grad-accum 4 --stage1-epochs 10

# 4. 추론 + 시각화
python src/qwen/vla/inference.py --checkpoint checkpoint_final.pt --visualize

# 5. 과적합 검증 (train set 궤적 재현 확인)
python src/qwen/vla/inference.py --checkpoint checkpoint_final.pt --split train --memorization-check
```

### 시각화 요구사항 (구현 필요)
- rollout video + language annotation 오버레이
- 예측 궤적 vs GT 궤적 비교
- gripper open/close 타이밍 표시
- MP4 출력 → `result/vla/`

## Model Path Override

```bash
export QWEN3VL_MODEL_PATH=/path/to/qwen3-vl-2b
export QWEN35_MODEL_PATH=/path/to/qwen35-0.8b
```

Default: `../models/qwen3-vl-2b`, `../models/qwen35-0.8b` (relative to repo root)

## Git Push

```bash
# GH_TOKEN 환경변수 필요
git remote set-url origin https://${GH_TOKEN}@github.com/MK040412/Weasel_toy_experiment.git
git push origin master
```
