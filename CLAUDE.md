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
