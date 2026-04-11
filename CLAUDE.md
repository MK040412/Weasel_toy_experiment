# Weasel Toy Experiment

TPU v4-8에서 돌리는 toy project 모음. VLM, offline RL, robot manipulation 실험.

## Setup

```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git
cd Weasel_toy_experiment
uv sync

# Qwen 모델 가중치 (HF_TOKEN 환경변수 필요)
export HF_TOKEN=<your_huggingface_token>
mkdir -p ../models/qwen3-vl-2b ../models/qwen35-0.8b
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --include "*.safetensors" --local-dir ../models/qwen3-vl-2b
huggingface-cli download Qwen/Qwen3.5-0.8B --include "*.safetensors" --local-dir ../models/qwen35-0.8b
```

## Directory Structure

```
src/qwen/           모델 코드 (패키지: from qwen.* import)
  qwen3vl/           Qwen3-VL 2B (vision+text, 28L decoder, GQA)
  qwen35/            Qwen3.5-0.8B (GDN linear attn + full attn)
  inference.py       통합 추론 스크립트
  train.py           학습 벤치마크 (single/multi-device)
bench/               벤치마크 실행 스크립트
  ogbench/           offline GCRL (외부 ogbench repo 래핑)
  calvin/            robot manipulation (외부 calvin repo 래핑)
compare/             HF↔JAX 수치 검증
result/              벤치마크 결과물 저장
  ogbench/           train.csv, eval.csv
  calvin/            rollout videos, metrics
```

## Quick Commands

```bash
# Inference
python src/qwen/inference.py --model qwen3vl
python src/qwen/inference.py --model qwen35

# Training benchmark
python src/qwen/train.py --mode single
python src/qwen/train.py --mode multi
python src/qwen/train.py --mode both

# OGBench
bash bench/ogbench/run.sh antmaze-large-navigate-v0 agents/gciql.py

# CALVIN
bash bench/calvin/run.sh

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
