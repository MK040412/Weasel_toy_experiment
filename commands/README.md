# commands/ — Shell Wrappers

이 디렉토리의 shell script들은 Python script들(`scripts/`)의 래퍼입니다.

- **환경변수 자동 설정** (PYTHONPATH, CALVIN_DIR, etc.)
- **vCPU 자동 감지** (preprocess.sh의 worker 수)
- **Preset 기반 실행** (env 이름 한 줄로 config 전환)
- **의존성 체크** (checkpoint, 외부 repo 존재 확인)

## 파일 목록

| 스크립트 | 용도 |
|---------|------|
| `download.sh` | HuggingFace → /dev/shm (RAM) 다운로드 |
| `preprocess.sh` | VLM embedding cache 생성 |
| `train.sh` | VLA action expert 학습 |
| `benchmark.sh` | CALVIN sim benchmark |
| `eval.sh` | Offline eval (val/test split) |
| `bench_ogbench.sh` | OGBench benchmark wrapper |

## VLA Pipeline (CALVIN)

```bash
# 1. 데이터 다운로드 (첫 실행 시 1회, 67 GB → /dev/shm, ~5 min)
export HF_TOKEN=<your_token>
bash commands/download.sh calvin-abcd

# 2. VLM cache 생성 (1회, ~30 min)
bash commands/preprocess.sh calvin-abcd-flower

# 3. 학습 (200 epochs × ~44s/epoch = ~2.5 hours)
bash commands/train.sh calvin-abcd-flower --epochs 200 --batch-size 128 --lr 1e-4

# 4. CALVIN sim benchmark (~20 min for 20 seqs × 8 workers)
bash commands/benchmark.sh calvin-abcd-flower --num-sequences 100 --num-workers 16

# (Optional) Offline eval without sim
bash commands/eval.sh calvin-abcd-flower val
```

### 지원 환경 (preset)

| Env | chunk | proprio | cams | 샘플 수 | 용도 |
|-----|-------|---------|------|--------|------|
| `calvin-debug` | 50 | 15 | top | 10 | Debug/개발 |
| `calvin-abcd` | 50 | 15 | top | 15k | 초기 baseline |
| **`calvin-abcd-flower`** | **10** | **8** | **top+wrist** | **53k** | **Recommended** (FLOWER recipe) |

## OGBench (offline GCRL)

```bash
# OGBench repo는 별도 설치 (이 repo와 독립)
git clone https://github.com/seohongpark/ogbench.git ~/ogbench
cd ~/ogbench && uv venv && uv pip install -e ".[all]"
cd impls && uv pip install -r requirements.txt

# 실행
export OGBENCH_DIR=~/ogbench
bash commands/bench_ogbench.sh                                         # default
bash commands/bench_ogbench.sh antmaze-large-navigate-v0 agents/gciql.py
bash commands/bench_ogbench.sh antmaze-large-navigate-v0 agents/hiql.py
```

### 지원 agents

GCBC, GCIVL, GCIQL, QRL, CRL, HIQL

## 환경변수

| Var | Default | 용도 |
|-----|---------|------|
| `HF_TOKEN` | — | HuggingFace 인증 (다운로드) |
| `QWEN3VL_MODEL_PATH` | `../models/qwen3-vl-2b` | Qwen3-VL 체크포인트 경로 |
| `CALVIN_DIR` | `$HOME/calvin` | CALVIN repo 경로 |
| `OGBENCH_DIR` | `$HOME/ogbench` | OGBench repo 경로 |

## 결과 파일 위치

```
result/
  vla/                  calvin-debug 결과
  vla_abcd/             calvin-abcd 결과
  vla_abcd_flower/      calvin-abcd-flower 결과 (권장 setting)
    vlm_cache/           embeddings.parquet (35 GB)
    checkpoint_train_final.npz
    train_log.csv        epoch별 loss, lr, throughput
    benchmark/           results.json + success/failure MP4s
  ogbench/              OGBench 결과 (<run_name>/train.csv, eval.csv)
```
