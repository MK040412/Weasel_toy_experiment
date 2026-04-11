# OGBench Benchmark

Offline Goal-Conditioned RL benchmark. Uses [seohongpark/ogbench](https://github.com/seohongpark/ogbench).

## Setup

```bash
# ogbench 별도 설치 (이 repo 외부)
git clone https://github.com/seohongpark/ogbench.git /path/to/ogbench
cd /path/to/ogbench && uv venv && uv pip install -e ".[all]"
cd impls && uv pip install -r requirements.txt
```

## Run

```bash
# 단일 에이전트 실행
bash bench/ogbench/run.sh

# 또는 직접:
cd /path/to/ogbench/impls
MUJOCO_GL=egl python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/gciql.py \
  --save_dir=$(pwd)/../../Weasel_toy_experiment/result/ogbench/
```

## Agents

GCBC, GCIVL, GCIQL, QRL, CRL, HIQL

## Results

Output: `result/ogbench/<run_name>/train.csv`, `eval.csv`, `flags.json`
