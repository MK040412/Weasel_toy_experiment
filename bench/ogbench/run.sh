#!/bin/bash
# OGBench benchmark runner
# Usage: bash bench/ogbench/run.sh [env_name] [agent]

set -euo pipefail

OGBENCH_DIR="${OGBENCH_DIR:-/home/perelman/ogbench}"
RESULT_DIR="$(cd "$(dirname "$0")/../../result/ogbench" && pwd)"
ENV_NAME="${1:-antmaze-large-navigate-v0}"
AGENT="${2:-agents/gciql.py}"

export MUJOCO_GL=egl

echo "=== OGBench Benchmark ==="
echo "Env: $ENV_NAME"
echo "Agent: $AGENT"
echo "Results: $RESULT_DIR"

cd "$OGBENCH_DIR/impls"
python main.py \
    --env_name="$ENV_NAME" \
    --agent="$AGENT" \
    --save_dir="$RESULT_DIR"
