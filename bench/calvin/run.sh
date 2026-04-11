#!/bin/bash
# CALVIN random rollout benchmark
# Usage: bash bench/calvin/run.sh

set -euo pipefail

CALVIN_DIR="${CALVIN_DIR:-/home/perelman/calvin}"
RESULT_DIR="$(cd "$(dirname "$0")/../../result/calvin" && pwd)"

export PYOPENGL_PLATFORM=osmesa
export MESA_GL_VERSION_OVERRIDE=3.3
unset DISPLAY 2>/dev/null || true

echo "=== CALVIN Benchmark ==="
echo "Calvin: $CALVIN_DIR"
echo "Results: $RESULT_DIR"

source "$CALVIN_DIR/.venv/bin/activate"
python "$(dirname "$0")/random_rollout.py" --calvin-dir "$CALVIN_DIR" --output-dir "$RESULT_DIR"
