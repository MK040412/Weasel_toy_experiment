#!/bin/bash
# CALVIN Benchmark — TPU JAX policy + multiprocess parallel sim envs
#
# Usage:
#   bash commands/benchmark.sh calvin-abcd                            # baseline ckpt
#   bash commands/benchmark.sh calvin-abcd-flower --num-sequences 100 --num-workers 16
#   bash commands/benchmark.sh calvin-abcd-flower --save-videos 10

set -euo pipefail
cd "$(dirname "$0")/.."

ENV="${1:-calvin-abcd}"
shift 2>/dev/null || true

case "$ENV" in
    calvin-abcd)
        CKPT="result/vla_abcd/checkpoint_train_final.npz"
        OUTPUT_DIR="result/vla_abcd/benchmark"
        PROPRIO_DIM=15
        CHUNK_SIZE=50
        ;;
    calvin-abcd-flower)
        CKPT="result/vla_abcd_flower/checkpoint_train_final.npz"
        OUTPUT_DIR="result/vla_abcd_flower/benchmark"
        PROPRIO_DIM=8
        CHUNK_SIZE=10
        ;;
    *)
        echo "Unknown env: $ENV"
        exit 1
        ;;
esac

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

export PYOPENGL_PLATFORM=osmesa
export MESA_GL_VERSION_OVERRIDE=3.3
unset DISPLAY 2>/dev/null || true

echo "=== CALVIN Benchmark: $ENV ==="
echo "Checkpoint: $CKPT"
echo "chunk_size=$CHUNK_SIZE, proprio_dim=$PROPRIO_DIM"

PYTHONUNBUFFERED=1 \
PYTHONPATH="src:/home/perelman/calvin/calvin_env:/home/perelman/calvin/calvin_models" \
.venv/bin/python scripts/benchmark_calvin_mp.py \
    --checkpoint "$CKPT" \
    --output-dir "$OUTPUT_DIR" \
    --chunk-size "$CHUNK_SIZE" \
    --proprio-dim "$PROPRIO_DIM" \
    "$@"
