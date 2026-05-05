#!/bin/bash
# VLA Training with two modes
#
# Mode options:
#   --mode cached (default)  Use pre-computed VLM cache (requires preprocess.sh first)
#   --mode online            Compute VLM on-the-fly each step (FLOWER-style, no cache)
#
# Multi-host (TPU v4-16, all 16 chips):
#   Run on ALL workers simultaneously via gcloud:
#   gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE> --worker=all \
#     --command="cd ~/Weasel_toy_experiment && bash commands/train.sh calvin-abcd-flower --batch-size 512"
#
# Single-host (TPU v4-8, 8 chips) or testing on one worker:
#   bash commands/train.sh calvin-abcd-flower --no-distributed --batch-size 256
#
# Usage:
#   # Split workflow (recommended for experimentation)
#   bash commands/preprocess.sh calvin-abcd-flower
#   bash commands/train.sh calvin-abcd-flower --mode cached --epochs 200
#
#   # All-in-one (cache auto-generated if missing)
#   bash commands/train.sh calvin-abcd-flower --mode cached
#
#   # Online mode (stride=1 full data, no cache)
#   bash commands/train.sh calvin-abcd-flower-full --mode online --epochs 5

set -euo pipefail
cd "$(dirname "$0")/.."

ENV="${1:-calvin-debug}"
shift 2>/dev/null || true

case "$ENV" in
    calvin-debug)
        OUTPUT_DIR="result/vla"
        ;;
    calvin-abcd)
        OUTPUT_DIR="result/vla_abcd"
        ;;
    calvin-abcd-flower)
        OUTPUT_DIR="result/vla_abcd_flower"
        ;;
    calvin-abcd-flower-full)
        OUTPUT_DIR="result/vla_abcd_flower_full"
        ;;
    *)
        echo "Unknown env: $ENV"
        exit 1
        ;;
esac

# Check if user passed --mode or use default "cached"
MODE="cached"
prev_arg=""
for arg in "$@"; do
    if [ "$prev_arg" = "--mode" ]; then
        MODE="$arg"
    fi
    prev_arg="$arg"
done

# Warn if cached mode without existing cache (auto-computed case is fine but slow)
if [ "$MODE" = "cached" ] && [ ! -f "$OUTPUT_DIR/vlm_cache/embeddings.parquet" ]; then
    echo "NOTE: VLM cache not found at $OUTPUT_DIR/vlm_cache/"
    echo "  Train will auto-compute it (slow first run)."
    echo "  For faster restart: pre-run 'bash commands/preprocess.sh $ENV'"
    echo ""
fi

echo "=== Training: $ENV (mode=$MODE) ==="
echo "Output: $OUTPUT_DIR"
echo "Extra args: $@"

PYTHONUNBUFFERED=1 PYTHONPATH=src python src/qwen/vla/train.py \
    --env "$ENV" --output-dir "$OUTPUT_DIR" "$@"
