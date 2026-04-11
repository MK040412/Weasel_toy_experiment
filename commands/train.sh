#!/bin/bash
# VLA Training (requires preprocessed VLM cache)
#
# Usage:
#   bash commands/train.sh                               # calvin-debug baseline
#   bash commands/train.sh calvin-abcd                   # ABCD-D baseline
#   bash commands/train.sh calvin-abcd --simulated-delay 15  # ABCD-D with RTC
#   bash commands/train.sh calvin-debug --epochs 200 --lr 2e-4

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
    *)
        echo "Unknown env: $ENV"
        exit 1
        ;;
esac

# Check VLM cache exists
if [ ! -f "$OUTPUT_DIR/vlm_cache/embeddings.parquet" ]; then
    echo "ERROR: VLM cache not found at $OUTPUT_DIR/vlm_cache/"
    echo "Run: bash commands/preprocess.sh $ENV"
    exit 1
fi

echo "=== Training: $ENV ==="
echo "Cache: $OUTPUT_DIR/vlm_cache/"
echo "Extra args: $@"

PYTHONPATH=src python src/qwen/vla/train.py --env "$ENV" --output-dir "$OUTPUT_DIR" "$@"
