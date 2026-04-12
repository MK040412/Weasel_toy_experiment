#!/bin/bash
# Offline evaluation: predict actions on val/test split, compute metrics.
#
# Usage:
#   bash commands/eval.sh                             # calvin-abcd val, default ckpt
#   bash commands/eval.sh calvin-abcd test            # test split
#   bash commands/eval.sh calvin-abcd val --n-steps 20  # custom denoising steps

set -euo pipefail
cd "$(dirname "$0")/.."

ENV="${1:-calvin-abcd}"
SPLIT="${2:-val}"
shift 2 2>/dev/null || true

case "$ENV" in
    calvin-debug) OUTPUT_DIR="result/vla" ;;
    calvin-abcd)  OUTPUT_DIR="result/vla_abcd" ;;
    *) echo "Unknown env: $ENV"; exit 1 ;;
esac

CKPT="$OUTPUT_DIR/checkpoint_train_final.npz"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

echo "=== Offline Eval: $ENV [$SPLIT] ==="
echo "Checkpoint: $CKPT"

PYTHONUNBUFFERED=1 PYTHONPATH=src python scripts/eval_offline.py \
    --env "$ENV" \
    --split "$SPLIT" \
    --checkpoint "$CKPT" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
