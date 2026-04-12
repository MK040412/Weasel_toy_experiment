#!/bin/bash
# Long-running VLA training (hours to days)
# - Runs in background with logging
# - Per-epoch checkpoint saving (for long runs)
# - Automatic log monitoring via tail
#
# Usage:
#   bash commands/train_long.sh calvin-abcd-flower-full online 3     # 3 epochs online
#   bash commands/train_long.sh calvin-abcd-flower cached 200        # 200 epochs cached
#   bash commands/train_long.sh calvin-abcd-flower-full online 3 --lr 2e-4  # override
#
# Monitor progress:
#   tail -f result/vla_{env}_full/train_long.log
#
# Kill:
#   ps aux | grep train.py | grep -v grep | awk '{print $2}' | xargs kill

set -euo pipefail
cd "$(dirname "$0")/.."

ENV="${1:-calvin-abcd-flower-full}"
MODE="${2:-online}"
EPOCHS="${3:-3}"
shift 3 2>/dev/null || true

case "$ENV" in
    calvin-debug)            OUTPUT_DIR="result/vla" ;;
    calvin-abcd)             OUTPUT_DIR="result/vla_abcd" ;;
    calvin-abcd-flower)      OUTPUT_DIR="result/vla_abcd_flower" ;;
    calvin-abcd-flower-full) OUTPUT_DIR="result/vla_abcd_flower_full" ;;
    *) echo "Unknown env: $ENV"; exit 1 ;;
esac

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train_long.log"
PID_FILE="$OUTPUT_DIR/train_long.pid"

# Check if already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "ERROR: Training already running (PID=$(cat $PID_FILE))"
    echo "Log: $LOG_FILE"
    echo "To kill: kill $(cat $PID_FILE)"
    exit 1
fi

echo "=== Long Training: $ENV (mode=$MODE, epochs=$EPOCHS) ==="
echo "Output: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Extra args: $@"
echo ""
echo "Per-epoch checkpoints: $OUTPUT_DIR/checkpoint_ep<N>.npz"
echo "Final checkpoint: $OUTPUT_DIR/checkpoint_train_final.npz"
echo ""

# Start training in background with output redirected to log file
PYTHONUNBUFFERED=1 PYTHONPATH=src nohup .venv/bin/python src/qwen/vla/train.py \
    --env "$ENV" \
    --mode "$MODE" \
    --epochs "$EPOCHS" \
    --output-dir "$OUTPUT_DIR" \
    "$@" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "Training started in background"
echo "  PID: $TRAIN_PID"
echo "  Log: $LOG_FILE"
echo ""
echo "Commands:"
echo "  tail -f $LOG_FILE          # live log"
echo "  ls -lh $OUTPUT_DIR/checkpoint_ep*.npz  # list saved checkpoints"
echo "  kill $TRAIN_PID            # stop training"
echo ""
sleep 2
# Show first few lines of log to confirm it started
if [ -f "$LOG_FILE" ]; then
    echo "=== Initial log output ==="
    head -20 "$LOG_FILE"
fi
