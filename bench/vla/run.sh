#!/bin/bash
set -euo pipefail

# VLA training and inference
# Usage:
#   bash bench/vla/run.sh train                          # standard training
#   bash bench/vla/run.sh train --simulated-delay 15     # with RTC
#   bash bench/vla/run.sh inference checkpoint_final.pt   # inference
#   bash bench/vla/run.sh compare ckpt_d0.pt ckpt_d15.pt # RTC ablation

RESULT_DIR="$(cd "$(dirname "$0")/../../result/vla" && pwd)"
MODE="${1:-train}"
shift || true

case "$MODE" in
    train)
        python src/qwen/vla/train.py "$@"
        ;;
    inference)
        CKPT="${1:?checkpoint path required}"
        shift
        python src/qwen/vla/inference.py --checkpoint "$CKPT" --output-dir "$RESULT_DIR" "$@"
        ;;
    compare)
        CKPT_B="${1:?baseline checkpoint required}"
        CKPT_R="${2:?rtc checkpoint required}"
        shift 2
        python compare/compare_rtc.py --ckpt-baseline "$CKPT_B" --ckpt-rtc "$CKPT_R" "$@"
        ;;
    *)
        echo "Usage: $0 {train|inference|compare} [args...]"
        exit 1
        ;;
esac
