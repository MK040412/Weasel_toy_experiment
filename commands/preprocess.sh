#!/bin/bash
# VLM Cache Preprocessing — auto-detects vCPU count for optimal parallelism
#
# Multi-host (TPU v4-16, all 16 chips):
#   gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE> --worker=all \
#     --command="cd ~/Weasel_toy_experiment && bash commands/preprocess.sh calvin-abcd-flower"
#
# Single-host (TPU v4-8) or testing:
#   bash commands/preprocess.sh calvin-abcd-flower -- --no-distributed
#
# Usage:
#   bash commands/preprocess.sh                          # calvin-debug
#   bash commands/preprocess.sh calvin-abcd              # ABCD-D from /dev/shm
#   bash commands/preprocess.sh calvin-abcd /custom/path # custom local path

set -euo pipefail
cd "$(dirname "$0")/.."

ENV="${1:-calvin-debug}"
LOCAL_PATH="${2:-}"
shift 2>/dev/null || true
EXTRA_ARGS="${*}"  # e.g. --no-distributed

VCPUS=$(nproc)
# Workers: use 75% of vCPUs (leave room for JAX runtime threads)
WORKERS=$(( VCPUS * 3 / 4 ))
[ "$WORKERS" -lt 4 ] && WORKERS=4

case "$ENV" in
    calvin-debug)
        OUTPUT_DIR="result/vla"
        ;;
    calvin-abcd)
        OUTPUT_DIR="result/vla_abcd"
        LOCAL_PATH="${LOCAL_PATH:-/dev/shm/calvin_abcd}"
        ;;
    calvin-abcd-flower)
        OUTPUT_DIR="result/vla_abcd_flower"
        LOCAL_PATH="${LOCAL_PATH:-/dev/shm/calvin_abcd}"
        ;;
    calvin-abcd-flower-full)
        OUTPUT_DIR="result/vla_abcd_flower_full"
        LOCAL_PATH="${LOCAL_PATH:-/dev/shm/calvin_abcd}"
        ;;
    *)
        echo "Unknown env: $ENV (supported: calvin-debug, calvin-abcd, calvin-abcd-flower, calvin-abcd-flower-full)"
        exit 1
        ;;
esac

echo "=== Preprocessing: $ENV ==="
echo "vCPUs: $VCPUS, workers: $WORKERS"
echo "Output: $OUTPUT_DIR/vlm_cache/"

ARGS="--env $ENV --output-dir $OUTPUT_DIR --workers $WORKERS"
[ -n "$LOCAL_PATH" ] && ARGS="$ARGS --local-path $LOCAL_PATH"

# Large datasets use float16 cache to fit in RAM
if [ "$ENV" = "calvin-abcd-flower-full" ]; then
    ARGS="$ARGS --obs-dtype float16"
fi

PYTHONUNBUFFERED=1 PYTHONPATH=src python scripts/preprocess_vlm_cache.py $ARGS $EXTRA_ARGS
