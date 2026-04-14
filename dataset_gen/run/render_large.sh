#!/bin/bash
# pod-0: render AntMaze large
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTPUT_DIR="${ANTMAZE_DATA_DIR:-/mnt/disks/data/antmaze}"
N_WORKERS="${N_WORKERS:-128}"

# safety: refuse tmpfs paths
resolved=$(readlink -f "$OUTPUT_DIR")
fs_type=$(df -T "$resolved" 2>/dev/null | awk 'NR==2 {print $2}')
if [[ "$resolved" == /dev/shm/* || "$resolved" == /tmp/* || "$fs_type" == "tmpfs" ]]; then
    echo "ERROR: $OUTPUT_DIR resolves to $resolved ($fs_type) — tmpfs not allowed."
    echo "Set ANTMAZE_DATA_DIR to a persistent disk path."
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1

echo "[render_large] output=$OUTPUT_DIR workers=$N_WORKERS"
python dataset_gen/scripts/make_paired_dataset_fast.py \
    --maze large \
    --output_dir "$OUTPUT_DIR" \
    --n_workers "$N_WORKERS"

echo "[validate_large]"
PYTHONPATH=src python dataset_gen/scripts/validate_antmaze_dataset.py \
    --zarr_dir "$OUTPUT_DIR/large" \
    --output_dir "$OUTPUT_DIR/large/validation" \
    --maze large \
    --n_verify_eps 3

echo "[done] large → $OUTPUT_DIR/large"
