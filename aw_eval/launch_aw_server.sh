#!/usr/bin/env bash
# Generic AndroidWorld policy-server launcher (deploy to TPU ~/launch_aw_server.sh).
# Usage: launch_aw_server.sh <decode> <bd> [model_dir]
#   decode : grounded_ar_jit | dvlm_bd4 | dual_dvlm_bd4
#   bd     : block size (1 for AR; 2/4/8/16/32 for dual-stream block-diffusion)
# Consolidates the former launch_ar_server.sh / launch_dual_bd.sh into one entry.
set -eu
DECODE=${1:-grounded_ar_jit}
BD=${2:-1}
MODEL=${3:-/home/dayeonhwang9/tpu_runs/boltzmann_20260606_092721/final}
cd ~/Weasel_toy_experiment
source .venv/bin/activate
exec python androidworld_tpu_jax_server.py \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port 8124 \
  --max-pixels 100352 --gen-len 96 \
  --decode "$DECODE" --bd "$BD"
