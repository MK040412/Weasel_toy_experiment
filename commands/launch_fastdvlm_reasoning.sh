#!/bin/bash
# REASONING SFT — Fast-dVLM 2B think-then-act (<think>...</think> + <tool_call>...</tool_call>) on
# phone-only GUI-Libra reasoning data. Run AFTER the action SFT finishes. Source of truth:
# commands/REASONING_SFT_RECIPE.md. Distillation is OFF (the base 2B has no reasoning, so the
# self-distillation kd terms would fight the CoT cross-entropy); reasoning is learned ONLY from
# the `reasoning` CoT column via CE (ce_noisy + ce_clean), which the loader patch now supervises.
#
# Deploy to every worker as ~/launch_fastdvlm.sh, then launch PER-WORKER IN PARALLEL (never --worker=all
# — it 255-retry-storms). Idempotent guard (so a gcloud retry cannot double-launch):
#   GUARD='if pgrep -f "[u]v run --no-sync python scripts/train_fastdvlm" >/dev/null; then echo ALREADY; \
#     else : > ~/train.log; setsid bash -lc "~/launch_fastdvlm.sh >> ~/train.log 2>&1" </dev/null >/dev/null 2>&1 & echo LAUNCHED; fi'
#   for w in 0 1 2 3; do gcloud compute tpus tpu-vm ssh weasel16 --zone asia-northeast1-b \
#     --project mobile-computing-new --worker $w --command "$GUARD" & done; wait
#
# Prereqs on every worker (see commands/tpu_v6e16_fastdvlm_zero1_recipe.md):
#   - CPU torch+torchvision in the venv (else AutoProcessor crashes)
#   - ~/.fastdvlm_secrets.env (HF_TOKEN)
#   - Data on every worker: ~/data/gui-libra-reasoning-phone (NESTED layout -> --data-pattern "*/*.parquet";
#     --data-mode episode is MANDATORY — row mode KeyErrors on `screenshot`)
#   - Start checkpoint = the BEST action-SFT checkpoint (see --model-dir below)
set -e
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH=$HOME/Weasel_toy_experiment/src PJRT_DEVICE=TPU PYTHONUNBUFFERED=1
source ~/.fastdvlm_secrets.env                      # HF_TOKEN (never hardcoded)
export JAX_COMPILATION_CACHE_DIR=$HOME/jax_ccache
mkdir -p $HOME/jax_ccache $HOME/runs

cd ~/Weasel_toy_experiment
# ============================================================================================
# PLACEHOLDER --model-dir: set this to the BEST action-SFT checkpoint once the action SFT run
# finishes. The action run uploads to KMK040412/fastdvlm-aw-guiowlvit; stitch+download the best
# step to ~/models/<best-action-sft-ckpt> and point --model-dir at it.
# DOCUMENTED FALLBACK (if the action SFT is not yet available): ~/models/boltzmann-final
# ============================================================================================
exec uv run --no-sync python scripts/train_fastdvlm_tpu.py --multihost --data-parallel \
  --model-dir ~/models/best-action-sft-ckpt-PLACEHOLDER \
  --data ~/data/gui-libra-reasoning-phone --data-pattern "*/*.parquet" \
  --out /dev/shm/v6e16_reasoning --data-mode episode --max-turns 12 \
  --max-samples 0 --samples-per-window 64 \
  --batch-size 16 --max-steps 3000 --epochs 2 \
  --bd-curriculum degree2 --bd-values "1,2,4,8,16,32" \
    --bd-lambda1 0.0 --bd-lambda2 1.04 --bd-lambda1-end -5.77 --bd-anneal-steps 1500 \
  --ctx-cap 4096 --pad-to 4096 --noisy-pad-to 1536 --vision-pad-to 1280 \
  --vision-precompute-batch-size 16 --pair-batch 1 --loss-token-cap 256 \
  --dtype bf16 --optim adamw_bf16 --lr 3e-6 --weight-decay 0.01 \
  --shard-opt-state --skip-nonfinite \
  --ce-noisy-weight 1.0 --ce-clean-weight 0.75 --kd-noisy-weight 0.0 --kd-fewstep-weight 0.0 \
  --hf-upload-repo KMK040412/fastdvlm-aw-reasoning \
  --hf-upload-every-steps 750 --hf-upload-final --delete-local-uploaded-checkpoints \
  --prefetch-windows 1 --log-every 1 --monitor-every 5
