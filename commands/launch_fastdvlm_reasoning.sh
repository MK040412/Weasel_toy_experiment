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
#
# ============================================================================================
# HANDOFF NOTES (read before launching — full rationale in docs/METHODOLOGY_AND_DECISIONS.md §6):
#
# (a) EPOCHS=2 IS A CEILING, NOT A PROVEN OPTIMUM. The reasoning set is small (~24k episodes /
#     ~42k steps) AND already distilled from a strong reasoner (GUI-Libra ASFT) -> overfitting
#     risk past ~1–1.5 epochs, under-training risk at <1 epoch. So: checkpoint every ~0.5 epoch
#     (--hf-upload-every-steps 750, already set; ~1497 steps/epoch) and EVAL-SELECT the best
#     checkpoint on AndroidWorld (the bd-sweep harness, aw_eval). DO NOT assume 2 is best —
#     1 epoch (~step 1500) may already be the winner.
#
# (b) OPTIONAL UPGRADE — MIXED-NOISE scheduler (uniform-vocab corruption + supervise corrupted
#     positions) gives denser supervision, which helps a small dataset. It is most valuable when
#     PAIRED with TOKEN-REVISION at decode (mixed-noise trains the fix-wrong-tokens skill that
#     token-revision exploits). Both are deferred follow-ups, NOT enabled here. See
#     docs/METHODOLOGY_AND_DECISIONS.md §5–§6 before adding either.
#
# (c) --model-dir MUST be the BEST action-SFT checkpoint WITH A COMPLETED IMAGE PROCESSOR.
#     Shipped action checkpoints may LACK preprocessor_config.json / special_tokens_map.json /
#     video_preprocessor_config.json. Before using a checkpoint for an image-processing run, copy
#     those 3 files from Qwen/Qwen3-VL-2B-Instruct (or ~/models/boltzmann-final) and verify with
#     AutoProcessor.from_pretrained(...). See docs/RESUME_FROM_GITHUB.md §1.
# ============================================================================================
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
# REMINDER: complete the image processor first (see HANDOFF NOTE (c) above / RESUME_FROM_GITHUB.md §1).
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
  --hf-upload-every-steps 750 --hf-upload-final \
  --prefetch-windows 1 --log-every 1 --monitor-every 5
# ============================================================================================
# CHECKPOINTING under multihost + ZeRO-1 (see commands/CHECKPOINT_DECOUPLED.md):
#   The --hf-upload-repo / --hf-upload-* flags above are effectively UNUSED by the DECOUPLED save
#   path. ZeRO-1 makes the vocab embedding dp-SHARDED, so an in-process upload deadlocks/fails;
#   instead each host dumps its LOCAL shards (no collective) to /dev/shm and the EXTERNAL
#   scripts/stitch_and_ship_checkpoint.py reassembles -> HF safetensors -> ships to
#   KMK040412/fastdvlm-aw-reasoning. The 750-step cadence still governs WHEN shards are dumped
#   (=> ~0.5-epoch checkpoints for the eval-select in note (a) above).
#
#   Removed the inert --delete-local-uploaded-checkpoints flag: with the decoupled path nothing is
#   uploaded in-process, so it had no effect; /dev/shm shards are managed by the external stitcher.
# ============================================================================================
