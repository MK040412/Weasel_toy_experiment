# v6e-16 PLAYBOOK — broad-corpus SFT → AndroidWorld (operational steps)

Do-this-in-order guide for when the v6e-16 pod is up. Pairs with `TRAINING_v6e16.md`
(exact launch flags) and `CLAUDE.md` (eval). Goal: SFT GUI-Owl-1.5-2B block-diffusion VLA
on the new AW-targeted mix to raise AndroidWorld success, then RL.

## 0. Provision
```bash
git clone https://github.com/MK040412/Weasel_toy_experiment.git && cd Weasel_toy_experiment
uv sync            # jax[tpu], flax, transformers, pyarrow, huggingface-hub
export HF_TOKEN=<token>  PYTHONPATH="$PWD/src"  PJRT_DEVICE=TPU  PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR=~/jax_ccache JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
python -c "import jax; print(jax.devices())"   # expect 16 TPU chips
```

## 1. WHICH HuggingFace MODEL TO PULL (decision table)

| Purpose | Pull this | Notes |
|---|---|---|
| **Base for the new broad SFT (recommended)** | **`mPLUG/GUI-Owl-1.5-2B-Instruct`** | Clean base; AVOID inheriting our aw-overfit checkpoints (they overfit to the narrow mix and collapse on the live emulator). Qwen3-VL-2B, vocab 151936, hidden 2048. |
| Continue/compare from our runs | `KMK040412/fast-dvlm-guiowl-kd-tpu` → `fast-dvlm-kd-tpu/aw-overfit-{boltzmann/final, bdcurric/checkpoint-step006000, continue/final}` | These are OVERFIT — use only for eval baselines / ablation, NOT as the broad-SFT base. |
| Eval baselines / bd-sweep ablation | same repo (boltzmann-final, bd-curric-6000, continue=baseline) | register in `aw_eval/config.py` CHECKPOINTS. |
| **Reasoning teacher (later, for distill/RLVR warm-start)** | `mPLUG/GUI-Owl-1.5-8B-Thinking` or `…-32B-Thinking` | NO 2B-Thinking exists. Same Qwen3 vocab → cross-size logit-KD feasible (top-k cached into `teacher_logprob_cache`). |

```bash
# base model (recommended start):
huggingface-cli download mPLUG/GUI-Owl-1.5-2B-Instruct --local-dir ~/models/gui-owl-1.5-2b
```

## 2. PULL THE DATASET (the new AW-targeted mix)
```bash
# pick one version (hybrid recommended: in-domain anchor + generic ballast, avoids overfit-collapse)
huggingface-cli download KMK040412/guiowl-aw-mix-hybrid --repo-type dataset --local-dir ~/aw_mix_hybrid
# versions: -full (broad/generalization), -targeted (in-domain/fast), -hybrid (balanced; recommended)
# raw 5-source corpus (if you want to re-mix): KMK040412/guiowl-curated-corpus
```
Data is unified 11-col norm1000 mobile_use parquet (`shard-*.parquet`). Each version's `balance_report.json` shows its action/app/coord/length balance.

## 3. TRAIN (see TRAINING_v6e16.md for full flags)
Key v6e-16 deltas vs the v6e-4 reference launch: `--model-dir ~/models/gui-owl-1.5-2b` (BASE), `--data ~/aw_mix_hybrid`, `--batch-size 128` (16 chips ×8; try 256), **omit `--skip-samples`** (fresh run), `--decay-steps = ceil(N_samples/batch × epochs)`, `--bd-anneal-steps ≈ 0.8×total`, `--hf-upload-prefix aw-broadmix-hybrid`. Launch under `tmux`. For multi-step/episode-packed data use `scripts/train_fastdvlm_episode.py`. Est ~2–4 h/epoch on v6e-16.

## 4. EVAL (after training)
```bash
# register the new ckpt in aw_eval/config.py CHECKPOINTS, then:
cd /home/perelman/aw_eval
python bd_sweep.py --checkpoints <new-ckpt> --bds 1,4 --repair both --task-set standard_full
# (start the TPU server via launch_aw_server.sh <decode> <bd>; bring up the Vultr tunnel/emulators per CLAUDE.md)
```
Compare to baselines (bd-sweep table in FINDINGS.md: overfit-final = 6%). Target: lift the non-toggle tasks.

## 5. NEXT (sequence)
broad-corpus SFT (here) → AW eval → **reasoning distill** (user-sourced reasoning data into `reasoning_plan`; or logit-KD from 8B/32B-Thinking, top-k cached) → **RLVR** (AndroidWorld success as verifiable reward; reasoning warm-start makes the sparse reward learnable).

## Reproducibility / sanity
- `bash /home/perelman/aw_eval/verify_repro.sh` confirms checkpoints/code/data/docs are all captured.
- Everything off-TPU: checkpoints on HF, code in this repo, recipe in TRAINING_v6e16.md, data on HF + Vultr.
