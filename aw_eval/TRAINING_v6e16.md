# Fast-dVLM training — reproducible recipe for v6e-16 (broad-corpus SFT → AndroidWorld)

Canonical, reproducible training spec for the next run: SFT the GUI-Owl-1.5-2B
block-diffusion VLA on the **new curated AW-targeted mix** (HF `KMK040412/guiowl-aw-mix-*`)
to raise AndroidWorld success, then RL. Captured 2026-06-07 from the working v6e-4
Boltzmann run before TPU teardown. Pairs with the eval harness in `./CLAUDE.md`.

## Model & objective (dual-stream block-diffusion KD)
- Base: GUI-Owl-1.5-2B-Instruct (Qwen3-VL-2B; arch `Qwen3VLForConditionalGeneration`, vocab 151936, hidden 2048, text 28L, M-RoPE mrope_section interleaved, DeepStack extract@ViT 5/11/17 → inject@LLM 0/1/2).
- **Loss = 1.0·ce_noisy + 0.75·ce_clean + 0.25·kd_noisy** (kd_temp 2.0). clean branch = same-model stop-grad internal teacher; noisy branch = block-diffusion (parallel-unmask) stream. Trainer: `scripts/train_fastdvlm_continue.py` (in Weasel_toy_experiment repo; local mirror `/home/perelman/episode_work/curation_src/train_fastdvlm_continue_TPU.py`).
- bd curriculum: **Boltzmann** P(bd) ∝ bd^(−λ), λ cosine-annealed start→end over `--bd-anneal-steps`. (bd-sweep showed decode near-lossless to bd4; this curriculum trains all bd 1–32.)

## Exact launch (v6e-4 reference — the run that produced boltzmann-final)
```bash
cd ~/Weasel_toy_experiment
export HF_TOKEN=<token>  PYTHONPATH="$PWD/src"  PJRT_DEVICE=TPU  PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR=~/jax_ccache    # persistent XLA cache (cuts recompiles)
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0  JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
.venv/bin/python scripts/train_fastdvlm_continue.py \
  --model-dir <CKPT> --data <DATA_DIR> --data-pattern 'shard-*.parquet' --out <OUT> \
  --max-samples <N> --max-steps 0 --samples-per-window 4096 --epochs 1 \
  --batch-size 32 --data-parallel \
  --bd-dist boltzmann --bd-values '1,2,4,8,16,32' --bd-lambda-start 1.5 --bd-lambda-end -0.5 --bd-anneal-steps 7000 \
  --ctx-cap 480 --pad-to 480 --noisy-pad-to 480 --vision-pad-to 96 \
  --vision-precompute-batch-size 16 --loss-token-cap 96 --dtype bf16 --optim adamw_bf16 \
  --lr 1e-6 --peak-lr 5e-6 --warmup-steps 100 --decay-steps <STEPS> --end-lr 1e-6 --grad-accum 1 --weight-decay 0.1 \
  --ce-noisy-weight 1.0 --ce-clean-weight 0.75 --kd-noisy-weight 0.25 --kd-temp 2.0 \
  --prefetch-prep --prefetch-windows 2 --log-every 20 --monitor-every 60 --save-final \
  --hf-upload-repo KMK040412/fast-dvlm-guiowl-kd-tpu --hf-upload-repo-type model \
  --hf-upload-prefix fast-dvlm-kd-tpu/<run-name> --hf-upload-every-steps 3000 --hf-upload-final \
  --delete-local-uploaded-checkpoints
```

## v6e-16 adaptations (the new run)
| knob | v6e-4 | **v6e-16** | why |
|---|---|---|---|
| devices | 4 | 16 | `--data-parallel` pmaps across all chips |
| `--batch-size` | 32 (8/chip) | **128** (8/chip) or 256 (16/chip) | must be multiple of 16; 128 safe (HBM ~27/33GB at bs32/4chip ⇒ headroom), try 256 if it fits |
| `--model-dir` | overfit ckpt | **BASE GUI-Owl-1.5-2B-Instruct** (fresh) | do NOT inherit the narrow aw-overfit; broad-corpus SFT from clean base avoids re-overfit |
| `--data` | ~/mixdata (overfit mix) | **new curated mix** (download HF `KMK040412/guiowl-aw-mix-hybrid` → local dir) | the AW-targeted balanced/episode mix |
| `--skip-samples` | 192000 (warm-resume) | **omit** (fresh run, see all data) | only for resuming a partial epoch |
| `--max-samples` / `--decay-steps` | 476043 / 8877 | **= new dataset size** / `ceil(N/batch * epochs)` | set decay-steps = total optimizer steps |
| `--bd-anneal-steps` | 7000 | scale to ~0.8× total steps | keep λ anneal spanning most of training |
| `--hf-upload-prefix` | aw-overfit-boltzmann | `aw-broadmix-<ver>` | new run id |

Pick dataset version (from `KMK040412/guiowl-aw-mix-{full,targeted,hybrid}`): **hybrid recommended** (in-domain + generic ballast, avoids overfit-collapse). Sizing: hybrid ~600k samples → at batch 128, 1 epoch ≈ 4,700 steps; v6e-16 ≈ **4× the v6e-4 throughput** → est. ~2–4h for 1 epoch (vs ~6h on v6e-4). 2–3 epochs for the long-horizon tail.

## Reproducibility gotchas (encoded, do not relearn)
- **JIT recompile**: `--loss-token-cap 96` fixes the sparse-loss tensor shape → single XLA compile (was 86s→1.4s/step). Keep it. Persistent `JAX_COMPILATION_CACHE_DIR` reuses compiles across restarts.
- **TPU paramiko**: nohup/systemd die after model load on these TPUs → launch under `tmux` (or a held SSH); see eval `CLAUDE.md`.
- **HF upload cadence**: `--hf-upload-every-steps 3000 --hf-upload-final --delete-local-uploaded-checkpoints` (saves disk; checkpoints land on HF).
- **Episode-packing** (if using the packed mix for multi-step): use `scripts/train_fastdvlm_episode.py` (same loss; packs img1,act1,…,imgN,actN with history/KV) — the proven fix for multi-step chaining (Clock/Contacts failures). All JIT fixes already ported.
- Reproducible seeds: `--task_random_seed`/data order fixed; log the full argv to `<OUT>/`.

## After training → eval
Use the eval harness here: serve the new ckpt on TPU (`launch_aw_server.sh <decode> <bd>`), then
`cd /home/perelman/aw_eval && python bd_sweep.py --checkpoints <new> --bds 1,4 --repair both`
(register the new ckpt in `config.py` CHECKPOINTS). Then RLVR with the AndroidWorld success verifier.

## Sequence
broad-corpus SFT (this doc) → AW eval (bd_sweep) → reasoning distill (user-sourced reasoning data, `reasoning_plan` field) → RLVR (AW success reward). See [[project_aw_curation_plan]], [[project_aw_bd_sweep_pipeline]].
