# TPU v6e-32 Fast-dVLM — Episode-Packing + kd_fewstep + degree-2 curriculum

Date: 2026-06-07

The locked AndroidWorld-SFT recipe for Fast-dVLM (GUI-Owl-1.5-2B, AR→block-diffusion).
Adds three things over `tpu_v6e_fastdvlm_kd_recipe.md`:

1. **Episode-packing** (`--data-mode episode`): a whole episode = one sequence
   (N images + N assistant turns) with **true cross-turn attention** (within-turn
   block-diagonal + noisy→past-clean causal). Trains on the validated packed-hybrid.
2. **degree-2 block-size curriculum** (`--bd-curriculum degree2`): `P(b) ∝ exp(-λ1·ln b - λ2·(ln b)²)`
   (W0 paper). `λ2=0` reduces to the Boltzmann power law.
3. **`kd_fewstep`** (`--kd-fewstep-weight 0.25`): step-axis self-distillation —
   the clean/AR branch (stop-grad) teaches the pair-0 (heavily-masked, large-block,
   few-effective-step) student via forward-KL, **on top of** the kept `kd_noisy`.
   bd-weighted, **b16-conservative** (`--kd-fewstep-bd-cap 4.0` ⇒ λ_fs saturates at b16).

## Memory note (why this fits on v6e-32 at "8k-ish" episodes)

The dual stream length is `total = noisy_pad_to + pad_to`. The **noisy branch is
text-only** (actions + minimal prompt, no vision), so it stays small. Keep:
- `--pad-to` = clean (full multimodal incl. vision) length — this is the real "context".
- `--noisy-pad-to` = just the non-vision token budget (≈ Σ turn actions + prompt).
- `--vision-pad-to` = `max_turns × per_image_tokens` (≈ 96 tokens/image at `--max-pixels 100352`).

So attention scales with `(pad_to + small)²`, NOT `(2·8k)²`. With `pad_to 4096` +
`noisy_pad_to 1024`, `total ≈ 5120`. Use `--max-turns` to bound the longest (60-turn) episodes;
episodes over `--ctx-cap` are trimmed to their most-recent turns (count logged as
`truncated_episodes` in `data_windows.jsonl` — never silently dropped).

> **Literal 8k tail (Phase-B, not in this recipe):** the dense `(total,total)` mask
> caps practical `total`. To pack the full 60-turn episodes at one shot, swap the dense
> attention for a splash/flash mask-mod kernel encoding `asymmetric_allowed`
> (see `src/qwen/qwen3vl/CLAUDE_modeling.md`). Phase-A below is runnable now.

## One-time setup

```bash
cd ~/Weasel_toy_experiment
# base instruct backbone (NOT a pixel-coordinate Gmail checkpoint)
huggingface-cli download mPLUG/GUI-Owl-1.5-2B-Instruct --local-dir ~/models/gui-owl-1.5-2b-instruct
# validated episode-packed hybrid dataset
huggingface-cli download KMK040412/guiowl-aw-mix-hybrid-packed --repo-type dataset \
  --local-dir ~/data/aw_mix_hybrid_packed
export HF_TOKEN=...   # only if uploading; never inline the token
```

## Launch (v6e-32, all 32 chips, data-parallel)

`jax.device_count()` auto-detects 32; the mesh and batch sharding need NO code change.
`--batch-size` must be a multiple of 32 (auto-rounds up with a logged warning otherwise).
Start at 32 (1 episode/chip); try 64 (2/chip) once the smoke run shows headroom.

```bash
cd ~/Weasel_toy_experiment

QWEN_TPU_DPA_ATTENTION=1 .venv/bin/python scripts/train_fastdvlm_tpu.py \
  --model-dir ~/models/gui-owl-1.5-2b-instruct \
  --data ~/data/aw_mix_hybrid_packed \
  --data-pattern 'packed-*.parquet' \
  --out ~/tpu_fastdvlm_runs/v6e32_episode_kdfs_degree2 \
  --data-mode episode \
  --max-turns 12 \
  --max-samples 0 \
  --epochs 1 \
  --batch-size 32 \
  --data-parallel \
  --bd 32 \
  --bd-curriculum degree2 \
  --bd-values "1,2,4,8,16,32" \
  --bd-lambda1 1.0 \
  --bd-lambda2 0.3 \
  --bd-lambda1-end -0.5 \
  --bd-anneal-steps 7000 \
  --ctx-cap 4096 \
  --pad-to 4096 \
  --noisy-pad-to 1024 \
  --vision-pad-to 1152 \
  --vision-precompute-batch-size 16 \
  --loss-token-cap 256 \
  --dtype bf16 \
  --optim adamw_bf16 \
  --lr 1e-6 \
  --ce-noisy-weight 1.0 \
  --ce-clean-weight 0.75 \
  --kd-noisy-weight 0.25 \
  --kd-temp 2.0 \
  --kd-fewstep-weight 0.25 \
  --kd-fewstep-bd-ref 4.0 \
  --kd-fewstep-bd-cap 4.0 \
  --kd-fewstep-warmup-steps 500 \
  --prefetch-prep \
  --prefetch-windows 2 \
  --samples-per-window 512 \
  --log-every 20 \
  --monitor-every 60 \
  --hf-upload-repo KMK040412/fast-dvlm-aw-episode-kdfs \
  --hf-upload-prefix v6e32-episode-kdfs-degree2 \
  --hf-upload-every-steps 3000 \
  --hf-upload-final \
  --save-final
```

## Smoke test first (≈20 steps, verify no OOM + all 5 loss terms finite)

```bash
... same flags ... --max-samples 256 --max-steps 20 --samples-per-window 64 \
  --hf-upload-every-steps 0 --no- (drop) --hf-upload-final
# check train_log.jsonl: each step has finite loss/ce_noisy/ce_clean/kd_noisy/kd_fewstep,
# kd_fewstep_lambda follows b16-cap (1.0 at b>=16 after warmup), and
# data_windows.jsonl reports truncated_episodes.
```

## Ablations (same harness)

- `--kd-fewstep-weight 0.0` → byte-identical 3-term loss (kd_fewstep OFF baseline).
- `--bd-curriculum static --bd-schedule "4:0.25,8:0.25,16:0.25,32:0.25"` → curriculum ablation.
- `--data-mode row` → single-step (history-as-text) baseline.

## Post-train

Pull the HF checkpoint into `aw_eval/` and run the bd-sweep; compare strict-JSON@b16/b32
against the kd_fewstep-OFF baseline to confirm validity recovery at the large-block frontier.
See `aw_eval/CLAUDE.md`.
