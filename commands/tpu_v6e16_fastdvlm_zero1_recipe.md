# TPU v6e-16 Fast-dVLM — ZeRO-1 AdamW continued-SFT (AndroidWorld)

Date: 2026-06-08

Reproducible runbook for continued-SFT of `boltzmann-final` (GUI-Owl-1.5-2B, AR→block-diffusion
dVLM) on a **v6e-16 spot TPU** (4 hosts × 4 chips, 33.55 GB usable HBM/chip), pure data-parallel,
to maximize the AndroidWorld benchmark. This is the v6e-16 sibling of
`tpu_v6e32_fastdvlm_episode_kd_recipe.md` (same episode-packing + kd_fewstep + degree-2 curriculum);
it adds the two things that make **AdamW** fit and stay finite on the smaller pod:

1. **ZeRO-1 optimizer-state sharding** (`--shard-opt-state`) — AdamW's mu+nu replicated is 8.13 GB/chip
   and OOMs at the one-time step-2 relayout (XLA holds old+new copies). Sharding mu/nu across the single
   `dp` axis shrinks it ~16× (→ ~0.5 GB/chip) and the relayout transient with it. Params stay replicated
   (this is ZeRO-1, NOT FSDP — the 2B model fits replicated; only the optimizer was the problem).
2. **`--skip-nonfinite` NaN guard** (`optax.apply_if_finite`) — skips the optimizer update for any step
   whose update is non-finite (weights preserved), so a rare bad batch cannot cascade into NaN weights.

## Provisioning (spot — EXPECT preemption)

- Project `plzsaveus`, zone `asia-south1-c`, accelerator `v6e-16` (4×4 topology), **spot/preemptible**.
- Node name `weasel-v6e16`; 4 workers SSH'd as `ses040515@<worker-ip>` (IPs are per-provision — re-read
  them from `gcloud compute tpus tpu-vm describe weasel-v6e16 --project=plzsaveus --zone=asia-south1-c`).
- **Spot reality:** this pod WAS preempted mid-run (≈step 475). A PREEMPTED node is stopped and not
  billing; delete it (`gcloud compute tpus tpu-vm delete weasel-v6e16 ...`) when done. Because preemption
  is frequent, **lower the checkpoint cadence** (see below) — the debug run lost everything because its
  first upload was scheduled at step 1000.

## Per-worker prerequisites (all 4 hosts, identical)

- `~/Weasel_toy_experiment` — this repo, branch `aw-blockdiffusion-eval-repro`, `uv sync` done.
- **CPU torch+torchvision in the venv (REQUIRED, easy to miss).** The trainer pre-imports `torch`/`torchvision`
  before jax (header comment, line ~44) because transformers' Qwen3-VL `AutoProcessor` eagerly instantiates
  `Qwen3VLVideoProcessor`, which hard-requires the torch+torchvision backends — even though this TPU trainer
  never processes video. `pyproject.toml` pins `torch>=2.7.1`/`torchvision>=0.22.1`, but a plain `uv sync` on a
  fresh TPU VM does NOT install them usable (the default index serves the CUDA build). Without them the run
  dies instantly at `AutoProcessor.from_pretrained` with `ImportError: Qwen3VLVideoProcessor requires the
  Torchvision/PyTorch library`. Fix on every worker (CPU-only, no TPU/HBM impact — used solely for the
  host-side processor):
  ```bash
  ~/.local/bin/uv pip install --python ~/Weasel_toy_experiment/.venv/bin/python \
    "torch>=2.7.1" "torchvision>=0.22.1" --index-url https://download.pytorch.org/whl/cpu
  ```
  Verify: `uv run --no-sync python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('~/models/boltzmann-final')"` → `Qwen3VLProcessor`.
- `~/models/boltzmann-final` — the continued-SFT start checkpoint (HF/jax weights + processor).
- `~/data/aw_mix_hybrid_packed` — **all 151 shards** of `KMK040412/guiowl-aw-mix-hybrid-packed`
  (57,669 episodes / 76.7 GB). Each host reads `sorted(glob("packed-*.parquet"))[proc_index::proc_count]`,
  so all 151 must be present on every host (the loader strides, it does not partition the download).
- `export JAX_COMPILATION_CACHE_DIR=$HOME/jax_ccache` — makes a preemption-restart a compile cache hit.
- `HF_TOKEN` from `~/.fastdvlm_secrets.env` (never inline it in a committed file).

## Launch (per worker; the `--multihost` trainer coordinates the 4 hosts)

```bash
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH=$HOME/Weasel_toy_experiment/src PJRT_DEVICE=TPU PYTHONUNBUFFERED=1
source ~/.fastdvlm_secrets.env            # provides HF_TOKEN — do NOT hardcode the token
export JAX_COMPILATION_CACHE_DIR=$HOME/jax_ccache

cd ~/Weasel_toy_experiment
uv run --no-sync python scripts/train_fastdvlm_tpu.py --multihost --data-parallel \
  --model-dir ~/models/boltzmann-final \
  --data ~/data/aw_mix_hybrid_packed --data-pattern "packed-*.parquet" \
  --out ~/runs/v6e16_adamw --data-mode episode --max-turns 12 \
  --max-samples 0 --samples-per-window 64 \         # <-- 0 = FULL DATA (see GOTCHA below)
  --batch-size 16 --max-steps 10122 --epochs 3 \   # CURATED phone-only = 53,991 eps / 16 = 3,374 steps/epoch x3. Epoch boundaries (=MAJOR ckpts): steps 3374 / 6748 / 10122.
  --bd-curriculum degree2 --bd-values "1,2,4,8,16,32" \   # bd32 INCLUDED so the eval-centered Gaussian (mode bd16) is symmetric {8,16,32}, not boundary-truncated.
    --bd-lambda1 0.0 --bd-lambda2 1.04 --bd-lambda1-end -5.77 --bd-anneal-steps 6749 \   # CLOSED-FORM derivation: degree-2 == discrete Gaussian in k=log2(b); λ2=1/(2(ln2)^2)=1.04, λ1=-μ_x/σ_x^2. EVAL=bd16 fixes END center μ_k=4; σ_k=1 octave (natural candidate spacing). Anneal center bd1->bd16 => λ1 0->-5.77. END P={8:.26,16:.42,32:.26}. (MaxEnt = least-assumptive dist for (center=eval, σ=1oct).)
  --kd-fewstep-weight 0.25 --kd-fewstep-bd-cap 4.0 --kd-fewstep-warmup-steps 500 \
  --ctx-cap 4096 --pad-to 4096 --noisy-pad-to 1024 --vision-pad-to 1280 \
  --vision-precompute-batch-size 16 --pair-batch 1 --loss-token-cap 256 \
  --dtype bf16 --optim adamw_bf16 --lr 3e-6 --weight-decay 0.01 \
  --shard-opt-state --skip-nonfinite \
  --ce-noisy-weight 1.0 --ce-clean-weight 0.75 --kd-noisy-weight 0.25 --kd-temp 2.0 \
  --hf-upload-repo KMK040412/fastdvlm-aw-adamw-guard \
  --hf-upload-every-steps 1687 --hf-upload-final \   # save every 1/2 epoch (1687 steps); epoch boundaries (3374/6748/10122/13496) are every-other save = the MAJOR per-epoch ckpts. NOTE: on spot preemption you lose <=1687 steps (~36min) -> resume via --start-step <last_saved> + --model-dir <last HF ckpt>.
  --prefetch-windows 1 --log-every 1 --monitor-every 5   # --log-every 1 = per-step JSON (loss + ce_noisy/ce_clean/kd_noisy/kd_fewstep + bd + lambda_fs + input_len + n_blocks + tokens/s + HBM via --monitor-every). Detailed by default.
```

### Finalized hyperparameters
| Knob | Value | Why |
|---|---|---|
| `--optim` | `adamw_bf16` | bf16 mu+nu. **Never fp32** (hard constraint). ZeRO-1 makes it fit. |
| `--lr` | `3e-6` | effective batch is small (loss-token-cap 256 ⇒ ~16×256 = 4k loss tokens/step); continued-SFT conservatism. Band [2e-6,5e-6]; 1e-6 under-trains AdamW. |
| `--weight-decay` | `0.01` | light decoupled WD. |
| `--pair-batch` | `1` | dual-stream pair-0 only. **pair-batch 2 OOMs TPU_0** (doubles the f32[2,16,5120,5120]≈3.4 GB attn buffer). |
| `--batch-size` | `16` | 1 episode/chip. |
| epochs | `3` (≈10,122 steps) | **curated** phone-only = 3,374 steps/epoch x3. `--bd-anneal-steps 6749` (NOT total): the curriculum reaches the eval center b*=16 by **epoch 2**, then epoch **3 consolidates at the bd16-centered distribution** (eval is at bd16). ETA ~4.0h no-preempt; ckpt every 1/2 epoch (1687) → epoch boundaries 3374/6748/10122. |
| data | **`KMK040412/guiowl-aw-mix-phoneonly-packed`** | the CURATED repo (tablets/landscape + empty-instruction + cross-shard removed; 53,991 eps, 6.59% fewer). Download it to `~/data/` on the workers instead of the raw `-hybrid-packed`. |

## NaN finding — the data is CLEAN; the guard is the right fix (verified by direct inspection)

The first AdamW+ZeRO-1 attempt NaN'd; the guarded re-run trained clean except a single skipped step
(`bd=2`, ≈0.5% of steps). We **directly inspected the actual data** (not metadata):

- Reconstructed the exact offending batch (`packed-0000.parquet`, the first 64-episode window): all
  episodes (`1001_MarkorEditNote`, a 24-step multi-app task, `11903` "click Settings", `440_ContactsNewContactDraft`)
  decode cleanly, every `mobile_use` action is valid JSON, every coordinate ∈ [0,1000].
- A raw scan of the first 80 episodes of `packed-0000` found **0** image-decode failures, **0** malformed
  actions, **0** out-of-range coordinates, **0** empty episodes, **0** non-finite pixels.
- Structurally a forward NaN cannot come from data content here: `decode_image` always
  `.convert("RGB").resize(FIXED_VISION_WH=(196,448))` (pixels always finite, images uniform-sized; a
  corrupt image makes the episode SKIP, never train), every loss denominator is `jnp.maximum(mask.sum(),1.0)`
  (a zero-loss-token sample gives `0/1=0`, never `0/0`), and labels are `jnp.maximum(shift_labels,0)`-clamped.

**Conclusion:** the rare NaN is a transient bf16 forward-compute edge at `bd=2` (masking RNG × block
structure × bf16 attention on a particular packed sequence), NOT a data-corruption problem. The dataset
needs no cleaning. `--skip-nonfinite` skips that ~0.5% of steps harmlessly (weights preserved, loss keeps
falling). `--skip-nonfinite` also auto-enables a per-step `GRADCHK` log (`gnorm / allfinite / firstbad`
param index, decode via the one-time `grad_debug_paths` event) to pinpoint the offending layer if needed.

## GOTCHAS (non-obvious, cost hours)

- **`--max-samples` defaults to 64 — that is a DEBUG cap, not full data.** With the default, the loader
  emits only the first 64 episodes of each host's stride and `window_idx` stays 0 forever (the run cycles
  the same 64 episodes). For the real full-data run you MUST pass **`--max-samples 0`**. (The `v6e16_nandbg`
  debug run used the default 64 deliberately, to reproduce the NaN fast on a tiny pool.)
- **grad-accum (`--grad-accum`) via `optax.MultiSteps` does NOT help here**: its `acc_grads` buffer stays
  ~+8 GB replicated under nnx.jit and OOMs the vision-constrained TPU_0. Use `--batch-size 16`, no accum.
- **pair_batch 2 OOMs TPU_0.** Keep `--pair-batch 1`.
- **One-time relayout warmup:** ZeRO-1 train_step compiles ~14 min; ~2 relayout compiles ≈ 28 min before
  steady state. `JAX_COMPILATION_CACHE_DIR` makes a preemption restart a cache hit. The "step-2 recompile"
  is normal XLA committed-layout settling, not a bug.
- **Lion fallback (no ZeRO-1):** `--optim lion --lr 1e-6 --pair-batch 1 --batch-size 16` (drop
  `--shard-opt-state`). Lion's single 4.06 GB momentum buffer fits replicated; ~2.6 h, no cross-host
  collectives. Use if ZeRO-1/AdamW is ever blocked.

## Verification (smoke gate before the full run)
1. No OOM through step 3+ (passes the relayout that kills replicated AdamW).
2. Memory probe `peak_gb_in_use` ≪ replicated-adamw 27.6 GB (expect ~16–17 GB); an opt-state leaf's
   `.addressable_shards[0].data.shape` is 1/16 of the global shape (confirms sharding, not replication).
3. All 5 loss terms finite at steps 3–20, values ≈ 0.7–1.6; steady ~1.0–1.3 s/step.

## Post-train
Load the HF checkpoint into `aw_eval` and run the bd-sweep (`bd_sweep.py`); compare AndroidWorld
strict-JSON success of `boltzmann-final` vs the new adamw-guard checkpoint.
See `aw_eval/CLAUDE.md` for the AndroidWorld harness, and `project_fastdvlm_v6e16_zero1_run` in memory.
