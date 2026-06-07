# Fast-dVLM TPU trainer — `scripts/train_fastdvlm_tpu.py`

The **main trainer** for Fast-dVLM (GUI-Owl-1.5-2B, AR→block-diffusion VLA). It runs a
**dual-stream continuation** objective on TPU (JAX/Flax NNX): a single forward pass over a
`[noisy_text | clean_full_multimodal]` concatenated sequence produces a **4-term loss** that
teaches an AR Qwen3-VL backbone to denoise masked-action blocks. **Why it matters:** this is the
file that actually converts the AR checkpoint into the block-diffusion VLA whose `b*` frontier and
b32 latency win are the W0 paper headline. Three new capabilities were just added and CPU-verified
(`scripts/test_fastdvlm_kd_curriculum.py`, all pass): **`kd_fewstep`** (4th loss term),
**degree-2 curriculum**, and **episode-packing**.

This doc is self-contained — you should not need to open the source. All function/arg names below
are real. Korean/English mix OK.

---

## TL;DR — copy-paste v6e-32 launch

Full recipe: `commands/tpu_v6e32_fastdvlm_episode_kd_recipe.md`. The launch (all 32 chips,
data-parallel, episode-packed, degree-2 curriculum, `kd_fewstep` ON):

```bash
cd ~/Weasel_toy_experiment

QWEN_TPU_DPA_ATTENTION=1 .venv/bin/python scripts/train_fastdvlm_tpu.py \
  --model-dir ~/models/gui-owl-1.5-2b-instruct \
  --data ~/data/aw_mix_hybrid_packed --data-pattern 'packed-*.parquet' \
  --out ~/tpu_fastdvlm_runs/v6e32_episode_kdfs_degree2 \
  --data-mode episode --max-turns 12 --max-samples 0 --epochs 1 \
  --batch-size 32 --data-parallel \
  --bd 32 --bd-curriculum degree2 --bd-values "1,2,4,8,16,32" \
  --bd-lambda1 1.0 --bd-lambda2 0.3 --bd-lambda1-end -0.5 --bd-anneal-steps 7000 \
  --ctx-cap 4096 --pad-to 4096 --noisy-pad-to 1024 --vision-pad-to 1152 \
  --vision-precompute-batch-size 16 --loss-token-cap 256 \
  --dtype bf16 --optim adamw_bf16 --lr 1e-6 \
  --ce-noisy-weight 1.0 --ce-clean-weight 0.75 --kd-noisy-weight 0.25 --kd-temp 2.0 \
  --kd-fewstep-weight 0.25 --kd-fewstep-bd-ref 4.0 --kd-fewstep-bd-cap 4.0 --kd-fewstep-warmup-steps 500 \
  --prefetch-prep --prefetch-windows 2 --samples-per-window 512 \
  --log-every 20 --monitor-every 60 \
  --hf-upload-repo KMK040412/fast-dvlm-aw-episode-kdfs \
  --hf-upload-prefix v6e32-episode-kdfs-degree2 \
  --hf-upload-every-steps 3000 --hf-upload-final --save-final
```

**Smoke test first** (≈20 steps, verify no OOM + all 5 numbers finite): add
`--max-samples 256 --max-steps 20 --samples-per-window 64 --hf-upload-every-steps 0`. Then check
`train_log.jsonl`: every step has finite `loss/ce_noisy/ce_clean/kd_noisy/kd_fewstep`,
`kd_fewstep_lambda` follows the b16-cap (`1.0` at b≥16 post-warmup), and `data_windows.jsonl`
reports `truncated_episodes`.

**v6e-32 needs NO code change.** `jax.device_count()` auto-detects 32; the mesh is
`Mesh(jax.devices(), ("dp",))`; `--batch-size` auto-rounds **up** to a multiple of the device count
with a logged `batch_size_rounded` event.

**Ablations** (same harness):
- `--kd-fewstep-weight 0.0` → byte-identical 3-term loss (kd_fewstep OFF baseline).
- `--bd-curriculum static --bd-schedule "4:0.25,8:0.25,16:0.25,32:0.25"` → curriculum ablation.
- `--data-mode row` → single-step (history-as-text) baseline.

---

## Architecture at a glance

```
parquet shard(s)
   │  iter_*_windows  (row OR episode mode; RAM-resident "windows" of N samples)
   ▼
raw samples  {input_ids, labels, mm_token_type_ids, pixel_values, image_grid_thw, ...}
   │  prepare_sample_window:
   │    compute_vision_embeds_for_window  → ViT+DeepStack (no grad, pmap), per-image loop, concat
   │    pad_vision_embeds → vision_pad_to
   ▼
prepared window (host RAM, numpy float32 vision)
   │  train_window dispatch loop (host):
   │    np_rng.choice(bd_values, p=bd_probs_fn(step))  → step_bd
   │    prepare_dual_arrays(sample, step_bd, ...)  → noising, dual layout, attn mask, sparse loss idx
   │    lambda_fs = host-side bd-weighted kd_fewstep scalar   ← bd lives ONLY host-side
   ▼
train_step  (@nnx.jit, data-parallel over "dp")
   │    dual_stream_loss_jax:
   │      embed → batched_merge_modalities(vision) → [noisy_emb | clean_pair]
   │      ONE language_model forward over total=(noisy_pad_to+pad_to)
   │      sparse_ce_from_hidden (clean) + sparse_ce_kd_from_hidden (noisy, returns token_kl)
   │      pair-0 reduction → kd_fewstep
   │    L = 1.00·ce_noisy + 0.75·ce_clean + 0.25·kd_noisy + lambda_fs·kd_fewstep
   ▼
optimizer.update(grads)  → log → (every N) HF safetensors upload
```

Backbone: `modeling.ModelConfig.qwen3vl_2b()`, loaded via
`params.create_model_from_safe_tensors`. **Vision tower is frozen / never traced for gradients** —
ViT+DeepStack run once per window outside JIT (`vision_grad: False` in `run_config.json`). mRoPE is
the exact Qwen3-VL interleaved 3D path; DeepStack features are injected into early text layers.

---

## 1. Data flow — row vs episode mode

Both modes share the **window** abstraction: data is streamed as RAM-resident windows of
`--samples-per-window` samples (0 = the whole selected set as one window) via a generator
(`sample_windows_for_epoch`), so memory stays bounded for large shards. `--prefetch-windows N`
runs a producer thread that materializes future raw windows (parquet→image→tokens) while the TPU
trains the current one (ViT precompute deliberately stays serial — it shares the TPU+model state).

| | **`--data-mode row`** (default, back-compat) | **`--data-mode episode`** (new) |
|---|---|---|
| Loader | `iter_row_sample_windows` | `iter_episode_windows` |
| Builder | `build_row_sample` | `build_episode_sample` |
| One sample = | one step (1 image + 1 assistant action) | a whole episode (N images + N assistant turns) |
| Image column | `image` | `image` (falls back to `screenshot`) — `_episode_image_bytes` handles both |
| History | as text only (none injected here) | true multi-turn sequence + cross-turn attention |
| Overflow policy | sample > `--ctx-cap` → **dropped** | drop OLDEST turns, keep most-recent-K, retry; counted as `truncated_episodes` |

### `iter_episode_windows`
Reads each shard whole (`ParquetFile(...).read().to_pandas()`), requires an **`episode_id`** column
(raises otherwise), groups rows by `episode_id` (`OrderedDict`, contiguous within a shard), sorts
each group by `step_id`, and emits one multi-turn sample per episode. Meta logged per window:
`seen_episodes / skipped_episodes / truncated_episodes`.

### `build_episode_sample(steps, processor, ctx_cap, max_pixels, max_turns)`
- `_build_episode_messages`: `[system] + per-turn (user[image (+ "Goal: …" only on turn 0)], assistant[native_action])`.
- `--max-turns N`: keep at most the **most-recent** N turns (`steps[-max_turns:]`) before tokenizing.
- **ctx-cap overflow handling (key):** tokenize → if `len(input_ids) > ctx_cap`, drop the oldest
  turn (`steps[1:]`) and re-tokenize in a loop; if a **single** turn still overflows → skip
  (returns `(None, 0, full_n)`). Goal text rides on the first retained turn. Returns
  `(sample, n_turns_used, n_turns_full)`; `n_used != n_full` ⇒ episode counted as truncated, **never
  silently dropped**.
- `assistant_labels` marks **EVERY** assistant turn (scans for `<|im_start|>assistant\n` … `<|im_end|>`),
  so all turns are supervised.

### `build_row_sample`
Single `[system, user(image + "Goal: …"), assistant(action)]`; `decode_image` → JPEG/PNG or raw RGB
fallback; `native_action` maps AITW `results_action_type` (or pre-classified `target_json`) into the
`<tool_call>{...mobile_use...}</tool_call>` string (coords normalized 0–1000). Returns `None`
(skip) if no image, no goal, no supervised tokens, or over `ctx_cap`.

### Vision precompute incl. multi-image — `compute_vision_embeds` / `compute_vision_embeds_for_window`
- `compute_vision_embeds_for_window`: groups pending samples by `(pixel_values.shape, grid_t,h,w)`
  and runs `make_pmap_vision_forward` (a `jax.pmap` of `visual.forward_static_with_deepstack`,
  cached per grid key) in chunks of `pmap_width = min(local_device_count, batch_size)`. Returns
  stats `vision_pmap_batches / vision_pmap_samples / vision_fallback_samples`. Single-image,
  same-grid samples take the fast pmap path; everything else falls back to `compute_vision_embeds`.
- **Multi-image (episode packing):** when `image_grid_thw` has >1 row, `compute_vision_embeds`
  forwards **each image separately** and **concatenates vision + the 3 DeepStack tensors IN IMAGE
  ORDER**. `pixel_values` is the row-concatenation of all images' patches; it is split by per-image
  patch counts `grid_t·grid_h·grid_w` (cumulative offsets). Downstream `batched_merge_modalities`
  consumes these via `cumsum(mask)-1`, i.e. scatters tokens against the image-token positions in
  exactly this order; trailing pad rows are never indexed.
- After precompute, `pad_vision_embeds` pads each sample's vision + 3 DeepStack tensors to
  `--vision-pad-to` rows (zeros), then they are moved to host as numpy `float32`.

### `pad_to` / `noisy_pad_to` / `vision_pad_to` and the memory rule
`prepare_dual_arrays` builds the dual layout `[ noisy_text (length lt) | clean_full (length clean_len) ]`:

| arg | binds | meaning |
|---|---|---|
| `--pad-to` | `clean_len` | the **clean** (full multimodal, incl. vision) length — the real "context". Sample longer than this → `ValueError`; window pre-filters `len(input_ids) <= pad_to`. |
| `--noisy-pad-to` | `lt` | the **noisy text-only** budget (Σ turn actions + minimal prompt, **no vision**). Default = `pad_to` if unset. `lt_actual > lt` → `ValueError`. |
| `--vision-pad-to` | vision rows | `max_turns × per_image_tokens` (≈96 tok/image at `--max-pixels 100352`). |

> **MEMORY RULE — `total = noisy_pad_to + pad_to`.** Dense attention is `(total, total)`. The
> **noisy branch is TEXT-ONLY (small)**, so cost scales with `(pad_to + small)²`, **NOT** `(2·8k)²`.
> Phase-A (`pad_to 4096`, `noisy_pad_to ~1024` → `total ≈ 5120`) runs on v6e-32 **now**. Literal
> full-8k 60-turn episodes need a Phase-B splash/flash mask-mod kernel encoding `asymmetric_allowed`
> (the dense `(total,total)` mask is what caps practical `total`).

---

## 2. Dual-stream 4-term loss

```
L = 1.00·ce_noisy + 0.75·ce_clean + 0.25·kd_noisy + lambda_fs·kd_fewstep
      └ ce_noisy_weight  └ ce_clean_weight  └ kd_noisy_weight  └ host-side scalar (bd-weighted)
```

All four are computed in `dual_stream_loss_jax` from **one** language-model forward over the
concatenated `[noisy | clean]` sequence (per noisy pair). The **teacher is the model's own clean/AR
branch hidden state with `stop_gradient`** — there is **no third forward**.

| term | where | head/positions | teacher | weight |
|---|---|---|---|---|
| `ce_noisy` | `sparse_ce_kd_from_hidden` | noisy hidden, shifted masked-token positions | — | `--ce-noisy-weight` 1.0 |
| `ce_clean` | `sparse_ce_from_hidden` | clean hidden, shifted response positions | — | `--ce-clean-weight` 0.75 |
| `kd_noisy` (KEPT) | `sparse_ce_kd_from_hidden` | noisy student logits vs clean teacher logits, **averaged over BOTH pairs** (context-axis) | clean branch (stop-grad) | `--kd-noisy-weight` 0.25 |
| `kd_fewstep` (NEW) | pair-0 reduction in `dual_stream_loss_jax` | **only pair-0** of `token_kl` (step-axis) | clean branch (stop-grad) | `lambda_fs(bd)` host-side |

**Two noisy pairs.** `prepare_dual_arrays` builds two noised views (`noisy_ids` shape
`(2, lt)`): `pair-0 = mask_idx` (each response block kept-or-masked with a per-block Bernoulli
`pblk`, heavily-masked / large-effective-block / few-effective-step view, closest to the bd32
failure regime), `pair-1 = comp_idx` (the complement). Both feed the forward as `pair_batch=2`.

**`sparse_ce_kd_from_hidden` now ALSO returns `(token_kl, weights)` unreduced.** Previously it
returned only `(ce, kd)`. The signature is now
`-> (ce, kd_noisy, token_kl, weights)`. `token_kl` is the per-token forward KL
`Σ p_t·(log p_t − log p_s)·temp²` (`temp = kd_temp`); `weights` is the loss mask. `kd_noisy` is
`(token_kl·weights).sum()/Σweights` over the flattened `(global_batch·pair_batch, cap)` — i.e. both
pairs.

**`kd_fewstep` reduction (the new code in `dual_stream_loss_jax`):**
```python
token_kl_pairs  = token_kl.reshape(global_batch, pair_batch, cap)   # un-flatten the pair axis
kd_weights_pairs = kd_weights.reshape(global_batch, pair_batch, cap)
fs_kl = token_kl_pairs[:, 0, :]   # PAIR INDEX 0 = mask_idx view (heavy / large-block / few-step)
fs_w  = kd_weights_pairs[:, 0, :]
kd_fewstep = (fs_kl * fs_w).sum() / jnp.maximum(fs_w.sum(), 1.0)   # masked-average, clamped denom
```
So `kd_fewstep` is the **same forward KL** as `kd_noisy` but isolated to **pair-0 only** (step axis)
instead of averaged over both pairs (context axis). No third forward; teacher = clean/AR hidden
(stop-grad). Divergence = forward KL at `--kd-temp`.

**`lambda_fs` is computed HOST-SIDE** in the dispatch loop (`train_window`), because `bd` is only
known host-side as `step_bd` (sampled per step) and we do **not** want to thread `bd` into the JIT
(would force a recompile). The formula:
```python
fs_ramp     = min((step + 1) / max(warmup, 1), 1.0)                       # linear warmup
fs_bd_factor= min(step_bd / max(bd_ref, 1e-9), bd_cap)                    # bd weighting + cap
lambda_fs   = kd_fewstep_weight * fs_ramp * fs_bd_factor                  # = lambda0 · ramp · min(b/b_ref, c)
```
With defaults `bd_ref=4`, `bd_cap=4.0`, `lambda0=0.25`, post-warmup: **b4→0.25, b8→0.5, b16→1.0,
b32→1.0 (capped)** — "b16-conservative" (saturates at b16, doesn't over-weight the b32 tail).
`lambda_fs` is passed to `train_step` as a **traced `jnp.asarray` scalar**, exactly like the other
three weights → **no recompile, no bd in JIT**. Logged per step as `kd_fewstep` (value) and
`kd_fewstep_lambda` (the scalar).

**Byte-identical when OFF.** `--kd-fewstep-weight 0.0` (default) ⇒ `lambda_fs = 0` ⇒ the 4th term
contributes nothing to `L` (and `kd_fewstep` itself uses a clamped denom, so no div-0 / NaN even
when pair-0 has zero supervised tokens). The result is **byte-identical to the prior 3-term loss**.
Verified by `test_pair0_reduction` (`0.0 * kd_fewstep == 0.0`, finite, no div0).

---

## 3. degree-2 curriculum — `degree2_bd_probs` + `bd_probs_fn`

Block size `b` is sampled per step from a **degree-2 Gaussian-in-log-b** distribution
(`--bd-curriculum degree2`):

```
P(b) ∝ exp(−λ1·ln b − λ2·(ln b)²)
```
`degree2_bd_probs(bd_values, lambda1, lambda2)` computes logits `−l1·ln b − l2·(ln b)²`, subtracts
the max for stability, exponentiates, normalizes. **`λ2 = 0` reduces EXACTLY to the Boltzmann power
law `b^{−λ1}`** (verified to `atol=1e-12` in `test_degree2_reduces_to_power_law`). Larger `λ1` →
mass on small blocks (AR-like, easy); `λ2 > 0` concentrates around a log-b mode.

**λ1 cosine annealing.** With `--bd-lambda1-end` + `--bd-anneal-steps`, `λ1` is cosine-annealed from
`--bd-lambda1` toward `--bd-lambda1-end` over the anneal window (`cos = 0.5·(1−cos(π·frac))`,
`frac = clip(step/anneal_steps, 0, 1)`), shifting mass from small blocks → large blocks over
training (curriculum: start easy AR-like, end on the hard b32 frontier).

**Wiring.** A `bd_probs_fn(step)` closure is built at startup. For `degree2` it recomputes the
annealed probs each call; for `static` it returns the fixed `--bd-schedule` probs
(`parse_bd_schedule`, e.g. `"4:0.1,8:0.2,16:0.3,32:0.4"`; bare `--bd` if no schedule). The sampler
in `make_prep_request` draws `step_bd = np_rng.choice(bd_values, p=bd_probs_fn(step))`.

---

## 4. Full argparse reference

| Flag | Default | Meaning |
|---|---|---|
| `--model-dir` | (required) | HF dir of the GUI-Owl/Qwen3-VL-2B backbone. |
| `--out` | `~/tpu_fastdvlm_runs/continue` | Run output dir (logs, checkpoints). |
| `--data` | `None` | Parquet file or directory (with `--data-pattern`). |
| `--data-pattern` | `*.parquet` | Glob used when `--data` is a directory. |
| `--data-mode` | `row` | `row` = one step/sample; `episode` = pack whole episode (multi-turn). |
| `--max-turns` | `0` | Episode mode: keep at most the most-recent N turns (0 = no cap; ctx-cap still trims). |
| `--hf-repo` | `cjfcsjt/AITW_General` | HF dataset repo for `--hf-file`. |
| `--hf-file` | `None` | Download this single parquet from `--hf-repo`. |
| `--download-dir` | `~/data/aitw_general` | Local dir for `--hf-file` download. |
| `--synthetic` | off | Use random synthetic samples (no parquet/processor; smoke test). |
| `--max-samples` | `64` | Cap total emitted samples (0 = all). |
| `--samples-per-window` | `0` | RAM-resident streaming window size (0 = one window for all data). |
| `--max-steps` | `20` | Stop after N train steps (0 = no cap). |
| `--epochs` | `1` | Epochs over the data. |
| `--batch-size` | `1` | **Global** batch; auto-rounds **up** to a multiple of device count when `--data-parallel`. |
| `--data-parallel` | off | Shard batch over all local TPU devices; replicate model/optimizer state. |
| `--bd` | `32` | Default block size (used when no schedule/curriculum). |
| `--bd-schedule` | `None` | Static comma schedule `"4:0.1,8:0.2,16:0.3,32:0.4"`. |
| `--bd-curriculum` | `static` | `static` (use `--bd-schedule`) or `degree2` (`P(b)∝exp(−λ1 ln b − λ2 (ln b)²)`). |
| `--bd-values` | `1,2,4,8,16,32` | Support set for `degree2` (comma ints). |
| `--bd-lambda1` | `1.0` | degree2 λ1 (larger ⇒ favor small blocks). |
| `--bd-lambda2` | `0.3` | degree2 λ2 (log-quadratic; 0 ⇒ Boltzmann power law). |
| `--bd-lambda1-end` | `None` | With `--bd-anneal-steps`, cosine-anneal λ1 to this (mass → large blocks). |
| `--bd-anneal-steps` | `0` | Steps to cosine-anneal λ1 over (0 disables). |
| `--ctx-cap` | `2048` | Drop/trim samples whose tokenized length exceeds this. |
| `--pad-to` | `0` | Clean (full multimodal) sequence length `clean_len`. 0 = per-window max. |
| `--noisy-pad-to` | `0` | Noisy text-only length `lt`. 0 ⇒ falls back to `--pad-to`. |
| `--vision-pad-to` | `0` | Pad vision/DeepStack to this many rows. 0 = per-window max. |
| `--vision-precompute-batch-size` | `16` | Batch for no-grad ViT+DeepStack pmap precompute (does NOT train vision). |
| `--loss-token-cap` | `128` | Max supervised shifted tokens per branch/sample (sparse LM-head loss). 0 = no truncation. |
| `--pad-token-id` | `0` | Pad token id for clean/noisy filler. |
| `--max-pixels` | `100352` | Processor `max_pixels` (≈96 image tokens at this value). |
| `--seq-len` | `256` | Synthetic-only sequence length. |
| `--response-len` | `64` | Synthetic-only response length. |
| `--lr` | `1e-6` | Learning rate. |
| `--weight-decay` | `0.0` | Weight decay (AdamW variants). |
| `--optim` | `sgd` | `sgd` / `adamw` / `adamw_bf16` (all with global-norm clip 1.0). |
| `--ce-noisy-weight` | `1.0` | Weight of `ce_noisy`. |
| `--ce-clean-weight` | `0.75` | Weight of `ce_clean`. |
| `--kd-noisy-weight` | `0.25` | Weight of `kd_noisy` (both-pair context-axis KD). |
| `--kd-temp` | `2.0` | KD temperature (shared by `kd_noisy` and `kd_fewstep`). |
| `--kd-fewstep-weight` | `0.0` | **λ0** for pair-0 step-axis KD. **0 = OFF = byte-identical 3-term loss.** 0.25 = on. |
| `--kd-fewstep-bd-ref` | `4.0` | Reference block size; λ_fs scales with `step_bd/bd_ref` (b4 = lossless anchor). |
| `--kd-fewstep-bd-cap` | `4.0` | Cap on `step_bd/bd_ref`. 4.0 = b16-conservative (saturates at λ0·4 for b≥16). |
| `--kd-fewstep-warmup-steps` | `500` | Linear warmup of λ_fs from 0 over N steps. |
| `--dtype` | `bf16` | `bf16` / `fp32` compute dtype. |
| `--min-noise` | `1e-3` | Floor on per-block keep prob `pblk`. |
| `--seed` | `7` | RNG seed (python/numpy). |
| `--log-every` | `1` | Print a step record every N steps (all steps are written to jsonl). |
| `--save-final` | off | Save final NNX state as `.npz`. |
| `--save-hf-final` | off | Export final HF safetensors locally. |
| `--hf-upload-repo` | `$HF_UPLOAD_REPO` | HF repo to upload checkpoints to (empty ⇒ no upload). |
| `--hf-upload-repo-type` | `model` | `model` / `dataset`. |
| `--hf-upload-prefix` | `fast-dvlm-kd-tpu` | Path prefix inside the repo. |
| `--hf-token-env` | `HF_TOKEN` | Env var holding the HF token. |
| `--hf-upload-private` | off | Create repo private. |
| `--hf-upload-strict` | off | Re-raise on upload failure (else log + continue). |
| `--hf-upload-every-steps` | `0` | Upload a checkpoint every N steps (0 = never). |
| `--hf-upload-final` | off | Upload a final checkpoint at the end. |
| `--delete-local-uploaded-checkpoints` | off | Delete non-final local bundles after upload. |
| `--monitor-every` | `5` | Seconds between `tpu_usage.jsonl` memory/host snapshots. |
| `--prefetch-prep` | off | Prepare next CPU/noising batch in a background thread (1 worker) during the TPU step. |
| `--prefetch-windows` | `0` | Raw-sample window prefetch depth (producer thread; ViT precompute stays serial). |

---

## 5. Block-diffusion turn/attention machinery

These functions are **REUSED unchanged** by episode-packing — multi-turn just feeds them longer
label/turn vectors. CPU-verified by `test_multiturn_block_index` and `test_cross_turn_attention`.

### `compute_response_block_idx(labels, block_size) -> (response_block_idx, turn_idx, n_blocks)`
Walks `labels` left→right. Within each contiguous response segment (`labels != -100`), assigns
`response_block_idx = current_block + (pos_in_seg // block_size)` (context tokens = `-1`). At the end
of a segment, `current_block += ceil(seg_len / block_size)` so the **next** response segment (next
turn) gets strictly larger block ids. `turn_idx[i]` increments whenever `response_block_idx`
changes vs `i-1`. → multiple turns in one sequence get increasing block ids and turn ids, exactly
what cross-turn attention needs.

### `asymmetric_allowed(q_idx, kv_idx, turn_idx_noisy, turn_idx_clean, n_noisy) -> bool mask`
Positions `< n_noisy` are noisy (`lt`), `>= n_noisy` are clean (the `x0` branch). It maps each
q/kv to its turn id (noisy vs clean lookup) and OR-s three allowances:
- **`block_diagonal`** — noisy↔noisy, **same turn** → within-turn **bidirectional** (block-diffusion).
- **`offset_block_causal`** — noisy q (turn `tq`) → clean kv (turn `tk`), allowed only if `tq > tk`
  → noisy attends to **strictly earlier** clean (past) turns; never future clean.
- **`x0_causal`** — clean→clean with `pos_q >= pos_kv` → the clean branch is plain **AR causal**.

Built once per sample as a dense `(total, total)` boolean mask (then `& valid_dual` to zero pad
rows). This dense mask is the practical cap on `total`; the Phase-B kernel would encode this
predicate directly. `compute_mrope_position_ids_np` loops modality runs (text / one image per
`image_grid_thw` row) to build the multi-image 3D mRoPE consumed by the forward.

---

## 6. Checkpoint + HF upload

- **`export_hf_safetensors`** — inverse of `params.create_model_from_safe_tensors`. Walks the NNX
  state and rebuilds HF-style tensor names for the full Qwen3-VL-2B (vision blocks, merger,
  `deepstack_merger_list`, language layers, tied `lm_head = embed_tokens`), applying `linear`
  (transpose) / `conv3d` (5D permute) transforms. Copies tokenizer/config/chat-template sidecar
  files from `--model-dir`. Writes `jax_export_summary.json` (`mrope_exact`, `deepstack_exact`).
- **`save_checkpoint_bundle`** — wraps the export into `final/` or `checkpoint-stepNNNNNN/`, plus
  copies of `train_log.jsonl`, `tpu_usage.jsonl`, `run_config.json`, `data_summary.json`,
  `summary.json`, and a `checkpoint_manifest.json`.
- **`maybe_upload_checkpoint_bundle`** — creates the HF repo (`exist_ok`) and `upload_folder` to
  `{prefix}/{ckpt_dir.name}`. Logs `hf_checkpoint_upload_{start,done,failed}` to `train_log.jsonl`.
  Non-fatal unless `--hf-upload-strict`. `--delete-local-uploaded-checkpoints` prunes non-final
  bundles after upload. Triggered mid-run every `--hf-upload-every-steps` and/or at the end via
  `--hf-upload-final`.
- **`save_nnx_state_npz`** (`--save-final`) — raw NNX state as a single `.npz` (JAX-native, not HF).

---

## 7. Mesh / data-parallel

**No mesh code change for v6e-32.** `n_devices = jax.device_count()` (auto 32 on v6e-32). When
`--data-parallel`: `Mesh(np.asarray(jax.devices()), ("dp",))`, batch sharded with `P("dp")`, model
and optimizer state **replicated** with `P()`. `--batch-size` is rounded up to a multiple of
`n_devices` (logged `batch_size_rounded`). `train_step` is `@nnx.jit`; the per-step weights
(`ce/kd/temp/lambda_fs`) are traced scalars so changing `bd`/`lambda_fs` per step causes **no
recompile**. Start at `--batch-size 32` (1 episode/chip); try 64 (2/chip) after a clean smoke run.

---

## 8. Gotchas

- **`bd` must NOT enter JIT.** It is sampled host-side (`step_bd`) and folded into the host-side
  `lambda_fs` scalar; threading `bd` into `train_step` would recompile per block size. The whole
  point of the traced-scalar design is one compiled `train_step`.
- **`kd_fewstep` uses pair INDEX 0**, which is the `mask_idx` (heavily-masked, large-effective-block)
  view — the bd32-failure-like regime. Pair-1 (`comp_idx`) is its complement and only feeds
  `kd_noisy`. Do not swap the pair index.
- **`--kd-fewstep-weight 0.0` ⇒ byte-identical** to the previous 3-term loss. This is the OFF
  baseline; use it for the ablation, not a separate code path.
- **Episode mode requires an `episode_id` column** (raises `ValueError` otherwise) and reads the
  `image` column first, falling back to `screenshot`
  (`row_value(step, "image", row_value(step, "screenshot"))` — packed-hybrid carries only
  `screenshot`, so it resolves via the fallback); both are handled by `_episode_image_bytes`.
- **ctx-cap overflow ≠ silent drop in episode mode.** Oldest turns are trimmed and the episode is
  counted as `truncated_episodes` in `data_windows.jsonl`. Only a single over-cap turn is skipped.
- **Vision is frozen / precomputed.** `compute_vision_embeds_for_window` runs ViT+DeepStack **outside**
  JIT (no grad) and groups by grid for pmap; multi-image episodes fall back to the per-image loop in
  `compute_vision_embeds` and concatenate **in image order** (don't reorder — `cumsum(mask)-1`
  depends on it).
- **`total = noisy_pad_to + pad_to`** drives HBM. Keep `noisy_pad_to` small (text-only) to fit large
  `pad_to`. Phase-A `4096 + 1024` is the runnable v6e-32 config; literal 8k needs the Phase-B kernel.
- **`--loss-token-cap`** truncates supervised tokens per branch (sparse LM-head); over-cap samples
  are flagged via `loss_tokens_truncated` (logged as `loss_tokens_truncated_count`). Raise it (256
  in the recipe) for long episodes, at the cost of memory.

---

## For the paper

Bridge file: **`/home/perelman/Weasel_toy_experiment/PAPER_BIB.md`** (maps this code to the W0 paper
`main.tex` / `w0_references.bib`). Status:

- **Already in the paper (reuse as scaffolding):** the **degree-2 curriculum**
  `P(b) ∝ exp(−λ1 ln b − λ2 (ln b)²)` (Prop. ~L1920-1941; `λ2=0` ⇒ Boltzmann power law `b^{−λ1}`,
  ~L1832-1861), the **`b*` critical block size** (strict-JSON 1.000@b1 … 0.945@b16, **0.569@b32**;
  "b=16 holds, b=32 breaks"; ~L975-994), and the **b32 2.8× latency** headline (1290→461 ms).
- **NOT yet in the paper — `kd_fewstep` is NOVEL vs `bard2026`.** BARD distills from a *fixed*
  small-block anchor across stages; ours distills the model's **own clean/AR branch (stop-grad)**
  into the **pair-0 large-block, few-effective-step** diffusion student, **bd-weighted and
  b16-targeted**. Lineage to cite (in `PAPER_BIB.md`, to add to `w0_references.bib`): **SDTT**
  (Deschenaux & Gulcehre, ICLR 2025), **Diffusion Duality / Duo** (Sahoo et al., ICML 2025),
  **Consistency Models** (Song et al., 2023), **Hinton et al. 2015**.

`kd_fewstep` in paper notation (as in `PAPER_BIB.md`):

```latex
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}^{\mathrm{noisy}}
            + \tfrac{3}{4}\,\mathcal{L}_{\mathrm{CE}}^{\mathrm{clean}}
            + \tfrac{1}{4}\,\mathcal{L}_{\mathrm{KD}}^{\mathrm{noisy}}
            + \lambda_{\mathrm{fs}}(b)\;\mathcal{L}_{\mathrm{KD\text{-}step}}
\qquad
\lambda_{\mathrm{fs}}(b) = \lambda_0 \cdot \min\!\big(b/b_{\mathrm{ref}},\, c\big),\;
b_{\mathrm{ref}}=4,\; c=4\ (\text{b16-cap})
```
where `L_{KD-step} = KL(p^{clean}_{AR} ‖ p^{noisy}_{pair-0})` at temperature `kd_temp` with
`stop_gradient` on the teacher, implemented in `dual_stream_loss_jax` and weight-scheduled host-side
(plus a linear warmup factor on `λ0` not shown above). Evidence target: recover **strict-JSON@b16/b32**
(paper ~L975-994) without re-introducing the b4-anchor staging of `bard2026`.
