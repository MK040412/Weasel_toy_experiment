# Qwen3-VL modeling — architecture + the attention/mask CONTRACT episode-packing depends on

Reference doc for `src/qwen/qwen3vl/modeling.py` (JAX/Flax NNX, `Qwen3VLForConditionalGeneration`,
Qwen3-VL-2B-Instruct == GUI-Owl-1.5-2B backbone). **Why this file exists:** Fast-dVLM's dual-stream
trainer (`scripts/train_fastdvlm_tpu.py`) and especially **episode-packing** (multi-turn, multi-image,
`--data-mode episode`) lean on a small number of *contracts* in this model — how vision tokens are
scattered into the text stream, how M-RoPE position ids are laid out per-image, where DeepStack injects,
and what shape/semantics the attention mask must have. Break any one and the model trains on garbage
**silently** (no shape error). This doc states each contract precisely so a reader understands the file
without opening it.

> 핵심: 이 모델은 `(B, T)` causal LM이 아니라 **`(B, total, total)` dense-bool-mask** 를 그대로 받는다.
> `total = noisy_pad_to + pad_to`. Vision token은 `cumsum(mask)-1` 로 **순서대로** 소비된다 — multi-image
> concat + trailing pad가 correct한 이유가 전부 여기서 나온다.

---

## TL;DR

- **Two towers, one merge.** Vision ViT (24 blocks, hidden 1024) → PatchMerger → `out_hidden_size=2048`
  tokens that live in the **text** embedding space. Text decoder = 28-layer Qwen3 (hidden 2048, GQA
  16 q / 8 kv heads, head_dim 128, vocab 151936, tied embeddings). `Qwen3VLForConditionalGeneration`
  is the top wrapper; `Qwen3VLModel` holds `.visual` + `.language_model`.
- **Merge is a scatter, not a concat.** `merge_modalities(img_emb, text_emb, token_mask)` writes the
  k-th vision token into the k-th masked position via `idx = cumsum(token_mask)-1`. Vision rows are
  consumed **in array order**; trailing pad rows are never indexed (`clip` keeps the index in range,
  `jnp.where` keeps the text value). → multi-image just means **concatenate the per-image vision blocks
  in image order**, and **zero-pad** to a static length. That is the whole multi-image story.
- **DeepStack is residual injection at 3 depths.** ViT extracts features after blocks **5 / 11 / 17**
  (`deepstack_visual_indexes`), each through its own post-shuffle PatchMerger → 3 tensors. The LLM
  *adds* them back at decoder layers **0 / 1 / 2** at the same image positions (`add_visual_embeds`,
  same `cumsum-1` scatter).
- **M-RoPE = interleaved 3-axis rope.** Position ids are `[3, B, T]` = (temporal/text, height, width).
  `_generate_interleaved_mrope` builds the temporal table, then *overwrites stride-3 channel slices*
  with the height/width tables (sections `mrope_section=(24,20,20)`). Text tokens have t==h==w so M-RoPE
  collapses to ordinary rope on text.
- **Attention contract (today).** Training passes a **dense `(B,1,total,total)` boolean** "allowed"
  mask straight into `Qwen3VLAttention`; `True`=attend. The mask *is* the policy — block-diagonal +
  offset-block-causal + AR-causal (the `asymmetric_allowed` of the dual stream) is fully encoded in
  those `total²` bits. The model does **not** know about turns, blocks, or noisy-vs-clean; it just
  obeys the mask.
- **The dense mask is the memory cap, not the FLOP cap.** `total = noisy_pad_to + pad_to`, and the
  noisy branch is **text-only / small**, so dense attention scales with `(pad_to + small)²`, NOT
  `(2·8k)²`. Phase-A (`pad_to≈4096`, `noisy_pad_to≈1024`) runs on **v6e-32 today**. Literal full-8k
  60-turn episodes need a **Phase-B splash/flash mask-mod kernel** that *computes* `asymmetric_allowed`
  on the fly instead of materializing the `(total,total)` bool.
- **For the paper:** this backbone is `bai2025qwen3vl` (GUI-Owl via `xu2026mobileagent35`); the
  dual-stream loss that consumes this contract carries `kd_fewstep` (novel vs `bard2026`). See
  `## For the paper`.

---

## Architecture (data flow)

```
pixel_values (P_total, C·tp·ps·ps)   input_ids (B, T)               # T = clean seq (+ noisy text rows upstream)
   │  per image: P_i = t·h·w patches    │
   ▼                                     ▼
Qwen3VLVisionModel (24 blocks)        embed_tokens  (Embed 151936×2048, tied)
   PatchEmbed (Conv3d t×16×16)           │
 + interpolated pos_embed                │   inputs_embeds (B, T, 2048)
 + 2D rope (_rot_pos_emb)                │
   ├─ extract @ blocks 5/11/17 ──► DeepStack mergers ──► ds0, ds1, ds2  (post-shuffle norm)
   ▼                                     │
   merger (PatchMerger, spatial 2×2)     │
   ▼                                     ▼
 vision_embeds (V, 2048) ───────────► merge_modalities  (cumsum(mask)-1 scatter)
                                         │   inputs_embeds with vision rows in place
                                         ▼
                         Qwen3VLTextModel (28 × Qwen3VLDecoderLayer)
                           layer 0/1/2: + add_visual_embeds(ds0/ds1/ds2)   # DeepStack inject
                           GQA + M-RoPE (sin,cos from interleaved mrope)
                           dense bool mask (B,1,total,total)
                           jax.checkpoint per layer (training, cache=None)
                         RMSNorm → logits = hidden @ embed.Tᵀ  (tied)
```

`from_pretrained(model_path)` loads HF safetensors via `params.create_model_from_safe_tensors`.
`forward_train` is the simple cache-free CE path; the **real** training path is the dual-stream
function in the trainer, which calls `model.model.language_model(...)` directly with the dense mask
and the DeepStack list (see Attention contract below).

---

## Components

| Component | Class / fn (modeling.py) | Shape / key facts |
|---|---|---|
| **Vision config** | `Qwen3VLVisionConfig.qwen3vl_2b` | depth=24, hidden=1024, heads=16, head_dim=64, patch=16, temporal_patch=2, spatial_merge=2, out_hidden=2048, `deepstack_visual_indexes=(5,11,17)`, gelu, rope_theta=1e4 |
| **Text config** | `Qwen3VLTextConfig.qwen3vl_2b` | vocab=151936, hidden=2048, inter=6144, **28 layers**, 16 q-heads / **8 kv-heads** (GQA n_rep=2), head_dim=128, rope_theta=**5e6**, `mrope_section=(24,20,20)`, silu, tied embeddings |
| **Model ids** | `ModelConfig` | image_token=151655, video_token=151656, vision_start=151652, vision_end=151653 |
| Patch embed | `Qwen3VLPatchEmbed` | Conv `(temporal_patch=2, 16, 16)` stride=kernel; reshapes each patch to `(3, 2, 16, 16)` = `C·tp·ps·ps = 1536` flattened, maps `(P,1536-in)→(P,1024)` (`hidden_size`) |
| ViT pos embed | `_fast_pos_embed_interpolate` | bilinear-interpolates a `2304=48²` grid pos table to the actual `grid_h×grid_w`, then re-orders into 2×2 merge blocks |
| ViT 2D rope | `_rot_pos_emb` | row/col freq tables concatenated → `(cos,sin)`; head_dim 64, rotary_dim 32 |
| ViT block | `Qwen3VLVisionBlock` | pre-norm LN → full (non-causal) attn → LN → gelu MLP; residual |
| Patch merger | `Qwen3VLPatchMerger` | reshapes `spatial_merge²=4` patches → one token; `use_postshuffle_norm` distinguishes the **main** merger (pre-norm, →2048) from the **DeepStack** mergers (post-shuffle norm) |
| Vision forward | `Qwen3VLVisionModel.__call__` / `forward_static_with_deepstack` | returns `(merged_hidden (V,2048), [ds0,ds1,ds2])`. `forward_static_*` take **python-int** `grid_h/grid_w/grid_t` so they are pmap/vmap-safe (no `int()` on a tracer) |
| Text attention | `Qwen3VLAttention` | per-head `q_norm`/`k_norm` (RMSNorm on head_dim), GQA `repeat_kv(n_rep=2)`, M-RoPE applied to q,k. **Three attention backends** (see below) |
| Decoder layer | `Qwen3VLDecoderLayer` | pre-norm RMSNorm → attn → RMSNorm → SwiGLU MLP; residual |
| Text model | `Qwen3VLTextModel.__call__` | loops 28 layers; **wraps each in `jax.checkpoint` when `cache is None`** (training memory); injects DeepStack at layers `i < len(deepstack_visual_embeds)` |
| KV cache | `LayerCache` / `init_cache` / `make_causal_mask` | inference-only; rounds cache size to next power-of-two; `cur_ind` advances by `seq_len`. Training uses `cache=None` |
| Merge / inject | `merge_modalities`, `add_visual_embeds`, `batched_*` | the `cumsum(mask)-1` scatter — the contract, detailed below |

Attention backends in `Qwen3VLAttention.__call__` (env-selected, all consume the **same dense bool mask**):

| Backend | Trigger | Notes |
|---|---|---|
| **Dense XLA** (default) | else-branch | `attn_weights = q·kᵀ·scale`; `jnp.where(mask, attn_weights, _K_MASK)` then softmax. `mask` is the `(B,1,T,T)` bool. **This is the path episode-packing uses today.** |
| Splash (Pallas TPU) | `QWEN_TPU_SPLASH_ATTENTION=1` & `cache is None` | `make_splash_mha_single_device(mask_one, ...)` per-batch via vmap; `mask` broadcast to `(B,H,Tq,Tk)`. **This is the Phase-B hook** (today it still takes a materialized mask). |
| `jax.nn.dot_product_attention` (XLA) | `QWEN_TPU_DPA_ATTENTION=1` & `cache is None` | mask passed as `mask=` kwarg |

---

## M-RoPE: interleaved 3-axis position ids (multi-image)

**Position ids are `[3, B, T]`**: axis 0 = temporal (== text counter for text tokens), axis 1 = height,
axis 2 = width. Two pieces:

**(a) Building the ids — `compute_mrope_position_ids_np` (trainer, host-side).**
Walks the sequence in maximal runs of equal `mm_token_type_ids` (0=text, 1=image):
- **text run** (len `L`): all three axes = `arange(L) + current_pos`; advance `current_pos += L`.
- **image run**: `get_vision_position_ids_np(current_pos, grid, spatial_merge_size)` lays out
  `(t,h,w)` per LLM grid cell (`llm_h = grid_h//2`, `llm_w = grid_w//2`); then
  **`current_pos += max(grid_h, grid_w)//spatial_merge_size`** — i.e. the *next* text token starts
  after the larger of the image's H/W spans, the standard Qwen-VL "image occupies a square block of
  position space" rule. Multi-image just iterates `image_i` over `image_grid_thw` in order; each image
  advances `current_pos` by its own H/W span. Final `[3, pad_to]` array, pad columns left at 0.

**(b) Consuming the ids — `_generate_interleaved_mrope(position_ids_3d, head_dim, rope_theta, mrope_section)`.**
```python
inv_freq = 1/ rope_theta**(arange(0,head_dim,2)/head_dim)      # half_dim = head_dim//2 = 64
freqs    = einsum("dbt,k->dbtk", pos_3d, inv_freq)             # d∈{0,1,2}
freqs_out = freqs[0]                                            # start = temporal/text table
for dim_idx, offset in ((1,1),(2,2)):                          # height@offset1, width@offset2
    length  = min(mrope_section[dim_idx]*3, half_dim)
    indices = arange(offset, length, 3)                        # every 3rd channel
    freqs_out[:,:,indices] = freqs[dim_idx][..., indices]      # interleave h/w in
return sin(freqs_out), cos(freqs_out)
```
So the 64 rope channels are **interleaved** t,h,w,t,h,w,… with `mrope_section=(24,20,20)` controlling
how far the h/w stripes extend. **Text tokens have t==h==w**, so the interleave is a no-op on text and
M-RoPE == plain rope there. Output `(sin,cos)` matches `_generate_rope`'s half-dim format and is fed to
`_apply_rope`, which doubles the **half-dim (64) cos/sin tables** to full head_dim (128) via
`cos_full=concat([cos,cos])`, `sin_full=concat([sin,sin])` and then calls
`apply_rotary_pos_emb(x, cos_full, sin_full)` (it is the cos/sin that are doubled, not `x`).

> Gotcha: `_generate_interleaved_mrope` is the **only** rope used in the dual-stream forward
> (`scripts/train_fastdvlm_tpu.py` calls it directly). The plain `_generate_rope` is only for the
> cache/AR path in `Qwen3VLForConditionalGeneration.__call__`. Multi-image correctness lives in the
> **host-side** `compute_mrope_position_ids_np`, not in the kernel.

---

## DeepStack: extract @ ViT 5/11/17 → inject @ LLM 0/1/2

**Extract (vision side, `Qwen3VLVisionModel`).** During the block loop, when `layer_idx in (5,11,17)`,
the running hidden state is passed through `deepstack_merger_list[ds_idx]` (a PatchMerger with
`use_postshuffle_norm=True`) and appended to `deepstack_features`. Order is fixed by
`deepstack_visual_indexes` → `ds_idx = indexes.index(layer_idx)` → `[ds@5, ds@11, ds@17]`. Each
`ds_k` has the **same token count `V`** and width 2048 as the main `vision_embeds` (post-merge).

**Inject (text side, `Qwen3VLTextModel.__call__`).** After decoder layer `i` runs,
```python
if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
    hidden = batched_add_visual_embeds(hidden, visual_pos_masks, deepstack_visual_embeds[i])
```
i.e. `ds0→after layer 0`, `ds1→after layer 1`, `ds2→after layer 2`. `add_visual_embeds` **adds**
the k-th deepstack token to the k-th image position using the *same* `cumsum(mask)-1` scatter as the
merge (so injection lands on exactly the image tokens, residually). `visual_pos_masks` is the image
mask over the **full** dual sequence (zeros for the noisy text block prepended, then `vision_mask`;
see trainer lines that build `visual_pos_masks = concat([zeros(lt), vision_mask])`).

> If `deepstack_visual_embeds` is `None` (e.g. `forward_train`, `get_hidden_states`), injection is
> skipped — those paths use only the main merge. DeepStack is wired **only** through the trainer's
> `language_model(...)` call.

---

## Modality merge contract — `cumsum(mask)-1` consumes vision tokens IN ORDER

This is the load-bearing invariant. From modeling.py:

```python
def merge_modalities(img_emb, text_emb, token_mask):       # all length-T (or (V,·)/(T,·))
    img_indices  = jnp.cumsum(token_mask) - 1              # 0-based running count of vision slots
    safe_indices = jnp.clip(img_indices, 0, img_emb.shape[0]-1)
    aligned      = img_emb[safe_indices]                   # gather k-th vision row at k-th vision slot
    return jnp.where(token_mask[:, None], aligned, text_emb)

def add_visual_embeds(hidden, visual_embeds, token_mask):  # identical scatter, but ADD (residual)
    if visual_embeds.shape[0] == 0: return hidden
    idx  = jnp.clip(jnp.cumsum(token_mask)-1, 0, visual_embeds.shape[0]-1)
    return jnp.where(token_mask[:,None], hidden + visual_embeds[idx], hidden)
```
`batched_merge_modalities` / `batched_add_visual_embeds` = `jax.vmap` over batch.

**What `cumsum(mask)-1` does.** At each token position, the index equals *(number of vision tokens at
or before this position) − 1*. So:
- the 1st masked position gets `img_emb[0]`, the 2nd gets `img_emb[1]`, … — **strictly in array order**;
- on text positions the index is "the last vision token seen" but `jnp.where` discards it (keeps text);
- if there are **more** vision rows than slots, the surplus rows are simply never selected;
- if there are **fewer** rows than slots (or zero, before the first image), `clip(...,0,V-1)` pins the
  index in range and `jnp.where` still keeps text on text positions. Before the first image
  `cumsum-1 = -1 → clip → 0`, harmless because those are text positions anyway.

**WHY multi-image concat + `pad_vision_embeds` is correct.** Episode-packing builds one sequence with
image blocks from several turns. `compute_vision_embeds` (trainer) forwards each image **separately**
and **concatenates `vision_embeds` and each DeepStack tensor in image order** (it splits `pixel_values`
by per-image patch counts `t·h·w`, loops `i`, `jnp.concatenate(..., axis=0)`):

```
# compute_vision_embeds, multi-image branch (trainer):
#   "concatenate the vision + deepstack tokens IN IMAGE ORDER. The downstream merge
#    (merge_modalities / add_visual_embeds) scatters tokens via cumsum(mask)-1, i.e. consumes them
#    in this exact order against the image-token positions of the packed sequence;
#    trailing pad rows are never indexed."
```
Because the merge consumes vision rows **in order**, and the packed sequence's image-token positions are
also **in image order** (image-1's tokens precede image-2's in `mm_token_type_ids`), the k-th vision row
lands on the k-th image slot — **for free**, no per-image bookkeeping at merge time. Then
`pad_vision_embeds(vision_embeds, target_len, 2048, dtype)` zero-pads the concatenated block to a
**static** length so JIT shapes are fixed. Those **trailing pad rows are never indexed** (total image
slots ≤ real vision rows), so padding is a pure no-op on the result. This is the entire reason
multi-image "just works" with the single-image merge kernel unchanged.

> Invariant to preserve: **(# image tokens in the packed sequence) == (# concatenated vision rows
> before padding)**, and **same image order on both sides**. Violate either and the scatter silently
> mis-aligns vision to the wrong positions — no exception, just a broken model. DeepStack uses the
> identical scatter, so the *same* concat order must hold for `ds0/ds1/ds2`.

---

## Attention / mask contract — dense `(B,1,total,total)` bool today

The model's attention is **mask-driven**. `Qwen3VLAttention.__call__(x, cache, sin, cos, mask)`:
in the default dense path it computes `attn_weights = q·kᵀ·scale` then
**`jnp.where(mask, attn_weights, _K_MASK)`** (`_K_MASK = bf16 min`) before softmax. So `mask[...,i,j]=True`
⇔ query *i* may attend to key *j*. The model has **no internal notion** of turns / blocks / causality —
all of it is precomputed into those bits.

**Shapes & `total`.** The dual stream stacks a **noisy (text-only) branch** on top of the **clean**
branch:
```
total = lt + clean_len = noisy_pad_to + pad_to        # trainer: total_len = lt + clean_len
demb  = concat([ noisy_emb (lt) , clean_emb (clean_len) ], axis=seq)   # one (B·pair, total, 2048)
mask  : (B·pair, 1, total, total) bool                # broadcast from asymmetric_allowed
```
The clean half carries the vision tokens; the **noisy half is text-only** (image positions are dropped
when extracting `text_positions`). The mask is built **host-side** by `asymmetric_allowed` and is the
*only* thing telling the model which of the three regions may talk to which:

```python
def asymmetric_allowed(q_idx, kv_idx, turn_idx_noisy, turn_idx_clean, n_noisy):
    x0_q, x0_kv = q_idx >= n_noisy, kv_idx >= n_noisy          # True ⇒ in the CLEAN (x0) half
    ... map to per-token turn ids tq, tk ...
    block_diagonal      = (~x0_q)&(~x0_kv)&(tq==tk)            # noisy↔noisy: WITHIN-TURN only (block-diag)
    offset_block_causal = (tq> tk)& x0_kv &(~x0_q)             # noisy(q) → PAST clean(kv): strictly earlier turns
    x0_causal           = x0_q & x0_kv &(pos_q>=pos_kv)        # clean↔clean: ordinary AR causal
    return block_diagonal | offset_block_causal | x0_causal
```
then masked again by `valid_dual` (drop padded rows/cols on both axes). Reading it:
- **noisy block-diagonal:** a masked/diffusion token attends only to other noisy tokens **of its own
  turn** (parallel within-block denoising, no leakage across turns).
- **offset-block-causal:** a noisy token of turn *t* may read **clean** tokens of **strictly earlier**
  turns (`tq>tk`) — its frozen context — but never the clean copy of its own turn (no answer leak).
- **x0-causal:** the clean branch is a plain left-to-right AR teacher.

This exact mask is **reused unchanged** by episode-packing; `compute_response_block_idx` produces the
per-turn `turn_idx` across all assistant segments of the packed multi-turn sequence, so the same three
rules tile across 60 turns with zero new mask logic. The model just sees a bigger `(total,total)` bool.

> Gotcha: the noisy branch length is `noisy_pad_to` and is **text-only**. That is *the* reason the dense
> mask is affordable (next section). Also: the same `mask` object flows into the splash and DPA backends
> unchanged — any future mask must stay a plain `(B,1,T,T)`-broadcastable bool.

---

## Phase-A vs Phase-B — the dense mask is the **memory** cap

`total = noisy_pad_to + pad_to`, and dense attention materializes `(total, total)` bool + `(total,total)`
scores per head. **But the noisy branch is text-only and small**, so:

```
mem(attention) ∝ total² = (pad_to + noisy_pad_to)²  ≈ (pad_to + small)²    # NOT (2·8k)²
```

| | pad_to (clean) | noisy_pad_to | total | dense `(total,total)` | status |
|---|---|---|---|---|---|
| **Phase-A** | ~4096 | ~1024 | ~5120 | ~26M bool/score | **runs on v6e-32 today** |
| **Phase-B** | 8192 (60-turn) | larger | up to ~16k | ~256M+ materialized | needs mask-mod kernel |

**Phase-B path to literal 8k.** Don't materialize the bool. Encode `asymmetric_allowed` as a
**splash/flash `mask_mod` function** `(q_idx, kv_idx) -> bool` that recomputes block-diagonal ∨
offset-block-causal ∨ x0-causal from the (precomputed, cheap) per-token turn ids — so attention never
allocates `(total,total)`. The Splash backend already exists in `Qwen3VLAttention`
(`QWEN_TPU_SPLASH_ATTENTION=1`, `make_splash_mha_single_device`); today it still *consumes a
materialized mask*. Phase-B = swap that materialized mask for a Pallas `mask_mod` closure encoding the
**asymmetric** (non-causal, non-symmetric) rule. The dense `(total,total)` mask is therefore the
**current cap** on episode length, and the only thing between Phase-A and full-8k 60-turn episodes.

> 요약: FLOP은 `(pad_to+small)²` 로 이미 작다 — 막는 건 dense bool **메모리**. Phase-B에서
> `asymmetric_allowed` 를 splash `mask_mod` 로 옮기면 8k가 열린다.

---

## Quick invariants checklist (don't break these)

1. **Merge order:** concatenated `vision_embeds` (and each `ds_k`) must be in **image order**, matching
   the order of image runs in `mm_token_type_ids`. `cumsum(mask)-1` assumes it.
2. **Count:** `#image-tokens == #vision-rows` (pre-pad). Surplus pad rows OK (never indexed); a shortfall
   silently reuses the last row.
3. **Mask is truth:** `True`=attend. All turn/block/causal structure must be baked into the
   `(B,1,total,total)` bool by `asymmetric_allowed` + `valid_dual`. The model adds nothing.
4. **`total = noisy_pad_to + pad_to`**, noisy half **text-only**. Keep it text-only or the memory
   argument (and the `text_positions` extraction) breaks.
5. **M-RoPE ids are `[3,B,T]`**, built host-side; text ⇒ t==h==w; each image advances `current_pos`
   by `max(h,w)//spatial_merge`.
6. **DeepStack:** exactly 3 tensors, injected after LLM layers 0/1/2, same scatter as merge.

---

## For the paper

Cite via **`/home/perelman/Weasel_toy_experiment/PAPER_BIB.md`** (which bridges this repo to
`W0Tech/paper/main.tex` + `w0_references.bib`).

- **Backbone:** `bai2025qwen3vl` (Qwen3-VL Technical Report) — the architecture documented here:
  M-RoPE `mrope_section=(24,20,20)`, DeepStack `(5,11,17)`, vocab 151936 / hidden 2048 / 28 layers.
  GUI-Owl-1.5-2B lineage: `xu2026mobileagent35` (+ `ye2025mobileagentv3`).
- **Block-diffusion action head** consuming this contract: `arriola2025bd3lm`, `sahoo2024mdlm`,
  `wu2026fastdvlm`, `liang2025discretevla`.
- **The asymmetric dual-stream mask** (`asymmetric_allowed`) is what makes the noisy (few-effective-step,
  large-block) student and the clean/AR teacher coexist in **one** forward over `(total,total)`. The
  step-axis self-distillation term that rides on it, `kd_fewstep`, is **novel vs `bard2026`** (BARD
  distills from a fixed small-block anchor; we distill the model's **own clean/AR branch, stop-grad**,
  into the **pair-0** large-block student — see `dual_stream_loss_jax`). Lineage to cite:
  SDTT (`https://arxiv.org/abs/2410.21035`), Diffusion Duality (`https://arxiv.org/abs/2506.10892`),
  Consistency Models, Hinton KD — all listed in PAPER_BIB.md "To ADD".

**`kd_fewstep` in paper notation** (total loss; teacher = clean/AR branch hidden, `stop_gradient`):

```latex
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}^{\mathrm{noisy}}
            + \tfrac{3}{4}\,\mathcal{L}_{\mathrm{CE}}^{\mathrm{clean}}
            + \tfrac{1}{4}\,\mathcal{L}_{\mathrm{KD}}^{\mathrm{noisy}}
            + \lambda_{\mathrm{fs}}(b)\;\mathcal{L}_{\mathrm{KD\text{-}step}},
\qquad
\mathcal{L}_{\mathrm{KD\text{-}step}}
   = \mathrm{KL}\!\big(p^{\mathrm{clean}}_{\mathrm{AR}} \,\big\|\, p^{\mathrm{noisy}}_{\text{pair-0}}\big)
   \ \text{at temp } \tau_{\mathrm{kd}},
```
```latex
\lambda_{\mathrm{fs}}(b) = \lambda_0 \cdot
   \min\!\Big(\tfrac{t+1}{T_{\mathrm{warmup}}},\,1\Big)\cdot
   \min\!\Big(\tfrac{b}{b_{\mathrm{ref}}},\, c\Big),
\quad b_{\mathrm{ref}}=4,\; c=4\ (\text{b16-cap}),\; \lambda_0=0.25 .
```
- `pair-0` = the `mask_idx` view (heavily-masked / large-block / **few-effective-step** student),
  selected as `token_kl_pairs[:, 0, :]` in `dual_stream_loss_jax`. Teacher is the clean branch hidden
  with `stop_gradient` ⇒ **no third forward**. Divergence is **forward KL** at `kd_temp`.
- `λ_fs` is computed **host-side** in the dispatch loop (block size `b == step_bd` is known only
  host-side), then passed as a **traced scalar** `jnp.asarray(lambda_fs)` exactly like the other loss
  weights ⇒ **no recompile, no `b` in the JIT**. With `λ0=0.25, b_ref=4, c=4`:
  `λ_fs(b4)=0.25, b8=0.50, b16=1.00, b32=1.00 (capped)` — **b16-targeted**.
- `--kd-fewstep-weight 0.0` (default) ⇒ term OFF ⇒ **byte-identical 3-term loss**.
- **Curriculum that feeds `b`:** degree-2 `degree2_bd_probs(bd, λ1, λ2) ∝ exp(-λ1·ln b - λ2·(ln b)²)`
  (paper Prop. ~L1920-1941); `λ2=0` reduces **exactly** to the Boltzmann power law `b^{-λ1}`
  (paper ~L1832-1861).
- **Evidence target:** recover strict-JSON validity at b16/b32 (paper `b*` finding ~L975-994:
  `1.000@b1 → 0.952@b4 → 0.945@b16 → 0.569@b32`; b32 latency 1290→461ms, 2.8×, abstract/Fig.1).

Implemented in `dual_stream_loss_jax` + the dispatch loop of `scripts/train_fastdvlm_tpu.py`; CPU-verified
by `scripts/test_fastdvlm_kd_curriculum.py`.
