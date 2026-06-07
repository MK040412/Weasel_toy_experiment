# Dual-stream Fast-dVLM decode — serving-time block-diffusion

Serving-time decode for the **Fast-dVLM** action head (GUI-Owl-1.5-2B, AR→block-diffusion VLA)
on AndroidWorld. This doc covers the **inference path only**: `dual_stream_decode_jax.py`
(the generalized dual-stream block-diffusion decoder) and how `androidworld_tpu_jax_server.py`
serves it. The matching *training* path lives in `scripts/train_fastdvlm_tpu.py`
(see `scripts/CLAUDE_train_fastdvlm_tpu.md`); the serving mask is a literal mirror of the training mask.

**Why it matters:** the bd-sweep (aw_eval) showed block-diffusion is *near-lossless to bd4* and
that result is produced by **this** decoder. The single biggest correctness property here is that
**bd4 serving is byte-identical to the old hardcoded bd4 code** — generalizing to arbitrary `bd`
introduced zero regression at the operating point the paper reports.

---

## TL;DR

```bash
# AR baseline (bd1):
python androidworld_tpu_jax_server.py --model-path <ckpt> --decode grounded_ar_jit
# block-diffusion dual-stream, any bd in {1,2,4,8,16,32} (GEN_LEN=96 must be divisible by bd):
python androidworld_tpu_jax_server.py --model-path <ckpt> --decode dual_dvlm_bd4 --bd 16 --tau 0.9
```

- **Layout:** one forward over a concatenated sequence `[ noisy_text (TEXT-ONLY) | clean_full_multimodal ]`.
  The **noisy** branch (length `NOISY_CAP=448`) holds masked response tokens + prompt *text*; the
  **clean** branch (length `CLEAN_CAP=PROMPT_CAP+GEN_LEN=736`) holds the full grounded multimodal prefix
  (image tokens + deepstack) and committed generated tokens. Logits are read **only off the noisy branch**.
- **bd is data, not control flow.** `bd` enters the kernel through exactly two places:
  `_turn_indices` (block stride `np.arange(gen_len)//bd`) and the host-side `active_len = min(bd, GEN_LEN-committed)`.
  The attention algebra, shapes, and JIT are bd-agnostic ⇒ **bd4 is byte-identical to the pre-generalization code**.
- **Decode loop:** outer loop over blocks (commit `active_len` tokens per block), inner confidence-threshold
  loop (Fast-dLLM style): unmask every position with `conf>tau`, else unmask the single most-confident one.
  Early-stop on `<|im_end|>` with no earlier `[MASK]`.
- **Mask = training's `asymmetric_allowed`**, term-for-term: within-turn **block-diagonal** (noisy↔noisy),
  noisy→past-clean **offset-block-causal**, clean→clean **AR x0-causal**.
- **kd_fewstep trains exactly this regime.** Large-bd / few-step diffusion (bd16/bd32) is the hard,
  lossy end of the sweep; `kd_fewstep` (trainer, pair-0) distills the clean/AR branch into precisely the
  heavily-masked large-block student this decoder runs at high bd. Serving consumes; training pays.

---

## Dual-stream serve layout `[noisy_text | clean_full]`

The decode is a **single concatenated sequence** of length `TOTAL_CAP = NOISY_CAP + CLEAN_CAP = 448 + 736 = 1184`,
built once in `_prepare_dual_prompt` and stepped by the JIT kernels. Two sub-streams:

| Stream | Range in concat | Length | Contents | Modality |
|---|---|---|---|---|
| **noisy** | `[0, NOISY_CAP)` | `NOISY_CAP=448` | prompt **text** tokens (image tokens dropped) + the `GEN_LEN=96` response slots (start all `[MASK]`) | **TEXT-ONLY** |
| **clean** | `[NOISY_CAP, TOTAL_CAP)` | `CLEAN_CAP=736` | full prompt incl. `IMG_TOKEN_ID` slots → vision/deepstack merged in; committed generated tokens | **multimodal (x0)** |

Caps (`dual_stream_decode_jax.py` top):

```python
PROMPT_CAP = 640      # max prompt_len (text+image tokens) on the clean branch
GEN_LEN    = 96       # response length, FIXED (server asserts --gen-len == 96)
NOISY_CAP  = 448      # noisy branch length: prompt_text_len + GEN_LEN must be <= this
CLEAN_CAP  = PROMPT_CAP + GEN_LEN   # = 736
TOTAL_CAP  = NOISY_CAP + CLEAN_CAP  # = 1184
```

**Construction (`_prepare_dual_prompt`):**
- `mm_types` = `mm_token_type_ids` if present else `(input_ids == IMG_TOKEN_ID)`. `prompt_text_positions = where(mm_types==0)`.
  The noisy branch takes **only** `input_ids[prompt_text_positions]` → image tokens never enter the noisy stream.
- Vision runs **once**: `model.model.visual.forward_static_with_deepstack(...)` → `vision_embeds` + 3 deepstack tensors,
  reused every step (not recomputed in the loop).
- mRoPE: clean positions from `compute_mrope_position_ids_np(clean_mm_types, grid, pad_to=CLEAN_CAP)` (multi-image-aware).
  Noisy positions are **copied** from the clean positions of the corresponding text/gen slots
  (`noisy_pos3[:, :text] = clean_pos3[:, prompt_text_positions]`; gen slots map to `prompt_len + arange(GEN_LEN)`),
  then `[noisy_pos3 | clean_pos3]` is concatenated and `sin/cos` precomputed via `modeling._generate_interleaved_mrope`.
  ⇒ a generated token sees the **same RoPE phase** on both branches.

**Memory note (load-bearing):** `total = NOISY_CAP + CLEAN_CAP`, and the noisy branch is **text-only / small**.
Dense attention is `(NOISY_CAP + PROMPT_CAP + GEN_LEN)² = 1184²`, **not** `(2·8k)²`. This is what makes the dual
stream cheap. Literal full-8k 60-turn episodes would need a Phase-B splash/flash mask-mod kernel encoding
`asymmetric_allowed`; the dense `(total,total)` boolean mask used here is the cap. Phase-A (these caps) runs today.

---

## bd-generalized decode (bd appears only in `_turn_indices` + `active_len`)

The file name still says `bd4`, but the decoder is **fully bd-parameterized**. `bd` reaches the computation
through exactly two surfaces — everything else is bd-invariant:

**(1) `_turn_indices(prompt_len, prompt_text_positions, gen_len, bd)`** — the ONLY place `bd` shapes the mask:

```python
clean_block[gen_clean_pos] = np.arange(gen_len) // int(bd)     # <-- block stride = bd
# clean_turn = cumulative "did the block id change?" -> turn index per clean position
# noisy_turn copies clean_turn at the text positions and the gen positions
```

`bd` only sets how many consecutive response slots share a block id (`//bd`). The downstream `clean_turn`/`noisy_turn`
arrays (turn index = running count of block-id changes) are what the mask consumes. Larger `bd` ⇒ fewer, fatter blocks
⇒ more positions share a turn ⇒ wider within-block bidirectional attention. This is identical to training's
`compute_response_block_idx(labels, block_size)` (which also does `pos_in_seg // block_size`).

**(2) `active_len`** — host-side per-block commit width, in the outer loop:

```python
active_len = min(bd, GEN_LEN - committed_len)   # last block may be short
```

`active_len` is passed in as a **traced scalar** to `_dual_bd4_step` (and recomputed inside `_dual_bd4_core`'s
`outer_body` as `jnp.minimum(BD4, GEN_LEN - c)`), so it gates *which* slots are unmaskable this block. It does
**not** change shapes ⇒ no recompile across bd.

**bd4 parity (zero regression).** At `bd=4`: `np.arange(96)//4` is exactly the old hardcoded block layout, and
`active_len=min(4, …)` is the old fixed step. Since `bd` touches nothing else (same `TOTAL_CAP`, same mask clauses,
same JIT), **bd4 output is byte-identical to the pre-generalization decoder** — the operating point the paper reports
(strict-JSON 0.952@b4) is unaffected by the generalization. (Encoded in aw_eval/CLAUDE.md gotcha: *"bd appears only in
`_turn_indices` (`//bd`) and `active_len` (`min(bd,…)`), so bd=4 is byte-identical to the old code"*.)

The server gates divisibility: `dual_dvlm_decode` raises if `GEN_LEN % bd != 0` (96 ⇒ bd ∈ {1,2,4,8,16,32}).

### Decode loop (Fast-dLLM-style confidence unmasking)

Two entry points, identical math:
- `_dual_bd4_step` — **one** forward, returns `(pred, conf)` per gen slot. Driven by the **Python** `while` loop in
  `dual_dvlm_decode` (host orchestrates commit / early-stop). This is the path the server actually calls.
- `_dual_bd4_core` — the **fully-fused** `jax.lax.while_loop` (outer block loop + inner unmask loop) version, kept
  for an all-on-TPU run; same algebra, mask precomputed once as `base_allowed`.

Per block (`dual_dvlm_decode`):
1. `active = (gen_idx >= committed_len) & (gen_idx < committed_len + active_len)`.
2. Inner loop while any active slot is still `[MASK]`:
   - forward → `pred/conf` (argmax + softmax-max confidence) read off rows `prompt_text_len + gen_idx - 1`
     (next-token convention: slot *i* is predicted from position *i−1* on the noisy branch).
   - `over_tau = (conf > tau) & can_update`. If any: commit **all** over-τ slots; else commit the **single** argmax-conf slot.
   - `nfe += 1` per forward. Early-stop: if `IM_END_ID` appears and no `[MASK]` precedes its first occurrence, `done`.
3. `committed_len += active_len`.

Output: truncate at first `IM_END_ID`, strip `EOS/IM_END/MASK`, `processor.decode(...)`. Returns `(raw, n_tokens, nfe)`.

---

## Attention mask logic

Built inside the JIT kernels from `noisy_turn`/`clean_turn` + the live `valid` lengths. With
`x0 := (idx >= NOISY_CAP)` meaning "this position is on the clean/x0 branch", `q_pos/kv_pos` the within-branch
positions, and `tq/tk` the turn indices, the **allowed** structure is three OR'd clauses (`dual_stream_decode_jax.py:189-193`):

```python
allowed = (
    ((~x0_q) & (~x0_kv) & (tq == tk))            # 1. noisy<->noisy: within-turn BLOCK-DIAGONAL (bidirectional)
    | ((tq > tk) & x0_kv & (~x0_q))              # 2. noisy q -> clean kv: OFFSET-BLOCK-CAUSAL (past turns only)
    | (x0_q & x0_kv & (q_pos >= kv_pos))         # 3. clean<->clean: AR x0-CAUSAL (lower-triangular)
)
```

| Clause | q → kv | Meaning |
|---|---|---|
| **block-diagonal** | noisy → noisy, same turn | A masked response block attends bidirectionally to itself (diffusion within block). |
| **offset-block-causal** | noisy → clean, `tq > tk` | The denoising block reads the *committed* (clean) past — earlier turns only, never its own clean copy. |
| **x0-causal** | clean → clean, `q_pos ≥ kv_pos` | The clean/AR prefix is plain causal — supplies grounded vision + committed tokens. |

Then **validity gating** turns the static structure into the live per-step mask:

```python
noisy_valid = noisy_idx < (prompt_text_len + committed_len + active_len)   # text + committed + this block
clean_valid = clean_idx < (prompt_len + committed_len)                     # full prompt + committed gen
valid = concat([noisy_valid, clean_valid])
attn  = allowed & valid[:,None] & valid[None,:]
attn  = attn | ((~valid)[:,None] & (eye))        # invalid rows attend to self (no NaN softmax)
```

The `~valid → self` diagonal fallback keeps padded rows from producing all-`-inf` softmax rows. `active_len`
(hence `bd`) only moves the noisy validity frontier; the `allowed` algebra is untouched. `clean_valid` excludes
the gen region until tokens are committed (a slot's *clean* copy only becomes visible after it is committed),
so the model can never peek at its own future via the clean branch.

---

## Relationship to TRAINING

The serving mask is a **line-by-line mirror** of the trainer's `asymmetric_allowed` (`scripts/train_fastdvlm_tpu.py:666-684`):

```python
# TRAINING  (asymmetric_allowed)            ==        SERVING (_dual_bd4_step allowed)
block_diagonal      = (~x0_q)&(~x0_kv)&(tq==tk)        ((~x0_q)&(~x0_kv)&(tq==tk))
offset_block_causal = (tq>tk)&x0_kv&(~x0_q)            ((tq>tk)&x0_kv&(~x0_q))
x0_causal           = x0_q&x0_kv&(pos_q>=pos_kv)       (x0_q&x0_kv&(q_pos>=kv_pos))
return block_diagonal | offset_block_causal | x0_causal
```

Same three clauses, same `n_noisy` split semantics (`lt`/`NOISY_CAP`), same turn-index inputs. The serving
`_turn_indices` reproduces the training `compute_response_block_idx` block layout (`// block_size`). So **serve =
train forward, minus the gradient and the second pair** — there is no train/serve skew in the attention pattern.

**kd_fewstep trains exactly the large-bd few-step regime this serves.** Training runs **two** noisy pairs per
sample: pair-0 = `mask_idx` (heavily-masked, large-block, **few-effective-step** student) and pair-1 = `comp_idx`
(complement). The 4th loss term, `kd_fewstep`, is evaluated **only on pair-0** (`dual_stream_loss_jax`:
`token_kl.reshape(global_batch, pair_batch, cap)[:, 0, :]`), with the **clean/AR branch hidden as the teacher**
(`stop_gradient`, forward-KL at `kd_temp`, no third forward). That pair-0 student *is* the high-bd diffusion mode
this decoder runs: when you serve `--bd 16`/`--bd 32`, every block has many masked slots resolved in few inner
steps — precisely what pair-0 + `kd_fewstep` optimizes. The host-side weight (`scripts/train_fastdvlm_tpu.py:2077-2087`)

```python
lambda_fs = kd_fewstep_weight * min((step+1)/warmup, 1.0) * min(step_bd/bd_ref, bd_cap)   # bd_ref=4, bd_cap=4.0
# lambda0=0.25  ->  b4:0.25  b8:0.50  b16:1.00  b32:1.00 (capped)  -- "b16-conservative"
```

up-weights `kd_fewstep` exactly when the trainer samples a large `step_bd`. So the curriculum that *trains* large
blocks and the distillation that *fixes* their few-step loss are aimed at the **same** decode this server runs at
high `--bd`. Serving here is the consumer; `kd_fewstep` is the training-time investment that should recover
strict-JSON@b16/b32 (the lossy tail in aw_eval's bd-sweep).

Two structural train↔serve differences (both intentional, neither changes the mask):
- Training masks via `mask_idx`/`comp_idx` (Bernoulli per-block noise `pblk`); serving starts the gen region all-`[MASK]`
  and **commits** progressively (the inner τ-loop), so `noisy_valid` grows by `active_len` per block.
- Training carries 2 pairs (context-axis kd) + multimodal teacher; serving carries 1 pair and reads logits only off noisy.

---

## Server flags (`--decode --bd`)

`androidworld_tpu_jax_server.py` (single TPU, `LOCK` + batch=1, FastAPI `/predict`, `/health`):

| Flag | Default | Effect |
|---|---|---|
| `--decode` | `grounded_ar_jit` | `grounded_ar_jit` (AR), `dvlm_bd4` (single-stream block-diffusion), **`dual_dvlm_bd4`** (this dual-stream decoder). |
| `--bd` | `4` | Block size for the `*_bd4` decodes. `GEN_LEN(96) % bd == 0` enforced. bd1≈AR, larger=more parallel/lossy. |
| `--tau` | `0.9` | Confidence threshold for the inner unmask loop. |
| `--gen-len` | `96` | **Must** equal `GEN_LEN=96` (asserted twice; dual decode rejects any other value). |
| `--max-pixels` | `100352` | Processor pixel cap (controls image-token count → prompt_len). |
| `--include-history` / `--include-ui` | off | Optional serving context; trimmed by `build_inputs_capped` to fit `prompt_cap`. |

**Decode dispatch** (`/predict`, under `LOCK`): `dual_dvlm_bd4` → `dual_dvlm_decode(model, config, enc, processor, gen_len, tau, block_size=bd)`.
When `--decode dual_dvlm_bd4`, the server sets `STATE["prompt_cap"] = dual_stream_decode_jax.PROMPT_CAP (640)` (vs the
grounded-AR `PROMPT_CAP`); `build_inputs_capped` then trims optional history/UI so `prompt_len <= 640` instead of HTTP-500'ing.
Response JSON echoes `decode, bd, tau, nfe, tokens, prompt_cap, gen_len`. Warmup runs one dummy decode at boot so the
JIT compile cost is paid before `READY`.

`bd=1 → grounded_ar_jit; bd>1 → dual_dvlm_bd4 --bd N` is the aw_eval `decode_for_bd` convention (the eval harness picks
AR for bd1 and this dual decoder for everything above).

### Gotchas

- **`NOISY_CAP=448` / `PROMPT_CAP=640` are training-regime caps.** Long-prompt AW tasks overflow
  (`prompt_text_len + GEN_LEN > NOISY_CAP` or `prompt_len > PROMPT_CAP`) → `ValueError` → that task fails. Raising the
  caps costs memory **and a recompile** (shapes change). The clean full-sweep fix is bigger caps; the current caps match
  the SFT context window.
- **`--gen-len` is frozen at 96.** Both server and `_prepare_dual_prompt` assert it. The JIT shapes (`GEN_LEN`, `TOTAL_CAP`)
  bake it in; do not pass another value.
- **bd never enters the JIT signature.** It only reshapes host-side numpy (`_turn_indices`) and the scalar `active_len`.
  Changing `--bd` does **not** trigger recompile — only changing a cap does. This is why the bd-sweep is cheap.
- **Logits are noisy-branch only** (`hidden[:, :NOISY_CAP, :]`). The clean branch is context, never a prediction source.
- **vision is computed once** in prep and threaded through every step; don't move it inside the loop.
- **`dvlm_bd4` ≠ `dual_dvlm_bd4`.** The former (`dvlm_decode_jax.py`) is single-stream; this doc is the dual-stream path.

---

## For the paper

See `/home/perelman/Weasel_toy_experiment/PAPER_BIB.md` for the full code→paper bridge and bib keys
(`arriola2025bd3lm`, `wu2026fastdvlm`, `wu2025fastdllm`, `bard2026`, `bai2025qwen3vl`, `xu2026mobileagent35`).

This decoder *produces* the paper's headline serving results: the **`b*` critical block size** finding
(strict-JSON 1.000@b1, 0.952@b4, 0.945@b16, **0.569@b32**; "b=16 holds, b=32 breaks", paper ~L975-994) and the
**b32 latency** headline (1290ms→461ms, **2.8×**, abstract/Fig.1 ~L123-125) are exactly the `--decode dual_dvlm_bd4 --bd {…}`
sweep run through `dual_stream_decode_jax.py`. Because **bd4 is byte-identical** to the pre-generalization code, those
numbers are regression-safe under the generalized decoder.

The lossy tail (b16/b32) is the target of **`kd_fewstep`** — the step-axis self-distillation term that is **novel vs
`bard2026`** (BARD distills from a *fixed* small-block anchor across stages; ours distills the model's **own** clean/AR
branch, stop-grad, into the pair-0 large-block few-step student, bd-weighted + b16-capped). `kd_fewstep` is **not yet in
`main.tex`**. In paper notation (PAPER_BIB.md §"kd_fewstep in paper notation"):

```latex
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}^{\mathrm{noisy}}
            + \tfrac{3}{4}\,\mathcal{L}_{\mathrm{CE}}^{\mathrm{clean}}
            + \tfrac{1}{4}\,\mathcal{L}_{\mathrm{KD}}^{\mathrm{noisy}}
            + \lambda_{\mathrm{fs}}(b)\;\mathcal{L}_{\mathrm{KD\text{-}step}},
\qquad
\lambda_{\mathrm{fs}}(b) = \lambda_0 \cdot \min\!\big(b/b_{\mathrm{ref}},\, c\big),\;
b_{\mathrm{ref}}=4,\; c=4\ (\text{b16-cap})
```

where `L_{KD-step} = KL( p^{clean}_{AR} \,\|\, p^{noisy}_{pair-0} )` at temperature `kd_temp` with `stop_gradient`
on the teacher. The **pair-0 large-block, few-effective-step student** in that equation is precisely the high-`--bd`
serving mode this file implements: train the regime, then serve it.
