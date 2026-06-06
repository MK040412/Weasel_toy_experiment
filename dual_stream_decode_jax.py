"""Dual-stream Fast-dVLM decode for AndroidWorld TPU serving.

This mirrors the training layout in ``scripts/train_fastdvlm_continue.py``:

    [noisy_text_tokens | clean_full_multimodal_tokens]

The noisy branch is text-only and predicts masked response tokens. It attends
bidirectionally within the current response block and attends to the clean
multimodal branch only for previous blocks. The clean branch supplies the
grounded visual prefix and committed generated tokens.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from qwen.qwen3vl import modeling
from grounded_ar_jax import compute_mrope_position_ids_np

IMG_TOKEN_ID = 151655
EOS_TOKEN_ID = 151643
IM_END_ID = 151645
MASK_TOKEN_ID = 151665
SPATIAL_MERGE = 2

# Training used --ctx-cap/--pad-to 480. Keep serving close to that regime while
# leaving headroom for slightly larger AndroidWorld screenshots.
PROMPT_CAP = 640
GEN_LEN = 96
BD4 = 4
NOISY_CAP = 448
CLEAN_CAP = PROMPT_CAP + GEN_LEN
TOTAL_CAP = NOISY_CAP + CLEAN_CAP


def _turn_indices(prompt_len: int, prompt_text_positions: np.ndarray, gen_len: int, bd: int = BD4) -> tuple[np.ndarray, np.ndarray]:
    clean_block = np.full((CLEAN_CAP,), -1, dtype=np.int32)
    gen_clean_pos = prompt_len + np.arange(gen_len, dtype=np.int32)
    clean_block[gen_clean_pos] = np.arange(gen_len, dtype=np.int32) // int(bd)

    clean_turn = np.zeros((CLEAN_CAP,), dtype=np.int32)
    for i in range(1, CLEAN_CAP):
        clean_turn[i] = clean_turn[i - 1] + (1 if clean_block[i] != clean_block[i - 1] else 0)

    noisy_turn = np.zeros((NOISY_CAP,), dtype=np.int32)
    text_len = int(prompt_text_positions.shape[0])
    noisy_turn[:text_len] = clean_turn[prompt_text_positions]
    noisy_turn[text_len : text_len + gen_len] = clean_turn[gen_clean_pos]
    return noisy_turn, clean_turn


def _prepare_dual_prompt(model, config, enc, gen_len: int, dtype=jnp.bfloat16, bd: int = BD4):
    if int(gen_len) != GEN_LEN:
        raise ValueError(f"dual-stream JIT decode expects gen_len={GEN_LEN}, got {gen_len}")

    input_ids = np.asarray(enc["input_ids"][0], dtype=np.int32)
    prompt_len = int(input_ids.shape[0])
    if prompt_len > PROMPT_CAP:
        raise ValueError(f"prompt_len {prompt_len} exceeds prompt_cap {PROMPT_CAP}")

    grid = np.asarray(enc["image_grid_thw"], dtype=np.int32)
    pixel_values = jnp.asarray(np.asarray(enc["pixel_values"]), dtype=dtype)
    mm = enc.get("mm_token_type_ids")
    if mm is not None:
        mm_types = np.asarray(mm[0], dtype=np.int32)
    else:
        mm_types = (input_ids == IMG_TOKEN_ID).astype(np.int32)

    prompt_text_positions = np.nonzero(mm_types == 0)[0].astype(np.int32)
    prompt_text_len = int(prompt_text_positions.shape[0])
    if prompt_text_len + GEN_LEN > NOISY_CAP:
        raise ValueError(f"noisy_text_len {prompt_text_len + GEN_LEN} exceeds noisy_cap {NOISY_CAP}")

    vision_embeds, deepstack = model.model.visual.forward_static_with_deepstack(
        pixel_values,
        grid_t=int(grid[0, 0]),
        grid_h=int(grid[0, 1]),
        grid_w=int(grid[0, 2]),
    )
    vision_embeds = vision_embeds.astype(dtype)
    ds0, ds1, ds2 = [d.astype(dtype)[None] for d in deepstack]

    clean_ids = np.zeros((CLEAN_CAP,), dtype=np.int32)
    clean_ids[:prompt_len] = input_ids
    clean_vision_mask = np.zeros((CLEAN_CAP,), dtype=bool)
    clean_vision_mask[:prompt_len] = mm_types > 0

    noisy_prompt = np.zeros((NOISY_CAP,), dtype=np.int32)
    noisy_prompt[:prompt_text_len] = input_ids[prompt_text_positions]

    clean_mm_types = np.zeros((CLEAN_CAP,), dtype=np.int32)
    clean_mm_types[:prompt_len] = mm_types
    clean_pos3, _ = compute_mrope_position_ids_np(
        clean_mm_types,
        grid,
        spatial_merge_size=SPATIAL_MERGE,
        pad_to=CLEAN_CAP,
    )

    gen_clean_positions = prompt_len + np.arange(GEN_LEN, dtype=np.int32)
    noisy_pos3 = np.zeros((3, NOISY_CAP), dtype=np.int32)
    noisy_pos3[:, :prompt_text_len] = clean_pos3[:, prompt_text_positions]
    noisy_pos3[:, prompt_text_len : prompt_text_len + GEN_LEN] = clean_pos3[:, gen_clean_positions]
    pos3 = np.concatenate([noisy_pos3, clean_pos3], axis=1)
    pos3 = jnp.asarray(pos3)[:, None, :]
    sin, cos = modeling._generate_interleaved_mrope(
        pos3,
        config.text_config.head_dim,
        config.text_config.rope_theta,
        config.text_config.mrope_section,
    )

    noisy_turn, clean_turn = _turn_indices(prompt_len, prompt_text_positions, GEN_LEN, bd)
    return {
        "clean_ids": jnp.asarray(clean_ids),
        "clean_vision_mask": jnp.asarray(clean_vision_mask[None]),
        "noisy_prompt": jnp.asarray(noisy_prompt),
        "prompt_len": prompt_len,
        "prompt_text_len": prompt_text_len,
        "noisy_turn": jnp.asarray(noisy_turn),
        "clean_turn": jnp.asarray(clean_turn),
        "vision_embeds": vision_embeds,
        "deepstack": (ds0, ds1, ds2),
        "sin": sin,
        "cos": cos,
    }


@nnx.jit
def _dual_bd4_step(
    model,
    clean_ids_base,
    noisy_prompt,
    gen_state,
    prompt_len,
    prompt_text_len,
    committed_len,
    active_len,
    noisy_turn,
    clean_turn,
    sin,
    cos,
    vision_embeds,
    ds0,
    ds1,
    ds2,
    clean_vision_mask,
):
    lm = model.model.language_model
    gen_idx = jnp.arange(GEN_LEN, dtype=jnp.int32)
    noisy_idx = jnp.arange(NOISY_CAP, dtype=jnp.int32)
    clean_idx = jnp.arange(CLEAN_CAP, dtype=jnp.int32)
    total_idx = jnp.arange(TOTAL_CAP, dtype=jnp.int32)

    noisy_tokens = noisy_prompt
    gen_noisy_pos = prompt_text_len + gen_idx
    noisy_tokens = noisy_tokens.at[gen_noisy_pos].set(gen_state)

    clean_tokens = clean_ids_base
    gen_clean_pos = prompt_len + gen_idx
    clean_tokens = clean_tokens.at[gen_clean_pos].set(gen_state)

    noisy_emb = lm.embed_tokens(noisy_tokens)[None]
    clean_emb = lm.embed_tokens(clean_tokens)[None]
    clean_emb = modeling.batched_merge_modalities(vision_embeds[None], clean_emb, clean_vision_mask)
    demb = jnp.concatenate([noisy_emb, clean_emb], axis=1)

    q_idx = total_idx[:, None]
    kv_idx = total_idx[None, :]
    x0_q = q_idx >= NOISY_CAP
    x0_kv = kv_idx >= NOISY_CAP
    q_pos = jnp.where(x0_q, q_idx - NOISY_CAP, q_idx)
    kv_pos = jnp.where(x0_kv, kv_idx - NOISY_CAP, kv_idx)
    tq = jnp.where(
        x0_q,
        jnp.take(clean_turn, jnp.clip(q_pos, 0, CLEAN_CAP - 1)),
        jnp.take(noisy_turn, jnp.clip(q_pos, 0, NOISY_CAP - 1)),
    )
    tk = jnp.where(
        x0_kv,
        jnp.take(clean_turn, jnp.clip(kv_pos, 0, CLEAN_CAP - 1)),
        jnp.take(noisy_turn, jnp.clip(kv_pos, 0, NOISY_CAP - 1)),
    )
    allowed = (
        ((~x0_q) & (~x0_kv) & (tq == tk))
        | ((tq > tk) & x0_kv & (~x0_q))
        | (x0_q & x0_kv & (q_pos >= kv_pos))
    )
    noisy_valid = noisy_idx < (prompt_text_len + committed_len + active_len)
    clean_valid = clean_idx < (prompt_len + committed_len)
    valid = jnp.concatenate([noisy_valid, clean_valid], axis=0)
    attn = allowed & valid[:, None] & valid[None, :]
    attn = attn | ((~valid)[:, None] & (total_idx[:, None] == total_idx[None, :]))

    hidden = lm(
        demb,
        None,
        sin,
        cos,
        attn[None, None, :, :],
        visual_pos_masks=jnp.concatenate([jnp.zeros((1, NOISY_CAP), dtype=jnp.bool_), clean_vision_mask], axis=1),
        deepstack_visual_embeds=[ds0, ds1, ds2],
    )
    noisy_hidden = hidden[:, :NOISY_CAP, :]
    if model.lm_head is not None:
        logits = model.lm_head(noisy_hidden)
    else:
        logits = noisy_hidden @ lm.embed_tokens.embedding[...].T

    rows = prompt_text_len + gen_idx - 1
    scores = jnp.take(logits[0], rows, axis=0).astype(jnp.float32)
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    probs = jax.nn.softmax(scores, axis=-1)
    pred = jnp.argmax(probs, axis=-1).astype(jnp.int32)
    conf = jnp.take_along_axis(probs, pred[:, None], axis=-1)[:, 0]
    return pred, conf


@nnx.jit
def _dual_bd4_core(
    model,
    clean_ids_base,
    noisy_prompt,
    prompt_len,
    prompt_text_len,
    noisy_turn,
    clean_turn,
    sin,
    cos,
    vision_embeds,
    ds0,
    ds1,
    ds2,
    clean_vision_mask,
    tau,
):
    lm = model.model.language_model
    gen_idx = jnp.arange(GEN_LEN, dtype=jnp.int32)
    noisy_idx = jnp.arange(NOISY_CAP, dtype=jnp.int32)
    clean_idx = jnp.arange(CLEAN_CAP, dtype=jnp.int32)
    total_idx = jnp.arange(TOTAL_CAP, dtype=jnp.int32)

    gen = jnp.full((GEN_LEN,), MASK_TOKEN_ID, dtype=jnp.int32)
    committed_len = jnp.array(0, dtype=jnp.int32)
    nfe = jnp.array(0, dtype=jnp.int32)
    done = jnp.array(False)

    q_idx = total_idx[:, None]
    kv_idx = total_idx[None, :]
    x0_q = q_idx >= NOISY_CAP
    x0_kv = kv_idx >= NOISY_CAP
    q_pos = jnp.where(x0_q, q_idx - NOISY_CAP, q_idx)
    kv_pos = jnp.where(x0_kv, kv_idx - NOISY_CAP, kv_idx)
    tq = jnp.where(
        x0_q,
        jnp.take(clean_turn, jnp.clip(q_pos, 0, CLEAN_CAP - 1)),
        jnp.take(noisy_turn, jnp.clip(q_pos, 0, NOISY_CAP - 1)),
    )
    tk = jnp.where(
        x0_kv,
        jnp.take(clean_turn, jnp.clip(kv_pos, 0, CLEAN_CAP - 1)),
        jnp.take(noisy_turn, jnp.clip(kv_pos, 0, NOISY_CAP - 1)),
    )
    base_allowed = (
        ((~x0_q) & (~x0_kv) & (tq == tk))
        | ((tq > tk) & x0_kv & (~x0_q))
        | (x0_q & x0_kv & (q_pos >= kv_pos))
    )

    def forward_tokens(gen_state, c, active_len):
        noisy_tokens = noisy_prompt
        gen_noisy_pos = prompt_text_len + gen_idx
        noisy_tokens = noisy_tokens.at[gen_noisy_pos].set(gen_state)

        clean_tokens = clean_ids_base
        gen_clean_pos = prompt_len + gen_idx
        clean_tokens = clean_tokens.at[gen_clean_pos].set(gen_state)

        noisy_emb = lm.embed_tokens(noisy_tokens)[None]
        clean_emb = lm.embed_tokens(clean_tokens)[None]
        clean_emb = modeling.batched_merge_modalities(vision_embeds[None], clean_emb, clean_vision_mask)
        demb = jnp.concatenate([noisy_emb, clean_emb], axis=1)

        noisy_valid = noisy_idx < (prompt_text_len + c + active_len)
        clean_valid = clean_idx < (prompt_len + c)
        valid = jnp.concatenate([noisy_valid, clean_valid], axis=0)
        attn = base_allowed & valid[:, None] & valid[None, :]
        attn = attn | ((~valid)[:, None] & (total_idx[:, None] == total_idx[None, :]))
        hidden = lm(
            demb,
            None,
            sin,
            cos,
            attn[None, None, :, :],
            visual_pos_masks=jnp.concatenate([jnp.zeros((1, NOISY_CAP), dtype=jnp.bool_), clean_vision_mask], axis=1),
            deepstack_visual_embeds=[ds0, ds1, ds2],
        )
        noisy_hidden = hidden[:, :NOISY_CAP, :]
        emb = lm.embed_tokens.embedding[...]
        if model.lm_head is not None:
            return model.lm_head(noisy_hidden)
        return noisy_hidden @ emb.T

    def outer_cond(state):
        _, c, _, is_done = state
        return (c < GEN_LEN) & (~is_done)

    def outer_body(state):
        gen_state, c, nfe_state, is_done = state
        active_len = jnp.minimum(jnp.array(BD4, dtype=jnp.int32), jnp.array(GEN_LEN, dtype=jnp.int32) - c)
        active_gen = (gen_idx >= c) & (gen_idx < c + active_len)

        def inner_cond(inner_state):
            inner_gen, _, inner_done = inner_state
            return jnp.any((inner_gen == MASK_TOKEN_ID) & active_gen) & (~inner_done)

        def inner_body(inner_state):
            inner_gen, inner_nfe, inner_done = inner_state
            logits = forward_tokens(inner_gen, c, active_len)
            rows = prompt_text_len + gen_idx - 1
            scores = jnp.take(logits[0], rows, axis=0).astype(jnp.float32)
            scores = scores - jnp.max(scores, axis=-1, keepdims=True)
            probs = jax.nn.softmax(scores, axis=-1)
            pred = jnp.argmax(probs, axis=-1).astype(jnp.int32)
            conf = jnp.take_along_axis(probs, pred[:, None], axis=-1)[:, 0]

            can_update = (inner_gen == MASK_TOKEN_ID) & active_gen
            over_tau = (conf > tau) & can_update
            any_take = jnp.any(over_tau)
            best_idx = jnp.argmax(jnp.where(can_update, conf, -1.0)).astype(jnp.int32)
            take = jnp.where(any_take, over_tau, gen_idx == best_idx)
            next_gen = jnp.where(take, pred, inner_gen)

            end_mask = next_gen == IM_END_ID
            has_end = jnp.any(end_mask)
            first_end = jnp.argmax(end_mask).astype(jnp.int32)
            no_mask_before = ~jnp.any((next_gen == MASK_TOKEN_ID) & (gen_idx < first_end))
            next_done = inner_done | (has_end & no_mask_before)
            return next_gen, inner_nfe + 1, next_done

        next_gen, next_nfe, inner_done = jax.lax.while_loop(
            inner_cond,
            inner_body,
            (gen_state, nfe_state, is_done),
        )
        return next_gen, c + active_len, next_nfe, inner_done

    gen, committed_len, nfe, done = jax.lax.while_loop(
        outer_cond,
        outer_body,
        (gen, committed_len, nfe, done),
    )
    return gen, nfe


def dual_dvlm_decode(model, config, enc, processor, gen_len: int, tau: float, block_size: int):
    bd = int(block_size)
    if GEN_LEN % bd != 0:
        raise ValueError(f"GEN_LEN={GEN_LEN} must be divisible by block_size={bd}")
    prep = _prepare_dual_prompt(model, config, enc, gen_len, bd=bd)
    t0 = time.time()
    ds0, ds1, ds2 = prep["deepstack"]
    gen_np = np.full((GEN_LEN,), MASK_TOKEN_ID, dtype=np.int32)
    nfe_int = 0
    committed_len = 0
    done = False
    gen_idx = np.arange(GEN_LEN, dtype=np.int32)
    tau_f = float(tau)
    while committed_len < GEN_LEN and not done:
        active_len = min(bd, GEN_LEN - committed_len)
        active = (gen_idx >= committed_len) & (gen_idx < committed_len + active_len)
        while np.any((gen_np == MASK_TOKEN_ID) & active) and not done:
            pred, conf = _dual_bd4_step(
                model,
                prep["clean_ids"],
                prep["noisy_prompt"],
                jnp.asarray(gen_np),
                jnp.asarray(prep["prompt_len"], dtype=jnp.int32),
                jnp.asarray(prep["prompt_text_len"], dtype=jnp.int32),
                jnp.asarray(committed_len, dtype=jnp.int32),
                jnp.asarray(active_len, dtype=jnp.int32),
                prep["noisy_turn"],
                prep["clean_turn"],
                prep["sin"],
                prep["cos"],
                prep["vision_embeds"],
                ds0,
                ds1,
                ds2,
                prep["clean_vision_mask"],
            )
            pred_np = np.asarray(pred, dtype=np.int32)
            conf_np = np.asarray(conf, dtype=np.float32)
            can_update = (gen_np == MASK_TOKEN_ID) & active
            over_tau = (conf_np > tau_f) & can_update
            if np.any(over_tau):
                take = over_tau
            else:
                take = np.zeros_like(can_update)
                take[int(np.argmax(np.where(can_update, conf_np, -1.0)))] = True
            gen_np[take] = pred_np[take]
            nfe_int += 1
            if IM_END_ID in gen_np:
                first_end = int(np.argmax(gen_np == IM_END_ID))
                done = not np.any(gen_np[:first_end] == MASK_TOKEN_ID)
        committed_len += active_len

    committed = gen_np.tolist()
    n = committed.index(IM_END_ID) + 1 if IM_END_ID in committed else int(gen_len)
    out = [t for t in committed[:n] if t not in (EOS_TOKEN_ID, IM_END_ID, MASK_TOKEN_ID)]
    raw = processor.decode(out, skip_special_tokens=True).strip()
    _ = (time.time() - t0) * 1000.0
    return raw, n, nfe_int
