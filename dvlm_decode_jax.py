"""Grounded block-diffusion decode for the JAX Qwen3-VL AndroidWorld server.

The public PyTorch probes decode from Python. On TPU that causes shape-specific
recompiles and timeout-level latency, so this file keeps the decode loop inside a
single NNX/JAX graph over fixed padded shapes.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from qwen.qwen3vl import modeling
from grounded_decode import (
    EOS_TOKEN_ID,
    IM_END_ID,
    SPATIAL_MERGE,
    compute_mrope_position_ids_np,
)

IMG_TOKEN_ID = 151655
MASK_TOKEN_ID = 151665
PROMPT_CAP = 1280
GEN_LEN = 96
BD4 = 4


def _block_attention_mask(total: int, prompt_len: int, committed_len: int) -> jax.Array:
    qi = np.arange(total, dtype=np.int32)
    allowed = (qi >= (prompt_len + committed_len))[:, None] | (qi[None, :] <= qi[:, None])
    return jnp.asarray(allowed)[None, None, :, :]


def _prepare_grounded_prompt(model, config, enc, gen_len: int, prompt_cap: int = PROMPT_CAP, dtype=jnp.bfloat16):
    input_ids = np.asarray(enc["input_ids"][0], dtype=np.int32)
    prompt_len = int(input_ids.shape[0])
    if prompt_len > prompt_cap:
        raise ValueError(f"prompt_len {prompt_len} exceeds prompt_cap {prompt_cap}")
    if int(gen_len) != GEN_LEN:
        raise ValueError(f"JIT dVLM server expects gen_len={GEN_LEN}, got {gen_len}")
    max_total = prompt_cap + GEN_LEN
    grid = np.asarray(enc["image_grid_thw"], dtype=np.int32)
    pixel_values = jnp.asarray(np.asarray(enc["pixel_values"]), dtype=dtype)

    mm = enc.get("mm_token_type_ids")
    if mm is not None:
        mm_types = np.asarray(mm[0], dtype=np.int32)
    else:
        mm_types = (input_ids == IMG_TOKEN_ID).astype(np.int32)

    vision_embeds, deepstack = model.model.visual.forward_static_with_deepstack(
        pixel_values,
        grid_t=int(grid[0, 0]),
        grid_h=int(grid[0, 1]),
        grid_w=int(grid[0, 2]),
    )
    vision_embeds = vision_embeds.astype(dtype)
    ds0, ds1, ds2 = [d.astype(dtype)[None] for d in deepstack]

    pos3, _ = compute_mrope_position_ids_np(
        mm_types,
        grid,
        spatial_merge_size=SPATIAL_MERGE,
        pad_to=max_total,
    )
    pos3 = jnp.asarray(pos3)[:, None, :]
    sin, cos = modeling._generate_interleaved_mrope(
        pos3,
        config.text_config.head_dim,
        config.text_config.rope_theta,
        config.text_config.mrope_section,
    )

    vision_mask = np.zeros((1, max_total), dtype=bool)
    vision_mask[0, :prompt_len] = mm_types > 0
    token_buf = np.zeros((1, max_total), dtype=np.int32)
    token_buf[0, :prompt_len] = input_ids

    return {
        "token_buf": jnp.asarray(token_buf),
        "prompt_len": prompt_len,
        "max_total": max_total,
        "vision_embeds": vision_embeds,
        "deepstack": (ds0, ds1, ds2),
        "sin": sin,
        "cos": cos,
        "vision_mask": vision_mask,
    }


@nnx.jit
def _dvlm_bd4_core(model, token_buf, prompt_len, sin, cos, vision_embeds, ds0, ds1, ds2, vision_mask, tau):
    lm = model.model.language_model
    gen_idx = jnp.arange(GEN_LEN, dtype=jnp.int32)
    seq_idx = jnp.arange(PROMPT_CAP + GEN_LEN, dtype=jnp.int32)
    gen = jnp.full((GEN_LEN,), MASK_TOKEN_ID, dtype=jnp.int32)
    committed_len = jnp.array(0, dtype=jnp.int32)
    nfe = jnp.array(0, dtype=jnp.int32)
    done = jnp.array(False)

    def forward_tokens(tokens, attn_mask):
        emb = lm.embed_tokens(tokens)
        emb = modeling.batched_merge_modalities(vision_embeds[None], emb, vision_mask)
        hidden = lm(emb, None, sin, cos, attn_mask, vision_mask, [ds0, ds1, ds2])
        if model.lm_head is not None:
            return model.lm_head(hidden)
        return hidden @ lm.embed_tokens.embedding[...].T

    def outer_cond(state):
        _, c, _, is_done = state
        return (c < GEN_LEN) & (~is_done)

    def outer_body(state):
        gen_state, c, nfe_state, is_done = state
        active_len = jnp.minimum(jnp.array(BD4, dtype=jnp.int32), jnp.array(GEN_LEN, dtype=jnp.int32) - c)
        active_gen = (gen_idx >= c) & (gen_idx < c + active_len)

        qi = seq_idx[:, None]
        ki = seq_idx[None, :]
        valid_key = ki < (prompt_len + c + active_len)
        valid_q = qi < (prompt_len + c + active_len)
        active_q = (qi >= (prompt_len + c)) & (qi < (prompt_len + c + active_len))
        prefix_q = qi < (prompt_len + c)
        pad_self = (~valid_q) & (qi == ki)
        attn_mask = ((valid_key & (active_q | (prefix_q & (ki <= qi)))) | pad_self)[None, None, :, :]

        def inner_cond(inner_state):
            inner_gen, _, inner_done = inner_state
            has_mask = jnp.any((inner_gen == MASK_TOKEN_ID) & active_gen)
            return has_mask & (~inner_done)

        def inner_body(inner_state):
            inner_gen, inner_nfe, inner_done = inner_state
            seq_positions = prompt_len + gen_idx
            tokens = token_buf.at[0, seq_positions].set(inner_gen)
            logits = forward_tokens(tokens, attn_mask)
            rows = prompt_len + gen_idx - 1
            scores = jnp.take(logits[0], rows, axis=0)
            scores = scores.astype(jnp.float32)
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


def dvlm_decode(model, config, enc, processor, gen_len: int, tau: float, block_size: int):
    """Block-wise diffusion decode.

    The committed prefix grows autoregressively over blocks; positions inside the
    active block can attend bidirectionally to each other, matching the training
    block-diffusion objective.
    """

    if int(block_size) != BD4:
        raise ValueError(f"JIT dVLM server currently supports bd={BD4}, got {block_size}")
    prep = _prepare_grounded_prompt(model, config, enc, gen_len)
    prompt_len = prep["prompt_len"]
    t0 = time.time()
    ds0, ds1, ds2 = prep["deepstack"]
    gen, nfe = _dvlm_bd4_core(
        model,
        prep["token_buf"],
        jnp.asarray(prompt_len, dtype=jnp.int32),
        prep["sin"],
        prep["cos"],
        prep["vision_embeds"],
        ds0,
        ds1,
        ds2,
        jnp.asarray(prep["vision_mask"]),
        jnp.asarray(float(tau), dtype=jnp.float32),
    )
    committed = np.asarray(gen, dtype=np.int32).tolist()
    nfe_int = int(np.asarray(nfe))
    n = committed.index(IM_END_ID) + 1 if IM_END_ID in committed else int(gen_len)
    out = [t for t in committed[:n] if t not in (EOS_TOKEN_ID, IM_END_ID, MASK_TOKEN_ID)]
    raw = processor.decode(out, skip_special_tokens=True).strip()
    _ = (time.time() - t0) * 1000.0
    return raw, n, nfe_int
