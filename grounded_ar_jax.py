"""Fixed-shape grounded AR decode for AndroidWorld JAX/TPU serving.

This follows the training-time grounding path: Qwen3-VL interleaved mRoPE for
the multimodal prompt and DeepStack visual features injected into the text
layers. The decode loop stays inside one NNX/JAX graph over padded shapes so the
AndroidWorld first request does not pay a per-token Python/XLA compile cost.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from qwen.qwen3vl import modeling

IMG_TOKEN_ID = 151655
EOS_TOKEN_ID = 151643
IM_END_ID = 151645
SPATIAL_MERGE = 2
PROMPT_CAP = 1280
GEN_LEN = 96


def get_vision_position_ids_np(start_position: int, grid_thw: np.ndarray, spatial_merge_size: int) -> np.ndarray:
    grid_t, grid_h, grid_w = [int(x) for x in grid_thw.tolist()]
    llm_grid_t = grid_t
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size
    position_temporal = np.arange(llm_grid_t, dtype=np.int32).repeat(llm_grid_h * llm_grid_w) + start_position
    position_height = np.arange(llm_grid_h, dtype=np.int32) + start_position
    position_height = np.repeat(position_height, llm_grid_w)
    position_height = np.tile(position_height, llm_grid_t)
    position_width = np.arange(llm_grid_w, dtype=np.int32) + start_position
    position_width = np.tile(position_width, llm_grid_h * llm_grid_t)
    return np.stack([position_temporal, position_height, position_width], axis=0)


def compute_mrope_position_ids_np(
    mm_token_type_ids: np.ndarray,
    image_grid_thw: np.ndarray,
    *,
    spatial_merge_size: int,
    pad_to: int,
) -> tuple[np.ndarray, int]:
    position_ids = np.zeros((3, pad_to), dtype=np.int32)
    current_pos = 0
    out_chunks: list[np.ndarray] = []
    image_i = 0
    i = 0
    n = int(mm_token_type_ids.shape[0])
    while i < n:
        modality = int(mm_token_type_ids[i])
        j = i + 1
        while j < n and int(mm_token_type_ids[j]) == modality:
            j += 1
        if modality == 0:
            text_len = j - i
            out_chunks.append(
                np.broadcast_to(np.arange(text_len, dtype=np.int32)[None, :] + current_pos, (3, text_len))
            )
            current_pos += text_len
        else:
            grid = image_grid_thw[image_i]
            vp = get_vision_position_ids_np(current_pos, grid, spatial_merge_size)
            out_chunks.append(vp)
            current_pos += max(int(grid[1]), int(grid[2])) // spatial_merge_size
            image_i += 1
        i = j
    pos = np.concatenate(out_chunks, axis=1)
    position_ids[:, :n] = pos
    if n < pad_to:
        tail = np.arange(pad_to - n, dtype=np.int32) + current_pos
        position_ids[:, n:] = np.broadcast_to(tail[None, :], (3, pad_to - n))
    return position_ids, current_pos


def prepare_grounded_prompt(
    model,
    config,
    enc: dict,
    *,
    prompt_cap: int = PROMPT_CAP,
    gen_len: int = GEN_LEN,
    dtype=jnp.bfloat16,
) -> dict[str, object]:
    input_ids = np.asarray(enc["input_ids"][0], dtype=np.int32)
    prompt_len = int(input_ids.shape[0])
    if prompt_len > prompt_cap:
        raise ValueError(f"prompt_len {prompt_len} exceeds prompt_cap {prompt_cap}")
    if int(gen_len) != GEN_LEN:
        raise ValueError(f"fixed grounded AR expects gen_len={GEN_LEN}, got {gen_len}")

    max_total = prompt_cap + gen_len
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

    pos3, _ = compute_mrope_position_ids_np(mm_types, grid, spatial_merge_size=SPATIAL_MERGE, pad_to=max_total)
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
        "vision_embeds": vision_embeds,
        "deepstack": (ds0, ds1, ds2),
        "sin": sin,
        "cos": cos,
        "vision_mask": jnp.asarray(vision_mask),
    }


@nnx.jit
def _grounded_ar_core(model, token_buf, prompt_len, sin, cos, vision_embeds, ds0, ds1, ds2, vision_mask):
    lm = model.model.language_model
    gen_idx = jnp.arange(GEN_LEN, dtype=jnp.int32)
    seq_idx = jnp.arange(PROMPT_CAP + GEN_LEN, dtype=jnp.int32)
    gen = jnp.zeros((GEN_LEN,), dtype=jnp.int32)
    step = jnp.array(0, dtype=jnp.int32)
    done = jnp.array(False)

    def forward_tokens(tokens, attn_mask):
        emb = lm.embed_tokens(tokens)
        emb = modeling.batched_merge_modalities(vision_embeds[None], emb, vision_mask)
        hidden = lm(emb, None, sin, cos, attn_mask, vision_mask, [ds0, ds1, ds2])
        if model.lm_head is not None:
            return model.lm_head(hidden)
        return hidden @ lm.embed_tokens.embedding[...].T

    def cond(state):
        _, k, is_done = state
        return (k < GEN_LEN) & (~is_done)

    def body(state):
        gen_state, k, is_done = state
        seq_positions = prompt_len + gen_idx
        tokens = token_buf.at[0, seq_positions].set(gen_state)

        qi = seq_idx[:, None]
        ki = seq_idx[None, :]
        valid_len = prompt_len + k
        valid_q = qi < valid_len
        valid_key = ki < valid_len
        pad_self = (~valid_q) & (qi == ki)
        attn_mask = ((valid_q & valid_key & (ki <= qi)) | pad_self)[None, None, :, :]

        logits = forward_tokens(tokens, attn_mask)
        next_token = jnp.argmax(logits[0, prompt_len + k - 1, :]).astype(jnp.int32)
        next_gen = gen_state.at[k].set(next_token)
        next_done = is_done | (next_token == EOS_TOKEN_ID) | (next_token == IM_END_ID)
        return next_gen, k + 1, next_done

    gen, step, done = jax.lax.while_loop(cond, body, (gen, step, done))
    return gen, step


def grounded_ar_decode(model, config, enc, processor, *, gen_len: int = GEN_LEN) -> tuple[str, int, int, float]:
    prep = prepare_grounded_prompt(model, config, enc, gen_len=gen_len)
    ds0, ds1, ds2 = prep["deepstack"]
    t0 = time.time()
    gen, steps = _grounded_ar_core(
        model,
        prep["token_buf"],
        jnp.asarray(prep["prompt_len"], dtype=jnp.int32),
        prep["sin"],
        prep["cos"],
        prep["vision_embeds"],
        ds0,
        ds1,
        ds2,
        prep["vision_mask"],
    )
    elapsed_ms = (time.time() - t0) * 1000.0
    out = np.asarray(gen, dtype=np.int32).tolist()
    used = int(np.asarray(steps))
    trimmed = out[:used]
    for stop_id in (EOS_TOKEN_ID, IM_END_ID):
        if stop_id in trimmed:
            trimmed = trimmed[: trimmed.index(stop_id) + 1]
            break
    raw = processor.decode([t for t in trimmed if t not in (EOS_TOKEN_ID, IM_END_ID)], skip_special_tokens=True).strip()
    return raw, len(trimmed), used, elapsed_ms
