"""Qwen3-VL JAX implementation adapted from jax-ml/bonsai for JAX 0.6.2.

Stripped all out_sharding / reshard APIs (requires JAX >= 0.8.0).
Runs on single device or with JAX default multi-device replication.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, TypeAlias

import jax
import jax.numpy as jnp
import optax
from flax import nnx

_K_MASK = jnp.finfo(jnp.bfloat16).min
_USE_SPLASH_ATTENTION = os.environ.get("QWEN_TPU_SPLASH_ATTENTION", "0") == "1"
_USE_DPA_ATTENTION = os.environ.get("QWEN_TPU_DPA_ATTENTION", "0") == "1"
_SPLASH_BLOCK_Q = int(os.environ.get("QWEN_TPU_SPLASH_BLOCK_Q", "128"))
_SPLASH_BLOCK_KV = int(os.environ.get("QWEN_TPU_SPLASH_BLOCK_KV", "128"))


# --- Configuration --- #


@dataclass(frozen=True)
class Qwen3VLVisionConfig:
    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple = (5, 11, 17)
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def qwen3vl_2b(cls):
        return cls(
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            out_hidden_size=2048,
            deepstack_visual_indexes=(5, 11, 17),
        )


@dataclass(frozen=True)
class Qwen3VLTextConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000
    mrope_section: tuple = (24, 20, 20)
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    @classmethod
    def qwen3vl_2b(cls):
        return cls(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            tie_word_embeddings=True,
        )


@dataclass(frozen=True)
class ModelConfig:
    vision_config: Qwen3VLVisionConfig
    text_config: Qwen3VLTextConfig
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653

    @classmethod
    def qwen3vl_2b(cls):
        return cls(vision_config=Qwen3VLVisionConfig.qwen3vl_2b(), text_config=Qwen3VLTextConfig.qwen3vl_2b())


# --- Layer Components --- #


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        x_f32 = x.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        out = (x_f32 * rms) * self.weight[...]
        return out.astype(x.dtype)


class Qwen3VLPatchEmbed(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nnx.Conv(
            in_features=config.in_channels,
            out_features=config.hidden_size,
            kernel_size=kernel,
            strides=kernel,
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        cfg = self.config
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size
        )
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)
        return self.proj(hidden_states).reshape(seq_len, cfg.hidden_size)


class Qwen3VLVisionMLP(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.linear_fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=True, rngs=rngs)
        self.linear_fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear_fc1(x)
        x = nnx.gelu(x, approximate=True)
        return self.linear_fc2(x)


def rotate_half(x: jax.Array) -> jax.Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    return (x * cos) + (rotate_half(x) * sin)


class Qwen3VLVisionAttention(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        hidden_size = config.hidden_size
        self.qkv = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=True, rngs=rngs)
        self.proj = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.scale = self.head_dim**-0.5

    def __call__(self, hidden_states: jax.Array, position_embeddings: Tuple[jax.Array, jax.Array]) -> jax.Array:
        seq_len = hidden_states.shape[0]
        cos, sin = position_embeddings
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        qkv_out = self.qkv(hidden_states)
        qkv = qkv_out.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
        attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) * self.scale
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q.dtype)
        out = jnp.matmul(attn_weights, v).transpose(1, 0, 2).reshape(seq_len, -1)
        return self.proj(out)


class Qwen3VLVisionBlock(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.norm2 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.attn = Qwen3VLVisionAttention(config, rngs=rngs)
        self.mlp = Qwen3VLVisionMLP(config, rngs=rngs)

    def __call__(self, hidden_states: jax.Array, position_embeddings: Tuple[jax.Array, jax.Array]) -> jax.Array:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3VLPatchMerger(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False, *, rngs: nnx.Rngs):
        self.config = config
        merge_factor = config.spatial_merge_size**2
        hidden_merged = config.hidden_size * merge_factor
        norm_dim = hidden_merged if use_postshuffle_norm else config.hidden_size
        self.norm = nnx.LayerNorm(norm_dim, epsilon=config.layer_norm_eps, rngs=rngs)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.linear_fc1 = nnx.Linear(hidden_merged, hidden_merged, use_bias=True, rngs=rngs)
        self.linear_fc2 = nnx.Linear(hidden_merged, config.out_hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        if not self.use_postshuffle_norm:
            x = self.norm(x)
        merge_factor = self.config.spatial_merge_size**2
        n_patches = x.shape[0] // merge_factor
        x = x.reshape(n_patches, -1)
        if self.use_postshuffle_norm:
            x = self.norm(x)
        x = self.linear_fc1(x)
        x = nnx.gelu(x)
        return self.linear_fc2(x)


class Qwen3VLVisionModel(nnx.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.patch_embed = Qwen3VLPatchEmbed(config, rngs=rngs)
        self.pos_embed = nnx.Embed(
            num_embeddings=config.num_position_embeddings, features=config.hidden_size, rngs=rngs
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.blocks = [Qwen3VLVisionBlock(config, rngs=rngs) for _ in range(config.depth)]
        self.merger = Qwen3VLPatchMerger(config, use_postshuffle_norm=False, rngs=rngs)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = [
            Qwen3VLPatchMerger(config, use_postshuffle_norm=True, rngs=rngs)
            for _ in range(len(config.deepstack_visual_indexes))
        ]

    def _fast_pos_embed_interpolate(self, grid_thw: jax.Array = None, *, grid_h: int = None, grid_w: int = None, grid_t: int = None) -> jax.Array:
        if grid_h is None:
            grid_h, grid_w = int(grid_thw[0, 1]), int(grid_thw[0, 2])
            grid_t = int(grid_thw[0, 0])
        if grid_t is None:
            grid_t = 1
        h_idxs = jnp.linspace(0, self.num_grid_per_side - 1, grid_h)
        w_idxs = jnp.linspace(0, self.num_grid_per_side - 1, grid_w)
        h_floor = jnp.floor(h_idxs).astype(jnp.int32)
        w_floor = jnp.floor(w_idxs).astype(jnp.int32)
        h_ceil = jnp.clip(h_floor + 1, 0, self.num_grid_per_side - 1)
        w_ceil = jnp.clip(w_floor + 1, 0, self.num_grid_per_side - 1)
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        base_h = h_floor * self.num_grid_per_side
        base_h_ceil = h_ceil * self.num_grid_per_side
        idx00 = (base_h[:, None] + w_floor[None, :]).flatten()
        idx01 = (base_h[:, None] + w_ceil[None, :]).flatten()
        idx10 = (base_h_ceil[:, None] + w_floor[None, :]).flatten()
        idx11 = (base_h_ceil[:, None] + w_ceil[None, :]).flatten()
        w00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten()
        w01 = ((1 - dh)[:, None] * dw[None, :]).flatten()
        w10 = (dh[:, None] * (1 - dw)[None, :]).flatten()
        w11 = (dh[:, None] * dw[None, :]).flatten()

        pos_embeds = (
            self.pos_embed(idx00) * w00[:, None]
            + self.pos_embed(idx01) * w01[:, None]
            + self.pos_embed(idx10) * w10[:, None]
            + self.pos_embed(idx11) * w11[:, None]
        )

        merge_size = self.config.spatial_merge_size
        pos_embeds = pos_embeds.reshape(grid_h, grid_w, -1)
        if grid_t > 1:
            pos_embeds = jnp.tile(pos_embeds[None], (grid_t, 1, 1, 1))
        else:
            pos_embeds = pos_embeds[None]
        merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
        pos_embeds = pos_embeds.reshape(grid_t, merged_h, merge_size, merged_w, merge_size, -1)
        pos_embeds = pos_embeds.transpose(0, 1, 3, 2, 4, 5)
        pos_embeds = pos_embeds.reshape(-1, pos_embeds.shape[-1])
        return pos_embeds

    def _rot_pos_emb(self, grid_thw: jax.Array = None, *, grid_h: int = None, grid_w: int = None, grid_t: int = None) -> Tuple[jax.Array, jax.Array]:
        merge_size = self.config.spatial_merge_size
        if grid_h is None:
            grid_h, grid_w = int(grid_thw[0, 1]), int(grid_thw[0, 2])
            grid_t = int(grid_thw[0, 0])
        if grid_t is None:
            grid_t = 1
        merged_h, merged_w = grid_h // merge_size, grid_w // merge_size

        block_rows = jnp.arange(merged_h)
        block_cols = jnp.arange(merged_w)
        intra_row = jnp.arange(merge_size)
        intra_col = jnp.arange(merge_size)
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
        row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
        col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

        if grid_t > 1:
            row_idx = jnp.tile(row_idx, grid_t)
            col_idx = jnp.tile(col_idx, grid_t)

        max_hw = max(grid_h, grid_w)
        head_dim = self.config.head_dim
        rotary_dim = head_dim // 2
        inv_freq = 1.0 / (self.config.rope_theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
        seq_positions = jnp.arange(max_hw, dtype=jnp.float32)
        freq_table = jnp.outer(seq_positions, inv_freq)
        row_emb = freq_table[row_idx]
        col_emb = freq_table[col_idx]
        emb = jnp.concatenate([row_emb, col_emb], axis=-1)
        emb = jnp.concatenate([emb, emb], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)

    def __call__(self, hidden_states: jax.Array, grid_thw: jax.Array) -> Tuple[jax.Array, list]:
        hidden_states = self.patch_embed(hidden_states)
        seq_len = hidden_states.shape[0]
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds[:seq_len]
        cos, sin = self._rot_pos_emb(grid_thw)
        position_embeddings = (cos[:seq_len], sin[:seq_len])

        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, position_embeddings)
            if layer_idx in self.deepstack_visual_indexes:
                ds_idx = list(self.deepstack_visual_indexes).index(layer_idx)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](hidden_states))
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_features

    def forward_static(self, hidden_states: jax.Array, *, grid_h: int, grid_w: int, grid_t: int = 1) -> jax.Array:
        """pmap/vmap-safe forward with pre-resolved grid dimensions (no int() tracing).

        Returns merged hidden states only (no deepstack features).
        """
        hidden_states = self.patch_embed(hidden_states)
        seq_len = hidden_states.shape[0]
        pos_embeds = self._fast_pos_embed_interpolate(grid_h=grid_h, grid_w=grid_w, grid_t=grid_t)
        hidden_states = hidden_states + pos_embeds[:seq_len]
        cos, sin = self._rot_pos_emb(grid_h=grid_h, grid_w=grid_w, grid_t=grid_t)
        position_embeddings = (cos[:seq_len], sin[:seq_len])

        for block in self.blocks:
            hidden_states = block(hidden_states, position_embeddings)
        hidden_states = self.merger(hidden_states)
        return hidden_states

    def forward_static_with_deepstack(
        self, hidden_states: jax.Array, *, grid_h: int, grid_w: int, grid_t: int = 1
    ) -> Tuple[jax.Array, list]:
        """Static-grid vision forward returning pooler and DeepStack features."""
        hidden_states = self.patch_embed(hidden_states)
        seq_len = hidden_states.shape[0]
        pos_embeds = self._fast_pos_embed_interpolate(grid_h=grid_h, grid_w=grid_w, grid_t=grid_t)
        hidden_states = hidden_states + pos_embeds[:seq_len]
        cos, sin = self._rot_pos_emb(grid_h=grid_h, grid_w=grid_w, grid_t=grid_t)
        position_embeddings = (cos[:seq_len], sin[:seq_len])

        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, position_embeddings)
            if layer_idx in self.deepstack_visual_indexes:
                ds_idx = list(self.deepstack_visual_indexes).index(layer_idx)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](hidden_states))
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_features


# --- Text Model --- #


class LayerCache(nnx.Module):
    def __init__(self, config: Qwen3VLTextConfig, batch_size: int, cache_size: int, dtype: jnp.dtype = jnp.bfloat16):
        cache_shape = (batch_size, cache_size, config.num_key_value_heads, config.head_dim)
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.size = cache_size
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


Cache: TypeAlias = list[LayerCache]


def init_cache(
    config: ModelConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
) -> Cache:
    cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))
    return [
        LayerCache(config.text_config, batch_size, cache_size, dtype)
        for _ in range(config.text_config.num_hidden_layers)
    ]


def _generate_rope(positions: jax.Array, head_dim: int, rope_theta: float) -> Tuple[jax.Array, jax.Array]:
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    sinusoid_inp = jnp.einsum(
        "bt,k->btk", positions.astype(jnp.float32), 1.0 / timescale, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def _generate_interleaved_mrope(
    position_ids_3d: jax.Array, head_dim: int, rope_theta: float, mrope_section: tuple
) -> Tuple[jax.Array, jax.Array]:
    """Qwen3-VL interleaved mRoPE.

    position_ids_3d: [3, B, T] for temporal/text, height, width.
    Returns sin/cos in the same half-dim format as _generate_rope.
    """
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.einsum(
        "dbt,k->dbtk", position_ids_3d.astype(jnp.float32), inv_freq, precision=jax.lax.Precision.HIGHEST
    )
    freqs_out = freqs[0]
    for dim_idx, offset in enumerate((1, 2), start=1):
        length = min(mrope_section[dim_idx] * 3, half_dim)
        indices = jnp.arange(offset, length, 3)
        freqs_out = freqs_out.at[:, :, indices].set(jnp.take(freqs[dim_idx], indices, axis=-1))
    return jnp.sin(freqs_out), jnp.cos(freqs_out)


def _apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    cos_full = jnp.concatenate([cos, cos], axis=-1)
    sin_full = jnp.concatenate([sin, sin], axis=-1)
    return apply_rotary_pos_emb(x, cos_full, sin_full)


def repeat_kv(hidden_states: jax.Array, n_rep: int) -> jax.Array:
    if n_rep == 1:
        return hidden_states
    b, t, kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = jnp.tile(hidden_states, (1, 1, 1, n_rep, 1))
    return hidden_states.reshape(b, t, kv_heads * n_rep, head_dim)


class Qwen3VLAttention(nnx.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nnx.Linear(
            config.hidden_size, self.num_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim, config.hidden_size, use_bias=config.attention_bias, rngs=rngs
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, rngs=rngs)

    def __call__(
        self, x: jax.Array, cache: Optional[LayerCache], sin: jax.Array, cos: jax.Array, mask: Optional[jax.Array]
    ) -> jax.Array:
        batch, seq_len, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = _apply_rope(q, sin, cos)
        k = _apply_rope(k, sin, cos)

        if cache is not None:
            cache_dtype = cache.k_cache[...].dtype
            slice_indices = (0, cache.cur_ind[...], 0, 0)
            cache.k_cache[...] = jax.lax.dynamic_update_slice(cache.k_cache[...], k.astype(cache_dtype), slice_indices)
            cache.v_cache[...] = jax.lax.dynamic_update_slice(cache.v_cache[...], v.astype(cache_dtype), slice_indices)
            k_full = repeat_kv(cache.k_cache[...], self.n_rep)
            v_full = repeat_kv(cache.v_cache[...], self.n_rep)
        else:
            k_full = repeat_kv(k, self.n_rep)
            v_full = repeat_kv(v, self.n_rep)

        if _USE_SPLASH_ATTENTION and cache is None and mask is not None:
            from jax.experimental.pallas.ops.tpu import splash_attention as splash

            splash_dtype = v_full.dtype
            q_s = q.astype(splash_dtype).transpose(0, 2, 1, 3)
            k_s = k_full.astype(splash_dtype).transpose(0, 2, 1, 3)
            v_s = v_full.astype(splash_dtype).transpose(0, 2, 1, 3)
            if mask.shape[1] == 1:
                splash_mask = jnp.broadcast_to(mask, (batch, self.num_heads, seq_len, k_full.shape[1]))
            else:
                splash_mask = mask

            block_sizes = splash.BlockSizes(
                block_q=_SPLASH_BLOCK_Q,
                block_kv=_SPLASH_BLOCK_KV,
                block_kv_compute=_SPLASH_BLOCK_KV,
                block_q_dkv=_SPLASH_BLOCK_Q,
                block_kv_dkv=_SPLASH_BLOCK_KV,
                block_kv_dkv_compute=_SPLASH_BLOCK_KV,
                block_q_dq=_SPLASH_BLOCK_Q,
                block_kv_dq=_SPLASH_BLOCK_KV,
            )

            def _one_splash(q_one, k_one, v_one, mask_one):
                splash_fn = splash.make_splash_mha_single_device(mask_one, block_sizes=block_sizes)
                return splash_fn(q_one, k_one, v_one)

            attn_out = jax.vmap(_one_splash)(q_s, k_s, v_s, splash_mask)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        elif _USE_DPA_ATTENTION and cache is None:
            dpa_dtype = v_full.dtype
            attn_out = jax.nn.dot_product_attention(
                q.astype(dpa_dtype),
                k_full.astype(dpa_dtype),
                v_full.astype(dpa_dtype),
                mask=mask,
                scale=self.scale,
                implementation="xla",
            ).reshape(batch, seq_len, -1)
        else:
            q = q.transpose(0, 2, 1, 3)  # (B, heads, T, dim)
            k_full = k_full.transpose(0, 2, 1, 3)
            v_full = v_full.transpose(0, 2, 1, 3)

            attn_weights = jnp.matmul(q, k_full.transpose(0, 1, 3, 2)) * self.scale
            if mask is not None:
                attn_weights = jnp.where(mask, attn_weights, _K_MASK)
            attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q.dtype)
            attn_out = jnp.matmul(attn_weights, v_full)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

        if cache is not None:
            cache.cur_ind[...] = cache.cur_ind[...] + seq_len
        return self.o_proj(attn_out)


class Qwen3VLMLP(nnx.Module):
    def __init__(self, config: Qwen3VLTextConfig, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLDecoderLayer(nnx.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.self_attn = Qwen3VLAttention(config, layer_idx, rngs=rngs)
        self.mlp = Qwen3VLMLP(config, rngs=rngs)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, rngs=rngs)

    def __call__(
        self, x: jax.Array, cache: Optional[LayerCache], sin: jax.Array, cos: jax.Array, mask: Optional[jax.Array]
    ) -> jax.Array:
        x = x + self.self_attn(self.input_layernorm(x), cache, sin, cos, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3VLTextModel(nnx.Module):
    def __init__(self, config: Qwen3VLTextConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embed_tokens = nnx.Embed(num_embeddings=config.vocab_size, features=config.hidden_size, rngs=rngs)
        self.layers = [Qwen3VLDecoderLayer(config, i, rngs=rngs) for i in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, rngs=rngs)

    def __call__(
        self,
        inputs_embeds: jax.Array,
        cache: Optional[Cache],
        sin: jax.Array,
        cos: jax.Array,
        mask: Optional[jax.Array],
        visual_pos_masks: Optional[jax.Array] = None,
        deepstack_visual_embeds: Optional[list[jax.Array]] = None,
    ) -> jax.Array:
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            if cache is None:
                # Training: gradient checkpointing to save activation memory
                def _ckpt_fn(h, fn=layer):
                    return fn(h, None, sin, cos, mask)

                hidden_states = jax.checkpoint(_ckpt_fn)(hidden_states)
            else:
                hidden_states = layer(hidden_states, layer_cache, sin, cos, mask)
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                hidden_states = batched_add_visual_embeds(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[i],
                )
        return self.norm(hidden_states)


# --- Merge Vision + Text --- #


def merge_modalities(img_emb: jax.Array, text_emb: jax.Array, token_mask: jax.Array) -> jax.Array:
    img_indices = jnp.cumsum(token_mask) - 1
    safe_indices = jnp.clip(img_indices, 0, img_emb.shape[0] - 1)
    aligned_images = img_emb[safe_indices]
    return jnp.where(token_mask[:, None], aligned_images, text_emb)


def batched_merge_modalities(img_emb: jax.Array, text_emb: jax.Array, token_mask: jax.Array) -> jax.Array:
    return jax.vmap(merge_modalities)(img_emb, text_emb, token_mask)


def add_visual_embeds(hidden_states: jax.Array, visual_embeds: jax.Array, token_mask: jax.Array) -> jax.Array:
    if visual_embeds.shape[0] == 0:
        return hidden_states
    img_indices = jnp.cumsum(token_mask) - 1
    safe_indices = jnp.clip(img_indices, 0, visual_embeds.shape[0] - 1)
    aligned_images = visual_embeds[safe_indices]
    return jnp.where(token_mask[:, None], hidden_states + aligned_images, hidden_states)


def batched_add_visual_embeds(hidden_states: jax.Array, visual_pos_masks: jax.Array, visual_embeds: jax.Array) -> jax.Array:
    return jax.vmap(add_visual_embeds)(hidden_states, visual_embeds, visual_pos_masks)


def make_train_causal_mask(seq_len: int) -> jax.Array:
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def make_causal_mask(cache: LayerCache, seq_len: int) -> jax.Array:
    cache_size = cache.size
    cur_pos = cache.cur_ind[...]
    seq_arange = jnp.arange(seq_len)
    cache_arange = jnp.arange(cache_size)
    mask = (seq_arange[:, None] + cur_pos) >= cache_arange[None, :]
    return mask[None, None, :, :]


# --- Top-level Model --- #


class Qwen3VLForConditionalGeneration(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.model = Qwen3VLModel(config, rngs=rngs)
        if config.text_config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nnx.Linear(
                config.text_config.hidden_size, config.text_config.vocab_size, use_bias=False, rngs=rngs
            )

    def __call__(
        self,
        input_ids: jax.Array,
        pixel_values: Optional[jax.Array] = None,
        image_grid_thw: Optional[jax.Array] = None,
        *,
        cache: Optional[Cache] = None,
        token_type_ids: Optional[jax.Array] = None,
    ) -> jax.Array:
        batch, seq_len = input_ids.shape
        if cache is not None:
            positions = jnp.arange(seq_len)[None, :] + cache[0].cur_ind[...]
        else:
            positions = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch, seq_len))
        sin, cos = _generate_rope(positions, self.config.text_config.head_dim, self.config.text_config.rope_theta)

        if cache is not None:
            mask = make_causal_mask(cache[0], seq_len)
        else:
            mask = make_train_causal_mask(seq_len)

        inputs_embeds = self.model.language_model.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None and token_type_ids is not None:
            vision_embeds, _ = self.model.visual(pixel_values, image_grid_thw)
            vision_embeds_batched = jnp.broadcast_to(
                vision_embeds[None], (batch, vision_embeds.shape[0], vision_embeds.shape[1])
            )
            inputs_embeds = batched_merge_modalities(vision_embeds_batched, inputs_embeds, token_type_ids)

        hidden_states = self.model.language_model(inputs_embeds, cache, sin, cos, mask)

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = hidden_states @ self.model.language_model.embed_tokens.embedding[...].T
        return logits

    def get_hidden_states(
        self,
        input_ids: jax.Array,
        pixel_values: Optional[jax.Array] = None,
        image_grid_thw: Optional[jax.Array] = None,
        token_type_ids: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Extract final hidden states (before LM head) for VLA obs embedding.

        Returns (B, seq_len, 2048).
        """
        batch, seq_len = input_ids.shape
        positions = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch, seq_len))
        sin, cos = _generate_rope(positions, self.config.text_config.head_dim, self.config.text_config.rope_theta)
        mask = make_train_causal_mask(seq_len)

        inputs_embeds = self.model.language_model.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None and token_type_ids is not None:
            vision_embeds, _ = self.model.visual(pixel_values, image_grid_thw)
            vision_embeds_batched = jnp.broadcast_to(
                vision_embeds[None], (batch, vision_embeds.shape[0], vision_embeds.shape[1])
            )
            inputs_embeds = batched_merge_modalities(vision_embeds_batched, inputs_embeds, token_type_ids)

        return self.model.language_model(inputs_embeds, None, sin, cos, mask)

    @classmethod
    def from_pretrained(cls, model_path: str, config: ModelConfig | None = None):
        """Load from local directory or HuggingFace model ID."""
        import os

        from qwen.qwen3vl import params

        if config is None:
            config = ModelConfig.qwen3vl_2b()
        if os.path.isdir(model_path):
            return params.create_model_from_safe_tensors(model_path, config)
        else:
            from huggingface_hub import snapshot_download

            ckpt_path = snapshot_download(repo_id=model_path, allow_patterns="*.safetensors")
            return params.create_model_from_safe_tensors(ckpt_path, config)


class Qwen3VLModel(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config, rngs=rngs)
        self.language_model = Qwen3VLTextModel(config.text_config, rngs=rngs)


@nnx.jit
def forward(model: Qwen3VLForConditionalGeneration, cache: Cache, input_ids: jax.Array) -> Tuple[jax.Array, Cache]:
    logits = model(input_ids, cache=cache)
    return logits[:, -1, :], cache


def forward_vision(
    model: Qwen3VLForConditionalGeneration,
    cache: Cache,
    input_ids: jax.Array,
    pixel_values: jax.Array,
    image_grid_thw: jax.Array,
    token_type_ids: jax.Array,
) -> Tuple[jax.Array, Cache]:
    logits = model(input_ids, pixel_values, image_grid_thw, cache=cache, token_type_ids=token_type_ids)
    return logits[:, -1, :], cache


def forward_train(model: Qwen3VLForConditionalGeneration, input_ids: jax.Array, labels: jax.Array) -> jax.Array:
    """Cache-free forward for training. Returns per-token cross-entropy loss."""
    logits = model(input_ids)  # (B, T, V)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
    mask = (shift_labels != -100).astype(jnp.float32)
    return (loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
