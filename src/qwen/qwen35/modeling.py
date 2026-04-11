"""Qwen3.5-0.8B VLM JAX implementation.

Combines:
- Vision encoder: adapted from bonsai (Qwen3-VL pattern, scaled down)
- Gated DeltaNet: extracted from MaxText
- Full Attention: adapted from bonsai
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, TypeAlias

import jax
import jax.numpy as jnp
from flax import nnx

from qwen.qwen35.gated_delta_net import GatedDeltaNetLayer, GDNCache

_K_MASK = jnp.finfo(jnp.bfloat16).min


# --- Configuration --- #


@dataclass(frozen=True)
class VisionConfig:
    depth: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_heads: int = 12
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 1024
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple = ()
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    # Full attention params
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    # Linear attention (Gated DeltaNet) params
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # Layer pattern: 3 linear + 1 full, repeated
    full_attention_interval: int = 4
    layer_types: tuple = (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
    # RoPE
    rope_theta: float = 10_000_000
    partial_rotary_factor: float = 0.25
    mrope_section: tuple = (11, 11, 10)
    # Other
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    attn_output_gate: bool = True


@dataclass(frozen=True)
class ModelConfig:
    vision_config: VisionConfig = VisionConfig()
    text_config: TextConfig = TextConfig()
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    @classmethod
    def qwen35_0_8b(cls):
        return cls()


# --- Shared Components --- #


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        x_f32 = x.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        return ((x_f32 * rms) * self.weight[...]).astype(x.dtype)


# --- Vision Encoder (from bonsai, scaled down) --- #


class VisionPatchEmbed(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
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

    def __call__(self, x: jax.Array) -> jax.Array:
        cfg = self.config
        seq_len = x.shape[0]
        x = x.reshape(seq_len, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        x = x.transpose(0, 2, 3, 4, 1)
        return self.proj(x).reshape(seq_len, cfg.hidden_size)


class VisionMLP(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=True, rngs=rngs)
        self.fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(nnx.gelu(self.fc1(x), approximate=True))


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class VisionAttention(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nnx.Linear(config.hidden_size, 3 * config.hidden_size, use_bias=True, rngs=rngs)
        self.proj = nnx.Linear(config.hidden_size, config.hidden_size, use_bias=True, rngs=rngs)
        self.scale = self.head_dim**-0.5

    def __call__(self, x: jax.Array, position_embeddings: Tuple) -> jax.Array:
        seq_len = x.shape[0]
        cos, sin = position_embeddings
        cos, sin = cos[:, None, :], sin[:, None, :]
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
        attn = jax.nn.softmax((jnp.matmul(q, k.transpose(0, 2, 1)) * self.scale).astype(jnp.float32), axis=-1).astype(
            q.dtype
        )
        return self.proj(jnp.matmul(attn, v).transpose(1, 0, 2).reshape(seq_len, -1))


class VisionBlock(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.norm2 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.attn = VisionAttention(config, rngs=rngs)
        self.mlp = VisionMLP(config, rngs=rngs)

    def __call__(self, x: jax.Array, pos_emb: Tuple) -> jax.Array:
        x = x + self.attn(self.norm1(x), pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionPatchMerger(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        merge_factor = config.spatial_merge_size**2
        hidden_merged = config.hidden_size * merge_factor
        self.norm = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.fc1 = nnx.Linear(hidden_merged, hidden_merged, use_bias=True, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_merged, config.out_hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.norm(x)
        merge_factor = self.config.spatial_merge_size**2
        x = x.reshape(x.shape[0] // merge_factor, -1)
        return self.fc2(nnx.gelu(self.fc1(x)))


class VisionEncoder(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.patch_embed = VisionPatchEmbed(config, rngs=rngs)
        self.pos_embed = nnx.Embed(
            num_embeddings=config.num_position_embeddings, features=config.hidden_size, rngs=rngs
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.blocks = [VisionBlock(config, rngs=rngs) for _ in range(config.depth)]
        self.merger = VisionPatchMerger(config, rngs=rngs)

    def _pos_embed_interpolate(self, grid_thw: jax.Array) -> jax.Array:
        grid_h, grid_w = int(grid_thw[0, 1]), int(grid_thw[0, 2])
        h_idxs = jnp.linspace(0, self.num_grid_per_side - 1, grid_h)
        w_idxs = jnp.linspace(0, self.num_grid_per_side - 1, grid_w)
        h_floor, w_floor = jnp.floor(h_idxs).astype(jnp.int32), jnp.floor(w_idxs).astype(jnp.int32)
        h_ceil = jnp.clip(h_floor + 1, 0, self.num_grid_per_side - 1)
        w_ceil = jnp.clip(w_floor + 1, 0, self.num_grid_per_side - 1)
        dh, dw = h_idxs - h_floor, w_idxs - w_floor
        base_h, base_hc = h_floor * self.num_grid_per_side, h_ceil * self.num_grid_per_side
        idx00 = (base_h[:, None] + w_floor[None, :]).flatten()
        idx01 = (base_h[:, None] + w_ceil[None, :]).flatten()
        idx10 = (base_hc[:, None] + w_floor[None, :]).flatten()
        idx11 = (base_hc[:, None] + w_ceil[None, :]).flatten()
        w00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten()
        w01 = ((1 - dh)[:, None] * dw[None, :]).flatten()
        w10 = (dh[:, None] * (1 - dw)[None, :]).flatten()
        w11 = (dh[:, None] * dw[None, :]).flatten()
        pe = (
            self.pos_embed(idx00) * w00[:, None]
            + self.pos_embed(idx01) * w01[:, None]
            + self.pos_embed(idx10) * w10[:, None]
            + self.pos_embed(idx11) * w11[:, None]
        )
        merge = self.config.spatial_merge_size
        grid_t = int(grid_thw[0, 0])
        pe = pe.reshape(grid_h, grid_w, -1)
        pe = jnp.tile(pe[None], (grid_t, 1, 1, 1)) if grid_t > 1 else pe[None]
        mh, mw = grid_h // merge, grid_w // merge
        pe = pe.reshape(grid_t, mh, merge, mw, merge, -1).transpose(0, 1, 3, 2, 4, 5).reshape(-1, pe.shape[-1])
        return pe

    def _rot_pos_emb(self, grid_thw: jax.Array) -> Tuple:
        merge = self.config.spatial_merge_size
        grid_h, grid_w, grid_t = int(grid_thw[0, 1]), int(grid_thw[0, 2]), int(grid_thw[0, 0])
        mh, mw = grid_h // merge, grid_w // merge
        br, bc = jnp.arange(mh), jnp.arange(mw)
        ir, ic = jnp.arange(merge), jnp.arange(merge)
        row_idx = jnp.broadcast_to(
            br[:, None, None, None] * merge + ir[None, None, :, None], (mh, mw, merge, merge)
        ).reshape(-1)
        col_idx = jnp.broadcast_to(
            bc[None, :, None, None] * merge + ic[None, None, None, :], (mh, mw, merge, merge)
        ).reshape(-1)
        if grid_t > 1:
            row_idx, col_idx = jnp.tile(row_idx, grid_t), jnp.tile(col_idx, grid_t)
        hd = self.config.head_dim
        rd = hd // 2
        inv_freq = 1.0 / (self.config.rope_theta ** (jnp.arange(0, rd, 2, dtype=jnp.float32) / rd))
        freq = jnp.outer(jnp.arange(max(grid_h, grid_w), dtype=jnp.float32), inv_freq)
        emb = jnp.concatenate([freq[row_idx], freq[col_idx]], axis=-1)
        emb = jnp.concatenate([emb, emb], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)

    def __call__(self, pixel_values: jax.Array, grid_thw: jax.Array) -> jax.Array:
        x = self.patch_embed(pixel_values)
        seq_len = x.shape[0]
        pe = self._pos_embed_interpolate(grid_thw)
        x = x + pe[:seq_len]
        cos, sin = self._rot_pos_emb(grid_thw)
        pos_emb = (cos[:seq_len], sin[:seq_len])
        for block in self.blocks:
            x = block(x, pos_emb)
        return self.merger(x)


# --- Text Model --- #


class LayerCache(nnx.Module):
    """KV-cache for full attention layers."""

    def __init__(self, config: TextConfig, batch_size: int, cache_size: int, dtype=jnp.bfloat16):
        cache_shape = (batch_size, cache_size, config.num_key_value_heads, config.head_dim)
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.size = cache_size
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


LayerCacheEntry: TypeAlias = LayerCache | GDNCache
Cache: TypeAlias = list[LayerCacheEntry]


def init_cache(config: ModelConfig, batch_size: int, token_len: int, gen_steps: int) -> Cache:
    cache_size = 2 ** math.ceil(math.log2(max(token_len + gen_steps, 1)))
    tc = config.text_config
    caches = []
    for lt in tc.layer_types:
        if lt == "full_attention":
            caches.append(LayerCache(tc, batch_size, cache_size))
        else:
            key_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
            value_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
            conv_dim = key_dim * 2 + value_dim
            caches.append(
                GDNCache(
                    batch_size=batch_size,
                    num_v_heads=tc.linear_num_value_heads,
                    k_head_dim=tc.linear_key_head_dim,
                    v_head_dim=tc.linear_value_head_dim,
                    conv_kernel_dim=tc.linear_conv_kernel_dim,
                    conv_dim=conv_dim,
                )
            )
    return caches


def _generate_interleaved_mrope(
    position_ids_3d: jax.Array, head_dim: int, rope_theta: float, partial_factor: float, mrope_section: tuple
) -> Tuple:
    """Interleaved multi-dimensional RoPE for Qwen3.5.

    Args:
        position_ids_3d: (3, B, T) — positions for T, H, W dimensions
        head_dim: full head dimension (e.g. 256)
        rope_theta: RoPE theta (e.g. 10_000_000)
        partial_factor: fraction of head_dim to rotate (e.g. 0.25)
        mrope_section: frequency section sizes (e.g. (11, 11, 10)), sum = rotary_dim//2

    Returns:
        cos, sin: (B, T, rotary_dim) — ready for rotate_half pattern
    """
    rotary_dim = int(head_dim * partial_factor)  # 64
    rotary_dim // 2  # 32 = sum(mrope_section)

    # inv_freq: (half_dim,)
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))

    # Compute freqs for each of 3 dimensions: (3, B, T, half_dim)
    # position_ids_3d: (3, B, T), inv_freq: (half_dim,)
    freqs = jnp.einsum(
        "dbt,k->dbtk", position_ids_3d.astype(jnp.float32), inv_freq, precision=jax.lax.Precision.HIGHEST
    )
    # freqs shape: (3, B, T, half_dim) = (3, B, T, 32)

    # Apply interleaved mRoPE: merge T/H/W frequencies
    # Start with freqs_t = freqs[0] (temporal/text): (B, T, 32)
    freqs_out = freqs[0]

    # Interleave H and W into T's frequency slots
    # mrope_section = (11, 11, 10) for T, H, W
    for dim_idx, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim_idx] * 3
        indices = jnp.arange(offset, length, 3)
        # Use explicit gather/scatter for proper indexing
        src = freqs[dim_idx]  # (B, T, 32)
        freqs_out = freqs_out.at[:, :, indices].set(src[:, :, indices])

    # Double for rotate_half pattern: (B, T, 32) -> (B, T, 64)
    emb = jnp.concatenate([freqs_out, freqs_out], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def _apply_rope_partial(x: jax.Array, cos: jax.Array, sin: jax.Array, partial_factor: float) -> jax.Array:
    """Apply partial RoPE — only rotate first `rotary_dim` of head_dim.

    cos, sin: (B, T, rotary_dim) from interleaved mRoPE
    x: (B, T, heads, head_dim)
    """
    head_dim = x.shape[-1]
    rotary_dim = int(head_dim * partial_factor)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    # Expand cos/sin for heads: (B, T, rotary_dim) -> (B, T, 1, rotary_dim)
    cos_exp = cos[:, :, None, :]
    sin_exp = sin[:, :, None, :]
    x_rot = apply_rotary_pos_emb(x_rot, cos_exp, sin_exp)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
    if n_rep == 1:
        return x
    b, t, kv_h, hd = x.shape
    return jnp.tile(x[:, :, :, None, :], (1, 1, 1, n_rep, 1)).reshape(b, t, kv_h * n_rep, hd)


class FullAttentionLayer(nnx.Module):
    """Standard GQA with Q/K norm + output gate fused in q_proj.

    q_proj outputs 2× (Q + gate fused): [hidden_size -> num_heads * head_dim * 2]
    The first half is Q, the second half is the output gate.
    """

    def __init__(self, config: TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        self.attn_output_gate = config.attn_output_gate

        # q_proj includes gate when attn_output_gate=True: output is 2× Q dim
        q_out_dim = self.num_heads * self.head_dim * (2 if config.attn_output_gate else 1)
        self.q_proj = nnx.Linear(config.hidden_size, q_out_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(self.num_heads * self.head_dim, config.hidden_size, use_bias=False, rngs=rngs)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, rngs=rngs)

    def __call__(
        self, x: jax.Array, cache: LayerCache, cos: jax.Array, sin: jax.Array, mask: Optional[jax.Array]
    ) -> jax.Array:
        B, T, _ = x.shape
        q_out = self.q_proj(x)  # (B, T, num_heads * head_dim * 2)

        if self.attn_output_gate:
            # HF splits per-head: view(B, T, num_heads, head_dim*2) then chunk(2, dim=-1)
            # This means Q and gate are interleaved per head in the weight matrix
            q_and_gate = q_out.reshape(B, T, self.num_heads, self.head_dim * 2)
            q_raw = q_and_gate[..., : self.head_dim]  # (B, T, num_heads, head_dim)
            gate_per_head = q_and_gate[..., self.head_dim :]  # (B, T, num_heads, head_dim)
            gate = gate_per_head.reshape(B, T, -1)  # (B, T, num_heads * head_dim)
        else:
            q_raw = q_out.reshape(B, T, self.num_heads, self.head_dim)
            gate = None

        q = self.q_norm(q_raw)
        k = self.k_norm(self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        q = _apply_rope_partial(q, cos, sin, self.config.partial_rotary_factor)
        k = _apply_rope_partial(k, cos, sin, self.config.partial_rotary_factor)

        slice_idx = (0, cache.cur_ind[...], 0, 0)
        cache.k_cache[...] = jax.lax.dynamic_update_slice(
            cache.k_cache[...], k.astype(cache.k_cache[...].dtype), slice_idx
        )
        cache.v_cache[...] = jax.lax.dynamic_update_slice(
            cache.v_cache[...], v.astype(cache.v_cache[...].dtype), slice_idx
        )

        k_full = repeat_kv(cache.k_cache[...], self.n_rep)
        v_full = repeat_kv(cache.v_cache[...], self.n_rep)

        q, k_full, v_full = q.transpose(0, 2, 1, 3), k_full.transpose(0, 2, 1, 3), v_full.transpose(0, 2, 1, 3)
        attn = jnp.matmul(q, k_full.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = jnp.where(mask, attn, _K_MASK)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)
        out = jnp.matmul(attn, v_full).transpose(0, 2, 1, 3).reshape(B, T, -1)

        if gate is not None:
            out = out * jax.nn.sigmoid(gate)

        cache.cur_ind[...] = cache.cur_ind[...] + T
        return self.o_proj(out)


class TextMLP(nnx.Module):
    def __init__(self, config: TextConfig, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nnx.Module):
    """Hybrid decoder layer: either GDN or Full Attention."""

    def __init__(self, config: TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, rngs=rngs)
        self.mlp = TextMLP(config, rngs=rngs)

        if self.layer_type == "full_attention":
            self.self_attn = FullAttentionLayer(config, layer_idx, rngs=rngs)
            self.gdn = None
        else:
            self.self_attn = None
            self.gdn = GatedDeltaNetLayer(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                conv_kernel_dim=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                rngs=rngs,
            )

    def __call__(
        self, x: jax.Array, cache: LayerCacheEntry, cos: jax.Array, sin: jax.Array, mask: Optional[jax.Array]
    ) -> jax.Array:
        normed = self.input_layernorm(x)
        if self.layer_type == "full_attention":
            x = x + self.self_attn(normed, cache, cos, sin, mask)
        else:
            x = x + self.gdn(normed, cache=cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TextModel(nnx.Module):
    def __init__(self, config: TextConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embed_tokens = nnx.Embed(num_embeddings=config.vocab_size, features=config.hidden_size, rngs=rngs)
        self.layers = [DecoderLayer(config, i, rngs=rngs) for i in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, rngs=rngs)

    def __call__(
        self, embeds: jax.Array, cache: Cache, cos: jax.Array, sin: jax.Array, mask: Optional[jax.Array]
    ) -> jax.Array:
        x = embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i], cos, sin, mask)
        return self.norm(x)


# --- Merge Vision + Text --- #


def merge_modalities(img_emb, text_emb, token_mask):
    idx = jnp.clip(jnp.cumsum(token_mask) - 1, 0, img_emb.shape[0] - 1)
    return jnp.where(token_mask[:, None], img_emb[idx], text_emb)


def make_causal_mask(cache: LayerCache, seq_len: int):
    cur = cache.cur_ind[...]
    return (jnp.arange(seq_len)[:, None] + cur) >= jnp.arange(cache.size)[None, :]


# --- Top-level Model --- #


class Qwen35ForConditionalGeneration(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.visual = VisionEncoder(config.vision_config, rngs=rngs)
        self.language_model = TextModel(config.text_config, rngs=rngs)
        if config.text_config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nnx.Linear(
                config.text_config.hidden_size, config.text_config.vocab_size, use_bias=False, rngs=rngs
            )

    def __call__(
        self, input_ids: jax.Array, pixel_values=None, image_grid_thw=None, *, cache: Cache, token_type_ids=None
    ) -> jax.Array:
        B, T = input_ids.shape
        tc = self.config.text_config

        # Find a full_attention cache for position tracking
        fa_cache = next(c for c in cache if isinstance(c, LayerCache))
        positions = jnp.arange(T)[None, :] + fa_cache.cur_ind[...]
        positions = jnp.broadcast_to(positions, (B, T))
        # Expand to 3D for interleaved mRoPE: (3, B, T) — text-only: all dims same position
        positions_3d = jnp.stack([positions, positions, positions], axis=0)
        cos, sin = _generate_interleaved_mrope(
            positions_3d, tc.head_dim, tc.rope_theta, tc.partial_rotary_factor, tc.mrope_section
        )

        mask = make_causal_mask(fa_cache, T)[None, None, :, :]

        embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None and token_type_ids is not None:
            vis_emb = self.visual(pixel_values, image_grid_thw)
            vis_batched = jnp.broadcast_to(vis_emb[None], (B, vis_emb.shape[0], vis_emb.shape[1]))
            embeds = jax.vmap(merge_modalities)(vis_batched, embeds, token_type_ids)

        hidden = self.language_model(embeds, cache, cos, sin, mask)

        if self.lm_head is not None:
            return self.lm_head(hidden)
        return hidden @ self.language_model.embed_tokens.embedding[...].T

    @classmethod
    def from_pretrained(cls, model_path: str, config: ModelConfig | None = None):
        import os

        from qwen.qwen35 import params

        if config is None:
            config = ModelConfig.qwen35_0_8b()
        if os.path.isdir(model_path):
            return params.create_model_from_safe_tensors(model_path, config)
        else:
            from huggingface_hub import snapshot_download

            ckpt = snapshot_download(repo_id=model_path, allow_patterns="*.safetensors")
            return params.create_model_from_safe_tensors(ckpt, config)


@nnx.jit
def forward(model, cache, input_ids):
    logits = model(input_ids, cache=cache)
    return logits[:, -1, :], cache


def forward_vision(model, cache, input_ids, pixel_values, image_grid_thw, token_type_ids):
    logits = model(input_ids, pixel_values, image_grid_thw, cache=cache, token_type_ids=token_type_ids)
    return logits[:, -1, :], cache
