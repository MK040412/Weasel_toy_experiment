"""Transformer building blocks for GemmaActionExpert in Flax NNX."""

import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        x_f32 = x.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        return ((x_f32 * rms) * self.weight[...]).astype(x.dtype)


def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
    """Repeat KV heads to match query heads. (B, T, n_kv_heads, D) -> (B, T, n_heads, D)."""
    if n_rep == 1:
        return x
    b, t, kv_heads, d = x.shape
    x = x[:, :, :, None, :]
    x = jnp.tile(x, (1, 1, 1, n_rep, 1))
    return x.reshape(b, t, kv_heads * n_rep, d)


class GQAttention(nnx.Module):
    """Grouped-Query Attention with optional prefix KV cache."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, *, rngs: nnx.Rngs):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.scale = head_dim**-0.5

        self.q_proj = nnx.Linear(d_model, n_heads * head_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, n_kv_heads * head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, n_kv_heads * head_dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(n_heads * head_dim, d_model, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        prefix_kv: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, t, self.n_kv_heads, self.head_dim)

        if prefix_kv is not None:
            pk, pv = prefix_kv
            k = jnp.concatenate([pk, k], axis=1)
            v = jnp.concatenate([pv, v], axis=1)

        k_full = repeat_kv(k, self.n_rep)
        v_full = repeat_kv(v, self.n_rep)

        q = q.transpose(0, 2, 1, 3)  # (B, heads, T, D)
        k_full = k_full.transpose(0, 2, 1, 3)
        v_full = v_full.transpose(0, 2, 1, 3)

        attn_weights = jnp.matmul(q, k_full.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(jnp.bfloat16).min)
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q.dtype)
        out = jnp.matmul(attn_weights, v_full)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)

        return self.o_proj(out), (k, v)


class GatedMLP(nnx.Module):
    """SwiGLU-style gated MLP."""

    def __init__(self, d_model: int, d_ff: int, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(d_ff, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))


class GemmaDecoderLayer(nnx.Module):
    """Pre-norm decoder layer: RMSNorm -> GQA -> residual -> RMSNorm -> MLP -> residual."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_kv_heads: int, head_dim: int, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(d_model, rngs=rngs)
        self.self_attn = GQAttention(d_model, n_heads, n_kv_heads, head_dim, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(d_model, rngs=rngs)
        self.mlp = GatedMLP(d_model, d_ff, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        prefix_kv: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        residual = x
        x = self.input_layernorm(x)
        attn_out, kv = self.self_attn(x, mask=mask, prefix_kv=prefix_kv)
        x = residual + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, kv
