"""Gated DeltaNet (Linear Attention) for Qwen3.5.

Extracted and simplified from MaxText (AI-Hypercomputer/maxtext).
Adapted for standalone use with JAX 0.6.2 + Flax NNX.
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx
from typing import Tuple, Optional


def l2norm(x, dim=-1, eps=1e-6):
    norm = jnp.sqrt(jnp.sum(x * x, axis=dim, keepdims=True) + eps)
    return x / norm


def jax_chunk_gated_delta_rule(
    query: jax.Array,  # (B, T, H, K_dim)
    key: jax.Array,    # (B, T, H, K_dim)
    value: jax.Array,  # (B, T, H, V_dim)
    g: jax.Array,      # (B, T, H)
    beta: jax.Array,   # (B, T, H)
    chunk_size: int = 64,
    initial_state: Optional[jax.Array] = None,
    use_qk_norm: bool = True,
) -> Tuple[jax.Array, Optional[jax.Array]]:
    """Optimized JAX Gated Delta Rule (from MaxText)."""
    initial_dtype = query.dtype

    if use_qk_norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    g = g.astype(jnp.float32)
    # HF reference computes everything in float32 — bfloat16 causes precision loss in recurrence
    compute_dtype = jnp.float32
    query = query.astype(compute_dtype)
    key = key.astype(compute_dtype)
    value = value.astype(compute_dtype)
    beta = beta.astype(compute_dtype)

    scale = jax.lax.rsqrt(jnp.array(query.shape[-1], dtype=jnp.float32)).astype(compute_dtype)
    query = query * scale

    B, seq_len, H, K_dim = key.shape
    V_dim = value.shape[-1]

    pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
    if pad_len > 0:
        def pad_fn(x, val=0.0):
            return jnp.pad(x, ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2), constant_values=val)
        query = pad_fn(query)
        key = pad_fn(key)
        value = pad_fn(value)
        g = pad_fn(g)
        beta = pad_fn(beta)

    num_chunks = query.shape[1] // chunk_size

    def to_chunk(x):
        return x.reshape(B, num_chunks, chunk_size, H, -1).transpose(0, 1, 3, 2, 4)

    def to_chunk_scalar(x):
        return x.reshape(B, num_chunks, chunk_size, H).transpose(0, 1, 3, 2)

    q_c = to_chunk(query)
    k_c = to_chunk(key)
    v_c = to_chunk(value)
    g_c = to_chunk_scalar(g)
    beta_c = to_chunk_scalar(beta)

    # STAGE 2: Intra-chunk (parallel)
    g_cumsum = jnp.cumsum(g_c, axis=-1)
    k_beta = k_c * beta_c[..., None]

    S = jnp.matmul(k_beta, k_c.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST)
    S = S.astype(jnp.float32)

    g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
    g_diff = jnp.where(mask, g_diff, -1e30)

    S = S * jnp.exp(g_diff)
    S = jnp.where(mask, S, 0.0)

    identity = jnp.eye(chunk_size, dtype=jnp.float32)
    identity_broadcasted = jnp.broadcast_to(identity, S.shape)
    A = jax.scipy.linalg.solve_triangular(identity + S, identity_broadcasted, lower=True, unit_diagonal=True)

    v_beta = v_c * beta_c[..., None]
    u_chunks = jnp.matmul(A, v_beta.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST).astype(compute_dtype)

    k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
    w_chunks = jnp.matmul(A, k_beta_g, precision=jax.lax.Precision.HIGHEST).astype(compute_dtype)

    # STAGE 3: Inter-chunk scan
    scan_perm_vec = (1, 0, 2, 3, 4)
    scan_perm_scl = (1, 0, 2, 3)

    w_scan = w_chunks.transpose(scan_perm_vec)
    u_scan = u_chunks.transpose(scan_perm_vec)
    k_scan = k_c.transpose(scan_perm_vec)
    q_scan = q_c.transpose(scan_perm_vec)
    g_scan = g_cumsum.transpose(scan_perm_scl)

    if initial_state is None:
        h_init = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        h_init = initial_state.astype(jnp.float32)

    xs = (w_scan, u_scan, q_scan, k_scan, g_scan)

    def scan_body(h, args):
        w, u, q, k, g = args
        prec = jax.lax.Precision.HIGHEST

        q_g = q.astype(jnp.float32) * jnp.exp(g)[..., None]
        attn_inter = jnp.matmul(q_g, h, precision=prec)

        v_prime = jnp.matmul(w.astype(jnp.float32), h, precision=prec)
        v_new = u.astype(jnp.float32) - v_prime

        attn = jnp.matmul(q, k.swapaxes(-1, -2), precision=prec).astype(jnp.float32)

        g_diff = g[..., :, None] - g[..., None, :]
        mask_intra = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
        g_diff = jnp.where(mask_intra, g_diff, -1e30)

        attn_i = attn * jnp.exp(g_diff)
        attn_i = jnp.where(mask_intra, attn_i, 0.0)

        term2 = jnp.matmul(attn_i, v_new, precision=prec)
        o_c = attn_inter + term2

        g_i_last_exp = jnp.exp(g[..., -1, None, None])
        h_new = h * g_i_last_exp

        g_diff_exp_state = jnp.exp(g[..., -1, None] - g)[..., None]
        k_i_g_diff = k.astype(jnp.float32) * g_diff_exp_state

        update_term = jnp.matmul(k_i_g_diff.swapaxes(-1, -2), v_new, precision=prec)
        h_new = h_new + update_term

        return h_new, o_c

    final_h, o_chunks = lax.scan(scan_body, h_init, xs)

    # STAGE 4: Finalize
    o = o_chunks.transpose(1, 0, 3, 2, 4).reshape(B, -1, H, V_dim)
    if pad_len > 0:
        o = o[:, :seq_len, :, :]

    o = o.astype(initial_dtype)
    return o, final_h


class RMSNormGated(nnx.Module):
    """RMSNorm with SiLU gating.

    Weight is per-head (value_head_dim), broadcast across all heads.
    x: (B, T, value_dim) = (B, T, num_v_heads * v_head_dim)
    z: (B, T, value_dim)
    """

    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.eps = eps
        self.dim = dim
        self.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        # x: (B, T, value_dim) where value_dim = num_heads * dim
        # Reshape to (B, T, num_heads, dim) for per-head norm, then back
        orig_shape = x.shape
        x_f32 = x.astype(jnp.float32).reshape(*orig_shape[:-1], -1, self.dim)
        rms = jax.lax.rsqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        normed = (x_f32 * rms) * self.weight[...]
        normed = normed.reshape(orig_shape)
        return (normed * nnx.silu(z.astype(jnp.float32))).astype(x.dtype)


class GDNCache(nnx.Module):
    """Cache for GDN layer: recurrent state + conv state."""

    def __init__(self, batch_size: int, num_v_heads: int, k_head_dim: int, v_head_dim: int,
                 conv_kernel_dim: int, conv_dim: int, dtype=jnp.bfloat16):
        # Recurrent state: (B, H_v, K_dim, V_dim)
        self.recurrent_state = nnx.Cache(
            jnp.zeros((batch_size, num_v_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
        )
        # Conv state: last (kernel-1) inputs, (B, kernel-1, conv_dim)
        self.conv_state = nnx.Cache(
            jnp.zeros((batch_size, conv_kernel_dim - 1, conv_dim), dtype=dtype)
        )


class GatedDeltaNetLayer(nnx.Module):
    """Qwen3.5 Gated DeltaNet layer with stateful recurrent cache.

    HF weight structure (separate projections):
      in_proj_qkv: hidden -> key_dim*2 + value_dim  (q, k, v concatenated)
      in_proj_z:   hidden -> value_dim
      in_proj_a:   hidden -> num_v_heads
      in_proj_b:   hidden -> num_v_heads
      conv1d.weight: (conv_dim, 1, kernel_size) — PyTorch grouped conv format
      A_log, dt_bias: (num_v_heads,)
      norm.weight: (value_head_dim,)
      out_proj: value_dim -> hidden
    """

    def __init__(self, hidden_size: int, num_key_heads: int, num_value_heads: int,
                 key_head_dim: int, value_head_dim: int, conv_kernel_dim: int = 4,
                 use_qk_norm: bool = True, chunk_size: int = 64,
                 rms_norm_eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.use_qk_norm = use_qk_norm
        self.chunk_size = chunk_size

        key_dim = num_key_heads * key_head_dim
        value_dim = num_value_heads * value_head_dim
        conv_dim = key_dim * 2 + value_dim

        # Separate projections matching HF weight keys
        self.in_proj_qkv = nnx.Linear(hidden_size, key_dim * 2 + value_dim, use_bias=False, rngs=rngs)
        self.in_proj_z = nnx.Linear(hidden_size, value_dim, use_bias=False, rngs=rngs)
        self.in_proj_a = nnx.Linear(hidden_size, num_value_heads, use_bias=False, rngs=rngs)
        self.in_proj_b = nnx.Linear(hidden_size, num_value_heads, use_bias=False, rngs=rngs)

        # Conv1D weight: stored as (conv_dim, kernel_size) in our convention
        # HF stores as (conv_dim, 1, kernel_size) for grouped conv — no bias
        self.conv1d_weight = nnx.Param(
            jax.random.normal(rngs.params(), (conv_dim, conv_kernel_dim), dtype=jnp.float32) * 0.02
        )

        # Learnable parameters for gating
        self.A_log = nnx.Param(jnp.zeros((num_value_heads,), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.zeros((num_value_heads,), dtype=jnp.float32))

        # Output — norm weight is per-head (value_head_dim), broadcast across heads
        self.norm = RMSNormGated(value_head_dim, eps=rms_norm_eps, rngs=rngs)
        self.out_proj = nnx.Linear(value_dim, hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, cache: Optional['GDNCache'] = None) -> jax.Array:
        """x: (B, T, hidden_size) -> (B, T, hidden_size)"""
        B, T, _ = x.shape
        key_dim = self.num_key_heads * self.key_head_dim
        value_dim = self.num_value_heads * self.value_head_dim

        # Step A: Separate projections
        qkv = self.in_proj_qkv(x)
        q_raw, k_raw, v_raw = jnp.split(qkv, [key_dim, key_dim * 2], axis=-1)
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)  # (B, T, num_v_heads)
        a = self.in_proj_a(x)  # (B, T, num_v_heads)

        # Step B: Causal Conv1D (depthwise) with state management
        conv_input = jnp.concatenate([q_raw, k_raw, v_raw], axis=-1)  # (B, T, conv_dim)
        conv_w = self.conv1d_weight[...]  # (conv_dim, kernel_size)
        kernel_size = conv_w.shape[1]

        if cache is not None:
            # Prepend conv state from previous step
            prev_state = cache.conv_state[...]  # (B, kernel-1, conv_dim)
            padded = jnp.concatenate([prev_state, conv_input], axis=1)  # (B, T+kernel-1, conv_dim)
            # Update conv state: last (kernel-1) positions
            new_conv_state = padded[:, -(kernel_size - 1):, :]
            cache.conv_state[...] = new_conv_state.astype(cache.conv_state[...].dtype)
        else:
            padded = jnp.pad(conv_input, ((0, 0), (kernel_size - 1, 0), (0, 0)))

        compute_dtype = padded.dtype
        padded_t = padded.transpose(0, 2, 1)
        w = conv_w[:, None, :].astype(compute_dtype)
        conv_out = jax.lax.conv_general_dilated(
            padded_t, w, window_strides=(1,), padding='VALID',
            feature_group_count=conv_w.shape[0]
        )
        conv_out = conv_out.transpose(0, 2, 1)  # (B, T, conv_dim)
        conv_out = nnx.silu(conv_out)

        q, k, v = jnp.split(conv_out, [key_dim, key_dim * 2], axis=-1)

        # Reshape to heads
        q = q.reshape(B, T, self.num_key_heads, self.key_head_dim)
        k = k.reshape(B, T, self.num_key_heads, self.key_head_dim)
        v = v.reshape(B, T, self.num_value_heads, self.value_head_dim)

        # Step C: Gates
        beta = jax.nn.sigmoid(b)  # (B, T, num_v_heads)
        # HF: g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        g = -jnp.exp(self.A_log[...].astype(jnp.float32)) * jax.nn.softplus(
            a.astype(jnp.float32) + self.dt_bias[...].astype(jnp.float32))  # (B, T, num_v_heads)

        # Expand k for v_heads > k_heads
        if self.num_value_heads > self.num_key_heads:
            repeats = self.num_value_heads // self.num_key_heads
            q = jnp.repeat(q, repeats, axis=2)
            k = jnp.repeat(k, repeats, axis=2)

        # Get initial recurrent state from cache
        initial_state = cache.recurrent_state[...] if cache is not None else None

        # Core recurrence — always pass initial_state to get final_state back
        core_out, final_state = jax_chunk_gated_delta_rule(
            q, k, v, g, beta,
            chunk_size=self.chunk_size,
            initial_state=initial_state if initial_state is not None else jnp.zeros((B, self.num_value_heads, self.key_head_dim, self.value_head_dim), dtype=jnp.float32),
            use_qk_norm=self.use_qk_norm,
        )

        # Update recurrent state cache
        if cache is not None and final_state is not None:
            cache.recurrent_state[...] = final_state

        # core_out: (B, T, H_v, V_dim)
        core_out = core_out.reshape(B, T, -1)

        # Step D: Norm + gate + output
        y = self.norm(core_out, z)
        return self.out_proj(y)
