"""End-to-end comparison: run full model forward pass in NumPy (HF algorithm) vs JAX.

Tests the entire pipeline including all 24 layers to find where outputs diverge.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

MODEL_PATH = os.environ.get(
    "QWEN35_MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "qwen35-0.8b")
)

# Load ALL weights
import safetensors

print("Loading weights...")
W = {}
with safetensors.safe_open(f"{MODEL_PATH}/model.safetensors", framework="numpy") as f:
    for k in f.keys():
        W[k] = f.get_tensor(k)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
B, T = 1, input_ids.shape[1]

# Embeddings
embed_w = W["model.language_model.embed_tokens.weight"].astype(np.float32)
x_ref = embed_w[input_ids[0]].reshape(1, T, 1024)

# ---- JAX model ----
from qwen.qwen35 import modeling

config = modeling.ModelConfig.qwen35_0_8b()
jax_model = modeling.Qwen35ForConditionalGeneration.from_pretrained(MODEL_PATH, config=config)
cache_jax = modeling.init_cache(config, 1, T, 10)

# Run JAX forward and capture per-layer outputs
jax_input_ids = jnp.array(input_ids)
x_jax = jax_model.language_model.embed_tokens(jax_input_ids)

# Position embeddings for JAX
tc = config.text_config
fa_cache = next(c for c in cache_jax if isinstance(c, modeling.LayerCache))
positions = jnp.arange(T)[None, :] + fa_cache.cur_ind[...]
positions = jnp.broadcast_to(positions, (B, T))
positions_3d = jnp.stack([positions, positions, positions], axis=0)
cos_jax, sin_jax = modeling._generate_interleaved_mrope(
    positions_3d, tc.head_dim, tc.rope_theta, tc.partial_rotary_factor, tc.mrope_section
)
mask_jax = modeling.make_causal_mask(fa_cache, T)[None, None, :, :]


# ---- NumPy reference (HF algorithm) ----
def rmsnorm(x, w, eps=1e-6):
    x_f = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f**2, axis=-1, keepdims=True) + eps)
    return (x_f / rms) * w.astype(np.float32)


def linear_np(x, w_key):
    return x.astype(np.float32) @ W[w_key].astype(np.float32).T


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


LAYER_TYPES = list(tc.layer_types)

print(f"\nComparing {len(LAYER_TYPES)} layers, input shape: ({B},{T},{1024})")
print(f"Embed diff: {np.abs(x_ref - np.array(x_jax)).max():.8f}")

x_np = x_ref.copy()

for layer_idx in range(len(LAYER_TYPES)):
    lt = LAYER_TYPES[layer_idx]
    prefix = f"model.language_model.layers.{layer_idx}"

    # LayerNorm
    ln_w = W[f"{prefix}.input_layernorm.weight"].astype(np.float32)
    h_np = rmsnorm(x_np, ln_w)

    # Run JAX layer
    jax_layer = jax_model.language_model.layers[layer_idx]
    x_jax_out = jax_layer(jnp.array(x_np), cache_jax[layer_idx], cos_jax, sin_jax, mask_jax)
    x_jax_out_np = np.array(x_jax_out)

    if lt == "linear_attention":
        # GDN reference in NumPy
        qkv = linear_np(h_np, f"{prefix}.linear_attn.in_proj_qkv.weight")
        z = linear_np(h_np, f"{prefix}.linear_attn.in_proj_z.weight")
        b_val = linear_np(h_np, f"{prefix}.linear_attn.in_proj_b.weight")
        a_val = linear_np(h_np, f"{prefix}.linear_attn.in_proj_a.weight")

        # Conv (PyTorch exact)
        conv_w = W[f"{prefix}.linear_attn.conv1d.weight"].astype(np.float32)
        qkv_pt = torch.tensor(qkv).transpose(1, 2)
        conv_out = F.silu(F.conv1d(qkv_pt, torch.tensor(conv_w), padding=3, groups=6144)[:, :, :T])
        conv_out = conv_out.transpose(1, 2).detach().numpy()

        q = conv_out[..., :2048].reshape(B, T, 16, 128)
        k = conv_out[..., 2048:4096].reshape(B, T, 16, 128)
        v = conv_out[..., 4096:].reshape(B, T, 16, 128)

        A_log = W[f"{prefix}.linear_attn.A_log"].astype(np.float32)
        dt_bias = W[f"{prefix}.linear_attn.dt_bias"].astype(np.float32)
        beta = 1.0 / (1.0 + np.exp(-b_val.astype(np.float32)))
        g = -np.exp(A_log) * np.log1p(np.exp(a_val.astype(np.float32) + dt_bias))

        # Use JAX GDN (already verified to match HF exactly)
        from qwen.qwen35.gated_delta_net import jax_chunk_gated_delta_rule

        gdn_out, _ = jax_chunk_gated_delta_rule(
            jnp.array(q),
            jnp.array(k),
            jnp.array(v),
            jnp.array(g),
            jnp.array(beta),
            chunk_size=64,
            initial_state=jnp.zeros((B, 16, 128, 128)),
            use_qk_norm=True,
        )
        gdn_out = np.array(gdn_out)

        # Norm + gate
        norm_w = W[f"{prefix}.linear_attn.norm.weight"].astype(np.float32)
        out_proj_w = W[f"{prefix}.linear_attn.out_proj.weight"].astype(np.float32)
        flat = gdn_out.reshape(-1, 128).astype(np.float32)
        z_flat = z.reshape(B, T, 16, 128).reshape(-1, 128).astype(np.float32)
        rms_val = np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + 1e-6)
        normed = (flat / rms_val) * norm_w
        gated = (normed * silu(z_flat)).reshape(B, T, -1)
        attn_out_np = gated @ out_proj_w.T

    else:  # full_attention
        # Q/K/V projections
        q_proj_w = W[f"{prefix}.self_attn.q_proj.weight"].astype(np.float32)
        k_proj_w = W[f"{prefix}.self_attn.k_proj.weight"].astype(np.float32)
        v_proj_w = W[f"{prefix}.self_attn.v_proj.weight"].astype(np.float32)
        o_proj_w = W[f"{prefix}.self_attn.o_proj.weight"].astype(np.float32)
        q_norm_w = W[f"{prefix}.self_attn.q_norm.weight"].astype(np.float32)
        k_norm_w = W[f"{prefix}.self_attn.k_norm.weight"].astype(np.float32)

        q_raw = h_np.astype(np.float32) @ q_proj_w.T  # (B,T,4096)
        # HF split: view(B,T,8,512) -> chunk(2,-1)
        q_viewed = q_raw.reshape(B, T, 8, 512)
        q_states = q_viewed[..., :256]
        gate = q_viewed[..., 256:].reshape(B, T, -1)

        q_normed = rmsnorm(q_states.reshape(-1, 256), q_norm_w).reshape(B, T, 8, 256)
        k_raw = (h_np.astype(np.float32) @ k_proj_w.T).reshape(B, T, 2, 256)
        k_normed = rmsnorm(k_raw.reshape(-1, 256), k_norm_w).reshape(B, T, 2, 256)
        v_states = (h_np.astype(np.float32) @ v_proj_w.T).reshape(B, T, 2, 256)

        # RoPE (use same cos/sin as JAX — already verified)
        cos_np = np.array(cos_jax)  # (B, T, 64)
        sin_np = np.array(sin_jax)

        # Transpose to (B,H,T,D) for HF convention
        q_t = q_normed.transpose(0, 2, 1, 3)  # (B,8,T,256)
        k_t = k_normed.transpose(0, 2, 1, 3)  # (B,2,T,256)
        v_t = v_states.transpose(0, 2, 1, 3)  # (B,2,T,256)

        # Apply RoPE (HF: unsqueeze(1) on cos/sin)
        rotary_dim = 64
        cos_r = cos_np[:, None, :, :]  # (B,1,T,64) — broadcast over heads
        sin_r = sin_np[:, None, :, :]

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return np.concatenate([-x2, x1], axis=-1)

        def apply_rope(x, cos, sin):
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
            x_rot = x_rot * cos + rotate_half(x_rot) * sin
            return np.concatenate([x_rot, x_pass], axis=-1)

        q_t = apply_rope(q_t, cos_r, sin_r)
        k_t = apply_rope(k_t, cos_r, sin_r)

        # Repeat KV for GQA: 2 -> 8 heads
        k_t = np.repeat(k_t, 4, axis=1)  # (B,8,T,256)
        v_t = np.repeat(v_t, 4, axis=1)

        # Attention
        scale = 1.0 / np.sqrt(256.0)
        attn = (q_t @ k_t.transpose(0, 1, 3, 2)) * scale
        # Causal mask
        causal = np.tril(np.ones((T, T)))[None, None, :, :]
        attn = np.where(causal > 0, attn, -1e30)
        attn = np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)

        attn_out = (attn @ v_t).transpose(0, 2, 1, 3).reshape(B, T, -1)
        attn_out = attn_out * (1.0 / (1.0 + np.exp(-gate)))  # sigmoid gate
        attn_out_np = attn_out @ o_proj_w.T

    # Residual
    x_ref_new = x_np + attn_out_np

    # MLP
    post_ln_w = W[f"{prefix}.post_attention_layernorm.weight"].astype(np.float32)
    h_mlp = rmsnorm(x_ref_new, post_ln_w)
    gate_proj = h_mlp @ W[f"{prefix}.mlp.gate_proj.weight"].astype(np.float32).T
    up_proj = h_mlp @ W[f"{prefix}.mlp.up_proj.weight"].astype(np.float32).T
    mlp_out = (silu(gate_proj) * up_proj) @ W[f"{prefix}.mlp.down_proj.weight"].astype(np.float32).T
    x_ref_new = x_ref_new + mlp_out

    # Compare
    diff = np.abs(x_ref_new - x_jax_out_np)
    print(
        f"Layer {layer_idx:2d} ({lt[:4]}): ref_std={x_ref_new.std():.4f}, jax_std={x_jax_out_np.std():.4f}, "
        f"max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}"
    )

    x_np = x_ref_new

# Final norm + logits
final_ln_w = W["model.language_model.norm.weight"].astype(np.float32)
x_final = rmsnorm(x_np, final_ln_w)
logits_ref = x_final @ embed_w.T
last_ref = logits_ref[0, -1, :]

# JAX logits
x_jax_final = jax_model.language_model.norm(jnp.array(x_np))
logits_jax = np.array(x_jax_final) @ embed_w.T
last_jax = logits_jax[0, -1, :]

# Also get actual JAX model logits
cache_fresh = modeling.init_cache(config, 1, T, 10)
logits_model = np.array(jax_model(jnp.array(input_ids), cache=cache_fresh))
last_model = logits_model[0, -1, :]

print(f"\n{'=' * 60}")
print("FINAL LOGITS COMPARISON")
print(f"{'=' * 60}")

top_ref = np.argsort(last_ref)[-5:][::-1]
top_model = np.argsort(last_model)[-5:][::-1]

print("NumPy reference top-5:")
for t in top_ref:
    print(f"  {repr(tokenizer.decode([int(t)]))}: {last_ref[t]:.2f}")

print("\nJAX model top-5:")
for t in top_model:
    print(f"  {repr(tokenizer.decode([int(t)]))}: {last_model[t]:.2f}")

diff_logits = np.abs(last_ref - last_model)
print(f"\nLogits diff: max={diff_logits.max():.4f}, mean={diff_logits.mean():.4f}")
