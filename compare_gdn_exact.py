"""Exact comparison of HF torch_chunk_gated_delta_rule vs JAX jax_chunk_gated_delta_rule.

Reimplements HF's exact algorithm in NumPy, then compares with JAX.
"""

import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", os.path.join(_SCRIPT_DIR, "..", "models", "qwen35-0.8b"))

# Load weights and prepare input (same as compare_blocks.py)
import safetensors
from transformers import AutoTokenizer

weights = {}
with safetensors.safe_open(f"{MODEL_PATH}/model.safetensors", framework="numpy") as f:
    for k in f.keys():
        weights[k] = f.get_tensor(k)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]

embed_w = weights["model.language_model.embed_tokens.weight"]
embeddings = embed_w[input_ids[0]].astype(np.float32).reshape(1, -1, 1024)
B, T, D = embeddings.shape

# LayerNorm
ln_w = weights["model.language_model.layers.0.input_layernorm.weight"].astype(np.float32)
x_f32 = embeddings.astype(np.float32)
rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + 1e-6)
x_normed = (x_f32 / rms) * ln_w

# Projections
def linear(x, w_key):
    w = weights[w_key].astype(np.float32)
    return x.astype(np.float32) @ w.T

qkv = linear(x_normed, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
z = linear(x_normed, "model.language_model.layers.0.linear_attn.in_proj_z.weight")
b = linear(x_normed, "model.language_model.layers.0.linear_attn.in_proj_b.weight")
a = linear(x_normed, "model.language_model.layers.0.linear_attn.in_proj_a.weight")

# Conv1D (use PyTorch for exact HF match)
import torch, torch.nn.functional as F
conv_w_raw = weights["model.language_model.layers.0.linear_attn.conv1d.weight"].astype(np.float32)
qkv_pt = torch.tensor(qkv).transpose(1, 2)
conv_w_pt = torch.tensor(conv_w_raw)
conv_out = F.silu(F.conv1d(qkv_pt, conv_w_pt, padding=3, groups=6144)[:, :, :T])
conv_out = conv_out.transpose(1, 2).detach().numpy()

# Split
key_dim, value_dim = 2048, 2048
query = conv_out[..., :key_dim].reshape(B, T, 16, 128)
key = conv_out[..., key_dim:key_dim*2].reshape(B, T, 16, 128)
value = conv_out[..., key_dim*2:].reshape(B, T, 16, 128)

# Gates
A_log = weights["model.language_model.layers.0.linear_attn.A_log"].astype(np.float32)
dt_bias = weights["model.language_model.layers.0.linear_attn.dt_bias"].astype(np.float32)
beta_np = 1.0 / (1.0 + np.exp(-b.astype(np.float32)))
g_np = -np.exp(A_log) * np.log1p(np.exp(a.astype(np.float32) + dt_bias))

# ==========================================================
# HF torch_chunk_gated_delta_rule — EXACT reimplementation in NumPy
# (from modeling_qwen3_5.py lines 234-311)
# ==========================================================
print("=" * 60)
print("HF torch_chunk_gated_delta_rule — exact NumPy reimplementation")
print("=" * 60)

def l2norm(x, eps=1e-6):
    return x / np.sqrt(np.sum(x*x, axis=-1, keepdims=True) + eps)

chunk_size = 64

# L2 norm on Q and K
q_in = l2norm(query.astype(np.float32))
k_in = l2norm(key.astype(np.float32))
v_in = value.astype(np.float32)
beta_in = beta_np.astype(np.float32)
g_in = g_np.astype(np.float32)

# Transpose to (B, H, T, D) — HF line 249-250
q_t = q_in.transpose(0, 2, 1, 3)  # (1, 16, T, 128)
k_t = k_in.transpose(0, 2, 1, 3)
v_t = v_in.transpose(0, 2, 1, 3)
beta_t = beta_in.transpose(0, 2, 1)  # (1, 16, T)
g_t = g_in.transpose(0, 2, 1)

num_heads = 16
k_head_dim = 128
v_head_dim = 128

# Padding
pad_size = (chunk_size - T % chunk_size) % chunk_size
if pad_size > 0:
    q_t = np.pad(q_t, ((0,0),(0,0),(0,pad_size),(0,0)))
    k_t = np.pad(k_t, ((0,0),(0,0),(0,pad_size),(0,0)))
    v_t = np.pad(v_t, ((0,0),(0,0),(0,pad_size),(0,0)))
    beta_t = np.pad(beta_t, ((0,0),(0,0),(0,pad_size)))
    g_t = np.pad(g_t, ((0,0),(0,0),(0,pad_size)))

total_len = T + pad_size
scale = 1.0 / np.sqrt(k_head_dim)
q_t = q_t * scale

v_beta = v_t * beta_t[..., None]
k_beta = k_t * beta_t[..., None]

# Reshape to chunks: (B, H, num_chunks, chunk_size, D)
num_chunks = total_len // chunk_size
q_c = q_t.reshape(B, num_heads, num_chunks, chunk_size, k_head_dim)
k_c = k_t.reshape(B, num_heads, num_chunks, chunk_size, k_head_dim)
v_c = v_t.reshape(B, num_heads, num_chunks, chunk_size, v_head_dim)
k_beta_c = k_beta.reshape(B, num_heads, num_chunks, chunk_size, k_head_dim)
v_beta_c = v_beta.reshape(B, num_heads, num_chunks, chunk_size, v_head_dim)
g_c = g_t.reshape(B, num_heads, num_chunks, chunk_size)

# Chunk decay (HF line 275-276)
g_cum = np.cumsum(g_c, axis=-1)
# decay_mask[i,j] = exp(g_cum[i] - g_cum[j]) for i >= j
g_diff = g_cum[..., :, None] - g_cum[..., None, :]  # (..., C, C)
tril_mask = np.tril(np.ones((chunk_size, chunk_size)))
decay_mask = np.exp(g_diff * tril_mask) * tril_mask

# HF line 277: attn = -(k_beta @ key.T) * decay_mask, masked upper tri to 0
upper_mask = np.triu(np.ones((chunk_size, chunk_size), dtype=bool))
attn = -(k_beta_c @ k_c.transpose(0, 1, 2, 4, 3)) * decay_mask
attn[..., upper_mask] = 0

# HF line 278-281: forward substitution for matrix inverse
for i in range(1, chunk_size):
    for b_idx in range(B):
        for h_idx in range(num_heads):
            for c_idx in range(num_chunks):
                row = attn[b_idx, h_idx, c_idx, i, :i].copy()
                sub = attn[b_idx, h_idx, c_idx, :i, :i].copy()
                attn[b_idx, h_idx, c_idx, i, :i] = row + np.sum(row[:, None] * sub, axis=0)

# HF line 282: add identity
attn = attn + np.eye(chunk_size)

# HF line 283-284
v_new = attn @ v_beta_c
k_cumdecay = attn @ (k_beta_c * np.exp(g_cum)[..., None])

# Recurrent state loop (HF line 294-304)
S = np.zeros((B, num_heads, k_head_dim, v_head_dim), dtype=np.float32)
core_out = np.zeros((B, num_heads, num_chunks, chunk_size, v_head_dim), dtype=np.float32)

upper_mask_2 = np.triu(np.ones((chunk_size, chunk_size), dtype=bool), k=1)

for i in range(num_chunks):
    q_i = q_c[:, :, i]        # (B, H, C, K)
    k_i = k_c[:, :, i]        # (B, H, C, K)
    v_i = v_new[:, :, i]      # (B, H, C, V)

    # Intra-chunk attention with decay
    attn_i = (q_i @ k_i.transpose(0, 1, 3, 2)) * decay_mask[:, :, i]
    attn_i[..., upper_mask_2] = 0

    # Inter-chunk: query @ recurrent_state
    v_prime = k_cumdecay[:, :, i] @ S
    v_delta = v_i - v_prime
    attn_inter = (q_i * np.exp(g_cum[:, :, i])[..., None]) @ S

    core_out[:, :, i] = attn_inter + attn_i @ v_delta

    # Update state
    g_last = g_cum[:, :, i, -1]
    S = S * np.exp(g_last)[:, :, None, None]
    k_decay = k_i * np.exp(g_cum[:, :, i, -1:] - g_cum[:, :, i])[..., None]
    S = S + k_decay.transpose(0, 1, 3, 2) @ v_delta

# Reshape and trim
hf_out = core_out.reshape(B, num_heads, -1, v_head_dim)[:, :, :T]
hf_out = hf_out.transpose(0, 2, 1, 3)  # (B, T, H, D)

print(f"HF exact output: mean={hf_out.mean():.6f}, std={hf_out.std():.6f}")
print(f"HF exact range: [{hf_out.min():.6f}, {hf_out.max():.6f}]")

# ==========================================================
# Our JAX jax_chunk_gated_delta_rule
# ==========================================================
print(f"\n{'='*60}")
print("JAX jax_chunk_gated_delta_rule")
print(f"{'='*60}")

from model35.gated_delta_net import jax_chunk_gated_delta_rule

q_jax = jnp.array(query)
k_jax = jnp.array(key)
v_jax = jnp.array(value)
g_jax = jnp.array(g_np)
beta_jax = jnp.array(beta_np)

jax_out, _ = jax_chunk_gated_delta_rule(
    q_jax, k_jax, v_jax, g_jax, beta_jax,
    chunk_size=64,
    initial_state=jnp.zeros((B, 16, 128, 128), dtype=jnp.float32),
    use_qk_norm=True,
)
jax_out_np = np.array(jax_out)
print(f"JAX output: mean={jax_out_np.mean():.6f}, std={jax_out_np.std():.6f}")
print(f"JAX range: [{jax_out_np.min():.6f}, {jax_out_np.max():.6f}]")

# ==========================================================
# Comparison
# ==========================================================
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
diff = np.abs(hf_out - jax_out_np)
print(f"Max diff:  {diff.max():.8f}")
print(f"Mean diff: {diff.mean():.8f}")
print(f"Relative diff: {diff.max() / (np.abs(hf_out).max() + 1e-10):.4f}")

# Check per-position
for t in range(T):
    d = np.abs(hf_out[0, t] - jax_out_np[0, t])
    print(f"  Position {t}: max_diff={d.max():.8f}, hf_mean={hf_out[0,t].mean():.8f}, jax_mean={jax_out_np[0,t].mean():.8f}")

if diff.max() < 0.001:
    print("\n✅ GDN core recurrence MATCH!")
else:
    print(f"\n❌ GDN core recurrence MISMATCH (max diff: {diff.max():.6f})")
