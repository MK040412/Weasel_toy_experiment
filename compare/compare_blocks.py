"""Block-by-block comparison: PyTorch (HF) vs JAX for Qwen3.5-0.8B.

Runs a single forward pass through each block type and compares outputs.
Uses the SAME weights and SAME input for both.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"  # Use CPU so both PyTorch and JAX run on same device

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "qwen35-0.8b"))

# ============================================================
# 1. Load weights from safetensors (shared by both)
# ============================================================
import safetensors

print("Loading weights...")
weights = {}
with safetensors.safe_open(f"{MODEL_PATH}/model.safetensors", framework="numpy") as f:
    for k in f.keys():
        weights[k] = f.get_tensor(k)
print(f"  Loaded {len(weights)} tensors")

# ============================================================
# 2. Create a fixed input
# ============================================================
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]  # (1, T)
print(f"Input: '{prompt}' -> {input_ids.shape[1]} tokens")

# ============================================================
# 3. Get embeddings (same for both)
# ============================================================
embed_w = weights["model.language_model.embed_tokens.weight"]  # (vocab, 1024)
embeddings = embed_w[input_ids[0]]  # (T, 1024)
embeddings_f32 = embeddings.astype(np.float32)
B, T, D = 1, embeddings_f32.shape[0], embeddings_f32.shape[1]
x_np = embeddings_f32.reshape(1, T, D)  # (1, T, 1024)


# ============================================================
# 4. RMSNorm (layer 0 input_layernorm)
# ============================================================
def np_rmsnorm(x, w, eps=1e-6):
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + eps)
    return ((x_f32 / rms) * w).astype(x.dtype)


ln_w = weights["model.language_model.layers.0.input_layernorm.weight"]
x_normed = np_rmsnorm(x_np, ln_w)

print(f"\n{'=' * 60}")
print("BLOCK 1: Input LayerNorm (layer 0)")
print(f"{'=' * 60}")
print(f"  Input:  mean={x_np.mean():.6f}, std={x_np.std():.6f}")
print(f"  Output: mean={x_normed.mean():.6f}, std={x_normed.std():.6f}")

# ============================================================
# 5. GDN Layer 0 — PyTorch manual forward
# ============================================================
print(f"\n{'=' * 60}")
print("BLOCK 2: GDN Layer 0 — Step-by-step comparison")
print(f"{'=' * 60}")


# Step A: Projections
def linear(x, w_key):
    """x: (B,T,in), weight: (out,in) -> (B,T,out)"""
    w = weights[w_key].astype(np.float32)
    return x.astype(np.float32) @ w.T


qkv = linear(x_normed, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
z = linear(x_normed, "model.language_model.layers.0.linear_attn.in_proj_z.weight")
b = linear(x_normed, "model.language_model.layers.0.linear_attn.in_proj_b.weight")
a = linear(x_normed, "model.language_model.layers.0.linear_attn.in_proj_a.weight")

print("\n  Step A: Projections")
print(f"    qkv: shape={qkv.shape}, mean={qkv.mean():.6f}, std={qkv.std():.6f}")
print(f"    z:   shape={z.shape}, mean={z.mean():.6f}, std={z.std():.6f}")
print(f"    b:   shape={b.shape}, values={b[0, 0, :4]}")
print(f"    a:   shape={a.shape}, values={a[0, 0, :4]}")

# Step B: Conv1D
# HF: transpose(1,2) -> conv1d(groups=conv_dim, padding=3) -> [:,:,:seq_len] -> silu -> transpose(1,2)
conv_w_raw = weights["model.language_model.layers.0.linear_attn.conv1d.weight"]  # (6144, 1, 4)
print("\n  Step B: Conv1D")
print(f"    conv weight shape (safetensors): {conv_w_raw.shape}")

# PyTorch way: transpose input, apply conv1d with padding=3, slice, silu
qkv_pt = torch.tensor(qkv).transpose(1, 2)  # (1, 6144, T)
conv_w_pt = torch.tensor(conv_w_raw.astype(np.float32))  # (6144, 1, 4)
conv_out_pt = F.conv1d(qkv_pt, conv_w_pt, padding=3, groups=6144)[:, :, :T]
conv_out_pt = F.silu(conv_out_pt)
conv_out_pt = conv_out_pt.transpose(1, 2)  # (1, T, 6144)
conv_out_pt_np = conv_out_pt.detach().numpy()
print(f"    PyTorch conv out: mean={conv_out_pt_np.mean():.6f}, std={conv_out_pt_np.std():.6f}")

# JAX way: pad left, depthwise conv, silu
qkv_jax = jnp.array(qkv)
conv_w_jax = jnp.array(conv_w_raw[:, 0, :].astype(np.float32))  # (6144, 4)
padded = jnp.pad(qkv_jax, ((0, 0), (3, 0), (0, 0)))
padded_t = padded.transpose(0, 2, 1)
w_jax = conv_w_jax[:, None, :]  # (6144, 1, 4)
conv_out_jax = jax.lax.conv_general_dilated(padded_t, w_jax, (1,), "VALID", feature_group_count=6144)
conv_out_jax = jax.nn.silu(conv_out_jax.transpose(0, 2, 1))
conv_out_jax_np = np.array(conv_out_jax)
print(f"    JAX conv out:     mean={conv_out_jax_np.mean():.6f}, std={conv_out_jax_np.std():.6f}")

diff_conv = np.abs(conv_out_pt_np - conv_out_jax_np)
print(f"    DIFF: max={diff_conv.max():.8f}, mean={diff_conv.mean():.8f}")

# Step C: Split QKV, reshape to heads
key_dim = 16 * 128  # 2048
value_dim = 16 * 128  # 2048

# Both use same split
q_np = conv_out_pt_np[..., :key_dim].reshape(B, T, 16, 128)
k_np = conv_out_pt_np[..., key_dim : key_dim * 2].reshape(B, T, 16, 128)
v_np = conv_out_pt_np[..., key_dim * 2 :].reshape(B, T, 16, 128)

print("\n  Step C: After split & reshape")
print(f"    q: shape={q_np.shape}, mean={q_np.mean():.6f}")
print(f"    k: shape={k_np.shape}, mean={k_np.mean():.6f}")
print(f"    v: shape={v_np.shape}, mean={v_np.mean():.6f}")

# Step D: Gates
A_log = weights["model.language_model.layers.0.linear_attn.A_log"].astype(np.float32)
dt_bias = weights["model.language_model.layers.0.linear_attn.dt_bias"].astype(np.float32)

beta = 1.0 / (1.0 + np.exp(-b.astype(np.float32)))  # sigmoid
g_pt = -np.exp(A_log) * np.log1p(np.exp(a.astype(np.float32) + dt_bias))  # -exp(A_log) * softplus(a+dt_bias)

print("\n  Step D: Gates")
print(f"    beta: mean={beta.mean():.6f}, range=[{beta.min():.4f}, {beta.max():.4f}]")
print(f"    g:    mean={g_pt.mean():.6f}, range=[{g_pt.min():.4f}, {g_pt.max():.4f}]")

# Step E: Core GDN recurrence (naive, in float32, matching HF exactly)
print("\n  Step E: Core GDN recurrence (naive float32)")


def l2norm_np(x, eps=1e-6):
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


# L2 normalize Q and K
q_n = l2norm_np(q_np.astype(np.float32))
k_n = l2norm_np(k_np.astype(np.float32))
v_f = v_np.astype(np.float32)

# Transpose to (B, H, T, D) matching HF
q_t = q_n.transpose(0, 2, 1, 3)  # (1, 16, T, 128)
k_t = k_n.transpose(0, 2, 1, 3)
v_t = v_f.transpose(0, 2, 1, 3)
beta_t = beta.transpose(0, 2, 1).astype(np.float32)  # (1, 16, T)
g_t = g_pt.transpose(0, 2, 1).astype(np.float32)  # (1, 16, T)

scale = 1.0 / np.sqrt(128.0)
q_t = q_t * scale

# Naive step-by-step recurrence (HF reference, line 293-304)
S = np.zeros((B, 16, 128, 128), dtype=np.float32)
outputs = np.zeros_like(q_t)  # (1, 16, T, 128)

for t_idx in range(T):
    qt = q_t[:, :, t_idx, :]  # (1, 16, 128)
    kt = k_t[:, :, t_idx, :]
    vt = v_t[:, :, t_idx, :]
    bt = beta_t[:, :, t_idx]  # (1, 16)
    gt = g_t[:, :, t_idx]

    # Decay state
    decay = np.exp(gt)[:, :, None, None]
    S = S * decay

    # Delta rule: v_beta = beta * v, k_beta = beta * k
    k_beta = kt * bt[:, :, None]
    v_beta = vt * bt[:, :, None]

    # v_new = v_beta - k_beta @ S (delta update)
    kS = np.einsum("bhk,bhkv->bhv", k_beta, S)
    v_new = v_beta - kS

    # Update state: S += k_beta^T @ v_new
    S = S + np.einsum("bhk,bhv->bhkv", k_beta, v_new)

    # Output: o = q @ S (using already-decayed S which includes this step's update)
    # Actually HF computes: attn_inter + attn @ v_new
    # attn_inter = (q * exp(g)) @ old_S_before_this_step
    # But in naive per-step: o = q @ S_current
    ot = np.einsum("bhk,bhkv->bhv", qt, S)
    outputs[:, :, t_idx, :] = ot

# Transpose back to (B, T, H, D)
gdn_out_ref = outputs.transpose(0, 2, 1, 3)
print(f"    Reference output: mean={gdn_out_ref.mean():.6f}, std={gdn_out_ref.std():.6f}")
print(f"    Reference range: [{gdn_out_ref.min():.6f}, {gdn_out_ref.max():.6f}]")

# Step F: Compare with our JAX jax_chunk_gated_delta_rule
from qwen.qwen35.gated_delta_net import jax_chunk_gated_delta_rule

q_jax = jnp.array(q_np)
k_jax = jnp.array(k_np)
v_jax = jnp.array(v_np)
g_jax = jnp.array(g_pt)
beta_jax = jnp.array(beta)

jax_out, _ = jax_chunk_gated_delta_rule(
    q_jax,
    k_jax,
    v_jax,
    g_jax,
    beta_jax,
    chunk_size=64,
    initial_state=jnp.zeros((B, 16, 128, 128), dtype=jnp.float32),
    use_qk_norm=True,
)
jax_out_np = np.array(jax_out)
print(f"\n    JAX chunked output: mean={jax_out_np.mean():.6f}, std={jax_out_np.std():.6f}")
print(f"    JAX chunked range: [{jax_out_np.min():.6f}, {jax_out_np.max():.6f}]")

diff_gdn = np.abs(gdn_out_ref - jax_out_np)
print(f"    DIFF (ref vs jax): max={diff_gdn.max():.6f}, mean={diff_gdn.mean():.6f}")

# Step G: RMSNormGated + output projection
print("\n  Step G: RMSNormGated + out_proj")
norm_w = weights["model.language_model.layers.0.linear_attn.norm.weight"].astype(np.float32)  # (128,)
out_proj_w = weights["model.language_model.layers.0.linear_attn.out_proj.weight"].astype(np.float32)  # (1024, 2048)

# Reference: reshape to (B*T*H, head_dim), norm, gate, reshape back
ref_flat = gdn_out_ref.reshape(-1, 128).astype(np.float32)
z_shaped = z.reshape(B, T, 16, 128).reshape(-1, 128).astype(np.float32)

# RMSNorm
rms = np.sqrt(np.mean(ref_flat**2, axis=-1, keepdims=True) + 1e-6)
ref_normed = (ref_flat / rms) * norm_w
# SiLU gate
silu_z = z_shaped * (1.0 / (1.0 + np.exp(-z_shaped)))
ref_gated = ref_normed * silu_z
ref_gated = ref_gated.reshape(B, T, -1)  # (1, T, 2048)
ref_output = ref_gated @ out_proj_w.T  # (1, T, 1024)

print(f"    Reference GDN block output: mean={ref_output.mean():.6f}, std={ref_output.std():.6f}")

# JAX full GDN block output
jax_flat = jax_out_np.reshape(-1, 128).astype(np.float32)
jax_normed = (jax_flat / np.sqrt(np.mean(jax_flat**2, axis=-1, keepdims=True) + 1e-6)) * norm_w
jax_gated = jax_normed * silu_z
jax_gated = jax_gated.reshape(B, T, -1)
jax_output = jax_gated @ out_proj_w.T

print(f"    JAX GDN block output:       mean={jax_output.mean():.6f}, std={jax_output.std():.6f}")
diff_block = np.abs(ref_output - jax_output)
print(f"    DIFF: max={diff_block.max():.6f}, mean={diff_block.mean():.6f}")

# ============================================================
# 6. Full Attention Layer 3 — comparison
# ============================================================
print(f"\n{'=' * 60}")
print("BLOCK 3: Full Attention Layer 3")
print(f"{'=' * 60}")

# For simplicity, use the layer-0 GDN output as a proxy input
# (In reality, layers 0,1,2 precede layer 3, but this tests the attention mechanics)
# We'll use x_normed as input to avoid cumulative errors

ln3_w = weights["model.language_model.layers.3.input_layernorm.weight"]
x3_normed = np_rmsnorm(x_np, ln3_w)

q_proj_w = weights["model.language_model.layers.3.self_attn.q_proj.weight"].astype(np.float32)  # (4096, 1024)
k_proj_w = weights["model.language_model.layers.3.self_attn.k_proj.weight"].astype(np.float32)  # (512, 1024)
v_proj_w = weights["model.language_model.layers.3.self_attn.v_proj.weight"].astype(np.float32)  # (512, 1024)
q_norm_w = weights["model.language_model.layers.3.self_attn.q_norm.weight"].astype(np.float32)  # (256,)
k_norm_w = weights["model.language_model.layers.3.self_attn.k_norm.weight"].astype(np.float32)  # (256,)

x3_f32 = x3_normed.astype(np.float32)

# q_proj -> view(B,T,H,Hd*2) -> chunk -> q(B,T,H,Hd), gate(B,T,H,Hd)
q_raw = x3_f32 @ q_proj_w.T  # (1, T, 4096)
print(f"\n  q_proj output: shape={q_raw.shape}, mean={q_raw.mean():.6f}")

# HF split: view(B,T,-1,head_dim*2) then chunk(2, dim=-1)
q_viewed = q_raw.reshape(B, T, 8, 512)  # 8 heads, 512=256*2
q_states = q_viewed[..., :256]  # (B,T,8,256) — Q
gate_states = q_viewed[..., 256:]  # (B,T,8,256) — gate
gate_flat = gate_states.reshape(B, T, -1)  # (B,T,2048)

print(f"  HF split: q={q_states.shape}, gate={gate_states.shape}")
print(f"    q first head: mean={q_states[0, 0, 0, :5].mean():.6f}")
print(f"    gate first head: mean={gate_states[0, 0, 0, :5].mean():.6f}")

# Our JAX split (AFTER the fix): same as HF
# q_and_gate = q_raw.reshape(B,T,8,512); q=[:256], gate=[256:]
# This should match now.

# Compare: what if we did the OLD wrong split?
q_old_wrong = q_raw[..., :2048]  # first half
gate_old_wrong = q_raw[..., 2048:]  # second half
q_old_reshaped = q_old_wrong.reshape(B, T, 8, 256)

print("\n  OLD (wrong) split comparison:")
print(f"    q correct[0,0,0,:5]:  {q_states[0, 0, 0, :5]}")
print(f"    q old_wrong[0,0,0,:5]: {q_old_reshaped[0, 0, 0, :5]}")
print(f"    Are they the same? {np.allclose(q_states, q_old_reshaped, atol=1e-5)}")


# RMSNorm on Q and K
def rmsnorm_per_head(x, w, eps=1e-6):
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + eps)
    return (x_f32 / rms) * w


q_normed = rmsnorm_per_head(q_states, q_norm_w)
k_raw = (x3_f32 @ k_proj_w.T).reshape(B, T, 2, 256)
k_normed = rmsnorm_per_head(k_raw, k_norm_w)

print("\n  After Q/K norm:")
print(f"    q_normed: mean={q_normed.mean():.6f}, std={q_normed.std():.6f}")
print(f"    k_normed: mean={k_normed.mean():.6f}, std={k_normed.std():.6f}")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  Conv1D:     {'PASS' if diff_conv.max() < 0.001 else 'FAIL'} (max diff: {diff_conv.max():.8f})")
print(f"  GDN core:   {'PASS' if diff_gdn.max() < 0.01 else 'FAIL'} (max diff: {diff_gdn.max():.6f})")
print(f"  GDN block:  {'PASS' if diff_block.max() < 0.01 else 'FAIL'} (max diff: {diff_block.max():.6f})")
print(f"  q_proj split: OLD wrong == HF correct? {np.allclose(q_states, q_old_reshaped, atol=1e-5)}")
