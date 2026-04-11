"""Transformer building blocks: RMSNorm, GQAttention, GatedMLP, GemmaDecoderLayer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class GQAttention(nn.Module):
    """Grouped-Query Attention with prefix KV cache support.

    When past_key_values is provided, the cached prefix K/V are concatenated
    with the suffix K/V along the sequence dim before computing attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Concat cached prefix KV with suffix KV
        if past_key_values is not None:
            cached_k, cached_v = past_key_values
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if attn_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class GatedMLP(nn.Module):
    """SwiGLU-style gated MLP."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GemmaDecoderLayer(nn.Module):
    """Standard Gemma decoder layer: RMSNorm -> Self-Attn -> RMSNorm -> MLP.

    Self-attention receives prefix KV cache so that suffix tokens can attend
    to prefix tokens without an explicit cross-attention layer.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model)
        self.self_attn = GQAttention(d_model, n_heads, n_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), past_key_values=past_key_values, attn_mask=attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
