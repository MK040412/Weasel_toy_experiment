"""GemmaActionExpert — ~311M pi0-style action expert for flow matching.

Training (MM DiT joint attention):
  forward_joint(obs_embed, noisy_actions, timestep)
  — obs + action tokens pass through transformer jointly with prefix-LM mask

Inference (cached KV, same latency as before):
  1. build_prefix_kv_cache(obs_embed)  — obs through full transformer, collect KV
  2. forward(noisy_actions, timestep, prefix_kv_cache) — N denoising steps
  3. denoise(obs_embed, ...) — convenience wrapper combining 1 + 2

openpi0.5 convention:
  - t=1 is pure noise, t=0 is clean data
  - Inference: Euler from t=1 -> t=0, dt = -1/n_steps
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen.vla.config import ActionExpertConfig
from qwen.vla.models.layers import GemmaDecoderLayer, RMSNorm


class GemmaActionExpert(nn.Module):
    """pi0-style ~311M Gemma action expert with MM DiT joint attention.

    Training: obs + action tokens attend jointly via prefix-LM mask.
    Inference: obs-only forward builds KV cache; action-only denoising loop.
    """

    def __init__(self, config: ActionExpertConfig | None = None):
        super().__init__()
        if config is None:
            config = ActionExpertConfig()
        self.config = config
        self.action_dim = config.action_dim
        self.d_model = config.d_model

        # Input projection
        self.action_in_proj = nn.Linear(config.action_dim, config.d_model, bias=False)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    head_dim=config.head_dim,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Output head
        self.output_norm = RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.action_dim, bias=False)

    def _make_attn_mask(
        self,
        n_obs: int,
        n_action: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Prefix-LM attention mask (OpenPI make_attn_mask pattern).

        - Prefix (obs) tokens: bidirectional among themselves, cannot attend to suffix.
        - Suffix (action) tokens: bidirectional attend to all (prefix + suffix).

        Returns:
            Boolean mask (1, 1, S, S) where S = n_obs + n_action.
        """
        ar_mask = [False] * n_obs + [True] + [False] * (n_action - 1)
        ar_mask = torch.tensor(ar_mask, device=device)
        cumsum = ar_mask.cumsum(0)
        # token i can attend to token j if cumsum[j] <= cumsum[i]
        attn_mask = cumsum.unsqueeze(0) <= cumsum.unsqueeze(1)  # (S, S)
        return attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    def forward_joint(
        self,
        obs_embed: torch.Tensor,  # (B, n_obs, d_model)
        noisy_actions: torch.Tensor,  # (B, chunk_size, action_dim)
        timestep: torch.Tensor,  # (B, T, 1)
    ) -> torch.Tensor:
        """Training: joint attention over [obs, action] with prefix-LM mask.

        Returns:
            Predicted velocity: (B, chunk_size, action_dim)
        """
        B, n_obs, _ = obs_embed.shape

        # Obs tokens with t=0 timestep embedding
        t_zero = torch.zeros(B, 1, 1, device=obs_embed.device, dtype=obs_embed.dtype)
        obs_tokens = obs_embed + self.timestep_mlp(t_zero)

        # Action tokens with per-token timestep embedding
        action_tokens = self.action_in_proj(noisy_actions) + self.timestep_mlp(timestep)

        # Joint sequence
        joint = torch.cat([obs_tokens, action_tokens], dim=1)
        n_action = noisy_actions.shape[1]
        attn_mask = self._make_attn_mask(n_obs, n_action, joint.device)

        for layer in self.layers:
            joint = layer(joint, attn_mask=attn_mask)

        # Extract action outputs only
        action_out = joint[:, n_obs:, :]
        return self.output_head(self.output_norm(action_out))

    def build_prefix_kv_cache(
        self,
        obs_embed: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Inference: run obs through full transformer, collect KV per layer.

        Because the prefix-LM mask prevents obs from attending to actions,
        this produces the same obs representations as forward_joint.

        Args:
            obs_embed: (B, n_obs, d_model) observation embeddings.

        Returns:
            List of (K, V) tuples, one per layer.
        """
        B, S, _ = obs_embed.shape
        t_zero = torch.zeros(B, 1, 1, device=obs_embed.device, dtype=obs_embed.dtype)
        x = obs_embed + self.timestep_mlp(t_zero)

        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            normed = layer.input_layernorm(x)
            attn = layer.self_attn

            q = attn.q_proj(normed).view(B, S, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(normed).view(B, S, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(normed).view(B, S, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
            cache.append((k, v))

            # Complete forward: self-attn + residual + MLP + residual
            k_exp = k.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else k
            v_exp = v.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else v
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, attn.n_heads * attn.head_dim)
            x = x + attn.o_proj(attn_out)
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        return cache

    def forward(
        self,
        noisy_actions: torch.Tensor,  # (B, chunk_size, action_dim)
        timestep: torch.Tensor,  # (B, T, 1) per-token or (B, 1) uniform
        prefix_kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Single denoising step: predict velocity field.

        Returns:
            Predicted velocity: (B, chunk_size, action_dim)
        """
        x = self.action_in_proj(noisy_actions)
        t_emb = self.timestep_mlp(timestep)  # (B, T, d_model) or (B, d_model)
        if t_emb.dim() == 2:
            t_emb = t_emb.unsqueeze(1)  # (B, 1, d_model) for broadcast
        x = x + t_emb

        for layer, kv in zip(self.layers, prefix_kv_cache):
            x = layer(x, past_key_values=kv)

        x = self.output_norm(x)
        return self.output_head(x)

    @torch.inference_mode()
    def denoise(
        self,
        obs_embed: torch.Tensor,  # (B, n_obs, d_model)
        chunk_size: int = 50,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Flow matching Euler integration: t=1 (noise) -> t=0 (data).

        openpi0.5 convention: reverse direction Euler.
        """
        B = obs_embed.shape[0]
        device = obs_embed.device
        dtype = obs_embed.dtype

        prefix_kv_cache = self.build_prefix_kv_cache(obs_embed)

        # Start from pure noise at t=1
        x_t = torch.randn(B, chunk_size, self.action_dim, device=device, dtype=dtype)
        dt = -1.0 / n_steps  # negative: t goes 1 -> 0
        t_curr = 1.0

        for _ in range(n_steps):
            t = torch.full((B, chunk_size, 1), t_curr, device=device, dtype=dtype)
            velocity = self(x_t, t, prefix_kv_cache)
            x_t = x_t + velocity * dt
            t_curr += dt

        return x_t
