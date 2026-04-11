"""GemmaActionExpert: pi0-style action denoising transformer in Flax NNX.

Supports:
- Joint training with prefix-LM mask (obs=prefix, actions=suffix)
- Cached KV inference for fast denoising
- Gripper as discrete token (BCE head) vs continuous flow matching for pos/orn
"""

import jax
import jax.numpy as jnp
from flax import nnx

from qwen.vla.models.layers import GemmaDecoderLayer, RMSNorm


def make_prefix_lm_mask(n_prefix: int, n_suffix: int) -> jax.Array:
    """Prefix-LM attention mask: prefix bidirectional, suffix attends to all.

    Returns bool mask (1, 1, S, S) where True = allowed.
    """
    total = n_prefix + n_suffix
    mask = jnp.ones((total, total), dtype=jnp.bool_)
    # Prefix tokens cannot attend to suffix tokens
    mask = mask.at[:n_prefix, n_prefix:].set(False)
    return mask[None, None, :, :]


class GemmaActionExpert(nnx.Module):
    """Action denoising transformer (~311M params).

    Architecture:
      action_in_proj(7 -> d_model) + timestep_mlp(1 -> d_model)
      12 x GemmaDecoderLayer (GQA, SwiGLU)
      RMSNorm -> action_out_proj(d_model -> 6)  [continuous: pos+orn]
      RMSNorm -> gripper_head(d_model -> 1)      [discrete: open/close]
    """

    def __init__(
        self,
        d_model: int = 1536,
        n_layers: int = 12,
        d_ff: int = 4096,
        n_heads: int = 12,
        n_kv_heads: int = 4,
        head_dim: int = 128,
        action_dim: int = 7,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.action_dim = action_dim
        self.continuous_dim = action_dim - 1  # 6: pos + orn
        self.gripper_dim = 1

        # Action input projection (continuous dims only for flow matching)
        self.action_in_proj = nnx.Linear(self.continuous_dim, d_model, use_bias=False, rngs=rngs)

        # Timestep MLP: scalar -> d_model
        self.timestep_proj = nnx.Linear(1, d_model, use_bias=True, rngs=rngs)
        self.timestep_out = nnx.Linear(d_model, d_model, use_bias=True, rngs=rngs)

        # Transformer layers
        self.layers = [
            GemmaDecoderLayer(d_model, d_ff, n_heads, n_kv_heads, head_dim, rngs=rngs) for _ in range(n_layers)
        ]

        # Output heads
        self.norm = RMSNorm(d_model, rngs=rngs)
        self.action_out_proj = nnx.Linear(d_model, self.continuous_dim, use_bias=False, rngs=rngs)
        self.gripper_head = nnx.Linear(d_model, 1, use_bias=False, rngs=rngs)

    def _timestep_embed(self, t: jax.Array) -> jax.Array:
        """Embed scalar timestep. t: (B, T, 1) or (B, 1)."""
        h = self.timestep_proj(t)
        h = nnx.silu(h)
        return self.timestep_out(h)

    def forward_joint(
        self,
        obs_embed: jax.Array,
        noisy_actions: jax.Array,
        timestep: jax.Array,
        gripper_gt: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Training forward with prefix-LM joint attention.

        Args:
            obs_embed: (B, n_obs, d_model) from VLM
            noisy_actions: (B, T, 6) continuous dims only
            timestep: (B, T, 1) per-token timesteps
            gripper_gt: (B, T, 1) ground-truth gripper for supervision

        Returns:
            velocity_pred: (B, T, 6) predicted velocity for continuous dims
            gripper_logits: (B, T, 1) gripper open/close logits
        """
        b, n_obs, _ = obs_embed.shape
        _, n_act, _ = noisy_actions.shape

        # Timestep embedding
        t_embed = self._timestep_embed(timestep)  # (B, T, d_model)
        t_zero = self._timestep_embed(jnp.zeros((b, n_obs, 1)))  # obs gets t=0

        # Token embeddings
        obs_tokens = obs_embed + t_zero
        act_tokens = self.action_in_proj(noisy_actions) + t_embed

        # Joint sequence
        tokens = jnp.concatenate([obs_tokens, act_tokens], axis=1)
        mask = make_prefix_lm_mask(n_obs, n_act)

        # Transformer
        for layer in self.layers:
            tokens, _ = layer(tokens, mask=mask)

        # Extract action tokens
        action_hidden = self.norm(tokens[:, n_obs:, :])
        velocity_pred = self.action_out_proj(action_hidden)
        gripper_logits = self.gripper_head(action_hidden)

        return velocity_pred, gripper_logits

    def build_prefix_kv_cache(self, obs_embed: jax.Array) -> list[tuple[jax.Array, jax.Array]]:
        """Cache KV from obs tokens for inference."""
        b, n_obs, _ = obs_embed.shape
        t_zero = self._timestep_embed(jnp.zeros((b, n_obs, 1)))
        tokens = obs_embed + t_zero

        kv_cache = []
        for layer in self.layers:
            tokens, kv = layer(tokens)
            kv_cache.append(kv)
        return kv_cache

    def forward_cached(
        self,
        noisy_actions: jax.Array,
        timestep: jax.Array,
        prefix_kv_cache: list[tuple[jax.Array, jax.Array]],
    ) -> tuple[jax.Array, jax.Array]:
        """Denoising forward with cached prefix KV."""
        t_embed = self._timestep_embed(timestep)
        tokens = self.action_in_proj(noisy_actions) + t_embed

        for layer, prefix_kv in zip(self.layers, prefix_kv_cache):
            tokens, _ = layer(tokens, prefix_kv=prefix_kv)

        hidden = self.norm(tokens)
        return self.action_out_proj(hidden), self.gripper_head(hidden)

    def denoise(
        self,
        obs_embed: jax.Array,
        chunk_size: int = 50,
        n_steps: int = 10,
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Full Euler denoising: t=1 (noise) -> t=0 (clean).

        Returns:
            actions_continuous: (B, chunk_size, 6)
            gripper_logits: (B, chunk_size, 1)
        """
        b = obs_embed.shape[0]
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Build prefix KV cache
        kv_cache = self.build_prefix_kv_cache(obs_embed)

        # Start from noise
        x_t = jax.random.normal(rng, (b, chunk_size, self.continuous_dim))
        dt = -1.0 / n_steps

        for step in range(n_steps):
            t_val = 1.0 + step * dt
            t = jnp.full((b, chunk_size, 1), t_val)
            velocity, gripper_logits = self.forward_cached(x_t, t, kv_cache)
            x_t = x_t + velocity * dt

        return x_t, gripper_logits
