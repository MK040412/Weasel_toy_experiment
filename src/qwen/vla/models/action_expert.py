"""GemmaActionExpert: pi0-style action denoising transformer in Flax NNX.

All 7 action dims (pos + orn + gripper) through flow matching (pi0 convention).
No separate gripper head — gripper is denoised as continuous value.
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
    mask = mask.at[:n_prefix, n_prefix:].set(False)
    return mask[None, None, :, :]


class GemmaActionExpert(nnx.Module):
    """Action denoising transformer (~311M params).

    Architecture:
      action_in_proj(7 -> d_model) + timestep_mlp(1 -> d_model)
      12 x GemmaDecoderLayer (GQA, SwiGLU)
      RMSNorm -> action_out_proj(d_model -> 7)
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
        proprio_dim: int = 15,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.action_dim = action_dim

        self.action_in_proj = nnx.Linear(action_dim, d_model, use_bias=False, rngs=rngs)
        self.proprio_proj = nnx.Linear(proprio_dim, d_model, use_bias=False, rngs=rngs)

        # Timestep MLP: scalar -> d_model
        self.timestep_proj = nnx.Linear(1, d_model, use_bias=True, rngs=rngs)
        self.timestep_out = nnx.Linear(d_model, d_model, use_bias=True, rngs=rngs)

        self.layers = [
            GemmaDecoderLayer(d_model, d_ff, n_heads, n_kv_heads, head_dim, rngs=rngs) for _ in range(n_layers)
        ]

        self.norm = RMSNorm(d_model, rngs=rngs)
        self.action_out_proj = nnx.Linear(d_model, action_dim, use_bias=False, rngs=rngs)

    def _timestep_embed(self, t: jax.Array) -> jax.Array:
        h = self.timestep_proj(t)
        h = nnx.silu(h)
        return self.timestep_out(h)

    def forward_joint(
        self,
        obs_embed: jax.Array,
        noisy_actions: jax.Array,
        timestep: jax.Array,
        proprio: jax.Array | None = None,
    ) -> jax.Array:
        """Training forward with prefix-LM joint attention.

        Args:
            obs_embed: (B, n_obs, d_model) from VLM
            noisy_actions: (B, T, 7) all action dims
            timestep: (B, T, 1) per-token timesteps
            proprio: (B, 1, proprio_dim) robot proprioceptive state

        Returns:
            velocity_pred: (B, T, 7) predicted velocity
        """
        b, n_obs, _ = obs_embed.shape
        _, n_act, _ = noisy_actions.shape

        t_embed = self._timestep_embed(timestep)

        # Build prefix: [proprio_token, obs_tokens]
        if proprio is not None:
            proprio_token = self.proprio_proj(proprio)  # (B, 1, d_model)
            t_zero = self._timestep_embed(jnp.zeros((b, n_obs + 1, 1)))
            obs_tokens = jnp.concatenate([proprio_token, obs_embed], axis=1) + t_zero
            n_prefix = n_obs + 1
        else:
            t_zero = self._timestep_embed(jnp.zeros((b, n_obs, 1)))
            obs_tokens = obs_embed + t_zero
            n_prefix = n_obs

        act_tokens = self.action_in_proj(noisy_actions) + t_embed

        tokens = jnp.concatenate([obs_tokens, act_tokens], axis=1)
        mask = make_prefix_lm_mask(n_prefix, n_act)

        for layer in self.layers:
            tokens, _ = layer(tokens, mask=mask)

        action_hidden = self.norm(tokens[:, n_prefix:, :])
        return self.action_out_proj(action_hidden)

    def build_prefix_kv_cache(
        self,
        obs_embed: jax.Array,
        proprio: jax.Array | None = None,
    ) -> list[tuple[jax.Array, jax.Array]]:
        b, n_obs, _ = obs_embed.shape
        if proprio is not None:
            proprio_token = self.proprio_proj(proprio)
            tokens = jnp.concatenate([proprio_token, obs_embed], axis=1)
            n_prefix = n_obs + 1
        else:
            tokens = obs_embed
            n_prefix = n_obs
        t_zero = self._timestep_embed(jnp.zeros((b, n_prefix, 1)))
        tokens = tokens + t_zero

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
    ) -> jax.Array:
        """Denoising forward with cached prefix KV. Returns velocity (B, T, 7)."""
        t_embed = self._timestep_embed(timestep)
        tokens = self.action_in_proj(noisy_actions) + t_embed

        for layer, prefix_kv in zip(self.layers, prefix_kv_cache):
            tokens, _ = layer(tokens, prefix_kv=prefix_kv)

        hidden = self.norm(tokens)
        return self.action_out_proj(hidden)

    def denoise(
        self,
        obs_embed: jax.Array,
        proprio: jax.Array | None = None,
        chunk_size: int = 50,
        n_steps: int = 10,
        rng: jax.Array | None = None,
    ) -> jax.Array:
        """Full Euler denoising: t=1 (noise) -> t=0 (clean).

        Returns: actions (B, chunk_size, 7)
        """
        b = obs_embed.shape[0]
        if rng is None:
            rng = jax.random.PRNGKey(0)

        kv_cache = self.build_prefix_kv_cache(obs_embed, proprio)

        x_t = jax.random.normal(rng, (b, chunk_size, self.action_dim))
        dt = -1.0 / n_steps

        for step in range(n_steps):
            t_val = 1.0 + step * dt
            t = jnp.full((b, chunk_size, 1), t_val)
            velocity = self.forward_cached(x_t, t, kv_cache)
            x_t = x_t + velocity * dt

        return x_t
