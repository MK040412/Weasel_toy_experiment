"""Flow matching scheduler for VLA training (JAX, openpi0.5 convention).

t=1: pure noise, t=0: clean data.
Velocity target: noise - actions.
"""

import jax
import jax.numpy as jnp


def sample_timesteps(rng: jax.Array, batch_size: int, chunk_size: int, beta_a: float = 1.5, beta_b: float = 1.0):
    """Sample timesteps from Beta distribution, mapped to [t_min, t_max].

    Returns: (batch_size, chunk_size, 1) timesteps.
    """
    t_scalar = jax.random.beta(rng, beta_a, beta_b, shape=(batch_size,))
    t_scalar = 0.001 + t_scalar * (1.0 - 0.001)  # map to [0.001, 1.0]
    return jnp.broadcast_to(t_scalar[:, None, None], (batch_size, chunk_size, 1))


def sample_timesteps_rtc(
    rng: jax.Array,
    batch_size: int,
    chunk_size: int,
    simulated_delay: int,
    beta_a: float = 1.5,
    beta_b: float = 1.0,
):
    """Sample per-token timesteps with RTC (Recurrent Time Chunking).

    Prefix tokens get t=0 (clean), postfix tokens get sampled t.
    Returns: timesteps (B, T, 1), loss_mask (B, T, 1).
    """
    rng_t, rng_delay = jax.random.split(rng)

    # Sample base timestep
    t_scalar = jax.random.beta(rng_t, beta_a, beta_b, shape=(batch_size,))
    t_scalar = 0.001 + t_scalar * (1.0 - 0.001)

    # Sample delay length per batch item
    delay_len = jax.random.randint(rng_delay, (batch_size,), 1, simulated_delay + 1)
    delay_len = jnp.minimum(delay_len, chunk_size - 1)

    # Build per-token timesteps: prefix=0, postfix=t_scalar
    positions = jnp.arange(chunk_size)[None, :]  # (1, T)
    is_postfix = positions >= delay_len[:, None]  # (B, T)
    timesteps = jnp.where(is_postfix, t_scalar[:, None], 0.0)
    loss_mask = is_postfix.astype(jnp.float32)

    return timesteps[:, :, None], loss_mask[:, :, None]


def make_noisy(actions: jax.Array, noise: jax.Array, timesteps: jax.Array) -> jax.Array:
    """Create noisy actions: x_t = t * noise + (1-t) * actions."""
    return timesteps * noise + (1.0 - timesteps) * actions


def velocity_target(actions: jax.Array, noise: jax.Array) -> jax.Array:
    """Target velocity: noise - actions (openpi0.5 convention)."""
    return noise - actions


def compute_loss(
    velocity_pred: jax.Array,
    velocity_gt: jax.Array,
    loss_mask: jax.Array,
) -> jax.Array:
    """MSE loss on predicted velocity, masked (pass ones for no masking)."""
    sq_err = (velocity_pred - velocity_gt) ** 2 * loss_mask
    return sq_err.sum() / jnp.maximum(loss_mask.sum() * velocity_pred.shape[-1], 1.0)
