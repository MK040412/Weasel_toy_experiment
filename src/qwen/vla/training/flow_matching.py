"""Flow matching scheduler — openpi0.5 convention.

Convention (openpi):
  - t=1 is pure noise, t=0 is clean data
  - x_t = t * noise + (1-t) * actions
  - target velocity = noise - actions
  - Inference: Euler from t=1 -> t=0, dt = -1/n_steps
  - t ~ Beta(1.5, 1.0), affine-transformed to [t_min, t_max]

Training-Time RTC (arXiv 2512.05964):
  - Simulates inference delay by fixing prefix actions to GT (t=0)
  - Loss computed only on postfix tokens
  - Per-token timestep: prefix=0 (clean), postfix=sampled t
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qwen.vla.config import FlowMatchingConfig


class FlowMatchingScheduler:
    """Conditional flow matching with openpi0.5 convention."""

    def __init__(self, config: FlowMatchingConfig | None = None):
        if config is None:
            config = FlowMatchingConfig()
        self.config = config

    def sample_training_pair(
        self,
        x_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample a noisy input, timestep, and target velocity for training.

        Args:
            x_1: Ground truth actions (normalized), (B, chunk_size, action_dim).

        Returns:
            x_t: Noisy actions at time t, (B, chunk_size, action_dim).
            t: Per-token timestep, (B, T, 1).
            target_velocity: noise - actions, (B, chunk_size, action_dim).
            loss_mask: (B, T) mask where 1=compute loss, or None if no masking.
        """
        B, T, _ = x_1.shape
        device = x_1.device
        dtype = x_1.dtype

        t_scalar = (
            torch.distributions.Beta(self.config.beta_a, self.config.beta_b)
            .sample((B, 1))
            .to(device=device, dtype=dtype)
        )
        t_scalar = t_scalar * (self.config.t_max - self.config.t_min) + self.config.t_min

        noise = torch.randn_like(x_1)
        target_velocity = noise - x_1

        sd = self.config.simulated_delay
        if sd <= 0:
            t = t_scalar.unsqueeze(-1).expand(B, T, 1)
            t_expand = t
            x_t = t_expand * noise + (1 - t_expand) * x_1
            return x_t, t, target_velocity, None

        # --- Training-Time RTC ---
        delays = torch.arange(sd, device=device, dtype=dtype)
        weights = torch.exp(delays.flip(0))
        probs = weights / weights.sum()
        d_indices = torch.multinomial(probs.expand(B, -1), num_samples=1).squeeze(-1)

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        prefix_mask = positions < d_indices.unsqueeze(1)

        t = t_scalar.unsqueeze(-1).expand(B, T, 1).clone()
        t[prefix_mask] = 0.0

        t_expand = t
        x_t = t_expand * noise + (1 - t_expand) * x_1
        loss_mask = (~prefix_mask).to(dtype)

        return x_t, t, target_velocity, loss_mask

    def compute_loss(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE loss between predicted and target velocity."""
        if loss_mask is None:
            return F.mse_loss(predicted_velocity, target_velocity)

        sq_err = (predicted_velocity - target_velocity).pow(2)
        mask = loss_mask.unsqueeze(-1)
        masked_err = (sq_err * mask).sum()
        n_elements = mask.sum() * sq_err.shape[-1]
        return masked_err / n_elements.clamp(min=1)
