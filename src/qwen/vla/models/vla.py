"""VLA Policy: Frozen Qwen3-VL encoder + GemmaActionExpert in JAX/Flax NNX.

Uses the existing JAX Qwen3-VL model as VLM backbone.
Obs projection: 2048 -> 1536 (VLM hidden -> action expert dim).
"""

import jax
from flax import nnx

from qwen.qwen3vl import modeling as qwen3vl
from qwen.vla.models.action_expert import GemmaActionExpert


class VLAPolicy(nnx.Module):
    """Vision-Language-Action policy.

    VLM (frozen) -> obs_proj -> GemmaActionExpert -> actions.
    """

    def __init__(
        self,
        vlm: qwen3vl.Qwen3VLForConditionalGeneration,
        vlm_hidden_dim: int = 2048,
        action_expert_config: dict | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.vlm = vlm
        self.vlm_hidden_dim = vlm_hidden_dim

        cfg = action_expert_config or {}
        d_model = cfg.get("d_model", 1536)

        self.obs_proj = nnx.Linear(vlm_hidden_dim, d_model, use_bias=False, rngs=rngs)
        self.action_expert = GemmaActionExpert(
            d_model=d_model,
            n_layers=cfg.get("n_layers", 12),
            d_ff=cfg.get("d_ff", 4096),
            n_heads=cfg.get("n_heads", 12),
            n_kv_heads=cfg.get("n_kv_heads", 4),
            head_dim=cfg.get("head_dim", 128),
            action_dim=cfg.get("action_dim", 7),
            rngs=rngs,
        )

    def encode_observations(
        self,
        input_ids: jax.Array,
        pixel_values: jax.Array | None = None,
        image_grid_thw: jax.Array | None = None,
        token_type_ids: jax.Array | None = None,
    ) -> jax.Array:
        """Encode images + language through frozen VLM, project to action expert dim.

        Returns: (B, seq_len, d_model)
        """
        hidden = self.vlm.get_hidden_states(input_ids, pixel_values, image_grid_thw, token_type_ids)
        return self.obs_proj(hidden)

    def predict_actions(
        self,
        obs_embed: jax.Array,
        chunk_size: int = 50,
        n_steps: int = 10,
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Predict actions via flow matching denoising.

        Returns:
            actions_continuous: (B, chunk_size, 6) pos + orn
            gripper_probs: (B, chunk_size, 1) probability of close
        """
        actions, gripper_logits = self.action_expert.denoise(obs_embed, chunk_size, n_steps, rng)
        gripper_probs = jax.nn.sigmoid(gripper_logits)
        return actions, gripper_probs
