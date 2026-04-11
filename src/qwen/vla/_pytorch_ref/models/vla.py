"""VLAPolicy — VLM encoder + ActionExpert combined VLA policy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen.vla.config import ActionExpertConfig, VLMConfig
from qwen.vla.models.action_expert import GemmaActionExpert


def monkey_patch_patch_embed(model: nn.Module) -> None:
    """Replace the vision encoder's Conv3d patch_embed with an equivalent F.linear call."""
    pe = model.model.visual.patch_embed
    W = pe.proj.weight.data.reshape(pe.embed_dim, -1).clone()
    b = pe.proj.bias.data.clone()

    def patched_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = W.dtype
        flat = hidden_states.to(dtype=target_dtype).reshape(hidden_states.shape[0], -1)
        return F.linear(flat, W, b)

    pe.forward = patched_forward


class VLAPolicy(nn.Module):
    """VLM encoder + ActionExpert combined VLA policy.

    The VLM (Qwen3-VL) encodes multi-camera images + language instructions into
    observation embeddings. These are projected to the action expert's dimension
    and used as prefix KV cache for flow matching denoising.
    """

    def __init__(self, vlm_config: VLMConfig, expert_config: ActionExpertConfig):
        super().__init__()
        self.vlm_config = vlm_config
        self.expert_config = expert_config

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_config.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(vlm_config.model_id)

        # Trainable bridge: VLM hidden dim -> action expert d_model
        self.obs_proj = nn.Linear(vlm_config.hidden_dim, expert_config.d_model)

        self.action_expert = GemmaActionExpert(expert_config)

        # Freeze VLM if configured
        if vlm_config.freeze:
            for p in self.vlm.parameters():
                p.requires_grad_(False)

        # Patch vision encoder for efficiency
        monkey_patch_patch_embed(self.vlm)

    def vlm_forward_hidden(
        self,
        images: list,
        language: str,
    ) -> torch.Tensor:
        """VLM forward pass returning raw hidden states (before obs_proj).

        Used by Stage 1 training to cache VLM outputs.

        Returns:
            hidden: (1, seq_len, vlm_hidden_dim) last hidden state.
        """
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": language})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.vlm.device)

        with torch.no_grad():
            outputs = self.vlm(
                **inputs,
                output_hidden_states=True,
            )

        hidden = outputs.hidden_states[-1]
        del outputs
        torch.cuda.empty_cache()

        return hidden

    def encode_observations(
        self,
        images: list,
        language: str,
    ) -> torch.Tensor:
        """Encode multi-camera images + language into observation embeddings.

        Args:
            images: List of PIL images.
            language: Task instruction string.

        Returns:
            obs_embed: (B, n_tokens, d_model) projected observation embeddings.
        """
        hidden = self.vlm_forward_hidden(images, language)
        obs_embed = self.obs_proj(hidden)
        del hidden
        torch.cuda.empty_cache()
        return obs_embed

    def forward(
        self,
        images: list,
        language: str,
    ) -> dict:
        """Training forward: encode observations.

        Flow matching loss is computed in the trainer using forward_joint.
        """
        obs_embed = self.encode_observations(images, language)
        return {"obs_embed": obs_embed}

    @torch.inference_mode()
    def predict_actions(
        self,
        images: list,
        language: str,
        chunk_size: int = 50,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Inference: encode observations -> denoise -> actions.

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        obs_embed = self.encode_observations(images, language)
        return self.action_expert.denoise(obs_embed, chunk_size=chunk_size, n_steps=n_steps)
