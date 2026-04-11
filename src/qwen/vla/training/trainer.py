"""VLA Trainer: 2-stage training with VLM embedding caching on TPU v4-8.

Stage 1: Cache VLM embeddings once, train obs_proj + action_expert.
Stage 2: Fine-tune with lower LR.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from transformers import AutoProcessor

from qwen.vla.data.lerobot_calvin import CalvinDataset
from qwen.vla.models.vla import VLAPolicy
from qwen.vla.training import flow_matching as fm


class VLATrainer:
    """2-stage VLA trainer with VLM embedding caching."""

    def __init__(
        self,
        policy: VLAPolicy,
        dataset: CalvinDataset,
        config: dict,
    ):
        self.policy = policy
        self.dataset = dataset
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.get("vlm_model_id", "Qwen/Qwen3-VL-2B-Instruct"))

    def _prepare_vlm_inputs(self, sample: dict) -> dict:
        """Prepare VLM inputs from dataset sample using HF processor."""
        language = sample["language"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["images"][0]},  # top camera
                    {"type": "text", "text": language},
                ],
            }
        ]

        # Use processor for tokenization + image preprocessing
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="np"
        )

        result = {"input_ids": jnp.array(inputs["input_ids"])}
        if "pixel_values" in inputs:
            result["pixel_values"] = jnp.array(inputs["pixel_values"])
            result["image_grid_thw"] = jnp.array(inputs["image_grid_thw"])
            result["token_type_ids"] = (result["input_ids"] == self.policy.vlm.config.image_token_id).astype(jnp.int32)
        return result

    def cache_vlm_embeddings(self) -> list[dict]:
        """Cache VLM hidden states for all dataset samples (one-time forward)."""
        print(f"Caching VLM embeddings for {len(self.dataset)} samples...")
        cached = []

        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            # Convert images from numpy (H,W,3) float to PIL for processor
            from PIL import Image

            pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in sample["images"]]
            sample_with_pil = {**sample, "images": pil_images}
            vlm_inputs = self._prepare_vlm_inputs(sample_with_pil)

            hidden = self.policy.vlm.get_hidden_states(
                vlm_inputs["input_ids"],
                vlm_inputs.get("pixel_values"),
                vlm_inputs.get("image_grid_thw"),
                vlm_inputs.get("token_type_ids"),
            )

            # Store on CPU as numpy to save TPU HBM
            cached.append(
                {
                    "obs_embed": np.array(self.policy.obs_proj(hidden)),
                    "actions_continuous": sample["actions_continuous"],
                    "gripper": sample["gripper"],
                    "language": sample["language"],
                    "episode": sample["episode"],
                }
            )

            if (i + 1) % 50 == 0:
                print(f"  Cached {i + 1}/{len(self.dataset)}")

        print(f"  Done. {len(cached)} embeddings cached.")
        return cached

    def train(self):
        """Full training pipeline: cache -> stage 1 -> stage 2."""
        # Cache VLM embeddings
        cached = self.cache_vlm_embeddings()

        # Stage 1: Train obs_proj + action_expert
        print("\n=== Stage 1: Training action expert ===")
        self._train_stage(
            cached,
            epochs=self.config.get("stage1_epochs", 30),
            lr=self.config.get("lr", 5e-5),
            stage_name="s1",
        )

        # Stage 2: Fine-tune with lower LR
        print("\n=== Stage 2: Fine-tuning ===")
        self._train_stage(
            cached,
            epochs=self.config.get("stage2_epochs", 20),
            lr=self.config.get("lr", 5e-5) * 0.1,
            stage_name="s2",
        )

        print("\nTraining complete!")

    def _train_stage(self, cached: list[dict], epochs: int, lr: float, stage_name: str):
        """Train action expert with cached embeddings."""
        grad_accum = self.config.get("gradient_accumulation_steps", 8)
        chunk_size = self.config.get("chunk_size", 50)
        simulated_delay = self.config.get("simulated_delay", 0)
        log_interval = self.config.get("log_interval", 10)
        save_interval = self.config.get("save_interval", 500)
        flow_beta_a = self.config.get("beta_a", 1.5)
        flow_beta_b = self.config.get("beta_b", 1.0)

        total_steps = len(cached) * epochs // grad_accum
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=min(100, total_steps // 10),
            decay_steps=total_steps,
            end_value=lr * 0.01,
        )

        # Optimizer only for trainable params (obs_proj + action_expert)
        optimizer = nnx.Optimizer(self.policy, optax.adamw(lr_schedule, weight_decay=0.01))

        rng = jax.random.PRNGKey(self.config.get("seed", 42))
        global_step = 0
        accum_loss = 0.0
        accum_count = 0

        @nnx.jit
        def train_step(policy, optimizer, obs_embed, actions_cont, gripper, noise, timesteps, loss_mask):
            def loss_fn(policy):
                vel_pred, grip_logits = policy.action_expert.forward_joint(
                    obs_embed, actions_cont + timesteps * (noise - actions_cont), timesteps, gripper
                )
                vel_target = fm.velocity_target(actions_cont, noise)
                l_vel = fm.compute_loss(vel_pred, vel_target, loss_mask)
                l_grip = fm.gripper_loss(grip_logits, gripper, loss_mask)
                return l_vel + 0.1 * l_grip

            loss, grads = nnx.value_and_grad(loss_fn)(policy)
            optimizer.update(grads)
            return loss

        t_start = time.time()
        for epoch in range(epochs):
            # Shuffle
            rng, shuffle_rng = jax.random.split(rng)
            indices = jax.random.permutation(shuffle_rng, len(cached))

            for idx in indices:
                sample = cached[int(idx)]
                obs_embed = jnp.array(sample["obs_embed"])
                actions_cont = jnp.array(sample["actions_continuous"])[None]  # (1, T, 6)
                gripper = jnp.array(sample["gripper"])[None]  # (1, T, 1)

                rng, noise_rng, t_rng = jax.random.split(rng, 3)
                noise = jax.random.normal(noise_rng, actions_cont.shape)

                if simulated_delay > 0:
                    timesteps, loss_mask = fm.sample_timesteps_rtc(
                        t_rng, 1, chunk_size, simulated_delay, flow_beta_a, flow_beta_b
                    )
                else:
                    timesteps = fm.sample_timesteps(t_rng, 1, chunk_size, flow_beta_a, flow_beta_b)
                    loss_mask = None

                loss = train_step(self.policy, optimizer, obs_embed, actions_cont, gripper, noise, timesteps, loss_mask)
                accum_loss += float(loss)
                accum_count += 1

                if accum_count % grad_accum == 0:
                    global_step += 1
                    if global_step % log_interval == 0:
                        avg_loss = accum_loss / accum_count
                        elapsed = time.time() - t_start
                        print(
                            f"  [{stage_name}] step {global_step}, "
                            f"epoch {epoch + 1}/{epochs}, "
                            f"loss={avg_loss:.4f}, "
                            f"time={elapsed:.1f}s"
                        )
                        accum_loss = 0.0
                        accum_count = 0

                    if global_step % save_interval == 0:
                        self._save_checkpoint(f"checkpoint_{stage_name}_{global_step}")

        self._save_checkpoint(f"checkpoint_{stage_name}_final")

    def _save_checkpoint(self, name: str):
        """Save trainable state (obs_proj + action_expert)."""
        state = nnx.state(self.policy.obs_proj, self.policy.action_expert)
        save_path = os.path.join(self.config.get("output_dir", "result/vla"), f"{name}.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        flat = {}
        flat_state = jax.tree.leaves(state)
        keys = [str(i) for i in range(len(flat_state))]
        for k, v in zip(keys, flat_state):
            flat[k] = np.array(v)
        np.savez(save_path, **flat)
        print(f"  Saved checkpoint: {save_path}")
