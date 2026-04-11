"""VLA Trainer: TPU HBM-resident cached training.

All cached embeddings stay on TPU HBM. Zero numpy<->JAX roundtrips during training.
Timestep sampling inside JIT to prevent retracing.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from PIL import Image as PILImage
from transformers import AutoTokenizer

from qwen.vla.data.lerobot_calvin import CalvinDataset
from qwen.vla.models.vla import VLAPolicy
from qwen.vla.training import flow_matching as fm

# Qwen3-VL special tokens
IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653


def _prepare_vision_inputs(image: np.ndarray, language: str, tokenizer, image_size: int = 320):
    """Prepare VLM inputs without PyTorch. image_size must be divisible by 32."""
    if image.shape[0] != image_size or image.shape[1] != image_size:
        pil = PILImage.fromarray((image * 255).astype(np.uint8))
        pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
        image = np.array(pil, dtype=np.float32) / 255.0

    patch_size, temporal_patches, merge_size = 16, 2, 2
    grid_h = image_size // patch_size
    grid_w = image_size // patch_size
    merged_h = grid_h // merge_size
    merged_w = grid_w // merge_size
    n_vision_tokens = merged_h * merged_w

    text_tokens = tokenizer.encode(language, add_special_tokens=False)
    input_ids = [VISION_START_ID] + [IMAGE_TOKEN_ID] * n_vision_tokens + [VISION_END_ID] + text_tokens
    input_ids = jnp.array([input_ids], dtype=jnp.int32)

    img_doubled = np.stack([image, image], axis=0)
    patches = []
    for h in range(0, image_size, patch_size):
        for w in range(0, image_size, patch_size):
            patch = img_doubled[:temporal_patches, h : h + patch_size, w : w + patch_size, :]
            patches.append(patch.transpose(3, 0, 1, 2).flatten())
    pixel_values = jnp.array(np.array(patches, dtype=np.float32))
    grid_thw = jnp.array([[1, grid_h, grid_w]], dtype=jnp.int32)
    token_type_ids = (input_ids == IMAGE_TOKEN_ID).astype(jnp.int32)

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
        "token_type_ids": token_type_ids,
    }


def _pad_seq(x: jax.Array, target_len: int) -> jax.Array:
    """Pad sequence dim (axis=0 for 2D, axis=1 for 3D) to target_len."""
    if x.ndim == 2:
        pad_len = target_len - x.shape[0]
        if pad_len <= 0:
            return x[:target_len, :]
        return jnp.pad(x, ((0, pad_len), (0, 0)))
    pad_len = target_len - x.shape[1]
    if pad_len <= 0:
        return x[:, :target_len, :]
    return jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))


class VLATrainer:
    """TPU HBM-resident VLA trainer. Zero host-device transfers during training."""

    def __init__(self, policy: VLAPolicy, dataset: CalvinDataset, config: dict):
        self.policy = policy
        self.dataset = dataset
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("vlm_model_id", "Qwen/Qwen3-VL-2B-Instruct"))
        # Will be set after caching
        self.cached_obs: jax.Array | None = None
        self.cached_actions: jax.Array | None = None
        self.cached_gripper: jax.Array | None = None
        self.n_cached: int = 0

    def cache_vlm_embeddings(self):
        """Cache VLM embeddings directly on TPU HBM. No numpy intermediary."""
        n = len(self.dataset)
        print(f"Caching VLM embeddings for {n} samples on TPU HBM...")

        obs_list, act_list, grip_list = [], [], []

        for i in range(n):
            sample = self.dataset[i]
            top_image = sample["images"][0]

            vlm_inputs = _prepare_vision_inputs(top_image, sample["language"], self.tokenizer)
            hidden = self.policy.vlm.get_hidden_states(
                vlm_inputs["input_ids"],
                vlm_inputs["pixel_values"],
                vlm_inputs["image_grid_thw"],
                vlm_inputs["token_type_ids"],
            )
            # obs_proj output stays as JAX array on TPU, squeeze batch dim
            obs_embed = self.policy.obs_proj(hidden)[0]  # (seq, 1536)
            obs_list.append(obs_embed)
            act_list.append(jnp.array(sample["actions_continuous"])[None])
            grip_list.append(jnp.array(sample["gripper"])[None])

            if (i + 1) % 5 == 0 or i == n - 1:
                print(f"  Cached {i + 1}/{n}")

        # Pad obs to uniform seq_len, stack into single array on TPU HBM
        max_seq = max(o.shape[0] for o in obs_list)
        self.cached_obs = jnp.stack([_pad_seq(o, max_seq) for o in obs_list])  # (N, S, 1536)
        self.cached_actions = jnp.concatenate(act_list, axis=0)  # (N, T, 6)
        self.cached_gripper = jnp.concatenate(grip_list, axis=0)  # (N, T, 1)
        self.n_cached = n

        obs_mb = self.cached_obs.nbytes / 1e6
        act_mb = self.cached_actions.nbytes / 1e6
        print(f"  HBM usage: obs={obs_mb:.1f}MB, actions={act_mb:.1f}MB")
        print(f"  Shapes: obs={self.cached_obs.shape}, actions={self.cached_actions.shape}")

    def train(self):
        self.cache_vlm_embeddings()

        print("\n=== Stage 1: Training action expert ===")
        self._train_stage(
            epochs=self.config.get("stage1_epochs", 30),
            lr=self.config.get("lr", 5e-5),
            tag="s1",
        )

        print("\n=== Stage 2: Fine-tuning ===")
        self._train_stage(
            epochs=self.config.get("stage2_epochs", 20),
            lr=self.config.get("lr", 5e-5) * 0.1,
            tag="s2",
        )
        print("\nTraining complete!")

    def _train_stage(self, epochs: int, lr: float, tag: str):
        chunk_size = self.config.get("chunk_size", 50)
        simulated_delay = self.config.get("simulated_delay", 0)
        log_interval = self.config.get("log_interval", 10)
        save_interval = self.config.get("save_interval", 500)

        total_steps = max(self.n_cached * epochs, 1)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=min(100, total_steps // 10),
            decay_steps=total_steps,
            end_value=lr * 0.01,
        )
        optimizer = nnx.Optimizer(self.policy, optax.adamw(lr_schedule, weight_decay=0.01))

        # All data arrays for direct TPU indexing
        all_obs = self.cached_obs
        all_acts = self.cached_actions
        all_grip = self.cached_gripper

        # Build train_step with everything inside JIT (including RNG + timestep sampling)
        @nnx.jit
        def train_step(policy, optimizer, obs, acts, grip, rng):
            rng, t_rng, n_rng = jax.random.split(rng, 3)
            noise = jax.random.normal(n_rng, acts.shape)

            if simulated_delay > 0:
                timesteps, loss_mask = fm.sample_timesteps_rtc(t_rng, 1, chunk_size, simulated_delay)
            else:
                timesteps = fm.sample_timesteps(t_rng, 1, chunk_size)
                loss_mask = jnp.ones_like(acts[..., :1])

            def loss_fn(policy):
                noisy = fm.make_noisy(acts, noise, timesteps)
                vel_pred, grip_logits = policy.action_expert.forward_joint(obs, noisy, timesteps)
                vel_target = fm.velocity_target(acts, noise)
                l_vel = fm.compute_loss(vel_pred, vel_target, loss_mask)
                l_grip = fm.gripper_loss(grip_logits, grip, loss_mask)
                return l_vel + 0.1 * l_grip

            loss, grads = nnx.value_and_grad(loss_fn)(policy)
            optimizer.update(grads)
            return loss, rng

        rng = jax.random.PRNGKey(self.config.get("seed", 42))
        global_step = 0
        accum_loss = 0.0
        t_start = time.time()

        for epoch in range(epochs):
            rng, shuffle_rng = jax.random.split(rng)
            indices = jax.random.permutation(shuffle_rng, self.n_cached)

            for raw_idx in indices:
                idx = int(raw_idx)
                # Pure TPU HBM indexing — zero host transfer
                obs = all_obs[idx : idx + 1]
                acts = all_acts[idx : idx + 1]
                grip = all_grip[idx : idx + 1]

                loss, rng = train_step(self.policy, optimizer, obs, acts, grip, rng)
                global_step += 1
                accum_loss += float(loss)

                if global_step % log_interval == 0:
                    avg = accum_loss / log_interval
                    elapsed = time.time() - t_start
                    steps_per_sec = global_step / elapsed
                    print(
                        f"  [{tag}] step {global_step}, ep {epoch + 1}/{epochs},"
                        f" loss={avg:.4f}, {steps_per_sec:.1f} steps/s"
                    )
                    accum_loss = 0.0

                if global_step % save_interval == 0:
                    self._save_checkpoint(f"checkpoint_{tag}_{global_step}")

        self._save_checkpoint(f"checkpoint_{tag}_final")

    def _save_checkpoint(self, name: str):
        output_dir = self.config.get("output_dir", "result/vla")
        save_path = os.path.join(output_dir, f"{name}.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        flat = {}
        # Get state for obs_proj and action_expert separately
        obs_state = nnx.state(self.policy.obs_proj)
        expert_state = nnx.state(self.policy.action_expert)
        leaves = jax.tree.leaves(obs_state) + jax.tree.leaves(expert_state)
        for i, v in enumerate(leaves):
            flat[f"p{i}"] = np.array(v)
        flat["q01"] = self.dataset.q01
        flat["q99"] = self.dataset.q99
        np.savez(save_path, **flat)
        print(f"  Saved: {save_path}")
