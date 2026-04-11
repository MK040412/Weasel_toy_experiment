"""VLA Trainer: config-driven, receives pre-computed VLMCache.

Responsibilities: training loop only. No VLM caching (see vlm_cache.py).
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from qwen.vla.config import FlowMatchingConfig, PipelineConfig, TrainingConfig
from qwen.vla.training import flow_matching as fm
from qwen.vla.training.vlm_cache import VLMCache


class VLATrainer:
    """Config-driven trainer. Receives VLMCache, not raw data."""

    def __init__(self, policy, cache: VLMCache, config: PipelineConfig, dataset=None):
        self.policy = policy
        self.cache = cache
        self.config = config
        self.dataset = dataset  # for checkpoint quantiles only

    def train(self):
        tc = self.config.training
        fc = self.config.flow_matching
        n = self.cache.n_samples

        print(f"\n=== Training (bf16), {tc.epochs} epochs, lr={tc.lr} ===")
        self._train_loop(
            epochs=tc.epochs,
            lr=tc.lr,
            batch_size=min(tc.batch_size, n),
            chunk_size=self.config.env.chunk_size,
            simulated_delay=fc.simulated_delay,
            log_interval=tc.log_interval,
            seed=tc.seed,
        )
        print("\nTraining complete!")

    def _train_loop(self, epochs, lr, batch_size, chunk_size, simulated_delay, log_interval, seed):
        n = self.cache.n_samples
        steps_per_epoch = (n + batch_size - 1) // batch_size
        total_steps = epochs * steps_per_epoch

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=min(10, total_steps // 5),
            decay_steps=total_steps,
            end_value=lr * 0.01,
        )
        optimizer = nnx.Optimizer(self.policy, optax.adamw(lr_schedule, weight_decay=self.config.training.weight_decay))

        all_obs = self.cache.obs
        all_acts = self.cache.actions
        all_proprio = self.cache.proprio

        @nnx.jit
        def train_step(policy, optimizer, obs, acts, proprio, rng):
            rng, t_rng, n_rng = jax.random.split(rng, 3)
            b = acts.shape[0]
            noise = jax.random.normal(n_rng, acts.shape)

            if simulated_delay > 0:
                timesteps, loss_mask = fm.sample_timesteps_rtc(t_rng, b, chunk_size, simulated_delay)
            else:
                timesteps = fm.sample_timesteps(t_rng, b, chunk_size)
                loss_mask = jnp.ones((b, chunk_size, 1))

            def loss_fn(policy):
                obs_bf = obs.astype(jnp.bfloat16)
                noisy = fm.make_noisy(acts, noise, timesteps).astype(jnp.bfloat16)
                ts_bf = timesteps.astype(jnp.bfloat16)
                proprio_bf = proprio.astype(jnp.bfloat16)
                vel_pred = policy.action_expert.forward_joint(obs_bf, noisy, ts_bf, proprio_bf)
                vel_pred = vel_pred.astype(jnp.float32)
                vel_target = fm.velocity_target(acts, noise)
                return fm.compute_loss(vel_pred, vel_target, loss_mask)

            loss, grads = nnx.value_and_grad(loss_fn)(policy)
            optimizer.update(grads)
            return loss, rng

        rng = jax.random.PRNGKey(seed)
        t_start = time.time()

        print(f"  batch={batch_size}, steps/epoch={steps_per_epoch}, total={total_steps}")

        for epoch in range(epochs):
            rng, shuffle_rng = jax.random.split(rng)
            indices = jax.random.permutation(shuffle_rng, n)

            for j in range(steps_per_epoch):
                start = j * batch_size
                batch_idx = indices[start : start + batch_size]
                if batch_idx.shape[0] < batch_size:
                    pad_len = batch_size - batch_idx.shape[0]
                    batch_idx = jnp.concatenate([batch_idx, batch_idx[:pad_len]])
                obs = all_obs[batch_idx]
                acts = all_acts[batch_idx]
                proprio = all_proprio[batch_idx]
                loss, rng = train_step(self.policy, optimizer, obs, acts, proprio, rng)

            if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
                epoch_loss = float(loss)
                elapsed = time.time() - t_start
                eps = (epoch + 1) / elapsed
                print(f"  ep {epoch + 1}/{epochs}, loss={epoch_loss:.4f}, {eps:.1f} ep/s")

        self._save_checkpoint()

    def _save_checkpoint(self):
        output_dir = self.config.training.output_dir
        save_path = os.path.join(output_dir, "checkpoint_train_final.npz")
        os.makedirs(output_dir, exist_ok=True)

        flat = {}
        obs_leaves = jax.tree.leaves(nnx.state(self.policy.obs_proj))
        expert_leaves = jax.tree.leaves(nnx.state(self.policy.action_expert))
        for i, v in enumerate(obs_leaves + expert_leaves):
            flat[f"p{i}"] = np.array(v)

        if self.dataset is not None:
            flat["q01"] = self.dataset.q01
            flat["q99"] = self.dataset.q99
            flat["q01_state"] = self.dataset.q01_state
            flat["q99_state"] = self.dataset.q99_state

        np.savez(save_path, **flat)
        print(f"  Saved: {save_path}")
