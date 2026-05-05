"""VLA Trainer: pmap data-parallel across all TPU devices.

Uses jax.pmap with pmean for 4-device data parallelism.
Cache stays in host RAM (numpy); per-batch transfer via pmap sharded input.
"""

import csv
import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from qwen.vla.config import PipelineConfig
from qwen.vla.training import flow_matching as fm
from qwen.vla.training.vlm_cache import VLMCache


class VLATrainer:
    """Data-parallel trainer using jax.pmap across all TPU devices."""

    def __init__(self, policy, cache: VLMCache, config: PipelineConfig, dataset=None):
        self.policy = policy
        self.cache = cache
        self.config = config
        self.dataset = dataset

    def train(self):
        tc = self.config.training
        fc = self.config.flow_matching
        n = self.cache.n_samples
        n_dev = jax.local_device_count()

        print(f"\n=== Training (bf16, pmap {n_dev}-dev local / {jax.device_count()}-dev global), {tc.epochs} epochs, lr={tc.lr} ===")
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
        n_dev = jax.local_device_count()

        # Round batch_size up to multiple of n_dev
        if batch_size % n_dev != 0:
            batch_size = ((batch_size + n_dev - 1) // n_dev) * n_dev
        per_dev = batch_size // n_dev

        steps_per_epoch = (n + batch_size - 1) // batch_size
        total_steps = epochs * steps_per_epoch

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=min(10, total_steps // 5),
            decay_steps=total_steps,
            end_value=lr * 0.01,
        )

        # Use flat optax directly (not nnx.Optimizer) for pmap compatibility
        tx = optax.adamw(lr_schedule, weight_decay=self.config.training.weight_decay)

        # Extract trainable state from policy (obs_proj + action_expert only)
        graph_def = nnx.graphdef(self.policy)
        state = nnx.state(self.policy)
        opt_state = tx.init(state)

        # Replicate across devices
        rep_state = jax.device_put_replicated(state, jax.local_devices())
        rep_opt_state = jax.device_put_replicated(opt_state, jax.local_devices())

        # Cache in host RAM (numpy)
        all_obs = self.cache.obs
        all_acts = self.cache.actions
        all_proprio = self.cache.proprio

        @functools.partial(jax.pmap, axis_name="batch")
        def pmap_step(state, opt_state, obs, acts, proprio, rng):
            rng, t_rng, n_rng = jax.random.split(rng, 3)
            b = acts.shape[0]
            noise = jax.random.normal(n_rng, acts.shape)

            if simulated_delay > 0:
                timesteps, loss_mask = fm.sample_timesteps_rtc(t_rng, b, chunk_size, simulated_delay)
            else:
                timesteps = fm.sample_timesteps(t_rng, b, chunk_size)
                loss_mask = jnp.ones((b, chunk_size, 1))

            def loss_fn(s):
                policy = nnx.merge(graph_def, s)
                obs_bf = obs.astype(jnp.bfloat16)
                noisy = fm.make_noisy(acts, noise, timesteps).astype(jnp.bfloat16)
                ts_bf = timesteps.astype(jnp.bfloat16)
                proprio_bf = proprio.astype(jnp.bfloat16)
                vel_pred = policy.action_expert.forward_joint(obs_bf, noisy, ts_bf, proprio_bf)
                vel_pred = vel_pred.astype(jnp.float32)
                vel_target = fm.velocity_target(acts, noise)
                return fm.compute_loss(vel_pred, vel_target, loss_mask)

            loss, grads = jax.value_and_grad(loss_fn)(state)
            # Average gradients across devices
            grads = jax.lax.pmean(grads, axis_name="batch")
            loss = jax.lax.pmean(loss, axis_name="batch")
            updates, new_opt_state = tx.update(grads, opt_state, state)
            new_state = optax.apply_updates(state, updates)
            return new_state, new_opt_state, loss, rng

        rng_keys = jax.random.split(jax.random.PRNGKey(seed), n_dev)
        t_start = time.time()

        print(f"  batch={batch_size} ({per_dev}/dev × {n_dev}), steps/epoch={steps_per_epoch}, total={total_steps}")

        # Training log CSV
        log_path = os.path.join(self.config.training.output_dir, "train_log.csv")
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["epoch", "step", "loss", "lr", "epoch_time_s", "samples_per_s"])
        global_step = 0

        def _prepare_batch(idx_slice):
            """Host gather → reshape → async device_put. Returns jax arrays (non-blocking)."""
            obs_np = all_obs[idx_slice].reshape(n_dev, per_dev, *all_obs.shape[1:])
            acts_np = all_acts[idx_slice].reshape(n_dev, per_dev, *all_acts.shape[1:])
            proprio_np = all_proprio[idx_slice].reshape(n_dev, per_dev, *all_proprio.shape[1:])
            return jnp.array(obs_np), jnp.array(acts_np), jnp.array(proprio_np)

        def _get_batch_idx(indices, j):
            start = j * batch_size
            batch_idx = indices[start : start + batch_size]
            if batch_idx.shape[0] < batch_size:
                pad_len = batch_size - batch_idx.shape[0]
                batch_idx = np.concatenate([batch_idx, batch_idx[:pad_len]])
            return batch_idx

        for epoch in range(epochs):
            perm_rng = jax.random.PRNGKey(seed + epoch)
            indices = np.array(jax.random.permutation(perm_rng, n))
            epoch_t0 = time.time()

            # Prefetch first batch (host gather + async device_put)
            prefetched = _prepare_batch(_get_batch_idx(indices, 0))

            for j in range(steps_per_epoch):
                # Consume current prefetch, kick off next prefetch concurrently
                obs, acts, proprio = prefetched
                if j + 1 < steps_per_epoch:
                    # Prefetch next batch: host gather runs on CPU, jnp.array is async
                    # TPU continues computing current step while transfer happens
                    prefetched = _prepare_batch(_get_batch_idx(indices, j + 1))

                rep_state, rep_opt_state, loss, rng_keys = pmap_step(
                    rep_state, rep_opt_state, obs, acts, proprio, rng_keys
                )
                global_step += 1

            epoch_time = time.time() - epoch_t0
            if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
                # pmap returns per-device loss; take device 0 (all same after pmean)
                epoch_loss = float(loss[0])
                elapsed = time.time() - t_start
                eps = (epoch + 1) / elapsed
                sps = n / epoch_time
                current_lr = float(lr_schedule(global_step))
                print(f"  ep {epoch + 1}/{epochs}, loss={epoch_loss:.4f}, lr={current_lr:.2e}, "
                      f"{eps:.1f} ep/s, {sps:.0f} samp/s")
                log_writer.writerow([epoch + 1, global_step, f"{epoch_loss:.6f}",
                                     f"{current_lr:.2e}", f"{epoch_time:.2f}", f"{sps:.0f}"])
                log_file.flush()

        log_file.close()
        print(f"  Log saved: {log_path}")

        # Unreplicate state (take device 0) and save checkpoint
        final_state = jax.tree.map(lambda x: x[0], rep_state)
        nnx.update(self.policy, final_state)
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
