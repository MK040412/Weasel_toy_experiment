"""Online VLA Trainer: VLM forward + action expert training in a single loop.

FLOWER-style: no pre-cache, compute VLM outputs on-the-fly during training.
Allows stride=1 full data utilization without cache memory constraints.

Architecture:
  Producer (CPU ThreadPool): dataset[i] → PNG decode → patches → queue
  Consumer (main thread):
    accumulate batch → VLM forward (no grad) → action expert train_step (pmap)
"""

from __future__ import annotations

import csv
import functools
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from transformers import AutoTokenizer

from qwen.vla.config import PipelineConfig
from qwen.vla.training import flow_matching as fm
from qwen.vla.training.vlm_cache import _prepare_vision_inputs_numpy


class OnlineVLATrainer:
    """Online training: VLM forward inline with action expert training."""

    def __init__(self, policy, dataset, config: PipelineConfig):
        self.policy = policy
        self.dataset = dataset
        self.config = config

    def train(self):
        tc = self.config.training
        fc = self.config.flow_matching
        n = len(self.dataset)
        n_dev = jax.device_count()

        print(f"\n=== Online Training (VLM on-the-fly, pmap {n_dev}-dev) ===")
        print(f"  Samples: {n}, Epochs: {tc.epochs}, Batch: {tc.batch_size}")
        self._train_loop(
            epochs=tc.epochs,
            lr=tc.lr,
            batch_size=tc.batch_size,
            chunk_size=self.config.env.chunk_size,
            simulated_delay=fc.simulated_delay,
            log_interval=tc.log_interval,
            seed=tc.seed,
            image_size=self.config.env.image_size,
            vlm_model_id=self.config.vlm.model_id,
        )
        print("\nTraining complete!")

    def _train_loop(self, epochs, lr, batch_size, chunk_size, simulated_delay,
                    log_interval, seed, image_size, vlm_model_id):
        from qwen.qwen3vl import modeling as qwen3vl

        n = len(self.dataset)
        n_dev = jax.device_count()

        # Round batch size
        if batch_size % n_dev != 0:
            batch_size = ((batch_size + n_dev - 1) // n_dev) * n_dev
        per_dev = batch_size // n_dev

        steps_per_epoch = (n + batch_size - 1) // batch_size
        total_steps = epochs * steps_per_epoch

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=min(100, total_steps // 10),
            decay_steps=total_steps,
            end_value=lr * 0.01,
        )
        tx = optax.adamw(lr_schedule, weight_decay=self.config.training.weight_decay)

        # VLM components (frozen, for inline forward)
        vlm = self.policy.vlm
        visual = vlm.model.visual
        lang_model = vlm.model.language_model
        vlm_config = vlm.config
        grid_h = image_size // 16

        # Train only action_expert + obs_proj (not VLM)
        # Split policy state: trainable (obs_proj + action_expert) and frozen (vlm)
        # Approach: use nnx.state on the policy, but we'll compute loss only through non-VLM parts
        # Simpler: extract trainable submodules explicitly
        obs_proj_graphdef = nnx.graphdef(self.policy.obs_proj)
        action_expert_graphdef = nnx.graphdef(self.policy.action_expert)

        trainable_state = {
            "obs_proj": nnx.state(self.policy.obs_proj),
            "action_expert": nnx.state(self.policy.action_expert),
        }
        opt_state = tx.init(trainable_state)

        rep_trainable = jax.device_put_replicated(trainable_state, jax.devices())
        rep_opt_state = jax.device_put_replicated(opt_state, jax.devices())

        # Vision pmap (frozen, no gradients)
        visual_state = nnx.state(visual)
        visual_graphdef = nnx.graphdef(visual)
        rep_vs = jax.device_put_replicated(visual_state, jax.devices())

        @functools.partial(jax.pmap)
        def pmap_vision(vs, pv):
            vis = nnx.merge(visual_graphdef, vs)
            return vis.forward_static(pv, grid_h=grid_h, grid_w=grid_h, grid_t=1)

        # Action expert pmap train step
        @functools.partial(jax.pmap, axis_name="batch")
        def pmap_action_step(trainable, opt_state, obs, acts, proprio, rng):
            rng, t_rng, n_rng = jax.random.split(rng, 3)
            b = acts.shape[0]
            noise = jax.random.normal(n_rng, acts.shape)

            if simulated_delay > 0:
                timesteps, loss_mask = fm.sample_timesteps_rtc(t_rng, b, chunk_size, simulated_delay)
            else:
                timesteps = fm.sample_timesteps(t_rng, b, chunk_size)
                loss_mask = jnp.ones((b, chunk_size, 1))

            def loss_fn(state):
                obs_proj_m = nnx.merge(obs_proj_graphdef, state["obs_proj"])
                action_expert_m = nnx.merge(action_expert_graphdef, state["action_expert"])

                # obs is raw VLM hidden states (B, seq, 2048). Project to 1536.
                obs_embed = obs_proj_m(obs.astype(jnp.bfloat16))

                noisy = fm.make_noisy(acts, noise, timesteps).astype(jnp.bfloat16)
                ts_bf = timesteps.astype(jnp.bfloat16)
                proprio_bf = proprio.astype(jnp.bfloat16)
                vel_pred = action_expert_m.forward_joint(obs_embed, noisy, ts_bf, proprio_bf)
                vel_pred = vel_pred.astype(jnp.float32)
                vel_target = fm.velocity_target(acts, noise)
                return fm.compute_loss(vel_pred, vel_target, loss_mask)

            loss, grads = jax.value_and_grad(loss_fn)(trainable)
            grads = jax.lax.pmean(grads, axis_name="batch")
            loss = jax.lax.pmean(loss, axis_name="batch")
            updates, new_opt_state = tx.update(grads, opt_state, trainable)
            new_trainable = optax.apply_updates(trainable, updates)
            return new_trainable, new_opt_state, loss, rng

        rng_keys = jax.random.split(jax.random.PRNGKey(seed), n_dev)
        t_start = time.time()
        n_workers = min(64, os.cpu_count() or 4)

        print(f"  batch={batch_size} ({per_dev}/dev × {n_dev}), steps/epoch={steps_per_epoch}, total={total_steps}")
        print(f"  CPU workers: {n_workers}")

        # Log CSV
        log_path = os.path.join(self.config.training.output_dir, "train_log.csv")
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["epoch", "step", "loss", "lr", "epoch_time_s", "samples_per_s"])

        # Tokenizer (shared across workers via ThreadPool)
        tokenizer = AutoTokenizer.from_pretrained(vlm_model_id)

        def _load_one(i):
            sample = self.dataset[i]
            text_tokens = tokenizer.encode(sample["language"], add_special_tokens=False)
            imgs = sample["images"] if sample["images"].shape[0] > 1 else sample["images"][0]
            vlm_inp = _prepare_vision_inputs_numpy(imgs, text_tokens, image_size)
            return vlm_inp, sample["actions"], sample["proprio"]

        def _vlm_forward_batch(vlm_inputs_list):
            """Run VLM on a batch of samples → obs_hidden (B, max_seq, 2048)."""
            bs = len(vlm_inputs_list)

            # Vision pmap: process per-device batch via vmap over images
            pv_list = [inp["pixel_values"] for inp in vlm_inputs_list]
            # Pad to multiple of n_dev
            while len(pv_list) % n_dev != 0:
                pv_list.append(pv_list[-1])

            # Process in chunks of n_dev for pmap
            all_ve = []
            for k in range(0, len(pv_list), n_dev):
                batch_pv = jnp.stack([jnp.array(pv) for pv in pv_list[k:k + n_dev]])
                ve_batch = pmap_vision(rep_vs, batch_pv)
                for j in range(n_dev):
                    if k + j < bs:
                        all_ve.append(ve_batch[j])

            # Batched language model
            max_seq = max(inp["input_ids"].shape[1] for inp in vlm_inputs_list)
            batch_ids = jnp.concatenate([
                jnp.pad(jnp.array(inp["input_ids"]),
                         ((0, 0), (0, max_seq - inp["input_ids"].shape[1])))
                for inp in vlm_inputs_list
            ], axis=0)
            batch_tt = jnp.concatenate([
                jnp.pad(jnp.array(inp["token_type_ids"]),
                         ((0, 0), (0, max_seq - inp["token_type_ids"].shape[1])))
                for inp in vlm_inputs_list
            ], axis=0)
            batch_ve = jnp.stack(all_ve[:bs])

            positions = jnp.broadcast_to(jnp.arange(max_seq)[None, :], (bs, max_seq))
            sin, cos = qwen3vl._generate_rope(
                positions, vlm_config.text_config.head_dim, vlm_config.text_config.rope_theta
            )
            mask = qwen3vl.make_train_causal_mask(max_seq)
            inputs_embeds = lang_model.embed_tokens(batch_ids)
            inputs_embeds = qwen3vl.batched_merge_modalities(batch_ve, inputs_embeds, batch_tt)
            hidden = lang_model(inputs_embeds, None, sin, cos, mask)  # (bs, seq, 2048)
            return hidden, max_seq

        global_step = 0
        for epoch in range(epochs):
            perm_rng = jax.random.PRNGKey(seed + epoch)
            indices = np.array(jax.random.permutation(perm_rng, n))
            epoch_t0 = time.time()

            # Producer/consumer with prefetch queue
            prefetch_q = queue.Queue(maxsize=4)  # prefetch 4 batches ahead
            producer_done = threading.Event()

            def _producer():
                pool = ThreadPoolExecutor(max_workers=n_workers)
                try:
                    for j in range(steps_per_epoch):
                        start = j * batch_size
                        batch_idx = indices[start : start + batch_size]
                        if batch_idx.shape[0] < batch_size:
                            pad_len = batch_size - batch_idx.shape[0]
                            batch_idx = np.concatenate([batch_idx, batch_idx[:pad_len]])
                        # Parallel load batch
                        results = list(pool.map(_load_one, batch_idx.tolist()))
                        vlm_inputs = [r[0] for r in results]
                        actions = np.stack([r[1] for r in results])
                        proprios = np.stack([r[2] for r in results])
                        prefetch_q.put((vlm_inputs, actions, proprios))
                finally:
                    pool.shutdown(wait=False)
                    producer_done.set()

            producer_thread = threading.Thread(target=_producer, daemon=True)
            producer_thread.start()

            for j in range(steps_per_epoch):
                vlm_inputs, actions_np, proprios_np = prefetch_q.get()

                # VLM forward (no grad)
                obs_hidden, _ = _vlm_forward_batch(vlm_inputs)  # (bs, seq, 2048)

                # Reshape for pmap: (n_dev, per_dev, ...)
                obs_h = obs_hidden.reshape(n_dev, per_dev, *obs_hidden.shape[1:])
                acts_d = jnp.array(actions_np).reshape(n_dev, per_dev, chunk_size, 7)
                proprio_d = jnp.array(proprios_np).reshape(n_dev, per_dev, 1, -1)

                rep_trainable, rep_opt_state, loss, rng_keys = pmap_action_step(
                    rep_trainable, rep_opt_state, obs_h, acts_d, proprio_d, rng_keys
                )
                global_step += 1

            producer_thread.join(timeout=5)
            epoch_time = time.time() - epoch_t0
            if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
                epoch_loss = float(loss[0])
                elapsed = time.time() - t_start
                eps = (epoch + 1) / elapsed
                sps = n / epoch_time
                current_lr = float(lr_schedule(global_step))
                print(f"  ep {epoch + 1}/{epochs}, loss={epoch_loss:.4f}, lr={current_lr:.2e}, "
                      f"{eps:.2f} ep/s, {sps:.0f} samp/s")
                log_writer.writerow([epoch + 1, global_step, f"{epoch_loss:.6f}",
                                     f"{current_lr:.2e}", f"{epoch_time:.2f}", f"{sps:.0f}"])
                log_file.flush()

        log_file.close()
        print(f"  Log saved: {log_path}")

        # Unreplicate and save
        final_trainable = jax.tree.map(lambda x: x[0], rep_trainable)
        nnx.update(self.policy.obs_proj, final_trainable["obs_proj"])
        nnx.update(self.policy.action_expert, final_trainable["action_expert"])
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

        ds = self.dataset
        flat["q01"] = ds.q01
        flat["q99"] = ds.q99
        flat["q01_state"] = ds.q01_state
        flat["q99_state"] = ds.q99_state

        np.savez(save_path, **flat)
        print(f"  Saved: {save_path}")
