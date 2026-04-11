"""VLM embedding cache: compute, save, and load.

Separated from trainer so it can be used independently
(e.g., from download scripts or preprocessing pipelines).
"""

from __future__ import annotations

import functools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from flax import nnx
from PIL import Image as PILImage
from transformers import AutoTokenizer

IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653


@dataclass
class VLMCache:
    """Cached VLM embeddings + actions + proprio on HBM."""

    obs: jax.Array  # (N, max_seq, d_model)
    actions: jax.Array  # (N, chunk_size, action_dim)
    proprio: jax.Array  # (N, 1, proprio_dim)
    n_samples: int


def _prepare_vision_inputs_numpy(image, language, tokenizer_name, image_size):
    """CPU-only preprocessing, returns numpy (no JAX). Picklable for multiprocessing."""
    if image.shape[0] != image_size or image.shape[1] != image_size:
        pil = PILImage.fromarray((image * 255).astype(np.uint8))
        pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
        image = np.array(pil, dtype=np.float32) / 255.0

    patch_size, temporal_patches, merge_size = 16, 2, 2
    grid_h = image_size // patch_size
    n_vision_tokens = (grid_h // merge_size) ** 2

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_tokens = tokenizer.encode(language, add_special_tokens=False)
    input_ids = np.array(
        [[VISION_START_ID] + [IMAGE_TOKEN_ID] * n_vision_tokens + [VISION_END_ID] + text_tokens],
        dtype=np.int32,
    )

    img_doubled = np.stack([image, image], axis=0)
    patches = []
    for h in range(0, image_size, patch_size):
        for w in range(0, image_size, patch_size):
            patch = img_doubled[:temporal_patches, h : h + patch_size, w : w + patch_size, :]
            patches.append(patch.transpose(3, 0, 1, 2).flatten())

    return {
        "input_ids": input_ids,
        "pixel_values": np.array(patches, dtype=np.float32),
        "token_type_ids": (input_ids == IMAGE_TOKEN_ID).astype(np.int32),
    }


def _preprocess_worker(args):
    """Module-level wrapper for ProcessPoolExecutor (must be picklable)."""
    return _prepare_vision_inputs_numpy(*args)


def _pad_seq(x, target_len):
    pad_len = target_len - x.shape[0]
    if pad_len <= 0:
        return x[:target_len, :]
    return jnp.pad(x, ((0, pad_len), (0, 0)))


class VLMCacher:
    """Compute, save, and load VLM embedding caches."""

    def __init__(self, output_dir: str):
        self._cache_dir = os.path.join(output_dir, "vlm_cache")
        self._cache_path = os.path.join(self._cache_dir, "embeddings.parquet")
        self._meta_path = os.path.join(self._cache_dir, "meta.json")

    def exists(self) -> bool:
        return os.path.exists(self._cache_path) and os.path.exists(self._meta_path)

    def load(self) -> VLMCache:
        t0 = time.time()
        with open(self._meta_path) as f:
            meta = json.load(f)

        n, max_seq = meta["n_samples"], meta["max_seq_len"]
        d_model, chunk_size = meta["d_model"], meta["chunk_size"]
        action_dim = meta.get("action_dim", 7)
        proprio_dim = meta.get("proprio_dim", 15)

        table = pq.read_table(self._cache_path)
        obs_col, act_col = table.column("obs"), table.column("actions")
        proprio_col = table.column("proprio")

        obs_np = np.zeros((n, max_seq, d_model), dtype=np.float32)
        act_np = np.zeros((n, chunk_size, action_dim), dtype=np.float32)
        proprio_np = np.zeros((n, 1, proprio_dim), dtype=np.float32)
        for i in range(n):
            arr = np.frombuffer(obs_col[i].as_py(), dtype=np.float32).reshape(-1, d_model)
            obs_np[i, : arr.shape[0], :] = arr
            act_np[i] = np.frombuffer(act_col[i].as_py(), dtype=np.float32).reshape(chunk_size, action_dim)
            proprio_np[i] = np.frombuffer(proprio_col[i].as_py(), dtype=np.float32).reshape(1, proprio_dim)

        cache = VLMCache(
            obs=jnp.array(obs_np),
            actions=jnp.array(act_np),
            proprio=jnp.array(proprio_np),
            n_samples=n,
        )
        elapsed = time.time() - t0
        print(f"Loaded VLM cache: {n} samples in {elapsed:.1f}s")
        print(f"  obs={cache.obs.shape}, acts={cache.actions.shape}, proprio={cache.proprio.shape}")
        return cache

    def compute(self, dataset, vlm, obs_proj, vlm_model_id: str, image_size: int) -> VLMCache:
        """Compute VLM embeddings with pmap vision + batched language."""
        from qwen.qwen3vl import modeling as qwen3vl

        n = len(dataset)
        n_dev = jax.device_count()
        visual = vlm.model.visual
        lang_model = vlm.model.language_model
        config = vlm.config
        grid_h = image_size // 16

        print(f"Computing VLM embeddings for {n} samples ({n_dev} devices)...")

        # Step 1: CPU preprocessing
        t0 = time.time()
        n_workers = min(8, os.cpu_count() or 1, n)
        work_items = []
        for i in range(n):
            sample = dataset[i]
            work_items.append((sample["images"][0], sample["language"], vlm_model_id, image_size))

        if n_workers > 1 and n > 4:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                all_inputs = list(pool.map(_preprocess_worker, work_items))
        else:
            all_inputs = [_prepare_vision_inputs_numpy(*w) for w in work_items]

        act_list, proprio_list = [], []
        for i in range(n):
            sample = dataset[i]
            act_list.append(sample["actions"])
            proprio_list.append(sample["proprio"])

        prep_time = time.time() - t0
        print(f"  CPU preprocessing: {prep_time:.1f}s ({n_workers} workers)")

        # Step 2: Vision encoder (pmap)
        t0 = time.time()
        visual_state = nnx.state(visual)
        visual_graphdef = nnx.graphdef(visual)
        rep_vs = jax.device_put_replicated(visual_state, jax.devices())

        @functools.partial(jax.pmap)
        def pmap_vision(vs, pv):
            vis = nnx.merge(visual_graphdef, vs)
            return vis.forward_static(pv, grid_h=grid_h, grid_w=grid_h, grid_t=1)

        pv_list = [all_inputs[i]["pixel_values"] for i in range(n)]
        while len(pv_list) % n_dev != 0:
            pv_list.append(pv_list[-1])

        warmup_pv = jnp.stack([jnp.array(pv_list[i]) for i in range(n_dev)])
        _ = pmap_vision(rep_vs, warmup_pv)
        jax.block_until_ready(_)

        all_ve = []
        for start in range(0, len(pv_list), n_dev):
            batch_pv = jnp.stack([jnp.array(pv_list[start + j]) for j in range(n_dev)])
            ve_batch = pmap_vision(rep_vs, batch_pv)
            jax.block_until_ready(ve_batch)
            for j in range(n_dev):
                if start + j < n:
                    all_ve.append(ve_batch[j])

        vis_time = time.time() - t0
        print(f"  Vision encoder (pmap {n_dev}-dev): {vis_time:.1f}s")

        # Step 3: Language model (batched)
        t0 = time.time()
        max_seq = max(inp["input_ids"].shape[1] for inp in all_inputs)
        lang_batch_size = min(128, n)

        all_obs_list = []
        for batch_start in range(0, n, lang_batch_size):
            batch_end = min(batch_start + lang_batch_size, n)
            bs = batch_end - batch_start

            batch_ids = jnp.concatenate([
                jnp.pad(jnp.array(all_inputs[i]["input_ids"]),
                         ((0, 0), (0, max_seq - all_inputs[i]["input_ids"].shape[1])))
                for i in range(batch_start, batch_end)
            ], axis=0)
            batch_tt = jnp.concatenate([
                jnp.pad(jnp.array(all_inputs[i]["token_type_ids"]),
                         ((0, 0), (0, max_seq - all_inputs[i]["token_type_ids"].shape[1])))
                for i in range(batch_start, batch_end)
            ], axis=0)
            batch_ve = jnp.stack(all_ve[batch_start:batch_end])

            positions = jnp.broadcast_to(jnp.arange(max_seq)[None, :], (bs, max_seq))
            sin, cos = qwen3vl._generate_rope(
                positions, config.text_config.head_dim, config.text_config.rope_theta
            )
            mask = qwen3vl.make_train_causal_mask(max_seq)
            inputs_embeds = lang_model.embed_tokens(batch_ids)
            inputs_embeds = qwen3vl.batched_merge_modalities(batch_ve, inputs_embeds, batch_tt)
            hidden = lang_model(inputs_embeds, None, sin, cos, mask)
            obs_batch = obs_proj(hidden)
            jax.block_until_ready(obs_batch)

            for j in range(bs):
                all_obs_list.append(np.array(obs_batch[j]))

        lang_time = time.time() - t0
        print(f"  Language model (batch={lang_batch_size}): {lang_time:.1f}s")

        # Step 4: Assemble
        max_obs_seq = max(o.shape[0] for o in all_obs_list)
        d_model = all_obs_list[0].shape[1]

        cache = VLMCache(
            obs=jnp.stack([_pad_seq(jnp.array(o), max_obs_seq) for o in all_obs_list]),
            actions=jnp.array(np.stack(act_list)),
            proprio=jnp.array(np.stack(proprio_list)),
            n_samples=n,
        )
        print(f"  obs={cache.obs.shape}, acts={cache.actions.shape}, proprio={cache.proprio.shape}")
        total_time = prep_time + vis_time + lang_time
        print(f"  Total: {total_time:.1f}s ({total_time / n * 1000:.0f}ms/sample)")

        self._save(cache, all_obs_list, max_obs_seq, d_model, dataset.chunk_size,
                   dataset.action_dim, dataset.proprio_dim)
        return cache

    def _save(self, cache, obs_raw_list, max_seq, d_model, chunk_size, action_dim, proprio_dim):
        os.makedirs(self._cache_dir, exist_ok=True)
        n = cache.n_samples
        act_np, proprio_np = np.array(cache.actions), np.array(cache.proprio)

        obs_bytes, act_bytes, proprio_bytes = [], [], []
        for i in range(n):
            obs_bytes.append(obs_raw_list[i].astype(np.float32).tobytes())
            act_bytes.append(act_np[i].astype(np.float32).tobytes())
            proprio_bytes.append(proprio_np[i].astype(np.float32).tobytes())

        table = pa.table({
            "obs": pa.array(obs_bytes, type=pa.binary()),
            "actions": pa.array(act_bytes, type=pa.binary()),
            "proprio": pa.array(proprio_bytes, type=pa.binary()),
        })
        pq.write_table(table, self._cache_path)

        meta = {
            "n_samples": n, "max_seq_len": max_seq, "d_model": d_model,
            "chunk_size": chunk_size, "action_dim": action_dim, "proprio_dim": proprio_dim,
        }
        with open(self._meta_path, "w") as f:
            json.dump(meta, f)

        size_mb = os.path.getsize(self._cache_path) / (1024 * 1024)
        print(f"  Saved: {self._cache_path} ({size_mb:.1f} MB)")
