"""VLM embedding cache: compute, save, and load.

Uses Ouroboros-style queue pipeline for zero idle time:
  CPU ThreadPool (N workers) → Queue → TPU consumer (vision pmap + lang batch)
  CPU and TPU run simultaneously, no idle on either side.
"""

from __future__ import annotations

import functools
import json
import os
import queue
import threading
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from flax import nnx
from PIL import Image as PILImage
from transformers import AutoTokenizer

IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653


@dataclass
class VLMCache:
    """Cached VLM embeddings + actions + proprio in host RAM (numpy).

    Training transfers per-batch to HBM via jnp.array(cache.obs[batch_idx]).
    Avoids HBM OOM for large caches (e.g., 53k × 115 × 1536 × 4 = 35 GB).
    """

    obs: np.ndarray  # (N, max_seq, d_model) float32
    actions: np.ndarray  # (N, chunk_size, action_dim)
    proprio: np.ndarray  # (N, 1, proprio_dim)
    n_samples: int


def _compose_images(images, image_size):
    """Compose multiple camera views into a single composite image.

    Strategy:
      1 cam:  full resize to (image_size × image_size)
      2 cams: Picture-in-Picture (PIP)
        - top camera: full 320×320 (aspect-preserving resize)
        - wrist camera: 80×80 inset at bottom-right corner

    Preserves top camera's full spatial resolution while still providing
    wrist view as a small inset (common in vision models for CALVIN).
    """
    n_cams = images.shape[0]

    # Primary image (top): full resize
    pil_top = PILImage.fromarray((images[0] * 255).astype(np.uint8))
    pil_top = pil_top.resize((image_size, image_size), PILImage.BILINEAR)
    top = np.array(pil_top, dtype=np.float32) / 255.0

    if n_cams == 1:
        return top

    # PIP: wrist in bottom-right corner
    inset_size = image_size // 4  # 80×80 for 320
    pil_wrist = PILImage.fromarray((images[1] * 255).astype(np.uint8))
    pil_wrist = pil_wrist.resize((inset_size, inset_size), PILImage.BILINEAR)
    wrist = np.array(pil_wrist, dtype=np.float32) / 255.0

    top[-inset_size:, -inset_size:, :] = wrist
    return top


def _prepare_vision_inputs_numpy(image_or_images, text_token_ids, image_size):
    """CPU-only: compose + resize + patch extraction + token assembly.

    Accepts single image (H,W,3) or stack of images (n_cams, H, W, 3).
    """
    if image_or_images.ndim == 4:
        image = _compose_images(image_or_images, image_size)
    else:
        image = image_or_images
        if image.shape[0] != image_size or image.shape[1] != image_size:
            pil = PILImage.fromarray((image * 255).astype(np.uint8))
            pil = pil.resize((image_size, image_size), PILImage.BILINEAR)
            image = np.array(pil, dtype=np.float32) / 255.0

    patch_size, temporal_patches, merge_size = 16, 2, 2
    grid_h = image_size // patch_size
    n_vision_tokens = (grid_h // merge_size) ** 2

    input_ids = np.array(
        [[VISION_START_ID] + [IMAGE_TOKEN_ID] * n_vision_tokens + [VISION_END_ID] + text_token_ids],
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

        cache = VLMCache(obs=obs_np, actions=act_np, proprio=proprio_np, n_samples=n)
        total_gb = (obs_np.nbytes + act_np.nbytes + proprio_np.nbytes) / 1024**3
        print(f"Loaded VLM cache: {n} samples in {time.time() - t0:.1f}s ({total_gb:.1f} GB RAM)")
        print(f"  obs={cache.obs.shape}, acts={cache.actions.shape}, proprio={cache.proprio.shape}")
        return cache

    def compute(self, dataset, vlm, obs_proj, vlm_model_id: str, image_size: int,
                n_workers: int | None = None) -> VLMCache:
        """Queue-based pipeline: CPU producers → Queue → TPU consumer. Zero idle."""
        from qwen.qwen3vl import modeling as qwen3vl

        n = len(dataset)
        n_dev = jax.device_count()
        visual = vlm.model.visual
        lang_model = vlm.model.language_model
        config = vlm.config
        grid_h = image_size // 16
        lang_bs = min(128, n)

        if n_workers is None:
            n_workers = max(4, (os.cpu_count() or 4) * 3 // 4)

        print(f"Pipeline: {n} samples, {n_dev} TPU, {n_workers} CPU workers")

        # ── Setup pmap vision (JIT compile) ──
        visual_state = nnx.state(visual)
        visual_graphdef = nnx.graphdef(visual)
        rep_vs = jax.device_put_replicated(visual_state, jax.devices())

        @functools.partial(jax.pmap)
        def pmap_vision(vs, pv):
            vis = nnx.merge(visual_graphdef, vs)
            return vis.forward_static(pv, grid_h=grid_h, grid_w=grid_h, grid_t=1)

        # JIT warmup with dummy data
        dummy_pv = jnp.zeros((n_dev, 400, grid_h * grid_h * 2 * 16 * 16 * 3 // (grid_h * grid_h) ), dtype=jnp.float32)
        # Actually just use correct shape
        sample0 = dataset[0]
        tok = AutoTokenizer.from_pretrained(vlm_model_id)
        t0_tokens = tok.encode(sample0["language"], add_special_tokens=False)
        dummy_inp = _prepare_vision_inputs_numpy(sample0["images"][0], t0_tokens, image_size)
        dummy_pv = jnp.stack([jnp.array(dummy_inp["pixel_values"])] * n_dev)
        print("  JIT compiling vision pmap...")
        t_jit = time.time()
        _ = pmap_vision(rep_vs, dummy_pv)
        jax.block_until_ready(_)
        print(f"  JIT done: {time.time() - t_jit:.1f}s")

        # ── Producer: CPU ThreadPool → Queue ──
        prefetch_q = queue.Queue(maxsize=n_workers * 3)
        producer_done = threading.Event()
        produce_count = [0]

        def _producer():
            tokenizer = AutoTokenizer.from_pretrained(vlm_model_id)

            def _load_one(i):
                sample = dataset[i]
                text_tokens = tokenizer.encode(sample["language"], add_special_tokens=False)
                # Pass all cameras; _prepare will compose if multi-cam
                imgs = sample["images"] if sample["images"].shape[0] > 1 else sample["images"][0]
                vlm_inp = _prepare_vision_inputs_numpy(imgs, text_tokens, image_size)
                return i, vlm_inp, sample["actions"], sample["proprio"]

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for result in pool.map(_load_one, range(n)):
                    prefetch_q.put(result)
                    produce_count[0] += 1

            producer_done.set()

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        # ── Consumer: TPU vision pmap + language batch ──
        t_start = time.time()

        # Accumulators
        vis_pv_buf = []       # pixel_values for pmap (accumulate n_dev)
        vis_inp_buf = []      # corresponding vlm_inputs for language model
        vis_idx_buf = []      # original indices
        lang_ve_buf = []      # vision embeddings waiting for language batch
        lang_inp_buf = []     # vlm_inputs for language
        lang_idx_buf = []     # indices

        act_dict = {}         # idx → actions
        proprio_dict = {}     # idx → proprio
        obs_dict = {}         # idx → obs numpy

        consumed = 0
        vis_done = 0
        lang_done = 0

        def _flush_vision():
            nonlocal vis_done
            if not vis_pv_buf:
                return
            # Pad to n_dev
            pv_list = list(vis_pv_buf)
            inp_list = list(vis_inp_buf)
            idx_list = list(vis_idx_buf)
            while len(pv_list) < n_dev:
                pv_list.append(pv_list[-1])
            batch_pv = jnp.stack([jnp.array(pv) for pv in pv_list])
            ve_batch = pmap_vision(rep_vs, batch_pv)
            for j in range(min(len(vis_pv_buf), n_dev)):
                lang_ve_buf.append(ve_batch[j])
                lang_inp_buf.append(inp_list[j])
                lang_idx_buf.append(idx_list[j])
            vis_done += len(vis_pv_buf)
            vis_pv_buf.clear()
            vis_inp_buf.clear()
            vis_idx_buf.clear()

        def _flush_language():
            nonlocal lang_done
            if not lang_ve_buf:
                return
            bs = len(lang_ve_buf)
            max_seq = max(inp["input_ids"].shape[1] for inp in lang_inp_buf)

            batch_ids = jnp.concatenate([
                jnp.pad(jnp.array(lang_inp_buf[j]["input_ids"]),
                         ((0, 0), (0, max_seq - lang_inp_buf[j]["input_ids"].shape[1])))
                for j in range(bs)
            ], axis=0)
            batch_tt = jnp.concatenate([
                jnp.pad(jnp.array(lang_inp_buf[j]["token_type_ids"]),
                         ((0, 0), (0, max_seq - lang_inp_buf[j]["token_type_ids"].shape[1])))
                for j in range(bs)
            ], axis=0)
            batch_ve = jnp.stack(lang_ve_buf[:bs])

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
                obs_dict[lang_idx_buf[j]] = np.array(obs_batch[j])

            lang_done += bs
            lang_ve_buf.clear()
            lang_inp_buf.clear()
            lang_idx_buf.clear()

        # Main consumer loop
        while True:
            # Try to get from queue (non-blocking if producer still running)
            try:
                item = prefetch_q.get(timeout=0.1)
            except queue.Empty:
                if producer_done.is_set() and prefetch_q.empty():
                    break
                continue

            idx, vlm_inp, actions, proprio = item
            act_dict[idx] = actions
            proprio_dict[idx] = proprio
            consumed += 1

            # Accumulate for vision
            vis_pv_buf.append(vlm_inp["pixel_values"])
            vis_inp_buf.append(vlm_inp)
            vis_idx_buf.append(idx)

            # Flush vision when we have n_dev samples
            if len(vis_pv_buf) >= n_dev:
                _flush_vision()

            # Flush language when we have lang_bs vision embeddings
            if len(lang_ve_buf) >= lang_bs:
                _flush_language()

            # Progress
            if consumed % 2000 == 0:
                elapsed = time.time() - t_start
                rate = consumed / elapsed
                q_size = prefetch_q.qsize()
                print(f"    {consumed}/{n} consumed, {vis_done} vis, {lang_done} lang "
                      f"({rate:.0f}/s, queue={q_size})")

        # Flush remaining
        _flush_vision()
        _flush_language()

        producer_thread.join(timeout=5)
        elapsed = time.time() - t_start
        print(f"  Pipeline done: {consumed} consumed, {vis_done} vis, {lang_done} lang in {elapsed:.0f}s")
        print(f"  Throughput: {consumed / elapsed:.0f} samples/s")

        # ── Assemble: numpy → single HBM transfer ──
        t0 = time.time()
        max_obs_seq = max(o.shape[0] for o in obs_dict.values())
        d_model = list(obs_dict.values())[0].shape[1]

        # Ordered assembly
        obs_np = np.zeros((n, max_obs_seq, d_model), dtype=np.float32)
        act_np = np.zeros((n, dataset.chunk_size, dataset.action_dim), dtype=np.float32)
        proprio_np = np.zeros((n, 1, dataset.proprio_dim), dtype=np.float32)

        for i in range(n):
            o = obs_dict[i]
            obs_np[i, : o.shape[0], :] = o
            act_np[i] = act_dict[i]
            proprio_np[i] = proprio_dict[i]

        del obs_dict, act_dict, proprio_dict
        cache = VLMCache(obs=obs_np, actions=act_np, proprio=proprio_np, n_samples=n)
        total_gb = (obs_np.nbytes + act_np.nbytes + proprio_np.nbytes) / 1024**3
        print(f"  Assemble: {time.time() - t0:.1f}s ({total_gb:.1f} GB RAM)")
        print(f"  obs={cache.obs.shape}, acts={cache.actions.shape}, proprio={cache.proprio.shape}")

        self._save(cache, n, max_obs_seq, d_model, dataset.chunk_size,
                   dataset.action_dim, dataset.proprio_dim)
        return cache

    def _save(self, cache, n, max_seq, d_model, chunk_size, action_dim, proprio_dim):
        os.makedirs(self._cache_dir, exist_ok=True)
        obs_np = cache.obs
        act_np = cache.actions
        proprio_np = cache.proprio

        obs_bytes, act_bytes, proprio_bytes = [], [], []
        for i in range(n):
            obs_bytes.append(obs_np[i].tobytes())
            act_bytes.append(act_np[i].tobytes())
            proprio_bytes.append(proprio_np[i].tobytes())

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
