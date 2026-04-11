"""Standalone VLM cache preprocessing script.

Generates VLM embedding cache independently of training.
Run once, reuse forever.

Usage:
    PYTHONPATH=src python scripts/preprocess_vlm_cache.py --env calvin-debug
    PYTHONPATH=src python scripts/preprocess_vlm_cache.py --env calvin-abcd --workers 180
"""

import argparse
import os
import time

import jax
from flax import nnx

from qwen.qwen3vl import modeling as qwen3vl
from qwen.vla.config import PipelineConfig
from qwen.vla.data.protocol import create_dataset
from qwen.vla.models.vla import VLAPolicy
from qwen.vla.training.vlm_cache import VLMCacher

DEFAULT_MODEL_PATH = os.environ.get("QWEN3VL_MODEL_PATH", "/home/perelman/models/qwen3-vl-2b")


def main():
    parser = argparse.ArgumentParser(description="VLM Cache Preprocessing")
    parser.add_argument("--env", default="calvin-debug", choices=["calvin-debug", "calvin-abcd"])
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--workers", type=int, default=None, help="ThreadPool workers (default: 75%% of vCPUs)")
    args = parser.parse_args()

    t_total = time.time()

    if args.env == "calvin-abcd":
        cfg = PipelineConfig.calvin_abcd()
    else:
        cfg = PipelineConfig.calvin_debug()

    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.local_path:
        cfg.env.local_path = args.local_path

    n_workers = args.workers or max(4, (os.cpu_count() or 4) * 3 // 4)

    print("=" * 60)
    print(f"VLM Cache Preprocessing — {cfg.env.name}")
    print("=" * 60)
    print(f"Devices: {jax.device_count()}, vCPUs: {os.cpu_count()}, workers: {n_workers}")
    print(f"Output: {cfg.training.output_dir}/vlm_cache/")

    cacher = VLMCacher(cfg.training.output_dir)
    if cacher.exists():
        print("\nCache already exists! Loading to verify...")
        cacher.load()
        print(f"Done in {time.time() - t_total:.1f}s")
        return

    print("\nLoading dataset...")
    ds = create_dataset(cfg.env, split="train")
    print(f"  {len(ds)} chunks, action_dim={ds.action_dim}, proprio_dim={ds.proprio_dim}")

    print("\nLoading Qwen3-VL 2B...")
    model_config = qwen3vl.ModelConfig.qwen3vl_2b()
    vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, config=model_config)
    policy = VLAPolicy(
        vlm=vlm, vlm_hidden_dim=cfg.vlm.hidden_dim,
        action_expert_config={"action_dim": cfg.env.action_dim, "proprio_dim": cfg.env.proprio_dim},
        rngs=nnx.Rngs(params=42),
    )

    print()
    cache = cacher.compute(ds, vlm, policy.obs_proj, cfg.vlm.model_id, cfg.env.image_size, n_workers=n_workers)

    print(f"\nTotal: {time.time() - t_total:.0f}s ({(time.time() - t_total) / 60:.1f} min)")


if __name__ == "__main__":
    main()
