"""VLA training CLI — config-driven, supports cached and online modes.

Two training modes:
  --mode cached (default): Use pre-computed VLM cache
      - Much faster per-step (no VLM forward each iteration)
      - Requires VLM cache on disk (run preprocess.sh first, or auto-compute)
      - Limited by RAM/disk size (stride=1 doesn't fit for ABCD-D)

  --mode online: Compute VLM on-the-fly during training (FLOWER-style)
      - Slower per-step (full VLM forward each iteration)
      - No cache needed — works with any data size (stride=1 OK)
      - Useful when cache size exceeds RAM/disk

Usage:
    # Split: preprocess once, train many times
    python scripts/preprocess_vlm_cache.py --env calvin-abcd-flower
    python src/qwen/vla/train.py --env calvin-abcd-flower --mode cached

    # All-in-one: train auto-computes cache if missing
    python src/qwen/vla/train.py --env calvin-abcd-flower --mode cached

    # Online: stride=1 full data, no cache
    python src/qwen/vla/train.py --env calvin-abcd-flower-full --mode online
"""

import argparse
import os

import jax
from flax import nnx

from qwen.vla.config import PipelineConfig
from qwen.vla.data.protocol import create_dataset
from qwen.vla.models.vla import VLAPolicy
from qwen.vla.training.online_trainer import OnlineVLATrainer
from qwen.vla.training.trainer import VLATrainer
from qwen.vla.training.vlm_cache import VLMCacher

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN3VL_MODEL_PATH",
    os.path.join(_ROOT, "..", "..", "..", "models", "qwen3-vl-2b"),
)


def main():
    parser = argparse.ArgumentParser(description="VLA Training (JAX/TPU)")
    parser.add_argument("--env", default="calvin-debug",
                        choices=["calvin-debug", "calvin-abcd", "calvin-abcd-flower", "calvin-abcd-flower-full"])
    parser.add_argument("--mode", default="cached", choices=["cached", "online"],
                        help="cached: pre-compute VLM embeddings; online: VLM forward each step (FLOWER-style)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--simulated-delay", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-distributed", action="store_true",
                        help="Skip jax.distributed.initialize() — use for single-host (v4-8) runs only")
    args = parser.parse_args()

    if not args.no_distributed:
        jax.distributed.initialize()

    if args.env == "calvin-abcd-flower-full":
        cfg = PipelineConfig.calvin_abcd_flower_full()
    elif args.env == "calvin-abcd-flower":
        cfg = PipelineConfig.calvin_abcd_flower()
    elif args.env == "calvin-abcd":
        cfg = PipelineConfig.calvin_abcd()
    else:
        cfg = PipelineConfig.calvin_debug()

    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.simulated_delay is not None:
        cfg.flow_matching.simulated_delay = args.simulated_delay
    if args.output_dir is not None:
        cfg.training.output_dir = args.output_dir
    if args.seed is not None:
        cfg.training.seed = args.seed
    cfg.vlm.model_path = args.model_path

    online_mode = args.mode == "online"

    print("=" * 60)
    print(f"VLA Training — {cfg.env.name} (mode={args.mode.upper()})")
    print("=" * 60)
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()}")
    print(f"Env: {cfg.env.name}, action_dim={cfg.env.action_dim}, proprio_dim={cfg.env.proprio_dim}")
    print(f"RTC delay: {cfg.flow_matching.simulated_delay}")

    dataset = create_dataset(cfg.env, split="train")
    print(f"Dataset: {len(dataset)} chunks")

    if online_mode:
        # On-the-fly: load VLM into policy, use OnlineVLATrainer
        print("Mode: on-the-fly VLM forward (no pre-cache)")
        from qwen.qwen3vl import modeling as qwen3vl

        model_config = qwen3vl.ModelConfig.qwen3vl_2b()
        vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(cfg.vlm.model_path, config=model_config)
        policy = VLAPolicy(
            vlm=vlm, vlm_hidden_dim=cfg.vlm.hidden_dim,
            action_expert_config={"action_dim": cfg.env.action_dim, "proprio_dim": cfg.env.proprio_dim},
            rngs=nnx.Rngs(params=cfg.training.seed),
        )
        trainer = OnlineVLATrainer(policy, dataset, cfg)
        trainer.train()
    else:
        # Cached: load or compute VLM cache, then train with VLATrainer
        cacher = VLMCacher(cfg.training.output_dir)

        if cacher.exists():
            print("VLM cache found.")
            cache = cacher.load()
            policy = VLAPolicy(
                vlm=None, vlm_hidden_dim=cfg.vlm.hidden_dim,
                action_expert_config={"action_dim": cfg.env.action_dim, "proprio_dim": cfg.env.proprio_dim},
                rngs=nnx.Rngs(params=cfg.training.seed),
            )
        else:
            print("Computing VLM cache...")
            from qwen.qwen3vl import modeling as qwen3vl

            model_config = qwen3vl.ModelConfig.qwen3vl_2b()
            vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(cfg.vlm.model_path, config=model_config)
            policy = VLAPolicy(
                vlm=vlm, vlm_hidden_dim=cfg.vlm.hidden_dim,
                action_expert_config={"action_dim": cfg.env.action_dim, "proprio_dim": cfg.env.proprio_dim},
                rngs=nnx.Rngs(params=cfg.training.seed),
            )
            cache = cacher.compute(dataset, vlm, policy.obs_proj, cfg.vlm.model_id, cfg.env.image_size)
            policy.vlm = None
            jax.clear_caches()

        trainer = VLATrainer(policy, cache, cfg, dataset=dataset)
        trainer.train()


if __name__ == "__main__":
    main()
