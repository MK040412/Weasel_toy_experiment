"""VLA training CLI — config-driven.

Usage:
    python src/qwen/vla/train.py                            # calvin-debug default
    python src/qwen/vla/train.py --env calvin-abcd          # ABCD-D preset
    python src/qwen/vla/train.py --simulated-delay 15       # RTC
    python src/qwen/vla/train.py --epochs 50 --lr 2e-4      # override
"""

import argparse
import os

import jax
from flax import nnx

from qwen.vla.config import PipelineConfig
from qwen.vla.data.protocol import create_dataset
from qwen.vla.models.vla import VLAPolicy
from qwen.vla.training.trainer import VLATrainer
from qwen.vla.training.vlm_cache import VLMCacher

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN3VL_MODEL_PATH",
    os.path.join(_ROOT, "..", "..", "..", "models", "qwen3-vl-2b"),
)


def main():
    parser = argparse.ArgumentParser(description="VLA Training (JAX/TPU)")
    parser.add_argument("--env", default="calvin-debug", choices=["calvin-debug", "calvin-abcd"])
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--simulated-delay", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Build config from preset + overrides
    if args.env == "calvin-abcd":
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

    print("=" * 60)
    print(f"VLA Training — {cfg.env.name}")
    print("=" * 60)
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()}")
    print(f"Env: {cfg.env.name}, action_dim={cfg.env.action_dim}, proprio_dim={cfg.env.proprio_dim}")
    print(f"RTC delay: {cfg.flow_matching.simulated_delay}")

    # Stage 1: VLM cache
    cacher = VLMCacher(cfg.training.output_dir)
    dataset = create_dataset(cfg.env, split="train")
    print(f"Dataset: {len(dataset)} chunks")

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

    # Stage 2: Train
    trainer = VLATrainer(policy, cache, cfg, dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    main()
