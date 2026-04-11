"""VLA training CLI for JAX/TPU v4-8.

Usage:
    python src/qwen/vla/train.py
    python src/qwen/vla/train.py --simulated-delay 15    # with RTC
    python src/qwen/vla/train.py --stage1-epochs 5       # quick test
"""

import argparse
import os

import jax
from flax import nnx

from qwen.qwen3vl import modeling as qwen3vl
from qwen.vla.data.lerobot_calvin import CalvinDataset
from qwen.vla.models.vla import VLAPolicy
from qwen.vla.training.trainer import VLATrainer

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN3VL_MODEL_PATH",
    os.path.join(_ROOT, "..", "..", "..", "models", "qwen3-vl-2b"),
)


def main():
    parser = argparse.ArgumentParser(description="VLA Training (JAX/TPU)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--repo-id", default="fywang/calvin-debug-lerobot")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--simulated-delay", type=int, default=0, help="RTC delay (0=disabled)")
    parser.add_argument("--output-dir", default="result/vla")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("VLA Training — JAX/TPU v4-8")
    print("=" * 60)
    print(f"JAX: {jax.__version__}, Devices: {jax.device_count()}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.repo_id}")
    print(f"RTC delay: {args.simulated_delay}")

    # Load VLM
    print("\nLoading Qwen3-VL 2B...")
    config = qwen3vl.ModelConfig.qwen3vl_2b()
    vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, config=config)

    # Build VLA policy
    rngs = nnx.Rngs(params=args.seed)
    policy = VLAPolicy(
        vlm=vlm,
        vlm_hidden_dim=2048,
        action_expert_config={
            "d_model": 1536,
            "n_layers": 12,
            "d_ff": 4096,
            "n_heads": 12,
            "n_kv_heads": 4,
            "head_dim": 128,
            "action_dim": 7,
        },
        rngs=rngs,
    )
    print("VLA policy initialized.")

    # Load dataset
    print(f"\nLoading dataset: {args.repo_id}")
    dataset = CalvinDataset(
        repo_id=args.repo_id,
        split="train",
        chunk_size=args.chunk_size,
    )
    print(f"  {len(dataset)} training chunks")

    # Train
    trainer_config = {
        "vlm_model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "lr": args.lr,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "simulated_delay": args.simulated_delay,
        "chunk_size": args.chunk_size,
        "output_dir": args.output_dir,
        "seed": args.seed,
    }

    trainer = VLATrainer(policy, dataset, trainer_config)
    trainer.train()


if __name__ == "__main__":
    main()
