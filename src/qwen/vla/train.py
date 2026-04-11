#!/usr/bin/env python3
"""CLI entry point for VLA training with CALVIN debug dataset (LeRobot v2.1).

Usage:
    python src/qwen/vla/train.py
    python src/qwen/vla/train.py --simulated-delay 15
    python src/qwen/vla/train.py --resume checkpoint_final.pt
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from qwen.vla.config import (
    ActionExpertConfig,
    DataConfig,
    FlowMatchingConfig,
    PipelineConfig,
    TrainingConfig,
    VLMConfig,
)
from qwen.vla.training.trainer import VLATrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VLA policy on CALVIN (LeRobot)")
    parser.add_argument("--repo-id", type=str, default="fywang/calvin-debug-lerobot")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--denoise-steps-inference", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vlm-model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--no-freeze-vlm", action="store_true", help="Unfreeze VLM backbone")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--simulated-delay",
        type=int,
        default=0,
        help="Training-time RTC: max prefix length (0=disabled, e.g. 15)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = PipelineConfig(
        action_expert=ActionExpertConfig(),
        vlm=VLMConfig(
            model_id=args.vlm_model_id,
            freeze=not args.no_freeze_vlm,
        ),
        data=DataConfig(
            repo_id=args.repo_id,
            chunk_size=args.chunk_size,
        ),
        flow_matching=FlowMatchingConfig(
            denoise_steps_inference=args.denoise_steps_inference,
            simulated_delay=args.simulated_delay,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr=args.lr,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            seed=args.seed,
        ),
    )

    print("Pipeline config:")
    print(f"  VLM: {config.vlm.model_id} (freeze={config.vlm.freeze})")
    print(f"  Action Expert: d_model={config.action_expert.d_model}, n_layers={config.action_expert.n_layers}")
    print(f"  Data: {config.data.repo_id}, chunk_size={config.data.chunk_size}")
    print(
        f"  Flow Matching: denoise_steps={config.flow_matching.denoise_steps_inference}, "
        f"simulated_delay={config.flow_matching.simulated_delay}"
    )
    print(
        f"  Training: batch_size={config.training.batch_size}, "
        f"lr={config.training.lr}, "
        f"stage1={config.training.stage1_epochs}ep, "
        f"stage2={config.training.stage2_epochs}ep, "
        f"grad_accum={config.training.gradient_accumulation_steps}"
    )

    trainer = VLATrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
