#!/usr/bin/env python3
"""Standalone VLA inference: load checkpoint, predict actions, print metrics.

Usage:
    python src/qwen/vla/inference.py --checkpoint checkpoint_final.pt
    python src/qwen/vla/inference.py --checkpoint checkpoint_final.pt --split val --output-dir result/vla/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

import qwen.vla.config

sys.modules["qwen3vl.config"] = qwen.vla.config  # backward compat with old checkpoints
sys.modules["weasel_vla.config"] = qwen.vla.config

from qwen.vla.config import PipelineConfig  # noqa: E402
from qwen.vla.data.lerobot_calvin import LeRobotCalvinDataset  # noqa: E402
from qwen.vla.models.vla import VLAPolicy  # noqa: E402


def load_model(ckpt_path: str):
    """Load checkpoint and return policy + action quantiles + config."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: PipelineConfig = ckpt["config"]
    policy = VLAPolicy(config.vlm, config.action_expert)
    policy.obs_proj.load_state_dict(ckpt["obs_proj"])
    policy.action_expert.load_state_dict(ckpt["action_expert"])
    policy.eval()
    return policy, ckpt["action_q01"], ckpt["action_q99"], config


def denormalize(actions: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> np.ndarray:
    """Inverse quantile normalization: [-1, 1] -> original scale."""
    return ((actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01).numpy()


@torch.inference_mode()
def predict(policy: VLAPolicy, hidden: torch.Tensor, chunk_size: int, n_steps: int, device: torch.device):
    """Run denoising from cached VLM hidden states."""
    hidden = hidden.to(device)
    obs_embed = policy.obs_proj(hidden)
    actions = policy.action_expert.denoise(obs_embed, chunk_size=chunk_size, n_steps=n_steps)
    return actions.cpu().float().squeeze(0)


DIM_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def main() -> None:
    parser = argparse.ArgumentParser(description="VLA inference: predict actions from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--denoise-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    policy, q01, q99, config = load_model(args.checkpoint)

    n_steps = args.denoise_steps or config.flow_matching.denoise_steps_inference
    chunk_size = config.data.chunk_size

    print(f"Loading {args.split} dataset...")
    dataset = LeRobotCalvinDataset(config.data, split=args.split, action_q01=q01, action_q99=q99)
    n = min(args.n_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)
    print(f"  {len(dataset)} total samples, evaluating {n}: {indices}")

    # Phase 1: Cache VLM embeddings
    print("Caching VLM embeddings...")
    policy.vlm.eval()
    vlm_cache: dict[int, torch.Tensor] = {}
    for idx in indices:
        sample = dataset[idx]
        hidden = policy.vlm_forward_hidden(sample["images"], sample["language"])
        vlm_cache[idx] = hidden.cpu()
        del hidden
        torch.cuda.empty_cache()

    # Phase 2: Offload VLM, move action expert to GPU
    print("Offloading VLM -> CPU, action expert -> GPU...")
    policy.vlm.to("cpu")
    torch.cuda.empty_cache()
    policy.obs_proj.to(device, dtype=torch.bfloat16)
    policy.action_expert.to(device, dtype=torch.bfloat16)

    # Predict and evaluate
    print(f"\nPredicting with {n_steps} denoising steps, chunk_size={chunk_size}...")
    print("=" * 70)

    results = []
    for idx in indices:
        sample = dataset[idx]
        gt_norm = sample["actions"]
        pred_norm = predict(policy, vlm_cache[idx], chunk_size, n_steps, device)

        gt_raw = denormalize(gt_norm, q01, q99)
        pred_raw = denormalize(pred_norm, q01, q99)

        mse = float(((gt_raw - pred_raw) ** 2).mean())
        mae = float(np.abs(gt_raw - pred_raw).mean())
        mae_per_dim = [float(v) for v in np.abs(gt_raw - pred_raw).mean(axis=0)]

        result = {
            "sample_idx": idx,
            "language": sample["language"],
            "mse": mse,
            "mae": mae,
            "mae_per_dim": dict(zip(DIM_NAMES, mae_per_dim)),
        }
        results.append(result)

        print(f'  Sample {idx}: "{sample["language"][:50]}"')
        print(f"    MSE={mse:.6f}  MAE={mae:.6f}")
        print(f"    Per-dim MAE: {' '.join(f'{n}={v:.4f}' for n, v in zip(DIM_NAMES, mae_per_dim))}")

    avg_mse = np.mean([r["mse"] for r in results])
    avg_mae = np.mean([r["mae"] for r in results])
    print("=" * 70)
    print(f"Average over {n} samples:  MSE={avg_mse:.6f}  MAE={avg_mae:.6f}")

    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / f"inference_{args.split}.json"
        with open(json_path, "w") as f:
            json.dump({"split": args.split, "n_steps": n_steps, "results": results}, f, indent=2)
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
