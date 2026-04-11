#!/usr/bin/env python3
"""Compare baseline (delay=0) vs RTC (delay=15) on train set overfitting.

Usage:
    python compare/compare_rtc.py
    python compare/compare_rtc.py --ckpt-baseline checkpoint_delay0.pt --ckpt-rtc checkpoint_delay15.pt

Produces:
  - result/vla/compare_rtc_trajectories.png
  - result/vla/compare_rtc_3d.png
  - result/vla/compare_rtc_errors.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import qwen.vla.config

sys.modules["qwen3vl.config"] = qwen.vla.config
sys.modules["weasel_vla.config"] = qwen.vla.config

from qwen.vla.config import PipelineConfig  # noqa: E402
from qwen.vla.data.lerobot_calvin import LeRobotCalvinDataset  # noqa: E402
from qwen.vla.models.vla import VLAPolicy  # noqa: E402

DIM_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
RESULT_DIR = Path("result/vla")


def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: PipelineConfig = ckpt["config"]
    policy = VLAPolicy(config.vlm, config.action_expert)
    policy.obs_proj.load_state_dict(ckpt["obs_proj"])
    policy.action_expert.load_state_dict(ckpt["action_expert"])
    policy.eval()
    return policy, ckpt["action_q01"], ckpt["action_q99"], config


def denormalize(actions, q01, q99):
    return ((actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01).numpy()


def delta_to_traj(deltas):
    pos = np.zeros((deltas.shape[0] + 1, 3))
    pos[1:] = np.cumsum(deltas[:, :3], axis=0)
    return pos


@torch.inference_mode()
def predict(policy, hidden, chunk_size, n_steps, device):
    hidden = hidden.to(device)
    obs_embed = policy.obs_proj(hidden)
    actions = policy.action_expert.denoise(obs_embed, chunk_size=chunk_size, n_steps=n_steps)
    return actions.cpu().float().squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-baseline", default="checkpoint_delay0.pt")
    parser.add_argument("--ckpt-rtc", default="checkpoint_delay15.pt")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Load baseline
    print("Loading baseline model...")
    pol_b, q01_b, q99_b, cfg_b = load_model(args.ckpt_baseline)

    print("Loading dataset...")
    train_ds = LeRobotCalvinDataset(cfg_b.data, split="train", action_q01=q01_b, action_q99=q99_b)
    n = min(args.n_samples, len(train_ds))
    indices = random.sample(range(len(train_ds)), n)
    print(f"  {n} samples selected: {indices}")

    # Cache VLM embeddings
    print("Caching VLM embeddings...")
    pol_b.vlm.eval()
    vlm_cache = {}
    for idx in indices:
        sample = train_ds[idx]
        hidden = pol_b.vlm_forward_hidden(sample["images"], sample["language"])
        vlm_cache[idx] = hidden.cpu()
        del hidden
        torch.cuda.empty_cache()

    pol_b.vlm.to("cpu")
    torch.cuda.empty_cache()
    pol_b.obs_proj.to(device, dtype=torch.bfloat16)
    pol_b.action_expert.to(device, dtype=torch.bfloat16)

    print("Predicting with baseline...")
    baseline_results = {}
    for idx in indices:
        sample = train_ds[idx]
        pred_norm = predict(
            pol_b, vlm_cache[idx], cfg_b.data.chunk_size, cfg_b.flow_matching.denoise_steps_inference, device
        )
        baseline_results[idx] = (
            denormalize(sample["actions"], q01_b, q99_b),
            denormalize(pred_norm, q01_b, q99_b),
            sample["language"],
        )

    del pol_b
    torch.cuda.empty_cache()

    # Load RTC model
    print("Loading RTC model...")
    pol_r, q01_r, q99_r, cfg_r = load_model(args.ckpt_rtc)
    pol_r.vlm.to("cpu")
    torch.cuda.empty_cache()
    pol_r.obs_proj.to(device, dtype=torch.bfloat16)
    pol_r.action_expert.to(device, dtype=torch.bfloat16)

    train_ds_r = LeRobotCalvinDataset(cfg_r.data, split="train", action_q01=q01_r, action_q99=q99_r)

    print("Predicting with RTC...")
    rtc_results = {}
    for idx in indices:
        sample = train_ds_r[idx]
        pred_norm = predict(
            pol_r, vlm_cache[idx], cfg_r.data.chunk_size, cfg_r.flow_matching.denoise_steps_inference, device
        )
        rtc_results[idx] = (
            denormalize(sample["actions"], q01_r, q99_r),
            denormalize(pred_norm, q01_r, q99_r),
            sample["language"],
        )

    del pol_r
    torch.cuda.empty_cache()

    # ==================== Metrics ====================
    print("\n" + "=" * 70)
    print("METRICS: Baseline (delay=0) vs RTC (delay=15)")
    print("=" * 70)

    for label, results in [("Baseline", baseline_results), ("RTC-15", rtc_results)]:
        max_errs, end_errs, mae_list = [], [], []
        for idx in indices:
            gt_raw, pred_raw, _ = results[idx]
            gt_traj = delta_to_traj(gt_raw)
            pred_traj = delta_to_traj(pred_raw)
            pos_err = np.linalg.norm(gt_traj - pred_traj, axis=1)
            max_errs.append(pos_err.max())
            end_errs.append(pos_err[-1])
            mae_list.append(np.abs(gt_raw - pred_raw).mean())

        avg_max = np.mean(max_errs)
        avg_end = np.mean(end_errs)
        avg_mae = np.mean(mae_list)
        ratio = avg_max / max(avg_end, 1e-8)
        print(f"\n  [{label}]")
        print(f"    avg max_pos_err = {avg_max:.4f}")
        print(f"    avg endpoint_err = {avg_end:.4f}")
        print(f"    max/endpoint ratio = {ratio:.2f}")
        print(f"    avg MAE (all dims) = {avg_mae:.4f}")

    # ==================== Plots ====================

    # 1. Per-dim trajectory comparison
    fig, axes = plt.subplots(n, 7, figsize=(28, 4 * n), squeeze=False)
    fig.suptitle("Train Overfit: GT vs Baseline vs RTC-15", fontsize=16, y=1.01)
    for row_i, idx in enumerate(indices):
        gt_b, pred_b, lang = baseline_results[idx]
        _, pred_r, _ = rtc_results[idx]
        timesteps = np.arange(gt_b.shape[0])
        for col in range(7):
            ax = axes[row_i, col]
            ax.plot(timesteps, gt_b[:, col], "b-", lw=1.5, label="GT")
            ax.plot(timesteps, pred_b[:, col], "r--", lw=1.2, alpha=0.8, label="Baseline")
            ax.plot(timesteps, pred_r[:, col], "g-.", lw=1.2, alpha=0.8, label="RTC-15")
            if row_i == 0:
                ax.set_title(DIM_NAMES[col], fontweight="bold")
            if col == 0:
                lbl = lang[:25] + "..." if len(lang) > 25 else lang
                ax.set_ylabel(f"#{idx}\n{lbl}", fontsize=8)
            ax.tick_params(labelsize=7)
            if row_i == 0 and col == 6:
                ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "compare_rtc_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {RESULT_DIR / 'compare_rtc_trajectories.png'}")

    # 2. 3D trajectory comparison
    fig = plt.figure(figsize=(6 * n, 6))
    fig.suptitle("3D Trajectories: GT (blue) vs Baseline (red) vs RTC-15 (green)", fontsize=14, y=1.02)
    for i, idx in enumerate(indices):
        gt_raw, pred_b, lang = baseline_results[idx]
        _, pred_r, _ = rtc_results[idx]
        gt_traj = delta_to_traj(gt_raw)
        b_traj = delta_to_traj(pred_b)
        r_traj = delta_to_traj(pred_r)

        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], "b-", lw=1.5, label="GT")
        ax.plot(b_traj[:, 0], b_traj[:, 1], b_traj[:, 2], "r--", lw=1.2, label="Baseline")
        ax.plot(r_traj[:, 0], r_traj[:, 1], r_traj[:, 2], "g-.", lw=1.2, label="RTC-15")
        ax.scatter(*gt_traj[0], c="blue", s=50, marker="o")
        ax.scatter(*gt_traj[-1], c="blue", s=80, marker="x", linewidths=2)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        title = lang[:25] + "..." if len(lang) > 25 else lang
        ax.set_title(title, fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "compare_rtc_3d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {RESULT_DIR / 'compare_rtc_3d.png'}")

    # 3. Cumulative position error
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for idx in indices:
        gt_raw, pred_b, _ = baseline_results[idx]
        _, pred_r, _ = rtc_results[idx]
        gt_traj = delta_to_traj(gt_raw)
        ax.plot(np.linalg.norm(gt_traj - delta_to_traj(pred_b), axis=1), "r-", alpha=0.5, lw=1)
        ax.plot(np.linalg.norm(gt_traj - delta_to_traj(pred_r), axis=1), "g-", alpha=0.5, lw=1)
    ax.plot([], [], "r-", lw=2, label="Baseline")
    ax.plot([], [], "g-", lw=2, label="RTC-15")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position Error (Euclidean)")
    ax.set_title("Cumulative Position Error Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    all_b = [
        np.linalg.norm(delta_to_traj(baseline_results[i][0]) - delta_to_traj(baseline_results[i][1]), axis=1)
        for i in indices
    ]
    all_r = [
        np.linalg.norm(delta_to_traj(rtc_results[i][0]) - delta_to_traj(rtc_results[i][1]), axis=1) for i in indices
    ]
    avg_b, avg_r = np.mean(all_b, axis=0), np.mean(all_r, axis=0)
    ax.plot(avg_b, "r-", lw=2, label=f"Baseline (max={avg_b.max():.3f}, end={avg_b[-1]:.3f})")
    ax.plot(avg_r, "g-", lw=2, label=f"RTC-15 (max={avg_r.max():.3f}, end={avg_r[-1]:.3f})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Avg Position Error")
    ax.set_title("Averaged Cumulative Position Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "compare_rtc_errors.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {RESULT_DIR / 'compare_rtc_errors.png'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
