"""
validate_antmaze_dataset.py — Dataset validation & visualisation for AntMaze VLA.

Outputs per maze size (in <output_dir>/<maze>/):
  1. verify_ep{i}.mp4       — top-down + 3rd-person side-by-side with goal marker
                               (3 episodes per maze size)
  2. stats_report.json       — episode counts, length distribution, goal_xy heatmap data
  3. goal_xy_heatmap.png     — 2-D scatter of all episode goals
  4. subgoal_heatmap.png     — 2-D scatter of hindsight sub-goal samples

Usage:
    MUJOCO_GL=osmesa python scripts/validate_antmaze_dataset.py \\
        --zarr_dir /data/antmaze/medium \\
        --output_dir /data/antmaze/medium/validation \\
        --n_verify_eps 3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_zarr_episode(root: zarr.Group, ep_idx: int) -> dict:
    """Load one episode from the zarr store (all arrays to numpy)."""
    grp = root["episodes"][str(ep_idx)]
    return dict(
        obs_topdown = grp["obs_topdown"][...],   # (T, H, W, 3)
        obs_third   = grp["obs_third"][...],     # (T, H, W, 3)
        actions     = grp["actions"][...],       # (T, 8)
        qpos        = grp["qpos"][...],          # (T, 15)
        ep_goal_xy  = grp["ep_goal_xy"][0],      # (2,)
    )


def overlay_goal_marker(
    frame: np.ndarray,
    goal_xy: np.ndarray | None,
    cur_xy: np.ndarray | None,
    maze: str,
    radius: int = 8,
) -> np.ndarray:
    """Draw a red cross at the goal position and a green dot at current position.

    Converts world-space xy to pixel coordinates using the top-down camera
    parameters used during rendering.
    """
    from make_paired_dataset_fast import TOPDOWN_CAM_PARAMS  # noqa: PLC0415

    frame = frame.copy()
    h, w = frame.shape[:2]
    p = TOPDOWN_CAM_PARAMS[maze]

    # Top-down camera projects xy linearly to pixel coords.
    # At elevation -90° the view is purely bird's-eye.
    # Approximate: lookat is the image centre, distance sets the scale.
    # For a square image the visible range is ≈ distance * tan(fov/2) on each side.
    # MuJoCo default fov = 45°, so half-range ≈ distance * tan(22.5°) ≈ distance * 0.414
    half_range = p["distance"] * 0.414

    def xy_to_px(xy: np.ndarray) -> tuple[int, int]:
        # MuJoCo top-down (azimuth=0, elevation=-90): world-y → image right, world-x → image down
        # Apply 90° CCW correction: px uses -world_y, py uses world_x
        px = int(w / 2 - (xy[1] - p["lookat_y"]) / half_range * w / 2)
        py = int(h / 2 - (xy[0] - p["lookat_x"]) / half_range * h / 2)
        return px, py

    # Draw goal (red cross)
    if goal_xy is not None:
        gx, gy = xy_to_px(goal_xy)
        for d in range(-radius, radius + 1):
            for dx, dy in [(d, 0), (0, d)]:
                px, py = gx + dx, gy + dy
                if 0 <= px < w and 0 <= py < h:
                    frame[py, px] = [255, 0, 0]

    # Draw current position (green dot)
    if cur_xy is not None:
        cx, cy = xy_to_px(cur_xy)
        for dy in range(-radius // 2, radius // 2 + 1):
            for dx in range(-radius // 2, radius // 2 + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    frame[py, px] = [0, 255, 0]

    return frame


# ---------------------------------------------------------------------------
# MP4 rendering
# ---------------------------------------------------------------------------

def write_mp4(
    frames: list[np.ndarray],  # each (H, W*2, 3) RGB
    path: Path,
    fps: int = 10,
) -> None:
    """Write frames to MP4 using imageio + ffmpeg backend (most compatible)."""
    import imageio
    
    # imageio-ffmpeg 백엔드가 가장 호환성 좋음
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec='libx264',
        quality=7,           # 0-10, higher = better
        pixelformat='yuv420p',
        macro_block_size=2,  # 홀수 해상도 허용
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  video saved: {path}  ({len(frames)} frames)")

def make_verification_video(
    root:       zarr.Group,
    ep_idx:     int,
    maze:       str,
    out_path:   Path,
    max_frames: int = 2001,
) -> None:
    """Side-by-side top-down | 3rd-person video with goal overlay."""
    ep = load_zarr_episode(root, ep_idx)
    goal_xy = ep["ep_goal_xy"]
    n_frames = min(len(ep["obs_topdown"]), max_frames)

    combined = []
    for t in range(n_frames):
        td   = ep["obs_topdown"][t]
        trd  = ep["obs_third"][t]
        cur  = ep["qpos"][t, :2]

        td_marked  = overlay_goal_marker(td,  goal_xy, cur,  maze)
        trd_marked = overlay_goal_marker(trd, goal_xy=None, cur_xy=None, maze=maze)

        side_by_side = np.concatenate([td_marked, trd_marked], axis=1)
        combined.append(side_by_side)

    write_mp4(combined, out_path)
    print(f"  video saved: {out_path}  ({n_frames} frames)")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(
    root:        zarr.Group,
    df:          pd.DataFrame,
    out_dir:     Path,
    maze:        str,
) -> dict:
    """Compute episode + sample statistics and save heatmaps."""
    n_eps    = len(root["episodes"])
    ep_lengths = []
    goal_xys   = []

    for ep_idx in range(n_eps):
        grp = root["episodes"][str(ep_idx)]
        ep_lengths.append(len(grp["qpos"]))
        goal_xys.append(grp["ep_goal_xy"][0].tolist())

    goal_xys = np.array(goal_xys)

    stats = dict(
        maze=maze,
        n_episodes=n_eps,
        n_training_samples=len(df),
        ep_length=dict(
            mean=float(np.mean(ep_lengths)),
            std=float(np.std(ep_lengths)),
            min=int(np.min(ep_lengths)),
            max=int(np.max(ep_lengths)),
            p25=float(np.percentile(ep_lengths, 25)),
            p50=float(np.percentile(ep_lengths, 50)),
            p75=float(np.percentile(ep_lengths, 75)),
        ),
        goal_xy=dict(
            x_range=[float(goal_xys[:, 0].min()), float(goal_xys[:, 0].max())],
            y_range=[float(goal_xys[:, 1].min()), float(goal_xys[:, 1].max())],
        ),
    )

    # --- goal_xy heatmap ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(goal_xys[:, 0], goal_xys[:, 1], s=4, alpha=0.4, c="steelblue")
    ax.set_title(f"AntMaze-{maze}: Episode terminal goal_xy distribution")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "goal_xy_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"  goal_xy heatmap saved: {out_dir / 'goal_xy_heatmap.png'}")

    # --- subgoal heatmap (from parquet) ---
    if "goal_xy_x" in df.columns:
        sub_xy = df[["goal_xy_x", "goal_xy_y"]].values
        # Subsample for speed
        if len(sub_xy) > 20_000:
            idx = np.random.choice(len(sub_xy), 20_000, replace=False)
            sub_xy = sub_xy[idx]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub_xy[:, 0], sub_xy[:, 1], s=2, alpha=0.2, c="coral")
        ax.set_title(f"AntMaze-{maze}: Hindsight sub-goal_xy distribution")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out_dir / "subgoal_heatmap.png", dpi=120)
        plt.close(fig)
        print(f"  subgoal heatmap saved: {out_dir / 'subgoal_heatmap.png'}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    zarr_dir   = Path(args.zarr_dir)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_path    = zarr_dir / "frames.zarr"
    parquet_path = zarr_dir / "samples.parquet"
    meta_path    = zarr_dir / "meta.json"

    if not zarr_path.exists():
        raise FileNotFoundError(f"zarr store not found: {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    df   = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # Infer maze type from meta.json or directory name
    maze = args.maze
    if maze is None and meta_path.exists():
        with open(meta_path) as f:
            maze = json.load(f).get("maze", "medium")
    if maze is None:
        maze = zarr_dir.parent.name  # guess from directory
    print(f"Maze: {maze}")

    # 1. Verification videos
    n_eps = len(root["episodes"])
    n_verify = min(args.n_verify_eps, n_eps)
    ep_indices = np.linspace(0, n_eps - 1, n_verify, dtype=int).tolist()
    print(f"\n--- Verification videos (episodes {ep_indices}) ---")
    for ep_idx in ep_indices:
        out_path = out_dir / f"verify_ep{ep_idx:04d}.mp4"
        make_verification_video(root, ep_idx, maze, out_path)

    # 2. Statistics
    print("\n--- Statistics ---")
    stats = compute_stats(root, df, out_dir, maze)
    stats_path = out_dir / "stats_report.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  stats saved: {stats_path}")

    # Print summary
    print(f"\n=== {maze} validation complete ===")
    print(f"  Episodes:       {stats['n_episodes']}")
    print(f"  Train samples:  {stats['n_training_samples']:,}")
    print(f"  Ep length:      {stats['ep_length']['mean']:.0f} ± {stats['ep_length']['std']:.0f} "
          f"  (min {stats['ep_length']['min']}, max {stats['ep_length']['max']})")
    print(f"  Goal xy range:  x {stats['goal_xy']['x_range']}  y {stats['goal_xy']['y_range']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate AntMaze VLA dataset.")
    p.add_argument("--zarr_dir", required=True,
                   help="Directory containing frames.zarr + samples.parquet")
    p.add_argument("--output_dir", required=True,
                   help="Where to write verification MP4s and stats")
    p.add_argument("--maze", default=None,
                   help="Maze size (medium|large|giant); auto-detected from meta.json if omitted")
    p.add_argument("--n_verify_eps", type=int, default=3,
                   help="Number of episodes to render as verification videos")
    return p.parse_args()


if __name__ == "__main__":
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "osmesa"
    run(parse_args())
