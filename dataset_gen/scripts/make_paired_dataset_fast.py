# scripts/make_paired_dataset_fast.py
"""Parallelized version — splits episodes across worker processes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from multiprocessing import Pool, current_process

# ★ 렌더링 전에 스레드 제한 (반드시 import 전에)
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import mujoco
import numpy as np
import pandas as pd
import zarr
from numcodecs import Zstd

_OGBENCH_DIR = Path(__file__).resolve().parent.parent / "ogbench"
if _OGBENCH_DIR.exists():
    sys.path.insert(0, str(_OGBENCH_DIR))

import ogbench.locomaze
import gymnasium
from ogbench.utils import DEFAULT_DATASET_DIR, download_datasets

# ---------------------------------------------------------------------------
# Constants (same as before)
# ---------------------------------------------------------------------------
MAZE_SIZES = ("medium", "large", "giant")
ENV_NAME_MAP = {
    "medium": "antmaze-medium-v0",
    "large":  "antmaze-large-v0",
    "giant":  "antmaze-giant-v0",
}
DATASET_NAME_MAP = {
    "medium": "antmaze-medium-navigate-v0",
    "large":  "antmaze-large-navigate-v0",
    "giant":  "antmaze-giant-navigate-v0",
}
TOPDOWN_CAM_PARAMS = {
    "medium": dict(lookat_x=10.0, lookat_y=10.0, distance=30.0),
    "large":  dict(lookat_x=18.0, lookat_y=10.0, distance=50.0),
    "giant":  dict(lookat_x=26.0, lookat_y=18.0, distance=70.0),
}
TERMINAL_GOAL_FRACTION = 0.25

_COMPRESSOR = Zstd(level=1)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def make_topdown_camera(maze: str) -> mujoco.MjvCamera:
    p = TOPDOWN_CAM_PARAMS[maze]
    cam = mujoco.MjvCamera()
    cam.lookat[0] = p["lookat_x"]
    cam.lookat[1] = p["lookat_y"]
    cam.lookat[2] = 0.0
    cam.distance  = p["distance"]
    cam.elevation = -90.0
    cam.azimuth   = 0.0
    return cam


# ---------------------------------------------------------------------------
# Worker function — each process renders a chunk of episodes
# ---------------------------------------------------------------------------
def render_episode_chunk(args_tuple):
    """Worker: render a list of episodes and save to zarr."""
    (maze, image_size, zarr_path_str, npz_path,
     ep_indices, ep_starts, ep_ends) = args_tuple

    # Each worker creates its own env + renderer
    env = gymnasium.make(ENV_NAME_MAP[maze], render_mode="rgb_array",
                         width=image_size, height=image_size)
    env.reset()
    raw = env.unwrapped
    renderer = mujoco.Renderer(raw.model, height=image_size, width=image_size)

    td_cam = make_topdown_camera(maze)
    track_cam = mujoco.MjvCamera()
    track_cam.lookat[2] = 0.5
    track_cam.distance  = 8.0
    track_cam.elevation = -45.0
    track_cam.azimuth   = 45.0

    # Load data (each worker reads the npz — memory mapped would be better but ok)
    f = np.load(npz_path)
    qpos    = f["qpos"]
    qvel    = f["qvel"]
    actions = f["actions"]

    # Open zarr store (append mode — multiple workers write different episode groups)
    store = zarr.DirectoryStore(zarr_path_str)
    root  = zarr.open_group(store, mode="a")
    if "episodes" not in root:
        root.require_group("episodes")

    pid = current_process().name
    t0 = time.time()

    for count, (ep_idx, ep_start, ep_end) in enumerate(
        zip(ep_indices, ep_starts, ep_ends)
    ):
        # Skip if already done
        if str(ep_idx) in root["episodes"]:
            continue

        ep_len = ep_end - ep_start + 1
        obs_td  = np.empty((ep_len, image_size, image_size, 3), dtype=np.uint8)
        obs_3rd = np.empty((ep_len, image_size, image_size, 3), dtype=np.uint8)

        data = raw.data
        for t_local, t_global in enumerate(range(ep_start, ep_end + 1)):
            # Direct qpos/qvel write (skip gymnasium wrapper)
            data.qpos[:] = qpos[t_global]
            data.qvel[:] = qvel[t_global]
            mujoco.mj_forward(raw.model, data)

            # Top-down
            renderer.update_scene(data, camera=td_cam)
            obs_td[t_local] = renderer.render()

            # Tracking
            track_cam.lookat[0] = float(qpos[t_global, 0])
            track_cam.lookat[1] = float(qpos[t_global, 1])
            renderer.update_scene(data, camera=track_cam)
            obs_3rd[t_local] = renderer.render()

        # Save to zarr
        grp = root["episodes"].require_group(str(ep_idx))
        kwargs = dict(compressor=_COMPRESSOR, overwrite=True)
        grp.array("obs_topdown", obs_td,
                   chunks=(min(50, ep_len), image_size, image_size, 3), **kwargs)
        grp.array("obs_third", obs_3rd,
                   chunks=(min(50, ep_len), image_size, image_size, 3), **kwargs)
        grp.array("actions", actions[ep_start:ep_end+1],
                   chunks=(min(200, ep_len), 8), **kwargs)
        grp.array("qpos", qpos[ep_start:ep_end+1].astype(np.float32),
                   chunks=(min(200, ep_len), 15), **kwargs)
        grp.array("qvel", qvel[ep_start:ep_end+1].astype(np.float32),
                   chunks=(min(200, ep_len), 14), **kwargs)
        grp.array("ep_goal_xy", qpos[ep_end, :2][np.newaxis].astype(np.float32),
                   chunks=(1, 2), **kwargs)

        if (count + 1) % 5 == 0:
            elapsed = time.time() - t0
            eps_done = count + 1
            eps_per_min = eps_done / (elapsed / 60)
            eps_left = len(ep_indices) - eps_done
            eta_min = eps_left / eps_per_min if eps_per_min > 0 else float('inf')
            print(f"  [{pid}] {eps_done}/{len(ep_indices)} eps | "
                  f"{eps_per_min:.1f} eps/min | ETA {eta_min:.0f}min")

    renderer.close()
    env.close()
    return len(ep_indices)


# ---------------------------------------------------------------------------
# Hindsight relabeling (same logic, unchanged)
# ---------------------------------------------------------------------------
def build_training_samples(ep_idx, ep_start, ep_end, qpos, chunk_size, rng,
                           terminal_fraction=TERMINAL_GOAL_FRACTION):
    ep_len = ep_end - ep_start + 1
    if ep_len < chunk_size + 1:
        return []
    rows = []
    for i in range(ep_len - chunk_size):
        if i + chunk_size >= ep_len - 1:
            j = ep_len - 1
        elif rng.random() < terminal_fraction:
            j = ep_len - 1
        else:
            j = int(rng.integers(i + 1, ep_len))
        goal_xy = qpos[ep_start + j, :2]
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        rows.append(dict(
            ep_idx=ep_idx, step_i=i, step_j=j,
            goal_xy_x=gx, goal_xy_y=gy,
            language=f"Go to ({gx:.2f}, {gy:.2f}).",
        ))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    maze       = args.maze
    output_dir = Path(args.output_dir) / maze
    image_size = args.image_size
    chunk_size = args.chunk_size
    n_workers  = args.n_workers
    seed       = args.seed
    dataset_dir = os.path.expanduser(args.dataset_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path    = output_dir / "frames.zarr"
    parquet_path = output_dir / "samples.parquet"
    meta_path    = output_dir / "meta.json"

    # 1. Download
    dataset_name = DATASET_NAME_MAP[maze]
    print(f"[1/4] Downloading {dataset_name} ...")
    download_datasets([dataset_name], dataset_dir)
    npz_path = os.path.join(dataset_dir, f"{dataset_name}.npz")

    # 2. Parse episodes
    print("[2/4] Parsing episodes ...")
    f = np.load(npz_path)
    terminals = f["terminals"].astype(bool)
    qpos = f["qpos"]
    ends   = np.where(terminals)[0].tolist()
    starts = [0] + [e + 1 for e in ends[:-1]]
    n_eps  = len(starts)

    ep_lengths = [ends[i] - starts[i] + 1 for i in range(n_eps)]
    print(f"      {n_eps} episodes | {len(qpos):,} steps | "
          f"mean len {np.mean(ep_lengths):.0f}")

    # Filter already done
    if zarr_path.exists():
        store = zarr.DirectoryStore(str(zarr_path))
        root  = zarr.open_group(store, mode="r")
        done  = set(root["episodes"].keys()) if "episodes" in root else set()
    else:
        done = set()

    todo_mask = [str(i) not in done for i in range(n_eps)]
    todo_indices = [i for i in range(n_eps) if todo_mask[i]]
    print(f"      Already done: {len(done)} | Remaining: {len(todo_indices)}")

    if len(todo_indices) == 0:
        print("      All episodes rendered!")
    else:
        # 3. Parallel rendering
        print(f"[3/4] Rendering with {n_workers} workers ...")

        # Split episodes across workers
        chunks = np.array_split(todo_indices, n_workers)
        worker_args = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            chunk = chunk.tolist()
            worker_args.append((
                maze, image_size, str(zarr_path), npz_path,
                chunk,
                [starts[i] for i in chunk],
                [ends[i] for i in chunk],
            ))

        t0 = time.time()
        if n_workers == 1:
            # Single process (easier to debug)
            for wa in worker_args:
                render_episode_chunk(wa)
        else:
            with Pool(processes=n_workers) as pool:
                pool.map(render_episode_chunk, worker_args)

        elapsed = time.time() - t0
        print(f"      Rendering done: {elapsed/60:.1f} min "
              f"({len(todo_indices)*np.mean(ep_lengths)/elapsed:.0f} frames/s)")

    # 4. Hindsight relabeling
    print(f"[4/4] Generating hindsight samples ...")
    qpos_full = np.load(npz_path)["qpos"]
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_eps):
        rows.extend(build_training_samples(
            i, starts[i], ends[i], qpos_full, chunk_size, rng))

    df = pd.DataFrame(rows)
    df["maze"] = maze
    df["zarr_path"] = str(zarr_path)
    df["chunk_size"] = chunk_size
    df["image_size"] = image_size
    df.to_parquet(parquet_path, index=False)

    # Meta
    meta = dict(
        maze=maze, n_episodes=n_eps, n_steps=len(qpos),
        image_size=image_size, chunk_size=chunk_size,
        ep_length_mean=float(np.mean(ep_lengths)),
        n_train_samples=len(df),
    )
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n=== Done: {maze} ===")
    print(f"  zarr:    {zarr_path}")
    print(f"  parquet: {parquet_path} ({len(df):,} rows)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--maze", choices=MAZE_SIZES, default="medium")
    p.add_argument("--output_dir", default="~/data/antmaze")
    p.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--image_size", type=int, default=224,
                   help="Render resolution (width=height). Must be a multiple of 28 "
                        "(Qwen3-VL ViT patch size). 224=8×8 patches. Same for all maze sizes.")
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    run(args)