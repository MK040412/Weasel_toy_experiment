"""Download calvin-task-ABCD-D-lerobot to RAM (/dev/shm) and cache VLM embeddings to disk.

Flow:
  1. Download parquet files to /dev/shm (tmpfs, 201 GB available)
  2. Build dataset from RAM-resident parquets
  3. Run VLM forward pass → save embeddings as parquet to disk (~12 GB)
  4. /dev/shm freed on exit (or reboot)

Usage:
  # Download only (keep in RAM for subsequent training)
  python data/download/fywang/calvin-task-ABCD-D-lerobot.py

  # Download + VLM cache in one shot
  python data/download/fywang/calvin-task-ABCD-D-lerobot.py --cache-vlm

  # Use custom output dir for VLM cache
  python data/download/fywang/calvin-task-ABCD-D-lerobot.py --cache-vlm --output-dir result/vla_abcd
"""

import argparse
import os
import shutil
import time

REPO_ID = "fywang/calvin-task-ABCD-D-lerobot"
RAM_DIR = "/dev/shm/hf_calvin_abcd"


def download_to_ram():
    """Download dataset to /dev/shm (tmpfs). No disk writes."""
    from huggingface_hub import snapshot_download

    avail_gb = shutil.disk_usage("/dev/shm").free / (1024**3)
    print(f"/dev/shm available: {avail_gb:.0f} GB")
    if avail_gb < 80:
        print(f"WARNING: need ~70 GB free, only {avail_gb:.0f} GB available")

    os.makedirs(RAM_DIR, exist_ok=True)
    print(f"Downloading {REPO_ID} → {RAM_DIR}")
    t0 = time.time()

    snapshot_dir = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        cache_dir=RAM_DIR,
        allow_patterns=["data/**/*.parquet", "meta/*"],
    )

    elapsed = time.time() - t0
    size_gb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(RAM_DIR)
        for f in fns
    ) / (1024**3)
    print(f"Done: {size_gb:.1f} GB in {elapsed:.0f}s ({size_gb / elapsed * 1024:.0f} MB/s)")
    print(f"Snapshot dir: {snapshot_dir}")
    return snapshot_dir


def cache_vlm_embeddings(snapshot_dir: str, output_dir: str):
    """Run VLM forward on all chunks, save embeddings to disk as parquet."""
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from flax import nnx
    from transformers import AutoTokenizer

    from qwen.qwen3vl import modeling as qwen3vl
    from qwen.vla.data.lerobot_calvin import CalvinDataset
    from qwen.vla.models.vla import VLAPolicy
    from qwen.vla.training.trainer import _prepare_vision_inputs

    MODEL_PATH = os.environ.get("QWEN3VL_MODEL_PATH", "/home/perelman/models/qwen3-vl-2b")
    cache_dir = os.path.join(output_dir, "vlm_cache")
    cache_file = os.path.join(cache_dir, "embeddings.parquet")
    meta_file = os.path.join(cache_dir, "meta.json")

    if os.path.exists(cache_file):
        print(f"VLM cache already exists: {cache_file}")
        return

    # Point HF cache to RAM so CalvinDataset finds the downloaded snapshot
    os.environ["HF_HOME"] = RAM_DIR
    print("Loading dataset from RAM...")
    ds = CalvinDataset(repo_id=REPO_ID, split="train", chunk_size=50)
    n = len(ds)
    print(f"  {n} training chunks")

    print("Loading Qwen3-VL 2B...")
    config = qwen3vl.ModelConfig.qwen3vl_2b()
    vlm = qwen3vl.Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, config=config)
    policy = VLAPolicy(vlm=vlm, vlm_hidden_dim=2048, rngs=nnx.Rngs(params=42))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    print(f"Caching VLM embeddings for {n} chunks...")
    obs_bytes_list, act_bytes_list, grip_bytes_list = [], [], []
    max_seq = 0
    d_model = None
    t0 = time.time()

    for i in range(n):
        sample = ds[i]
        vlm_inputs = _prepare_vision_inputs(sample["images"][0], sample["language"], tokenizer)
        hidden = policy.vlm.get_hidden_states(
            vlm_inputs["input_ids"],
            vlm_inputs["pixel_values"],
            vlm_inputs["image_grid_thw"],
            vlm_inputs["token_type_ids"],
        )
        obs_embed = np.array(policy.obs_proj(hidden)[0])  # (seq, d_model)
        max_seq = max(max_seq, obs_embed.shape[0])
        if d_model is None:
            d_model = obs_embed.shape[1]

        obs_bytes_list.append(obs_embed.astype(np.float32).tobytes())
        act_bytes_list.append(sample["actions_continuous"].astype(np.float32).tobytes())
        grip_bytes_list.append(sample["gripper"].astype(np.float32).tobytes())

        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  {i + 1}/{n}  ({rate:.1f} samples/s, ETA {eta / 60:.0f} min)")

    # Save to disk
    os.makedirs(cache_dir, exist_ok=True)
    table = pa.table({
        "obs": pa.array(obs_bytes_list, type=pa.binary()),
        "actions": pa.array(act_bytes_list, type=pa.binary()),
        "gripper": pa.array(grip_bytes_list, type=pa.binary()),
    })
    pq.write_table(table, cache_file)

    meta = {"n_samples": n, "max_seq_len": max_seq, "d_model": d_model, "chunk_size": 50}
    with open(meta_file, "w") as f:
        json.dump(meta, f)

    size_gb = os.path.getsize(cache_file) / (1024**3)
    print(f"Saved: {cache_file} ({size_gb:.1f} GB)")

    # Free VLM
    del vlm, policy
    jax.clear_caches()
    print("VLM released.")


def cleanup_ram():
    """Remove downloaded data from /dev/shm."""
    if os.path.exists(RAM_DIR):
        shutil.rmtree(RAM_DIR)
        print(f"Cleaned up {RAM_DIR}")


def main():
    parser = argparse.ArgumentParser(description=f"Download {REPO_ID} to RAM")
    parser.add_argument("--cache-vlm", action="store_true", help="Also compute & save VLM embeddings")
    parser.add_argument("--output-dir", default="result/vla_abcd", help="Disk dir for VLM cache")
    parser.add_argument("--cleanup", action="store_true", help="Remove RAM data after caching")
    args = parser.parse_args()

    snapshot_dir = download_to_ram()

    if args.cache_vlm:
        cache_vlm_embeddings(snapshot_dir, args.output_dir)
        if args.cleanup:
            cleanup_ram()

    print("\nDone. Next steps:")
    if not args.cache_vlm:
        print(f"  # Generate VLM cache:")
        print(f"  python {__file__} --cache-vlm --output-dir {args.output_dir} --cleanup")
    print(f"  # Train with RTC:")
    print(f"  python src/qwen/vla/train.py --simulated-delay 15 --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
