"""LeRobot CALVIN dataset loader for JAX VLA training.

Loads from HuggingFace fywang/calvin-*-lerobot (LeRobot v2.1 format).
Uses PyArrow for fast parquet loading — no HF `datasets` overhead.

Implements VLADataset protocol (see data/protocol.py).
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

from qwen.vla.data.protocol import register_dataset


@register_dataset("calvin-debug")
@register_dataset("calvin-abcd")
class CalvinDataset:
    """CALVIN debug dataset via PyArrow (4x faster than HF datasets).

    Each sample: images (n_cameras, H, W, 3), actions (T, 6), gripper (T, 1), language.
    Actions normalized to [-1, 1] via quantile normalization.
    Gripper (dim 6) binarized to {0, 1}.
    """

    def __init__(
        self,
        repo_id: str = "fywang/calvin-debug-lerobot",
        split: str = "train",
        cameras: list[str] | None = None,
        chunk_size: int = 50,
        image_size: int = 336,
        local_path: str | None = None,
    ):
        self.repo_id = repo_id
        self.split = split
        self.cameras = cameras or ["top", "wrist"]
        self.chunk_size = chunk_size
        self.image_size = image_size

        if local_path and Path(local_path).exists():
            self._snapshot_dir = local_path
        else:
            self._snapshot_dir = snapshot_download(
                repo_id, repo_type="dataset", allow_patterns=["data/**/*.parquet", "meta/*"]
            )

        # Collect parquets from all chunk dirs
        data_dir = Path(self._snapshot_dir) / "data"
        self._parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))

        # Load metadata
        self._load_tasks()

        # Phase 1: fast metadata load (actions + indices only, no images)
        self._load_metadata_fast()

        # Build episodes, chunks, quantiles
        self._build_episodes(split)
        self._build_chunks()
        self._compute_quantiles()

        # Episode-level caches
        self._action_cache: dict[int, np.ndarray] = {}
        self._image_table_cache: dict[int, pa.Table] = {}

    def _load_tasks(self):
        """Load task descriptions from meta/tasks.jsonl."""
        tasks_path = Path(self._snapshot_dir) / "meta" / "tasks.jsonl"
        self.tasks = {}
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    task = json.loads(line)
                    self.tasks[task["task_index"]] = task["task"]

    def _load_metadata_fast(self):
        """Load only metadata columns from all parquets in parallel (~10ms)."""
        meta_cols = ["action", "episode_index", "task_index", "frame_index", "observation.state"]

        def _read_one(path):
            return pq.read_table(path, columns=meta_cols)

        n_workers = min(32, len(self._parquet_files))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            tables = list(pool.map(_read_one, self._parquet_files))

        import pyarrow as pa

        combined = pa.concat_tables(tables)
        self._all_actions = np.array(combined["action"].to_pylist(), dtype=np.float32)
        self._all_states = np.array(combined["observation.state"].to_pylist(), dtype=np.float32)
        self._all_episode_idx = np.array(combined["episode_index"].to_pylist(), dtype=np.int64)
        self._all_task_idx = np.array(combined["task_index"].to_pylist(), dtype=np.int64)
        self._all_frame_idx = np.array(combined["frame_index"].to_pylist(), dtype=np.int64)
        self._n_rows = len(self._all_actions)

        # Map episode -> parquet file for image loading
        self._ep_to_file: dict[int, Path] = {}
        for f in self._parquet_files:
            ep_num = int(f.stem.split("_")[-1])
            self._ep_to_file[ep_num] = f

    def _build_episodes(self, split: str):
        """Group rows by episode, filter by split."""
        all_episodes = np.unique(self._all_episode_idx)
        n_total = len(all_episodes)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)

        if split == "train":
            self.episodes = all_episodes[:n_train]
        elif split == "val":
            self.episodes = all_episodes[n_train : n_train + n_val]
        else:
            self.episodes = all_episodes[n_train + n_val :]

        self.episode_ranges = {}
        for ep in self.episodes:
            mask = self._all_episode_idx == ep
            rows = np.where(mask)[0]
            if len(rows) > 0:
                self.episode_ranges[ep] = (rows[0], rows[-1] + 1)

    def _build_chunks(self):
        """Build (episode, start_row) pairs for chunk extraction."""
        self.chunks = []
        for ep, (start, end) in self.episode_ranges.items():
            ep_len = end - start
            if ep_len < self.chunk_size:
                continue
            for i in range(0, ep_len - self.chunk_size + 1, self.chunk_size // 2):
                self.chunks.append((ep, start + i))

    def _compute_quantiles(self):
        """Compute q01/q99 for action and proprio normalization."""
        all_actions, all_states = [], []
        for _, (start, end) in self.episode_ranges.items():
            all_actions.append(self._all_actions[start:end])
            all_states.append(self._all_states[start:end])

        if all_actions:
            cat = np.concatenate(all_actions, axis=0)
            self.q01 = np.percentile(cat, 1, axis=0).astype(np.float32)
            self.q99 = np.percentile(cat, 99, axis=0).astype(np.float32)
            cat_s = np.concatenate(all_states, axis=0)
            self.q01_state = np.percentile(cat_s, 1, axis=0).astype(np.float32)
            self.q99_state = np.percentile(cat_s, 99, axis=0).astype(np.float32)
        else:
            self.q01 = np.zeros(7, dtype=np.float32)
            self.q99 = np.ones(7, dtype=np.float32)
            self.q01_state = np.zeros(15, dtype=np.float32)
            self.q99_state = np.ones(15, dtype=np.float32)

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        return (actions - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0

    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        return (actions + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.q01_state) / (self.q99_state - self.q01_state + 1e-6) * 2.0 - 1.0

    def _get_episode_images(self, ep: int) -> pa.Table:
        """Lazy-load full parquet table for image access (cached per episode)."""
        if ep not in self._image_table_cache:
            parquet_file = self._ep_to_file.get(ep)
            if parquet_file is None:
                raise ValueError(f"No parquet file for episode {ep}")
            img_cols = [f"observation.images.{cam}" for cam in self.cameras]
            self._image_table_cache[ep] = pq.read_table(parquet_file, columns=img_cols)
        return self._image_table_cache[ep]

    def _decode_image(self, img_struct: dict) -> np.ndarray:
        """Decode PNG bytes from parquet struct to numpy array."""
        img_bytes = img_struct["bytes"]
        if isinstance(img_bytes, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            img = Image.open(io.BytesIO(img_bytes.as_py())).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    @property
    def action_dim(self) -> int:
        return self._all_actions.shape[1]

    @property
    def proprio_dim(self) -> int:
        return self._all_states.shape[1]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        ep, start = self.chunks[idx]
        end = start + self.chunk_size

        # Actions from pre-loaded numpy (instant, no I/O)
        raw_actions = self._all_actions[start:end].copy()
        actions_norm = self.normalize_actions(raw_actions)  # all 7 dims normalized
        # Proprio: first frame state (conditioning for action expert)
        proprio = self.normalize_state(self._all_states[start:start + 1])  # (1, 15)

        # Images from first frame (lazy parquet load + PNG decode)
        ep_start = self.episode_ranges[ep][0]
        frame_in_ep = start - ep_start
        img_table = self._get_episode_images(ep)
        images = []
        for cam in self.cameras:
            col = f"observation.images.{cam}"
            img_struct = img_table[col][frame_in_ep].as_py()
            images.append(self._decode_image(img_struct))

        # Language
        task_idx = int(self._all_task_idx[start])
        language = self.tasks.get(task_idx, "manipulate the object")

        return {
            "images": np.stack(images),  # (n_cameras, H, W, 3)
            "actions": actions_norm,  # (T, 7) all dims normalized
            "proprio": proprio,  # (1, 15) first frame state, normalized
            "raw_actions": raw_actions,  # (T, 7)
            "language": language,
            "episode": int(ep),
        }


def collate_batch(samples: list[dict]) -> dict:
    """Collate samples into batched numpy arrays for jax.device_put."""
    return {
        "images": np.stack([s["images"] for s in samples]),
        "actions_continuous": np.stack([s["actions_continuous"] for s in samples]),
        "gripper": np.stack([s["gripper"] for s in samples]),
        "raw_actions": np.stack([s["raw_actions"] for s in samples]),
        "language": [s["language"] for s in samples],
        "episode": np.array([s["episode"] for s in samples]),
    }
