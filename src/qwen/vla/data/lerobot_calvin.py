"""LeRobot CALVIN dataset loader for JAX VLA training.

Loads from HuggingFace fywang/calvin-debug-lerobot (LeRobot v2.1 format).
Returns numpy arrays for JAX consumption.
"""

import json

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image


class CalvinDataset:
    """CALVIN debug dataset in LeRobot v2.1 format.

    Each sample: images (list of PIL), actions (chunk_size, 7), language instruction.
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
    ):
        self.repo_id = repo_id
        self.split = split
        self.cameras = cameras or ["top", "wrist"]
        self.chunk_size = chunk_size
        self.image_size = image_size

        # Load dataset
        self.ds = load_dataset(repo_id, split="train")

        # Load metadata
        tasks_path = hf_hub_download(repo_id, "meta/tasks.jsonl", repo_type="dataset")
        self.tasks = {}
        with open(tasks_path) as f:
            for line in f:
                task = json.loads(line)
                self.tasks[task["task_index"]] = task["task"]

        info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
        with open(info_path) as f:
            self.info = json.load(f)

        # Build episode index ranges
        self._build_episodes(split)
        self._build_chunks()
        self._compute_quantiles()

    def _build_episodes(self, split: str):
        """Group dataset rows by episode, filter by split."""
        ep_indices = np.array(self.ds["episode_index"])
        all_episodes = np.unique(ep_indices)
        n_total = len(all_episodes)

        # Simple split: 80% train, 10% val, 10% test
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        if split == "train":
            self.episodes = all_episodes[:n_train]
        elif split == "val":
            self.episodes = all_episodes[n_train : n_train + n_val]
        else:
            self.episodes = all_episodes[n_train + n_val :]

        # Build per-episode row ranges
        self.episode_ranges = {}
        for ep in self.episodes:
            mask = ep_indices == ep
            rows = np.where(mask)[0]
            self.episode_ranges[ep] = (rows[0], rows[-1] + 1)

    def _build_chunks(self):
        """Build (episode, start_idx) pairs for chunk extraction."""
        self.chunks = []
        for ep, (start, end) in self.episode_ranges.items():
            ep_len = end - start
            if ep_len < self.chunk_size:
                continue
            for i in range(0, ep_len - self.chunk_size + 1, self.chunk_size // 2):
                self.chunks.append((ep, start + i))

    def _compute_quantiles(self):
        """Compute q01/q99 for action normalization (from training data only)."""
        all_actions = []
        for ep, (start, end) in self.episode_ranges.items():
            actions = np.array(self.ds[start:end]["action"])
            all_actions.append(actions)

        if all_actions:
            all_actions = np.concatenate(all_actions, axis=0)
            self.q01 = np.percentile(all_actions, 1, axis=0).astype(np.float32)
            self.q99 = np.percentile(all_actions, 99, axis=0).astype(np.float32)
        else:
            self.q01 = np.zeros(7, dtype=np.float32)
            self.q99 = np.ones(7, dtype=np.float32)

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Quantile normalize to [-1, 1]."""
        return (actions - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0

    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Inverse quantile normalization."""
        return (actions + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01

    def _load_image(self, row: dict, camera: str) -> np.ndarray:
        """Load and resize image from dataset row."""
        key = f"observation.images.{camera}"
        img = row[key]
        if isinstance(img, dict) and "path" in img:
            img = Image.open(img["path"]).convert("RGB")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        ep, start = self.chunks[idx]
        end = start + self.chunk_size
        rows = self.ds[start:end]

        # Images from first frame
        images = []
        for cam in self.cameras:
            first_row = self.ds[start]
            img = self._load_image(first_row, cam)
            images.append(img)

        # Actions
        actions = np.array(rows["action"], dtype=np.float32)
        raw_actions = actions.copy()

        # Binarize gripper (dim 6): threshold at 0
        gripper = (actions[:, 6:7] > 0).astype(np.float32)

        # Normalize continuous dims (0:6)
        actions[:, :6]
        continuous_norm = self.normalize_actions(actions)[:, :6]

        # Language
        task_idx = rows["task_index"][0]
        language = self.tasks.get(task_idx, "manipulate the object")

        return {
            "images": np.stack(images),  # (n_cameras, H, W, 3)
            "actions_continuous": continuous_norm,  # (T, 6) normalized
            "gripper": gripper,  # (T, 1) binary {0, 1}
            "raw_actions": raw_actions,  # (T, 7) original scale
            "language": language,
            "episode": int(ep),
        }


def collate_batch(samples: list[dict]) -> dict:
    """Collate list of samples into batched numpy arrays."""
    return {
        "images": np.stack([s["images"] for s in samples]),
        "actions_continuous": np.stack([s["actions_continuous"] for s in samples]),
        "gripper": np.stack([s["gripper"] for s in samples]),
        "raw_actions": np.stack([s["raw_actions"] for s in samples]),
        "language": [s["language"] for s in samples],
        "episode": np.array([s["episode"] for s in samples]),
    }
