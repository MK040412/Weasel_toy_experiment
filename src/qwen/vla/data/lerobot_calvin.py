"""CALVIN dataset in LeRobot v2.1 format from HuggingFace.

Pipeline:
  1. HuggingFace datasets loads parquet episodes
  2. episode-level grouping
  3. action chunking (sliding window)
  4. task_index -> language instruction mapping
  5. Action normalization (quantile: q01/q99 -> [-1, 1]) — openpi0.5 style
"""

from __future__ import annotations

import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from qwen.vla.config import DataConfig


class LeRobotCalvinDataset(Dataset):
    """CALVIN dataset in LeRobot v2.1 format from HuggingFace.

    Action normalization is critical: flow matching operates in normalized space,
    raw action scales differ across dimensions (position vs rotation vs gripper).

    Uses quantile normalization (openpi0.5 style): (x - q01) / (q99 - q01) * 2 - 1 -> [-1, 1]

    Attributes:
        action_q01: (action_dim,) 1st percentile of all actions in this split.
        action_q99: (action_dim,) 99th percentile of all actions in this split.
    """

    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        action_q01: torch.Tensor | None = None,
        action_q99: torch.Tensor | None = None,
    ):
        self.config = config
        self.split = split

        # 1. Load dataset from HuggingFace
        from datasets import load_dataset

        full_ds = load_dataset(config.repo_id)["train"]

        # 2. Load task descriptions from meta/tasks.jsonl
        from huggingface_hub import hf_hub_download

        tasks_path = hf_hub_download(config.repo_id, "meta/tasks.jsonl", repo_type="dataset")
        self.tasks: dict[int, str] = {}
        with open(tasks_path) as f:
            for line in f:
                obj = json.loads(line)
                self.tasks[obj["task_index"]] = obj["task"]

        # 3. Load info.json for splits
        info_path = hf_hub_download(config.repo_id, "meta/info.json", repo_type="dataset")
        with open(info_path) as f:
            info = json.load(f)

        # Parse split range (e.g. "0:9" -> episodes 0..8)
        if split == "train":
            split_key = "train"
        else:
            for key in [split, "val", "validation", "test"]:
                if key in info["splits"]:
                    split_key = key
                    break
            else:
                raise KeyError(f"No split found for '{split}' in {list(info['splits'].keys())}")
        split_range = info["splits"][split_key]
        start_ep, end_ep = map(int, split_range.split(":"))

        # 4. Filter by split episodes
        self.data = full_ds.filter(lambda x: start_ep <= x["episode_index"] < end_ep)

        # 5. Compute or reuse action quantiles (openpi0.5 style)
        if action_q01 is not None and action_q99 is not None:
            self.action_q01 = action_q01
            self.action_q99 = action_q99
        else:
            self.action_q01, self.action_q99 = self._compute_action_quantiles()

        # 6. Group by episode + build chunk indices
        self.episodes: dict[int, list[int]] = {}
        for i in range(len(self.data)):
            ep = self.data[i]["episode_index"]
            self.episodes.setdefault(ep, []).append(i)

        self.samples: list[tuple[int, int, list[int]]] = []
        for ep_idx, row_indices in sorted(self.episodes.items()):
            n_frames = len(row_indices)
            if n_frames >= config.chunk_size:
                for start in range(n_frames - config.chunk_size + 1):
                    self.samples.append((ep_idx, start, row_indices))
            else:
                self.samples.append((ep_idx, 0, row_indices))

        # Image resize for Qwen3VL
        self.resize = transforms.Resize(
            (config.resize_to, config.resize_to),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

    def _compute_action_quantiles(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute q01/q99 per action dimension (openpi0.5 style)."""
        all_actions = torch.tensor(self.data["action"], dtype=torch.float32)
        q01 = torch.quantile(all_actions, 0.01, dim=0)
        q99 = torch.quantile(all_actions, 0.99, dim=0)
        return q01, q99

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start, row_indices = self.samples[idx]
        chunk_size = self.config.chunk_size
        n_frames = len(row_indices)

        end = min(start + chunk_size, n_frames)
        chunk_indices = row_indices[start:end]

        first_row = self.data[chunk_indices[0]]

        images = []
        for cam in self.config.cameras:
            key = f"observation.images.{cam}"
            img = first_row[key]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = self.resize(img)
            images.append(img)

        task_idx = first_row["task_index"]
        language = self.tasks.get(task_idx, "")

        actions_raw = [self.data[i]["action"] for i in chunk_indices]
        actions = torch.tensor(actions_raw, dtype=torch.float32)

        if actions.shape[0] < chunk_size:
            pad_size = chunk_size - actions.shape[0]
            last_action = actions[-1:].expand(pad_size, -1)
            actions = torch.cat([actions, last_action], dim=0)

        actions = (actions - self.action_q01) / (self.action_q99 - self.action_q01 + 1e-6) * 2.0 - 1.0

        return {
            "images": images,
            "actions": actions,
            "language": language,
        }


def calvin_collate_fn(batch: list[dict]) -> dict:
    """Collate for batch_size=1 (primary use case on RTX 4060)."""
    return {
        "images": batch[0]["images"],
        "actions": batch[0]["actions"].unsqueeze(0),
        "language": batch[0]["language"],
    }
