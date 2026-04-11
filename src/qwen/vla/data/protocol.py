"""Dataset protocol and registry for VLA training.

All datasets must return dicts with:
  images:      (n_cameras, H, W, 3)  float32 [0, 1]
  actions:     (T, action_dim)       float32 normalized [-1, 1]
  proprio:     (1, proprio_dim)      float32 normalized [-1, 1]
  language:    str
  episode:     int
  raw_actions: (T, action_dim)       float32 original scale

Properties:
  action_dim, proprio_dim, chunk_size, q01, q99, q01_state, q99_state
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from qwen.vla.config import EnvConfig


@runtime_checkable
class VLADataset(Protocol):
    """Protocol for VLA training datasets."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict: ...

    @property
    def action_dim(self) -> int: ...

    @property
    def proprio_dim(self) -> int: ...

    @property
    def chunk_size(self) -> int: ...

    q01: np.ndarray
    q99: np.ndarray
    q01_state: np.ndarray
    q99_state: np.ndarray


DATASET_REGISTRY: dict[str, type] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class."""

    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


def create_dataset(env_config: EnvConfig, split: str = "train") -> VLADataset:
    """Create dataset from env config using registry."""
    key = env_config.name
    if key not in DATASET_REGISTRY:
        # Fallback: try repo_id-based lookup
        for rkey, cls in DATASET_REGISTRY.items():
            if rkey in env_config.repo_id:
                kw = dict(
                    repo_id=env_config.repo_id, split=split, cameras=env_config.cameras,
                    chunk_size=env_config.chunk_size, image_size=env_config.image_size,
                )
                if hasattr(env_config, "local_path") and env_config.local_path:
                    kw["local_path"] = env_config.local_path
                return cls(**kw)
        raise KeyError(f"No dataset registered for '{key}'. Available: {list(DATASET_REGISTRY.keys())}")

    cls = DATASET_REGISTRY[key]
    kwargs = dict(
        repo_id=env_config.repo_id,
        split=split,
        cameras=env_config.cameras,
        chunk_size=env_config.chunk_size,
        image_size=env_config.image_size,
    )
    if hasattr(env_config, "local_path") and env_config.local_path:
        kwargs["local_path"] = env_config.local_path
    return cls(**kwargs)
