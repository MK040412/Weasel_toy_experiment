"""Image transforms and action normalization for VLA training."""

from __future__ import annotations

import torch
from torchvision import transforms


def build_image_transforms(resize_to: int, is_train: bool) -> transforms.Compose:
    """Build image transforms for CALVIN images -> Qwen3VL ViT input."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((resize_to, resize_to)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((resize_to, resize_to)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class ActionNormalizer:
    """Normalize / denormalize actions using running mean and std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.mean.to(actions.device)) / (self.std.to(actions.device) + self.eps)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * (self.std.to(actions.device) + self.eps) + self.mean.to(actions.device)
