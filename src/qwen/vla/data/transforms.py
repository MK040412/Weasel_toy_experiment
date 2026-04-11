"""Image and action transforms for VLA data pipeline."""

import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """ImageNet normalization on HWC float32 image in [0, 1]."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Inverse ImageNet normalization."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return image * std + mean
