from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import config


def window_image(image: np.ndarray, center: float = config.DEFAULT_WINDOW_CENTER, width: float = config.DEFAULT_WINDOW_WIDTH) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / max(high - low, 1)
    return (normalized * 255).astype(np.uint8)


def bone_mask(volume_hu: np.ndarray, threshold: float = config.BONE_THRESHOLD_HU) -> np.ndarray:
    mask = volume_hu >= threshold
    mask = ndi.binary_opening(mask, structure=np.ones((1, 2, 2)), iterations=1)
    return mask


def normalize_slice_rgb(image: np.ndarray, center: float, width: float) -> np.ndarray:
    gray = window_image(image, center=center, width=width)
    return np.repeat(gray[:, :, None], 3, axis=2)
