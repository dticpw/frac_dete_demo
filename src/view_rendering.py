from __future__ import annotations

import numpy as np

from .candidate_detection import Candidate
from .volume_processing import normalize_slice_rgb


def render_views(
    volume_hu: np.ndarray,
    axial_index: int,
    coronal_index: int,
    sagittal_index: int,
    candidates: list[Candidate],
    center: float,
    width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = int(np.clip(axial_index, 0, volume_hu.shape[0] - 1))
    y = int(np.clip(coronal_index, 0, volume_hu.shape[1] - 1))
    x = int(np.clip(sagittal_index, 0, volume_hu.shape[2] - 1))

    axial = normalize_slice_rgb(volume_hu[z], center, width)
    coronal = normalize_slice_rgb(volume_hu[:, y, :], center, width)
    sagittal = normalize_slice_rgb(volume_hu[:, :, x], center, width)

    for cand in candidates:
        if abs(cand.z - z) <= 2:
            _draw_cross(axial, cand.x, cand.y)
        if abs(cand.y - y) <= 2:
            _draw_cross(coronal, cand.x, cand.z)
        if abs(cand.x - x) <= 2:
            _draw_cross(sagittal, cand.y, cand.z)

    return axial, np.flipud(coronal), np.flipud(sagittal)


def _draw_cross(image: np.ndarray, x: int, y: int, radius: int = 12) -> None:
    h, w = image.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    color = np.array([255, 40, 40], dtype=np.uint8)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    image[y, x0:x1] = color
    image[y0:y1, x] = color
    if y0 < h and x0 < w:
        image[y0:y1, x0] = color
        image[y0:y1, x1 - 1] = color
        image[y0, x0:x1] = color
        image[y1 - 1, x0:x1] = color
