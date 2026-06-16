from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

from . import config
from .volume_processing import bone_mask


@dataclass
class Candidate:
    candidate_id: str
    case_id: str
    slice_index: int
    x: int
    y: int
    z: int
    score: float
    reason: str
    status: str = "unreviewed"

    def to_dict(self) -> dict:
        return asdict(self)


def detect_candidates(case_id: str, volume_hu: np.ndarray, max_candidates: int = config.MAX_CANDIDATES) -> list[Candidate]:
    mask = bone_mask(volume_hu)
    small_fragments = _small_fragment_candidates(case_id, volume_hu, mask)
    edge_candidates = _edge_candidates(case_id, volume_hu, mask)
    merged = _deduplicate(small_fragments + edge_candidates)
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:max_candidates]


def candidate_table(candidates: list[Candidate]) -> list[list]:
    return [
        [
            cand.candidate_id,
            cand.slice_index,
            cand.x,
            cand.y,
            f"{cand.score:.2f}",
            cand.reason,
            cand.status,
        ]
        for cand in candidates
    ]


def _small_fragment_candidates(case_id: str, volume_hu: np.ndarray, mask: np.ndarray) -> list[Candidate]:
    labels, count = ndi.label(mask)
    if count == 0:
        return []
    objects = ndi.find_objects(labels)
    candidates: list[Candidate] = []
    for idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        zslice, yslice, xslice = slc
        size = int(np.sum(labels[slc] == idx))
        if size < 8 or size > 800:
            continue
        zc = int((zslice.start + zslice.stop - 1) / 2)
        yc = int((yslice.start + yslice.stop - 1) / 2)
        xc = int((xslice.start + xslice.stop - 1) / 2)
        local_max = float(np.max(volume_hu[slc]))
        score = min(0.95, 0.35 + size / 1400 + max(local_max - 700, 0) / 3000)
        candidates.append(
            Candidate(
                candidate_id=f"{case_id}_frag_{idx:04d}",
                case_id=case_id,
                slice_index=zc,
                x=xc,
                y=yc,
                z=zc,
                score=score,
                reason="small high-density component",
            )
        )
    return candidates


def _edge_candidates(case_id: str, volume_hu: np.ndarray, mask: np.ndarray) -> list[Candidate]:
    candidates: list[Candidate] = []
    sample_step = max(1, volume_hu.shape[0] // 80)
    for z in range(0, volume_hu.shape[0], sample_step):
        image = volume_hu[z]
        local_mask = mask[z]
        if not np.any(local_mask):
            continue
        gy, gx = np.gradient(np.clip(image, -1000, 2000))
        grad = np.hypot(gx, gy)
        edge = grad * local_mask
        threshold = np.percentile(edge[edge > 0], 99.7) if np.any(edge > 0) else 0
        points = np.argwhere(edge >= threshold)
        if points.size == 0:
            continue
        y, x = points[len(points) // 2]
        score = min(0.9, float(edge[y, x]) / 2200)
        if score < 0.35:
            continue
        candidates.append(
            Candidate(
                candidate_id=f"{case_id}_edge_{z:04d}",
                case_id=case_id,
                slice_index=int(z),
                x=int(x),
                y=int(y),
                z=int(z),
                score=score,
                reason="sharp bone-window edge change",
            )
        )
    return candidates


def _deduplicate(candidates: list[Candidate], radius: int = 18) -> list[Candidate]:
    kept: list[Candidate] = []
    for cand in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = False
        for existing in kept:
            if abs(cand.z - existing.z) <= 6 and abs(cand.x - existing.x) <= radius and abs(cand.y - existing.y) <= radius:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept
