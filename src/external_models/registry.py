from __future__ import annotations

import numpy as np

from .base import ExternalCandidate, ExternalModelAdapter
from .cspine_rsna_adapter import RsnaCSpine8thReferenceAdapter
from .heuristic_adapter import HeuristicAdapter
from .placeholders import (
    BoneSegmentationReferenceAdapter,
    RibFractureReferenceAdapter,
)


def get_all_adapters() -> list[ExternalModelAdapter]:
    return [
        HeuristicAdapter(),
        RibFractureReferenceAdapter(),
        RsnaCSpine8thReferenceAdapter(),
        BoneSegmentationReferenceAdapter(),
    ]


def get_enabled_adapters() -> list[ExternalModelAdapter]:
    return [adapter for adapter in get_all_adapters() if adapter.enabled]


def run_enabled_adapters(case_id: str, volume_hu: np.ndarray, metadata: dict | None = None) -> list[ExternalCandidate]:
    candidates: list[ExternalCandidate] = []
    for adapter in get_enabled_adapters():
        candidates.extend(adapter.predict(case_id, volume_hu, metadata=metadata))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates
