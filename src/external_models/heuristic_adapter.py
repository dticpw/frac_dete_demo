from __future__ import annotations

import numpy as np

from ..candidate_detection import detect_candidates
from .base import ExternalCandidate


class HeuristicAdapter:
    name = "heuristic"
    display_name = "Heuristic Rules"
    enabled = True
    out_of_domain = False

    def predict(self, case_id: str, volume_hu: np.ndarray, metadata: dict | None = None) -> list[ExternalCandidate]:
        candidates = detect_candidates(case_id, volume_hu)
        return [
            ExternalCandidate(
                source=self.name,
                candidate_id=cand.candidate_id,
                case_id=cand.case_id,
                slice_index=cand.slice_index,
                x=cand.x,
                y=cand.y,
                z=cand.z,
                score=cand.score,
                label="weak_candidate",
                note=cand.reason,
                status=cand.status,
            )
            for cand in candidates
        ]
