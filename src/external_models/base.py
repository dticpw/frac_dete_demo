from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np


@dataclass
class ExternalCandidate:
    source: str
    candidate_id: str
    case_id: str
    slice_index: int
    x: int
    y: int
    z: int
    score: float
    label: str
    note: str
    status: str = "unreviewed"

    def to_dict(self) -> dict:
        return asdict(self)


class ExternalModelAdapter(Protocol):
    name: str
    display_name: str
    enabled: bool
    out_of_domain: bool

    def predict(self, case_id: str, volume_hu: np.ndarray, metadata: dict | None = None) -> list[ExternalCandidate]:
        """Return weak reference candidates in unified coordinates."""
