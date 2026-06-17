from __future__ import annotations

import os

import numpy as np

from .base import ExternalCandidate
from .cspine_8th_runtime import (
    KAGGLE_MODEL_URL,
    REQUIRED_IMPORTS,
    REQUIRED_WEIGHT_DIRS,
    SOLUTION_NAME,
    WEIGHTS_DIR,
    check_readiness,
)


class RsnaCSpine8thReferenceAdapter:
    name = "rsna_cspine_8th_reference"
    display_name = "RSNA C-Spine 8th Reference"
    out_of_domain = True

    solution_name = SOLUTION_NAME
    kaggle_model_url = KAGGLE_MODEL_URL
    required_weight_dirs = REQUIRED_WEIGHT_DIRS
    required_imports = REQUIRED_IMPORTS

    def __init__(self) -> None:
        self.enabled = os.environ.get("FRAC_ENABLE_RSNA_CSPINE_8TH", "").lower() in {"1", "true", "yes"}
        self.weights_dir = WEIGHTS_DIR

    def readiness(self) -> dict:
        readiness = check_readiness()
        data = readiness.to_dict()
        data.update(
            {
                "enabled": self.enabled,
                "missing_imports": data["missing_imports_after_vendor"],
                "note": "Notebook has been selected as the next target, but inference wrapper is not implemented yet.",
            }
        )
        return data

    def predict(self, case_id: str, volume_hu: np.ndarray, metadata: dict | None = None) -> list[ExternalCandidate]:
        return []
