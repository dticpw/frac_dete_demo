from __future__ import annotations

import os
from importlib import util
from pathlib import Path

import numpy as np

from .. import config
from .base import ExternalCandidate


class RsnaCSpine8thReferenceAdapter:
    name = "rsna_cspine_8th_reference"
    display_name = "RSNA C-Spine 8th Reference"
    out_of_domain = True

    solution_name = "8th-place-solution"
    kaggle_model_url = "https://www.kaggle.com/models/zixuanh/cspine-8th-place-solution-model-weights"
    required_weight_dirs = (
        "try2-seg-b1v10-sagview-full",
        "try2-seg-b1v1-full",
        "try17-b5-v5-t4-pseudo-round1",
        "b5-v5-t4-pseudo-round1-seq-v2",
    )
    required_imports = (
        "torch",
        "cv2",
        "timm",
        "segmentation_models_pytorch",
        "efficientnet_pytorch",
        "pretrainedmodels",
    )

    def __init__(self) -> None:
        self.enabled = os.environ.get("FRAC_ENABLE_RSNA_CSPINE_8TH", "").lower() in {"1", "true", "yes"}
        self.output_root = config.DEMO_DIR / "outputs" / "cspine_reference_probe"
        self.repo_dir = self.output_root / "repo" / self.solution_name
        self.weights_dir = self.output_root / "weights" / self.solution_name

    def readiness(self) -> dict:
        missing_imports = [name for name in self.required_imports if util.find_spec(name) is None]
        missing_weight_dirs = [name for name in self.required_weight_dirs if not (self.weights_dir / name).exists()]
        return {
            "enabled": self.enabled,
            "repo_exists": self.repo_dir.exists(),
            "weights_dir": str(self.weights_dir),
            "missing_imports": missing_imports,
            "missing_weight_dirs": missing_weight_dirs,
            "runtime_ready": False,
            "note": "Notebook has been selected as the next target, but inference wrapper is not implemented yet.",
        }

    def predict(self, case_id: str, volume_hu: np.ndarray, metadata: dict | None = None) -> list[ExternalCandidate]:
        return []
