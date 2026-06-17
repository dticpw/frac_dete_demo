from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import util
from pathlib import Path

from .. import config


SOLUTION_NAME = "8th-place-solution"
KAGGLE_MODEL_URL = "https://www.kaggle.com/models/zixuanh/cspine-8th-place-solution-model-weights"
OUTPUT_ROOT = config.DEMO_DIR / "outputs" / "cspine_reference_probe"
REPO_DIR = OUTPUT_ROOT / "repo" / SOLUTION_NAME
WEIGHTS_DIR = OUTPUT_ROOT / "weights" / SOLUTION_NAME

REQUIRED_WEIGHT_DIRS = (
    "try2-seg-b1v10-sagview-full",
    "try2-seg-b1v1-full",
    "try17-b5-v5-t4-pseudo-round1",
    "b5-v5-t4-pseudo-round1-seq-v2",
)

VENDORED_PATHS = (
    REPO_DIR / "timm-pytorch-image-models" / "pytorch-image-models-master",
    REPO_DIR / "segmentation-models-pytorch" / "segmentation_models.pytorch-master",
    REPO_DIR / "efficientnet-pytorch" / "EfficientNet-PyTorch-master",
    REPO_DIR / "pretrainedmodels" / "pretrainedmodels-0.7.4" / "pretrainedmodels-0.7.4",
)

REQUIRED_IMPORTS = (
    "torch",
    "cv2",
    "timm",
    "segmentation_models_pytorch",
    "efficientnet_pytorch",
    "pretrainedmodels",
)


@dataclass(frozen=True)
class CSpine8thReadiness:
    repo_exists: bool
    notebook_exists: bool
    weights_dir: Path
    missing_weight_dirs: tuple[str, ...]
    missing_imports_before_vendor: tuple[str, ...]
    missing_imports_after_vendor: tuple[str, ...]
    added_vendor_paths: tuple[str, ...]

    @property
    def runtime_ready(self) -> bool:
        return (
            self.repo_exists
            and self.notebook_exists
            and not self.missing_weight_dirs
            and not self.missing_imports_after_vendor
        )

    def to_dict(self) -> dict:
        return {
            "repo_exists": self.repo_exists,
            "notebook_exists": self.notebook_exists,
            "weights_dir": str(self.weights_dir),
            "missing_weight_dirs": list(self.missing_weight_dirs),
            "missing_imports_before_vendor": list(self.missing_imports_before_vendor),
            "missing_imports_after_vendor": list(self.missing_imports_after_vendor),
            "added_vendor_paths": list(self.added_vendor_paths),
            "runtime_ready": self.runtime_ready,
        }


def add_vendor_paths() -> tuple[str, ...]:
    added = []
    for path in VENDORED_PATHS:
        if path.exists():
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
                added.append(path_text)
    return tuple(added)


def check_imports() -> tuple[str, ...]:
    return tuple(name for name in REQUIRED_IMPORTS if util.find_spec(name) is None)


def check_readiness(add_vendor: bool = True) -> CSpine8thReadiness:
    missing_before = check_imports()
    added_paths = add_vendor_paths() if add_vendor else ()
    missing_after = check_imports()
    missing_weight_dirs = tuple(name for name in REQUIRED_WEIGHT_DIRS if not (WEIGHTS_DIR / name).exists())
    return CSpine8thReadiness(
        repo_exists=REPO_DIR.exists(),
        notebook_exists=(REPO_DIR / "8th-place-inference.ipynb").exists(),
        weights_dir=WEIGHTS_DIR,
        missing_weight_dirs=missing_weight_dirs,
        missing_imports_before_vendor=missing_before,
        missing_imports_after_vendor=missing_after,
        added_vendor_paths=added_paths,
    )


def format_readiness(readiness: CSpine8thReadiness) -> str:
    lines = [
        "RSNA C-Spine 8th Reference",
        "用途：跨域颈椎骨折参考模型，不是当前手腕/足部骨折诊断模型。",
        f"Kaggle weights: {KAGGLE_MODEL_URL}",
        f"Expected weights dir: {readiness.weights_dir}",
        f"Repo exists: {readiness.repo_exists}",
        f"Notebook exists: {readiness.notebook_exists}",
        f"Runtime ready: {readiness.runtime_ready}",
    ]
    if readiness.added_vendor_paths:
        lines.append("Vendored paths added:")
        lines.extend(f"  - {path}" for path in readiness.added_vendor_paths)
    if readiness.missing_imports_after_vendor:
        lines.append("Missing imports:")
        lines.extend(f"  - {name}" for name in readiness.missing_imports_after_vendor)
    if readiness.missing_weight_dirs:
        lines.append("Missing weight folders:")
        lines.extend(f"  - {readiness.weights_dir / name}" for name in readiness.missing_weight_dirs)
    if readiness.runtime_ready:
        lines.append("Next: implement notebook-extracted inference wrapper.")
    else:
        lines.append("Current status: not runnable yet; waiting for weights and/or dependency resolution.")
    return "\n".join(lines)


def run_cspine_reference_case(case_path: str) -> dict:
    readiness = check_readiness()
    if not readiness.runtime_ready:
        raise RuntimeError(format_readiness(readiness))
    raise NotImplementedError("8th-place notebook inference has not been extracted into Python runtime yet.")
