from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydicom


SKIP_NAMES = {"DICOMDIR", "LOCKFILE", "VERSION"}


@dataclass
class VolumeData:
    case_id: str
    volume_hu: np.ndarray
    files: list[str]
    metadata: dict

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.volume_hu.shape


def find_dicom_files(case_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in case_dir.rglob("*"):
        if not path.is_file() or path.name.upper() in SKIP_NAMES:
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "Rows", None) and getattr(ds, "Columns", None):
            files.append(path)
    return files


@lru_cache(maxsize=4)
def load_case(case_id: str, case_dir_str: str) -> VolumeData:
    case_dir = Path(case_dir_str)
    dicom_files = find_dicom_files(case_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM image files found under {case_dir}")

    slices = []
    for path in dicom_files:
        ds = pydicom.dcmread(str(path), stop_before_pixels=False, force=True)
        order_key = _slice_order_key(ds, path)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = ds.pixel_array.astype(np.float32) * slope + intercept
        slices.append((order_key, path, arr, ds))

    slices.sort(key=lambda item: item[0])
    volume = np.stack([item[2] for item in slices], axis=0)
    first = slices[0][3]
    metadata = {
        "rows": int(getattr(first, "Rows", volume.shape[1])),
        "columns": int(getattr(first, "Columns", volume.shape[2])),
        "slice_count": int(volume.shape[0]),
        "pixel_spacing": [float(x) for x in getattr(first, "PixelSpacing", [])] if getattr(first, "PixelSpacing", None) else None,
        "slice_thickness": float(getattr(first, "SliceThickness", 0) or 0),
        "modality": str(getattr(first, "Modality", "")),
    }
    return VolumeData(case_id=case_id, volume_hu=volume, files=[str(item[1]) for item in slices], metadata=metadata)


def _slice_order_key(ds, path: Path) -> tuple[int, float, str]:
    instance = getattr(ds, "InstanceNumber", None)
    if instance is not None:
        try:
            return (0, float(instance), str(path))
        except Exception:
            pass

    position = getattr(ds, "ImagePositionPatient", None)
    if position is not None and len(position) >= 3:
        try:
            return (1, float(position[2]), str(path))
        except Exception:
            pass

    location = getattr(ds, "SliceLocation", None)
    if location is not None:
        try:
            return (2, float(location), str(path))
        except Exception:
            pass

    return (3, 0.0, str(path))
