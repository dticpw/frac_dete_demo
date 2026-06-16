from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    path: Path
    image_count: int | None = None
    rows: int | None = None
    columns: int | None = None
    pixel_spacing: str | None = None
    slice_thickness: float | None = None

    @property
    def label(self) -> str:
        if self.image_count:
            return f"Case {self.case_id} ({self.image_count} slices)"
        return f"Case {self.case_id}"


def _load_summary() -> dict[str, dict]:
    if not config.SUMMARY_CSV.exists():
        return {}
    df = pd.read_csv(config.SUMMARY_CSV, dtype={"case": str})
    return {str(row["case"]): row.to_dict() for _, row in df.iterrows()}


def list_cases() -> list[CaseInfo]:
    summary = _load_summary()
    cases: list[CaseInfo] = []
    if not config.DATA_DIR.exists():
        return cases

    for path in sorted([p for p in config.DATA_DIR.iterdir() if p.is_dir()], key=lambda p: p.name):
        row = summary.get(path.name, {})
        cases.append(
            CaseInfo(
                case_id=path.name,
                path=path,
                image_count=_optional_int(row.get("dicom_image_count")),
                rows=_optional_int(row.get("rows")),
                columns=_optional_int(row.get("columns")),
                pixel_spacing=_optional_str(row.get("pixel_spacing")),
                slice_thickness=_optional_float(row.get("slice_thickness")),
            )
        )
    return cases


def case_labels(cases: list[CaseInfo]) -> list[str]:
    return [case.label for case in cases]


def case_by_label(label: str, cases: list[CaseInfo]) -> CaseInfo:
    for case in cases:
        if case.label == label:
            return case
    if not cases:
        raise ValueError("No DICOM cases found.")
    return cases[0]


def describe_case(case: CaseInfo) -> str:
    parts = [
        f"Case: {case.case_id}",
        f"Path: {case.path}",
    ]
    if case.image_count:
        parts.append(f"Slices: {case.image_count}")
    if case.rows and case.columns:
        parts.append(f"Size: {case.rows} x {case.columns}")
    if case.pixel_spacing:
        parts.append(f"Pixel spacing: {case.pixel_spacing}")
    if case.slice_thickness:
        parts.append(f"Slice thickness: {case.slice_thickness} mm")
    return "\n".join(parts)


def _optional_int(value) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _optional_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _optional_str(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)
