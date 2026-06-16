from __future__ import annotations

import numpy as np

from .base import ExternalCandidate


class PlaceholderAdapter:
    enabled = False
    out_of_domain = True

    def __init__(self, name: str, display_name: str, note: str):
        self.name = name
        self.display_name = display_name
        self.note = note

    def predict(self, case_id: str, volume_hu: np.ndarray, metadata: dict | None = None) -> list[ExternalCandidate]:
        return []


def RibFractureReferenceAdapter() -> PlaceholderAdapter:
    return PlaceholderAdapter(
        name="ribfrac_reference",
        display_name="Rib Fracture Reference",
        note="Reserved for a rib fracture CT model; out-of-domain for wrist/foot/ankle/elbow CT.",
    )


def SpineFractureReferenceAdapter() -> PlaceholderAdapter:
    return PlaceholderAdapter(
        name="spine_reference",
        display_name="Spine Fracture Reference",
        note="Reserved for a spine/cervical fracture model; out-of-domain for wrist/foot/ankle/elbow CT.",
    )


def BoneSegmentationReferenceAdapter() -> PlaceholderAdapter:
    return PlaceholderAdapter(
        name="bone_segmentation_reference",
        display_name="Bone Segmentation Reference",
        note="Reserved for a bone/structure segmentation model used as spatial context, not fracture diagnosis.",
    )
