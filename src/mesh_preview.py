from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from . import config
from .candidate_detection import Candidate


MESH_DIR = config.CACHE_DIR / "mesh"


def build_or_load_mesh(case_id: str, volume_hu: np.ndarray, candidates: list[Candidate]) -> Path:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = MESH_DIR / f"case_{case_id}_bone_preview.obj"
    if obj_path.exists() and not _uses_external_material(obj_path):
        return obj_path

    factor = _downsample_factor(volume_hu.shape)
    sampled = volume_hu[::factor, ::factor, ::factor]
    mask = sampled >= config.BONE_THRESHOLD_HU
    mask = ndi.binary_opening(mask, structure=np.ones((1, 2, 2)), iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((1, 2, 2)), iterations=1)

    if np.count_nonzero(mask) < 100:
        _write_empty_obj(obj_path)
        return obj_path

    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0.5)
    faces = _limit_faces(faces, max_faces=80000)
    _write_obj(obj_path, verts, faces, candidates, factor)
    return obj_path


def _downsample_factor(shape: tuple[int, int, int], max_dim: int = 180) -> int:
    return max(1, int(np.ceil(max(shape) / max_dim)))


def _limit_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if len(faces) <= max_faces:
        return faces
    step = int(np.ceil(len(faces) / max_faces))
    return faces[::step]


def _uses_external_material(obj_path: Path) -> bool:
    try:
        with obj_path.open("r", encoding="utf-8") as handle:
            for _ in range(5):
                line = handle.readline()
                if not line:
                    break
                if line.startswith("mtllib "):
                    return True
    except OSError:
        return True
    return False


def _write_empty_obj(obj_path: Path) -> None:
    obj_path.write_text("o empty\n", encoding="utf-8")


def _write_obj(obj_path: Path, verts: np.ndarray, faces: np.ndarray, candidates: list[Candidate], factor: int) -> None:
    lines: list[str] = ["o bone_surface"]

    # marching_cubes returns z, y, x. OBJ uses x, y, z.
    for z, y, x in verts:
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
    for face in faces:
        a, b, c = face + 1
        lines.append(f"f {a} {b} {c}")

    vertex_offset = len(verts)
    marker_vertices: list[tuple[float, float, float]] = []
    marker_faces: list[tuple[int, int, int]] = []
    for cand in candidates[: config.MAX_CANDIDATES]:
        cx = cand.x / factor
        cy = cand.y / factor
        cz = cand.z / factor
        radius = max(2.5, min(6.0, float(np.max(np.ptp(verts, axis=0))) / 50))
        start = vertex_offset + len(marker_vertices) + 1
        marker_vertices.extend(_octahedron_vertices(cx, cy, cz, radius))
        marker_faces.extend([(start + a, start + b, start + c) for a, b, c in _OCTAHEDRON_FACES])

    if marker_vertices:
        lines.append("o weak_candidates")
        for x, y, z in marker_vertices:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        for a, b, c in marker_faces:
            lines.append(f"f {a} {b} {c}")

    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _octahedron_vertices(cx: float, cy: float, cz: float, r: float) -> list[tuple[float, float, float]]:
    return [
        (cx + r, cy, cz),
        (cx - r, cy, cz),
        (cx, cy + r, cz),
        (cx, cy - r, cz),
        (cx, cy, cz + r),
        (cx, cy, cz - r),
    ]


_OCTAHEDRON_FACES = [
    (0, 2, 4),
    (2, 1, 4),
    (1, 3, 4),
    (3, 0, 4),
    (2, 0, 5),
    (1, 2, 5),
    (3, 1, 5),
    (0, 3, 5),
]
