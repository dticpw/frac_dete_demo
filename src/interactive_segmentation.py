from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from . import config


MODEL_REPO_ID = "nnInteractive/nnInteractive"
MODEL_SUBDIR = "nnInteractive_v1.0"
MODEL_ROOT = config.DEMO_DIR / "data" / "models" / "nninteractive"
OUTPUT_ROOT = config.DEMO_DIR / "outputs" / "nninteractive_gradio"


@dataclass
class InteractiveSegmentationResult:
    mask: np.ndarray
    overlay: np.ndarray
    mask_path: Path
    point_zyx: tuple[int, int, int]
    elapsed_seconds: float
    inference_seconds: float
    mask_voxels: int
    license: str


def run_interactive_point_segmentation(
    case_id: str,
    volume_hu: np.ndarray,
    point_zyx: tuple[int, int, int],
    device: str = "cuda",
    torch_threads: int = 8,
) -> InteractiveSegmentationResult:
    output_dir = OUTPUT_ROOT / f"case_{case_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    model_dir = ensure_model()
    mask, model_license, inference_seconds = _run_nninteractive(
        volume_hu=volume_hu,
        point_zyx=point_zyx,
        model_dir=model_dir,
        device=device,
        torch_threads=torch_threads,
    )

    mask_path = output_dir / f"case_{case_id}_nninteractive_mask_z{point_zyx[0]}_y{point_zyx[1]}_x{point_zyx[2]}.nii.gz"
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    sitk.WriteImage(mask_image, str(mask_path))

    overlay = render_axial_overlay(volume_hu, mask, point_zyx)
    elapsed_seconds = time.time() - started
    return InteractiveSegmentationResult(
        mask=mask,
        overlay=overlay,
        mask_path=mask_path,
        point_zyx=point_zyx,
        elapsed_seconds=round(elapsed_seconds, 2),
        inference_seconds=round(inference_seconds, 2),
        mask_voxels=int(mask.sum()),
        license=model_license,
    )


def ensure_model() -> Path:
    model_dir = MODEL_ROOT / MODEL_SUBDIR
    checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"
    if checkpoint.exists():
        return model_dir

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=str(MODEL_ROOT),
        allow_patterns=[f"{MODEL_SUBDIR}/*"],
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"nnInteractive checkpoint not found after download: {checkpoint}")
    return model_dir


def render_axial_overlay(volume_hu: np.ndarray, mask: np.ndarray, point_zyx: tuple[int, int, int]) -> np.ndarray:
    z, y, x = point_zyx
    z = int(np.clip(z, 0, volume_hu.shape[0] - 1))
    y = int(np.clip(y, 0, volume_hu.shape[1] - 1))
    x = int(np.clip(x, 0, volume_hu.shape[2] - 1))

    image_slice = volume_hu[z]
    mask_slice = mask[z] > 0
    low = config.DEFAULT_WINDOW_CENTER - config.DEFAULT_WINDOW_WIDTH / 2
    high = config.DEFAULT_WINDOW_CENTER + config.DEFAULT_WINDOW_WIDTH / 2
    gray = (np.clip((image_slice - low) / max(high - low, 1), 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    overlay = rgb.copy()
    overlay[mask_slice, 0] = 255
    overlay[mask_slice, 1] = (overlay[mask_slice, 1] * 0.4).astype(np.uint8)
    overlay[mask_slice, 2] = 0
    rgb = np.where(mask_slice[..., None], (0.55 * rgb + 0.45 * overlay).astype(np.uint8), rgb)

    _draw_cross(rgb, x, y)
    return rgb


def _run_nninteractive(
    volume_hu: np.ndarray,
    point_zyx: tuple[int, int, int],
    model_dir: Path,
    device: str,
    torch_threads: int,
) -> tuple[np.ndarray, str, float]:
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    from nnunetv2.utilities.helpers import empty_cache

    torch_device = torch.device(device)
    session = nnInteractiveInferenceSession(
        device=torch_device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=torch_threads,
        do_autozoom=True,
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_dir),
        use_fold=0,
        checkpoint_name="checkpoint_final.pth",
    )
    model_license = session.license or "unknown"

    session.set_image(volume_hu[None].astype(np.float32))
    target_buffer = torch.zeros(volume_hu.shape, dtype=torch.uint8, device="cpu")
    session.set_target_buffer(target_buffer)

    started = time.time()
    session.add_point_interaction(point_zyx, include_interaction=True, run_prediction=True)
    inference_seconds = time.time() - started
    mask = session.target_buffer.cpu().numpy().astype(np.uint8)
    del session
    empty_cache(torch_device)
    return mask, model_license, inference_seconds


def _draw_cross(image: np.ndarray, x: int, y: int, radius: int = 12) -> None:
    h, w = image.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    color = np.array([0, 255, 255], dtype=np.uint8)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    image[y, x0:x1] = color
    image[y0:y1, x] = color
