from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = DEMO_DIR / "outputs" / "nninteractive_probe"
DEFAULT_MODEL_ROOT = DEMO_DIR / "data" / "models" / "nninteractive"
MODEL_REPO_ID = "nnInteractive/nnInteractive"
MODEL_SUBDIR = "nnInteractive_v1.0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated nnInteractive point-prompt probe.")
    parser.add_argument("--case", default="1", help="Case folder name under ../测试, e.g. 1")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda, cuda:0, cpu")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--keep-existing", action="store_true", help="Do not remove previous probe files.")
    parser.add_argument("--download-only", action="store_true", help="Only download or locate the model.")
    parser.add_argument("--no-inference", action="store_true", help="Export input and summary without running prediction.")
    parser.add_argument("--point", default=None, help="Optional positive point as z,y,x. Defaults to a bone-threshold point.")
    parser.add_argument("--bone-threshold", type=float, default=300.0, help="HU threshold for choosing the default point.")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_root) / f"case_{args.case}_{args.device.replace(':', '')}"
    if output_dir.exists() and not args.keep_existing:
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    summary: dict = {
        "case": args.case,
        "device": args.device,
        "model_repo_id": MODEL_REPO_ID,
        "model_subdir": MODEL_SUBDIR,
        "output_dir": str(output_dir),
    }

    try:
        model_dir = ensure_model(Path(args.model_root))
        summary["model_dir"] = str(model_dir)

        if args.download_only:
            summary["status"] = "downloaded"
            write_summary(output_dir, summary, started)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        case_dir = PROJECT_ROOT / "测试" / args.case
        image, array_zyx, spacing_zyx = read_case(case_dir)
        input_path = output_dir / f"case_{args.case}_input.nii.gz"
        sitk.WriteImage(image, str(input_path))

        point_zyx = parse_point(args.point) if args.point else choose_default_point(array_zyx, args.bone_threshold)
        summary.update(
            {
                "case_dir": str(case_dir),
                "input_path": str(input_path),
                "shape_zyx": [int(x) for x in array_zyx.shape],
                "spacing_zyx": [float(x) for x in spacing_zyx],
                "positive_point_zyx": [int(x) for x in point_zyx],
                "bone_threshold": args.bone_threshold,
            }
        )

        if args.no_inference:
            summary["status"] = "prepared"
            write_summary(output_dir, summary, started)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        mask = run_point_prompt(
            array_zyx=array_zyx,
            spacing_zyx=spacing_zyx,
            point_zyx=point_zyx,
            model_dir=model_dir,
            device=args.device,
            torch_threads=args.torch_threads,
        )

        mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_image.CopyInformation(image)
        mask_path = output_dir / f"case_{args.case}_nninteractive_mask.nii.gz"
        sitk.WriteImage(mask_image, str(mask_path))

        summary.update(
            {
                "status": "ok",
                "mask_path": str(mask_path),
                "mask_voxels": int(mask.sum()),
            }
        )
        write_summary(output_dir, summary, started)

        preview_path = output_dir / f"case_{args.case}_mask_overlay.png"
        try:
            save_overlay_png(array_zyx, mask, point_zyx, preview_path)
            summary["preview_path"] = str(preview_path)
        except Exception as exc:
            summary["preview_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        summary.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_summary(output_dir, summary, started)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise

    write_summary(output_dir, summary, started)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def ensure_model(model_root: Path) -> Path:
    model_dir = model_root / MODEL_SUBDIR
    checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"
    if checkpoint.exists():
        return model_dir

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=str(model_root),
        allow_patterns=[f"{MODEL_SUBDIR}/*"],
        local_dir_use_symlinks=False,
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"nnInteractive checkpoint not found after download: {checkpoint}")
    return model_dir


def read_case(case_dir: Path) -> tuple[sitk.Image, np.ndarray, tuple[float, float, float]]:
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    series_dir = find_first_series_dir(case_dir)
    reader = sitk.ImageSeriesReader()
    series = reader.GetGDCMSeriesIDs(str(series_dir))
    files = reader.GetGDCMSeriesFileNames(str(series_dir), series[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    array_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return image, array_zyx, spacing_zyx


def find_first_series_dir(case_dir: Path) -> Path:
    for path in [case_dir, *case_dir.rglob("*")]:
        if not path.is_dir():
            continue
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(path))
        if series:
            return path
    raise FileNotFoundError(f"No DICOM series found under {case_dir}")


def parse_point(raw: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--point must be formatted as z,y,x")
    return parts[0], parts[1], parts[2]


def choose_default_point(array_zyx: np.ndarray, threshold: float) -> tuple[int, int, int]:
    bone = array_zyx > threshold
    if not bone.any():
        return tuple(int(x // 2) for x in array_zyx.shape)

    center = np.array(array_zyx.shape, dtype=np.float32) / 2.0
    coords = np.argwhere(bone)
    distances = ((coords - center) ** 2).sum(axis=1)
    point = coords[int(np.argmin(distances))]
    return int(point[0]), int(point[1]), int(point[2])


def run_point_prompt(
    array_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    point_zyx: tuple[int, int, int],
    model_dir: Path,
    device: str,
    torch_threads: int,
) -> np.ndarray:
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    from nnunetv2.utilities.helpers import empty_cache

    torch_device = torch.device(device)
    session = nnInteractiveInferenceSession(
        device=torch_device,
        use_torch_compile=False,
        verbose=True,
        torch_n_threads=torch_threads,
        do_autozoom=True,
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_dir),
        use_fold=0,
        checkpoint_name="checkpoint_final.pth",
    )
    session.set_image(array_zyx[None].astype(np.float32), image_properties={"spacing": spacing_zyx})
    target_buffer = torch.zeros(array_zyx.shape, dtype=torch.uint8, device="cpu")
    session.set_target_buffer(target_buffer)
    session.add_point_interaction(point_zyx, include_interaction=True, run_prediction=True)
    mask = session.target_buffer.cpu().numpy().astype(np.uint8)
    del session
    empty_cache(torch_device)
    return mask


def save_overlay_png(array_zyx: np.ndarray, mask: np.ndarray, point_zyx: tuple[int, int, int], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    z, y, x = point_zyx
    image_slice = array_zyx[z]
    mask_slice = mask[z] > 0

    low, high = -200.0, 1000.0
    gray = (np.clip((image_slice - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    overlay = rgb.copy()
    overlay[mask_slice, 0] = 255
    overlay[mask_slice, 1] = (overlay[mask_slice, 1] * 0.45).astype(np.uint8)
    overlay[mask_slice, 2] = 0
    rgb = np.where(mask_slice[..., None], (0.55 * rgb + 0.45 * overlay).astype(np.uint8), rgb)

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    radius = 10
    draw.line((x - radius, y, x + radius, y), fill=(0, 255, 255), width=2)
    draw.line((x, y - radius, x, y + radius), fill=(0, 255, 255), width=2)
    image.save(output_path)


def write_summary(output_dir: Path, summary: dict, started: float) -> None:
    summary["elapsed_seconds"] = round(time.time() - started, 2)
    (output_dir / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
