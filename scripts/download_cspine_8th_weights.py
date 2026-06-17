from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LINK_DIR = DEMO_DIR / "outputs" / "cspine_reference_probe" / "weights" / "8th-place-solution"
MODEL_HANDLE = "zixuanh/cspine-8th-place-solution-model-weights/pyTorch/default"
REQUIRED_WEIGHT_DIRS = (
    "try2-seg-b1v10-sagview-full",
    "try2-seg-b1v1-full",
    "try17-b5-v5-t4-pseudo-round1",
    "b5-v5-t4-pseudo-round1-seq-v2",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RSNA C-Spine 8th-place weights via KaggleHub.")
    parser.add_argument("--link-dir", default=str(DEFAULT_LINK_DIR), help="Project path expected by the Gradio runtime.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating a junction/symlink.")
    args = parser.parse_args()

    cache_dir = download_with_kagglehub()
    link_dir = Path(args.link_dir)
    link_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.copy:
        copy_weights(cache_dir, link_dir)
        mode = "copy"
    else:
        link_weights(cache_dir, link_dir)
        mode = "link"

    print(f"Downloaded model: {MODEL_HANDLE}")
    print(f"KaggleHub cache: {cache_dir}")
    print(f"Project weights dir: {link_dir}")
    print(f"Placement mode: {mode}")


def download_with_kagglehub() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Install with: "
            "D:/python/anaconda/envs/fracmed/python.exe -m pip install kagglehub"
        ) from exc
    return Path(kagglehub.model_download(MODEL_HANDLE))


def copy_weights(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists, refusing to overwrite: {destination}")
    shutil.copytree(source, destination)


def link_weights(source: Path, destination: Path) -> None:
    if destination.exists():
        if is_link_like(destination) or has_required_weight_dirs(destination):
            return
        raise FileExistsError(f"Destination already exists and is not a link: {destination}")
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(destination), str(source)], check=True)
    else:
        destination.symlink_to(source, target_is_directory=True)


def is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name == "nt":
        return bool(path.stat().st_file_attributes & 0x400)
    return False


def has_required_weight_dirs(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_dir() for name in REQUIRED_WEIGHT_DIRS)


if __name__ == "__main__":
    main()
