from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = DEMO_DIR / "outputs" / "cspine_reference_probe"
REPO_URL = "https://github.com/zxjasonhu/cspine-2022-challenge-inference-collection.git"
REPO_NAME = "cspine-2022-challenge-inference-collection"
KAGGLE_7TH_MODEL_URL = "https://www.kaggle.com/models/zixuanh/cspine-7th-place-solution-model-weights"
KAGGLE_7TH_MODEL_SLUG = "zixuanh/cspine-7th-place-solution-model-weights"


REQUIRED_IMPORTS_7TH = {
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "pydicom": "pydicom",
    "SimpleITK": "SimpleITK",
    "skimage": "scikit-image",
    "nibabel": "nibabel",
    "scipy": "scipy",
    "cc3d": "connected-components-3d",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe RSNA C-Spine 2022 reference model integration.")
    parser.add_argument("--case", default="1", help="Case folder name under ../测试.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--solution", default="7th-place-solution", help="Solution folder to inspect.")
    parser.add_argument("--refresh-repo", action="store_true", help="Delete and re-clone the external repo.")
    parser.add_argument("--download-weights", action="store_true", help="Try Kaggle weight download if kaggle CLI is available.")
    parser.add_argument("--skip-repo", action="store_true", help="Do not clone/pull repo, only inspect existing files.")
    args = parser.parse_args()

    started = time.time()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    repo_dir = output_root / "repo"
    input_dir = output_root / "input"
    weights_dir = output_root / "weights" / args.solution

    summary: dict = {
        "model": "RSNA C-Spine 2022 reference",
        "selected_solution": args.solution,
        "source_repo": args.repo_url,
        "kaggle_model_url": KAGGLE_7TH_MODEL_URL,
        "out_of_domain": True,
        "status": "started",
        "outputs": [],
        "warnings": [],
        "next_steps": [],
    }

    try:
        if not args.skip_repo:
            ensure_repo(repo_dir, args.repo_url, args.refresh_repo)
        summary["repo_dir"] = str(repo_dir)
        summary["repo_commit"] = git_commit(repo_dir)

        case_info = prepare_case_manifest(args.case, input_dir)
        summary.update(case_info)

        solution_dir = repo_dir / args.solution
        solution_info = inspect_solution(solution_dir)
        summary["solution"] = solution_info

        env_info = inspect_environment()
        summary["environment"] = env_info

        weight_info = inspect_weights(solution_dir, weights_dir)
        summary["weights"] = weight_info

        if args.download_weights:
            summary["weight_download_attempt"] = try_download_weights(weights_dir)
            summary["weights"] = inspect_weights(solution_dir, weights_dir)
        else:
            summary["weight_download_attempt"] = {
                "attempted": False,
                "reason": "Use --download-weights to try Kaggle download. Kaggle credentials are usually required.",
            }

        summary["status"] = classify_status(summary)
        summary["next_steps"] = recommend_next_steps(summary)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        write_summary(output_root, summary, started)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise

    write_summary(output_root, summary, started)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def ensure_repo(repo_dir: Path, repo_url: str, refresh: bool) -> None:
    if refresh and repo_dir.exists():
        shutil.rmtree(repo_dir)
    if (repo_dir / ".git").exists():
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)


def git_commit(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def prepare_case_manifest(case_id: str, input_dir: Path) -> dict:
    case_dir = PROJECT_ROOT / "测试" / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    dicom_files = find_dicom_files(case_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM image files found under {case_dir}")

    first = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True, force=True)
    study_uid = str(getattr(first, "StudyInstanceUID", case_id))
    manifest_dir = input_dir / f"case_{case_id}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "test.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", "image_folder"])
        writer.writeheader()
        writer.writerow({"StudyInstanceUID": study_uid, "image_folder": str(case_dir)})

    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "study_instance_uid": study_uid,
        "dicom_file_count": len(dicom_files),
        "input_manifest": str(manifest_path),
        "outputs": [str(manifest_path)],
    }


def find_dicom_files(case_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in case_dir.rglob("*"):
        if not path.is_file() or path.name.upper() in {"DICOMDIR", "LOCKFILE", "VERSION"}:
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "Rows", None) and getattr(ds, "Columns", None):
            files.append(path)
    return sorted(files)


def inspect_solution(solution_dir: Path) -> dict:
    if not solution_dir.exists():
        raise FileNotFoundError(f"Solution directory not found: {solution_dir}")

    final_inference = solution_dir / "Training" / "Task_203_FractureDetection_Real5Fold" / "final_inference_CV.py"
    remain_inference = solution_dir / "Training" / "Task_203_FractureDetection_Real5Fold" / "final_inference_remain.py"
    requirements = solution_dir / "requirements.txt"
    readme = solution_dir / "README.md"
    path_py = solution_dir / "path.py"

    expected_model_patterns = []
    if final_inference.exists():
        text = final_inference.read_text(encoding="utf-8", errors="ignore")
        expected_model_patterns = sorted(set(re.findall(r"Task_[^'\"]+?model_final_checkpoint\.model", text)))

    return {
        "solution_dir": str(solution_dir),
        "readme_exists": readme.exists(),
        "requirements_exists": requirements.exists(),
        "path_py_exists": path_py.exists(),
        "final_inference_exists": final_inference.exists(),
        "remain_inference_exists": remain_inference.exists(),
        "expected_model_patterns": expected_model_patterns,
        "requires_old_stack": True,
        "declared_gpu_memory": "At least 24 GB GPU memory",
        "declared_torch": "torch==1.11.0+cu115",
    }


def inspect_environment() -> dict:
    packages = {}
    for import_name, dist_name in REQUIRED_IMPORTS_7TH.items():
        packages[dist_name] = installed_version(dist_name, import_name)
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": packages,
        "torch_cuda": torch_cuda_info(),
        "risk": [
            "7th solution declares torch==1.11.0+cu115 and Python 3.7+, while fracmed uses a newer stack.",
            "Directly installing 7th-place requirements into fracmed is not recommended.",
            "The model declares at least 24 GB GPU memory; current RTX 4060 Laptop GPU may be insufficient for full inference.",
        ],
    }


def installed_version(dist_name: str, import_name: str) -> dict:
    try:
        version = metadata.version(dist_name)
        installed = True
    except metadata.PackageNotFoundError:
        version = None
        installed = False

    import_ok = False
    import_error = None
    try:
        __import__(import_name)
        import_ok = True
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"

    return {"installed": installed, "version": version, "import_ok": import_ok, "import_error": import_error}


def torch_cuda_info() -> dict:
    try:
        import torch

        return {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def inspect_weights(solution_dir: Path, weights_dir: Path) -> dict:
    local_weight_files = []
    local_plan_files = []
    for root in [solution_dir, weights_dir]:
        if root.exists():
            local_weight_files.extend(
                str(path)
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".model", ".pth", ".pt"}
            )
            local_plan_files.extend(
                str(path)
                for path in root.rglob("plans.pkl")
                if path.is_file()
            )
    return {
        "kaggle_model_slug": KAGGLE_7TH_MODEL_SLUG,
        "kaggle_model_url": KAGGLE_7TH_MODEL_URL,
        "local_weights_dir": str(weights_dir),
        "local_weight_file_count": len(local_weight_files),
        "local_weight_files_sample": local_weight_files[:20],
        "local_plan_file_count": len(local_plan_files),
        "local_plan_files_sample": local_plan_files[:20],
        "kaggle_cli_available": shutil.which("kaggle") is not None,
        "kaggle_json_exists": kaggle_json_exists(),
    }


def kaggle_json_exists() -> bool:
    candidates = []
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        candidates.append(Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json")
    candidates.append(Path.home() / ".kaggle" / "kaggle.json")
    return any(path.exists() for path in candidates)


def try_download_weights(weights_dir: Path) -> dict:
    weights_dir.mkdir(parents=True, exist_ok=True)
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        return {
            "attempted": False,
            "reason": "kaggle CLI not found in PATH. Install/configure Kaggle separately if needed.",
            "manual_url": KAGGLE_7TH_MODEL_URL,
        }
    if not kaggle_json_exists():
        return {
            "attempted": False,
            "reason": "kaggle.json credentials not found.",
            "manual_url": KAGGLE_7TH_MODEL_URL,
        }

    cmd = [
        kaggle,
        "models",
        "instances",
        "versions",
        "download",
        KAGGLE_7TH_MODEL_SLUG,
        "-p",
        str(weights_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "attempted": True,
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def classify_status(summary: dict) -> str:
    if not summary.get("solution", {}).get("final_inference_exists"):
        return "repo_checked_inference_entry_missing"
    weights = summary.get("weights", {})
    if weights.get("local_weight_file_count", 0) == 0:
        return "repo_checked_weights_missing"
    return "ready_for_manual_inference_probe"


def recommend_next_steps(summary: dict) -> list[str]:
    steps = []
    if summary.get("weights", {}).get("local_weight_file_count", 0) == 0:
        steps.append("Download 7th-place weights from Kaggle and extract them into outputs/cspine_reference_probe/weights/7th-place-solution.")
    if not summary.get("weights", {}).get("kaggle_cli_available"):
        steps.append("If automatic weight download is desired, install/configure Kaggle CLI outside the main fracmed dependency path first.")
    steps.append("Do not install 7th-place requirements into fracmed directly; use a dedicated cspine environment if full inference is attempted.")
    steps.append("After weights exist, patch path.py or run a wrapper that points path_root/path_competition_data to the probe output structure.")
    return steps


def write_summary(output_root: Path, summary: dict, started: float) -> None:
    summary["elapsed_seconds"] = round(time.time() - started, 2)
    summary_path = output_root / "probe_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
