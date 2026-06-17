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
KAGGLE_MODELS = {
    "7th-place-solution": {
        "url": "https://www.kaggle.com/models/zixuanh/cspine-7th-place-solution-model-weights",
        "slug": "zixuanh/cspine-7th-place-solution-model-weights",
    },
    "8th-place-solution": {
        "url": "https://www.kaggle.com/models/zixuanh/cspine-8th-place-solution-model-weights",
        "slug": "zixuanh/cspine-8th-place-solution-model-weights",
    },
}


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

REQUIRED_IMPORTS_BY_SOLUTION = {
    "4th-place-solution": {
        "torch": "torch",
        "numpy": "numpy",
        "pydicom": "pydicom",
        "cv2": "opencv-python",
        "albumentations": "albumentations",
        "skimage": "scikit-image",
        "mmengine": "mmengine",
    },
    "5th-place-solution": {
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "pydicom": "pydicom",
        "cv2": "opencv-python",
        "timm": "timm",
    },
    "7th-place-solution": REQUIRED_IMPORTS_7TH,
    "8th-place-solution": {
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "pydicom": "pydicom",
        "cv2": "opencv-python",
        "timm": "timm",
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "efficientnet_pytorch": "efficientnet-pytorch",
        "pretrainedmodels": "pretrainedmodels",
        "skimage": "scikit-image",
    },
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
        "kaggle_model_url": KAGGLE_MODELS.get(args.solution, {}).get("url"),
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

        env_info = inspect_environment(args.solution)
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

    solution_name = solution_dir.name
    if solution_name == "4th-place-solution":
        return inspect_4th_solution(solution_dir)
    if solution_name == "5th-place-solution":
        return inspect_5th_solution(solution_dir)
    if solution_name == "8th-place-solution":
        return inspect_8th_solution(solution_dir)
    return inspect_7th_solution(solution_dir)


def inspect_7th_solution(solution_dir: Path) -> dict:
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
        "integration_priority": "defer",
        "integration_note": "Full multi-stage 3D nnU-Net pipeline; useful reference but too heavy for first Gradio integration.",
    }


def inspect_4th_solution(solution_dir: Path) -> dict:
    model_py = solution_dir / "model.py"
    model_cam_py = solution_dir / "model_cam.py"
    text = model_py.read_text(encoding="utf-8", errors="ignore") if model_py.exists() else ""
    checkpoint_refs = sorted(set(re.findall(r'"([^"]*(?:swa_|256_ResNet)[^"]*)"', text)))
    return {
        "solution_dir": str(solution_dir),
        "entry_type": "python_class",
        "entrypoint": "model.py::MDAIModel.predict",
        "model_py_exists": model_py.exists(),
        "model_cam_exists": model_cam_py.exists(),
        "expected_checkpoint_refs": checkpoint_refs,
        "uses_3d_models": True,
        "uses_grad_cam": model_cam_py.exists(),
        "requires_old_stack": False,
        "required_missing_packages_likely": ["albumentations", "mmengine"],
        "declared_gpu_memory": "Not declared in repo; 3D CSN ensemble may still be heavy.",
        "integration_priority": "candidate_after_8th",
        "integration_note": "Clear Python API and Grad-CAM path, but uses 3D CSN ensemble plus mmengine and five checkpoints.",
    }


def inspect_5th_solution(solution_dir: Path) -> dict:
    notebook = solution_dir / "rsna2022-5th-place-solution-inference.ipynb"
    codebase = solution_dir / "rsna2022-codebase"
    configs = sorted(str(path.relative_to(solution_dir)) for path in codebase.rglob("configs/*.py")) if codebase.exists() else []
    models = sorted(str(path.relative_to(solution_dir)) for path in codebase.rglob("models/*.py")) if codebase.exists() else []
    return {
        "solution_dir": str(solution_dir),
        "entry_type": "notebook_plus_codebase",
        "entrypoint": str(notebook),
        "notebook_exists": notebook.exists(),
        "codebase_exists": codebase.exists(),
        "config_count": len(configs),
        "model_file_count": len(models),
        "config_files_sample": configs[:12],
        "model_files_sample": models[:12],
        "uses_3d_models": True,
        "requires_old_stack": "unknown",
        "declared_gpu_memory": "Not declared in inspected files.",
        "integration_priority": "defer",
        "integration_note": "Full competition codebase with multiple stages/configs; too much glue for the next quick Gradio proof.",
    }


def inspect_8th_solution(solution_dir: Path) -> dict:
    notebook = solution_dir / "8th-place-inference.ipynb"
    text = read_notebook_source(notebook) if notebook.exists() else ""
    folder_refs = sorted(set(re.findall(r"folders = \[(.*?)\]", text)))
    torch_load_refs = sorted(set(re.findall(r"torch\.load\([^\n]+", text)))
    vendored_dirs = [
        "timm-pytorch-image-models/pytorch-image-models-master",
        "segmentation-models-pytorch/segmentation_models.pytorch-master",
        "efficientnet-pytorch/EfficientNet-PyTorch-master",
        "pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4",
    ]
    return {
        "solution_dir": str(solution_dir),
        "entry_type": "notebook_pipeline",
        "entrypoint": str(notebook),
        "notebook_exists": notebook.exists(),
        "input_csv_placeholder": "YOUR_TEST_FILE" in text,
        "folder_refs": folder_refs,
        "torch_load_refs_sample": torch_load_refs[:12],
        "vendored_dependency_dirs": [item for item in vendored_dirs if (solution_dir / item).exists()],
        "uses_2d_segmentation": True,
        "uses_sequence_classifier": True,
        "uses_3d_models": False,
        "requires_old_stack": "probably lower than 7th; still needs notebook extraction and package checks",
        "declared_gpu_memory": "Not declared in inspected notebook.",
        "integration_priority": "recommended_next",
        "integration_note": "Best next target: 2D segmentation + sequence classifier notebook, likely easier to slim into a reference adapter than 7th/5th.",
    }


def read_notebook_source(notebook: Path) -> str:
    data = json.loads(notebook.read_text(encoding="utf-8"))
    chunks = []
    for cell in data.get("cells", []):
        source = cell.get("source", [])
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


def inspect_environment(solution_name: str) -> dict:
    packages = {}
    imports = REQUIRED_IMPORTS_BY_SOLUTION.get(solution_name, REQUIRED_IMPORTS_7TH)
    for import_name, dist_name in imports.items():
        packages[dist_name] = installed_version(dist_name, import_name)
    missing = [dist for dist, info in packages.items() if not info["import_ok"]]
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": packages,
        "missing_imports": missing,
        "torch_cuda": torch_cuda_info(),
        "risk": environment_risks(solution_name),
    }


def environment_risks(solution_name: str) -> list[str]:
    if solution_name == "7th-place-solution":
        return [
            "7th solution declares torch==1.11.0+cu115 and Python 3.7+, while fracmed uses a newer stack.",
            "Directly installing 7th-place requirements into fracmed is not recommended.",
            "The model declares at least 24 GB GPU memory; current RTX 4060 Laptop GPU may be insufficient for full inference.",
        ]
    if solution_name == "8th-place-solution":
        return [
            "8th solution is notebook-based and must be extracted into a deterministic Python wrapper before Gradio use.",
            "Some dependencies are vendored inside the external repo, but segmentation_models_pytorch / efficientnet_pytorch may still be missing from fracmed.",
            "Do not install missing packages into fracmed until their version constraints are checked.",
        ]
    if solution_name == "4th-place-solution":
        return [
            "4th solution has a clear Python class entrypoint but requires mmengine and albumentations.",
            "It loads a 3D CSN ensemble and may still exceed laptop GPU memory depending on checkpoint size.",
        ]
    if solution_name == "5th-place-solution":
        return [
            "5th solution is a full multi-stage codebase; wrapping it is likely slower than extracting the 8th-place notebook.",
        ]
    return ["Unknown solution; only static file and weight checks are available."]


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
                if path.is_file() and path.suffix.lower() in {".ckpt", ".model", ".pth", ".pt"}
            )
            local_plan_files.extend(
                str(path)
                for path in root.rglob("plans.pkl")
                if path.is_file()
            )
    return {
        "kaggle_model_slug": KAGGLE_MODELS.get(solution_dir.name, {}).get("slug"),
        "kaggle_model_url": KAGGLE_MODELS.get(solution_dir.name, {}).get("url"),
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
            "manual_url": KAGGLE_MODELS.get(weights_dir.name, {}).get("url"),
        }
    if not kaggle_json_exists():
        return {
            "attempted": False,
            "reason": "kaggle.json credentials not found.",
            "manual_url": KAGGLE_MODELS.get(weights_dir.name, {}).get("url"),
        }

    cmd = [
        kaggle,
        "models",
        "instances",
        "versions",
        "download",
        KAGGLE_MODELS.get(weights_dir.name, {}).get("slug") or weights_dir.name,
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
    solution = summary.get("solution", {})
    has_entry = (
        solution.get("final_inference_exists")
        or solution.get("model_py_exists")
        or solution.get("notebook_exists")
    )
    if not has_entry:
        return "repo_checked_inference_entry_missing"
    weights = summary.get("weights", {})
    if weights.get("local_weight_file_count", 0) == 0:
        return "repo_checked_weights_missing"
    return "ready_for_manual_inference_probe"


def recommend_next_steps(summary: dict) -> list[str]:
    steps = []
    solution_name = summary.get("selected_solution", "")
    model_url = summary.get("weights", {}).get("kaggle_model_url")
    if summary.get("weights", {}).get("local_weight_file_count", 0) == 0:
        if model_url:
            steps.append(f"Download {solution_name} weights from Kaggle and extract them into outputs/cspine_reference_probe/weights/{solution_name}.")
        else:
            steps.append(f"No Kaggle model URL is configured for {solution_name}; locate the matching checkpoint package before runtime integration.")
    if not summary.get("weights", {}).get("kaggle_cli_available"):
        steps.append("If automatic weight download is desired, install/configure Kaggle CLI outside the main fracmed dependency path first.")
    if solution_name == "8th-place-solution":
        steps.append("Extract the 8th-place notebook into a small Python wrapper that accepts a DICOM folder and returns study/C1-C7 probabilities.")
        steps.append("Run a no-weight import probe first; only then decide whether to install missing dependencies into a dedicated cspine environment.")
    elif solution_name == "7th-place-solution":
        steps.append("Do not install 7th-place requirements into fracmed directly; use a dedicated cspine environment if full inference is attempted.")
        steps.append("After weights exist, patch path.py or run a wrapper that points path_root/path_competition_data to the probe output structure.")
    else:
        steps.append("Use this static probe result to decide whether to write a thin wrapper or defer this solution.")
    return steps


def write_summary(output_root: Path, summary: dict, started: float) -> None:
    summary["elapsed_seconds"] = round(time.time() - started, 2)
    summary_path = output_root / "probe_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
