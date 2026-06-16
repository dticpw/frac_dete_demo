from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = DEMO_DIR / "outputs" / "totalseg_probe"
FRACMED_PYTHON = Path("D:/python/anaconda/envs/fracmed/python.exe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated TotalSegmentator probe on one DICOM case.")
    parser.add_argument("--case", default="1", help="Case folder name under ../测试, e.g. 1")
    parser.add_argument("--task", default="appendicular_bones", help="TotalSegmentator task")
    parser.add_argument("--device", default="gpu", help="TotalSegmentator device: gpu, gpu:0, cpu")
    parser.add_argument("--quality", choices=["normal", "fast", "fastest"], default="fastest")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--keep-existing", action="store_true", help="Do not remove the previous probe output folder.")
    parser.add_argument("--input-mode", choices=["nifti", "dicom"], default="nifti")
    args = parser.parse_args()

    case_dir = PROJECT_ROOT / "测试" / args.case
    output_dir = Path(args.output_root) / f"case_{args.case}_{args.task}_{args.quality}_{args.device.replace(':', '')}"
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_mode == "nifti":
        input_path = output_dir / f"case_{args.case}_input.nii.gz"
        export_case_to_nifti(case_dir, input_path)
    else:
        input_path = find_first_series_dir(case_dir)

    cmd = [
        str(FRACMED_PYTHON),
        "-m",
        "totalsegmentator.bin.TotalSegmentator",
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "-ta",
        args.task,
        "-d",
        args.device,
        "-q",
    ]
    if args.quality == "fast":
        cmd.append("--fast")
    elif args.quality == "fastest":
        cmd.append("--fastest")

    started = time.time()
    result = subprocess.run(cmd, cwd=str(DEMO_DIR), text=True, capture_output=True)
    elapsed = time.time() - started

    summary = {
        "case": args.case,
        "case_dir": str(case_dir),
        "input_mode": args.input_mode,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "task": args.task,
        "device": args.device,
        "quality": args.quality,
        "elapsed_seconds": round(elapsed, 2),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "outputs": sorted([str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()]) if output_dir.exists() else [],
    }

    summary_path = output_dir / "probe_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if result.returncode != 0:
        raise SystemExit(result.returncode)


def find_first_series_dir(case_dir: Path) -> Path:
    for path in [case_dir, *case_dir.rglob("*")]:
        if not path.is_dir():
            continue
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(path))
        if series:
            return path
    raise FileNotFoundError(f"No DICOM series found under {case_dir}")


def export_case_to_nifti(case_dir: Path, output_path: Path) -> None:
    series_dir = find_first_series_dir(case_dir)
    reader = sitk.ImageSeriesReader()
    series = reader.GetGDCMSeriesIDs(str(series_dir))
    files = reader.GetGDCMSeriesFileNames(str(series_dir), series[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_path))


if __name__ == "__main__":
    main()
