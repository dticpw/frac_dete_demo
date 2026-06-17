from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import config
from src.external_models.cspine_8th_runtime import run_cspine_reference_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the extracted RSNA C-Spine 8th-place runtime on one local DICOM case.")
    parser.add_argument("--case", default="1", help="Case folder name under ../测试.")
    parser.add_argument("--case-path", default=None, help="Explicit DICOM case path.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    case_path = Path(args.case_path) if args.case_path else config.DATA_DIR / args.case
    result = run_cspine_reference_case(str(case_path))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
