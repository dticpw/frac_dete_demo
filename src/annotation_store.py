from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config


def save_annotations(case_id: str, rows: list[list]) -> tuple[Path, Path]:
    config.ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = config.ANNOTATION_DIR / f"case_{case_id}_annotations_{timestamp}"
    columns = ["candidate_id", "slice_index", "x", "y", "score", "reason", "status"]
    records = [dict(zip(columns, row)) for row in rows]

    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(records, columns=columns).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path
