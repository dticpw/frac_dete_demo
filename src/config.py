from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEMO_DIR.parent

DATA_DIR = PROJECT_ROOT / "测试"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "样例图片"
PREVIEW_DIR = PROJECT_ROOT / "解析预览"
SUMMARY_CSV = PREVIEW_DIR / "dicom_summary.csv"

ANNOTATION_DIR = DEMO_DIR / "data" / "annotations"
CACHE_DIR = DEMO_DIR / "data" / "cache"
PREVIEW_OUTPUT_DIR = DEMO_DIR / "outputs" / "previews"
REPORT_OUTPUT_DIR = DEMO_DIR / "outputs" / "reports"

DEFAULT_WINDOW_CENTER = 500
DEFAULT_WINDOW_WIDTH = 2000

BONE_THRESHOLD_HU = 350
MAX_CANDIDATES = 12


def ensure_dirs() -> None:
    for path in [ANNOTATION_DIR, CACHE_DIR, PREVIEW_OUTPUT_DIR, REPORT_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
