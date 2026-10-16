"""全員で共通にするパスと定数。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "utokyo-mlapp-2026-ex1"
SUBMISSION_DIR = ROOT / "submissions"
# 外部データのダウンロード先（git の管理外）
EXTERNAL_DIR = ROOT / "external_data"

TARGET = "取引価格（総額）"
SEED = 42
