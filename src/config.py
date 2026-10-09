"""全員で共通にするパスと定数。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "utokyo-mlapp-2026-ex1"
EXPERIMENTS_DIR = ROOT / "experiments"
SUBMISSION_DIR = ROOT / "submissions"

TARGET = "取引価格（総額）"
SEED = 42
