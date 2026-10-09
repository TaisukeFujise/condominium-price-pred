"""データの読み取り・書き出し。

train.csv / test.csv の読み込みと、提出ファイル（id, price）の書き出しのみを担当する。
カラムの変換・前処理は preprocessing 側の責務。
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "utokyo-mlapp-2026-ex1"

TARGET_COLUMN = "取引価格（総額）"


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "train.csv")


def load_test(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "test.csv")


def write_submission(ids: pd.Series, prices, path: Path) -> None:
    """id, price の2列でCSVを書き出す（ヘッダ付き）。"""
    submission = pd.DataFrame({"id": ids, "price": prices})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


if __name__ == "__main__":
    train = load_train()
    test = load_test()
    print("train:", train.shape, "test:", test.shape)
    print("columns match (test + target == train):",
          set(test.columns) | {TARGET_COLUMN} == set(train.columns))
