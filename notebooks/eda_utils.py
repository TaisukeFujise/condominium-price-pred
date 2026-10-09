"""EDA 用の共通処理。ノートブックから `from eda_utils import load, setup_plot` で使う。

データは元の値のまま読み込む。数値への変換などは、各ノートブックで必要に応じて行う。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

DATA_DIR = Path(__file__).resolve().parent.parent / "utokyo-mlapp-2026-ex1"
TARGET = "取引価格（総額）"


JAPANESE_FONTS = ["Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP"]


def setup_plot() -> None:
    """グラフで日本語が表示されるように、フォントを設定する。"""
    # 入っていないフォントを指定すると大量に警告が出るので、入っているものだけを選ぶ
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [name for name in JAPANESE_FONTS if name in installed]
    if not fonts:
        print("日本語フォントが見つかりません。グラフの日本語が文字化けします。")
        return
    plt.rcParams["font.family"] = fonts[0]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """train と test を元の値のまま読み込む。

    Returns:
        train と test。
    """
    # 面積は "45" と "2,000㎡以上" が混ざっていて、数値と文字列が混在した列になるので、文字列でそろえる
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"面積（㎡）": str})
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"面積（㎡）": str})
    return train, test
