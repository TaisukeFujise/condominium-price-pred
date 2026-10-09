"""EDA 用の共通処理。ノートブックから `from eda_utils import load, setup_plot` で使う。

ここでの変換は EDA で見やすくするためのもの。モデルの前処理とは別。
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
    """train と test を読み込み、EDA でよく使う列を足す。

    足す列:
        時期: "2019Q1" の形の文字列（並べ替えると時間順になる）
        取引年, 取引四半期, 面積_数値, 建築年_数値, 築年数, 駅距離_分: 文字列の列を数値にしたもの
        ㎡単価: 価格 ÷ 面積（train のみ）

    Returns:
        train と test。
    """
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"面積（㎡）": str})
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"面積（㎡）": str})

    for df in (train, test):
        period = df["取引時期"].str.extract(r"(\d{4})年第(\d)四半期").astype(int)
        df["取引年"] = period[0]
        df["取引四半期"] = period[1]
        df["時期"] = df["取引年"].astype(str) + "Q" + df["取引四半期"].astype(str)
        df["面積_数値"] = pd.to_numeric(
            df["面積（㎡）"].replace("2,000㎡以上", "2000"), errors="coerce"
        )
        df["建築年_数値"] = pd.to_numeric(
            df["建築年"].str.replace("年", "").replace("戦前", "1945"),
            errors="coerce",
        )
        df["築年数"] = df["取引年"] - df["建築年_数値"]
        distance_map = {"30分～60分": 45, "1H～1H30": 75, "1H30～2H": 105, "2H～": 120}
        df["駅距離_分"] = pd.to_numeric(
            df["最寄駅：距離（分）"].replace(distance_map), errors="coerce"
        )

    train["㎡単価"] = train[TARGET] / train["面積_数値"]
    return train, test
