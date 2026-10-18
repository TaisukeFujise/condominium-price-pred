"""カテゴリの値ごとの件数（count encoding）。

珍しい間取りや構造は、件数の少なさそのものが特殊な物件の目印になる。
価格を使わないので、fold に分ける前に train と test をまとめて数える。
地区（市区町村名 × 地区名）と駅の件数は group_aggregates.py が作るので、ここでは作らない。
"""

import pandas as pd

COUNT_COLS = [
    "市区町村名",
    "地区名",
    "間取り",
    "建物の構造",
    "用途",
    "今後の利用目的",
    "都市計画",
    "改装",
    "取引の事情等",
]


def make(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリの列ごとに、その行の値の件数を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        件数の列の表。値が欠損の行は NaN。
    """
    out = pd.DataFrame(index=df.index)
    for col in COUNT_COLS:
        counts = df[col].value_counts()
        out[f"{col}_件数"] = df[col].map(counts).astype(float)
    return out
