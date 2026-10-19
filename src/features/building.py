"""同じ建物らしい行のまとまり（疑似建物）ごとの、価格を使わない集計。

建物名の列がないので、市区町村・地区・駅・駅距離・建築年・構造がすべて同じ行を、同じ建物とみなす（Issue #41 と同じ）。
件数・面積の幅・間取りの種類数から、大規模な物件か、投資用のワンルーム棟かを見分けさせる。

価格を使わないので、fold に分ける前に、train と test をまとめて集計する（group_aggregates.py と同じ）。
"""

import pandas as pd

BUILDING_KEYS = [
    "市区町村名",
    "地区名",
    "最寄駅：名称",
    "駅距離_分",
    "建築年_数値",
    "建物の構造",
]


def make(df: pd.DataFrame) -> pd.DataFrame:
    """疑似建物ごとに、件数・面積の最大・最小・標準偏差・間取りの種類数を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        集計の列の表。キーのどれかが欠損の行は、別の建物と混ざるので NaN。
    """
    out = pd.DataFrame(index=df.index)
    groups = df.groupby(BUILDING_KEYS, observed=True)
    out["建物_件数"] = groups["id"].transform("size")
    area = groups["面積_数値"]
    out["建物_面積_最大"] = area.transform("max")
    out["建物_面積_最小"] = area.transform("min")
    out["建物_面積_標準偏差"] = area.transform("std")
    out["建物_間取り_種類数"] = groups["間取り"].transform("nunique")
    return out
