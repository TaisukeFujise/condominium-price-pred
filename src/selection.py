"""学習データの選択。どの行を、どの重みで学習に使うか。

列（特徴量）の選択は src/features/ で行う。ここは行の選択。

CV では fold ごとに、その fold の学習データだけに対して呼ばれる。

相場から極端にずれた取引（外れ値）を学習から外せる（Issue #6）。相場は「似た物件の log(㎡単価) の中央値」で、
同じ市区町村・地区・取引年・面積帯・築年数帯から始めて、件数が MIN_COUNT に足りなければ粗い組に下げる。
相場は価格から作るので、ここ（fold の学習データだけを受け取る場所）で計算する。
しきい値は環境変数 OUTLIER_CUT（log のずれ。2.0 なら約 7 倍）で与え、無ければ何も外さない。
関係者間取引・調停・競売等は、全体的に安い取引で test にもあるので外さない。
"""

import os

import numpy as np
import pandas as pd

from src.config import TARGET

MIN_COUNT = 10
# 細かい組から順に試す。件数が足りなければ次の組にする
GROUPS = [
    ["市区町村名", "地区名", "取引年", "面積帯", "築年帯"],
    ["市区町村名", "地区名", "取引年"],
    ["市区町村名", "取引年", "面積帯", "築年帯"],
    ["市区町村名", "取引年"],
]
KEEP_REASONS = ("関係者間取引", "調停・競売等")


def select(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """学習に使う行と、行ごとの重みを決める。OUTLIER_CUT があれば外れ値を外す。重みは全行同じ。

    Args:
        tr: 学習データ。CV では、その fold の学習データ。

    Returns:
        学習に使う行と、行ごとの重み。重みが None なら全行同じ重み。
    """
    cut = os.environ.get("OUTLIER_CUT")
    if not cut:
        return tr, None
    deviation = _deviation_from_market(tr)
    keep = (deviation.abs() <= float(cut)) | tr["取引の事情等"].isin(KEEP_REASONS)
    return tr[keep.to_numpy()], None


def _deviation_from_market(tr: pd.DataFrame) -> pd.Series:
    """行ごとに、log(㎡単価) の相場（似た物件の中央値）からのずれを返す。

    Args:
        tr: 学習データ。

    Returns:
        log(㎡単価) − 相場。相場を決められない行は 0。
    """
    log_unit = np.log(tr[TARGET] / tr["面積_数値"])
    keys = pd.DataFrame(
        {
            "市区町村名": tr["市区町村名"].astype(object),
            "地区名": tr["地区名"].astype(object),
            "取引年": tr["取引年"],
            "面積帯": pd.cut(
                tr["面積_数値"], [0, 20, 30, 40, 50, 60, 70, 80, 100, 2000]
            ).astype(object),
            "築年帯": pd.cut(
                tr["築年数"].fillna(-1), [-2, 5, 10, 20, 30, 40, 50, 100]
            ).astype(object),
        },
        index=tr.index,
    )
    market = pd.Series(np.nan, index=tr.index)
    for group in GROUPS:
        grouped = log_unit.groupby([keys[c] for c in group])
        median = grouped.transform("median")
        count = grouped.transform("size")
        fill = market.isna() & (count >= MIN_COUNT)
        market[fill] = median[fill]
    return (log_unit - market).fillna(0.0)
