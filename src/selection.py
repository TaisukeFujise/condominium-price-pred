"""学習データの選択。どの行を、どの重みで学習に使うか。

列（特徴量）の選択は src/features/ で行う。ここは行の選択。

CV では fold ごとに、その fold の学習データだけに対して呼ばれる。
"""

import numpy as np
import pandas as pd

from src.config import TARGET

# 相場と log(㎡単価) でこれ以上ずれる行を外れ値とする（1.5 なら約 4.5 倍）
OUTLIER_THRESHOLD = 1.5
# "drop" なら外れ値を学習から外す。"weight" なら OUTLIER_WEIGHT の重みで学習する
OUTLIER_MODE = "weight"
OUTLIER_WEIGHT = 0.1

# 全体的に安い取引で test にもあるので、外れ値に見えても外さない（Issue #6）
KEEP_CIRCUMSTANCES = ["関係者間取引", "調停・競売等"]

# 似た物件の条件。細かい順に試し、MIN_COUNT 件以上ある最初の条件の中央値を相場にする
REFERENCE_LEVELS = [
    ["地区名", "取引年", "面積帯", "築年帯"],
    ["市区町村名", "取引年", "面積帯", "築年帯", "駅距離帯"],
    ["市区町村名", "取引年", "面積帯", "駅距離帯"],
    ["市区町村名", "面積帯", "駅距離帯"],
    ["市区町村名", "面積帯"],
]
MIN_COUNT = 10
AREA_BINS = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 2000]
BUILT_YEAR_BINS = [1900, 1975, 1985, 1995, 2005, 2015, 2030]
DISTANCE_BINS = [-1, 5, 10, 15, 20, 30, 1000]


def _band(values: pd.Series, bins: list[int]) -> pd.Series:
    """数値を帯に分ける。欠損も 1 つの帯として扱う。"""
    # 欠損の行も相場を決められるように、groupby で落ちない文字列にする
    return pd.cut(values, bins).astype(str)


def _market_deviation(tr: pd.DataFrame) -> pd.Series:
    """各行の log(㎡単価) が、似た物件の相場（中央値）からどれだけずれているかを計算する。

    Args:
        tr: 価格の列を含む学習データ。

    Returns:
        相場とのずれ（log(㎡単価) − 相場）。どの条件でも件数が足りない行は NaN。
    """
    log_unit = np.log(tr[TARGET] / tr["面積_数値"])
    keys = pd.DataFrame(
        {
            "地区名": tr["地区名"].astype(str),
            "市区町村名": tr["市区町村名"].astype(str),
            "取引年": tr["取引年"],
            "面積帯": _band(tr["面積_数値"], AREA_BINS),
            "築年帯": _band(tr["建築年_数値"], BUILT_YEAR_BINS),
            "駅距離帯": _band(tr["駅距離_分"], DISTANCE_BINS),
        },
        index=tr.index,
    )
    deviation = pd.Series(np.nan, index=tr.index)
    for level in REFERENCE_LEVELS:
        group = log_unit.groupby([keys[k] for k in level])
        enough = deviation.isna() & (group.transform("size") >= MIN_COUNT)
        deviation[enough] = (log_unit - group.transform("median"))[enough]
    return deviation


def select(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """相場から極端にずれた行を、学習から外すか重みを下げる。

    相場は価格から計算するので、渡された学習データだけで計算する。

    Args:
        tr: 学習データ。CV では、その fold の学習データ。

    Returns:
        学習に使う行と、行ごとの重み。重みが None なら全行同じ重み。
    """
    outlier = _market_deviation(tr).abs() > OUTLIER_THRESHOLD
    outlier &= ~tr["取引の事情等"].isin(KEEP_CIRCUMSTANCES)
    if OUTLIER_MODE == "drop":
        return tr[~outlier], None
    weight = pd.Series(np.where(outlier, OUTLIER_WEIGHT, 1.0), index=tr.index)
    return tr, weight
