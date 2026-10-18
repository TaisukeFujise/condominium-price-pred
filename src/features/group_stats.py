"""グループの中での、その行の位置（z スコア・順位）と、グループのばらつき（標準偏差）。

group_aggregates.py の平均との差は、グループごとにばらつきが違っても同じ尺度で並ぶ。
ここでは標準偏差で割ったり順位にしたりして、グループの中で珍しい値かどうかを表す。
価格を使わないので、fold に分ける前に train と test をまとめて集計する。

LGBM だけの CV ではどれも良くならなかったので、MAKERS には入れていない（PR の本文を参照）。
"""

import pandas as pd

# 集計の単位。地区名は市区町村をまたいで同じ名前があるので、市区町村名と組にする
LOCATION_KEYS = {
    "地区": ["市区町村名", "地区名"],
    "駅": ["最寄駅：名称"],
}
LOCATION_COLS = ["面積_数値", "建築年_数値", "駅距離_分"]
# 取引年ごとの z スコア。年ごとに取引される物件の構成が変わるのを打ち消す
YEAR_COLS = ["面積_数値", "建築年_数値"]


def make_location_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """地区ごと・最寄駅ごとの、面積・建築年・駅距離の z スコアを作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        z スコアの列の表。
    """
    out = pd.DataFrame(index=df.index)
    for name, keys in LOCATION_KEYS.items():
        out = out.join(_zscore(df, keys, LOCATION_COLS, f"{name}内"))
    return out


def make_location_std(df: pd.DataFrame) -> pd.DataFrame:
    """地区ごと・最寄駅ごとの、面積・建築年・駅距離の標準偏差を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        標準偏差の列の表。
    """
    out = pd.DataFrame(index=df.index)
    for name, keys in LOCATION_KEYS.items():
        groups = df.groupby(keys, observed=True)
        for col in LOCATION_COLS:
            out[f"{name}_標準偏差_{col}"] = groups[col].transform("std")
    return out


def make_location_rank(df: pd.DataFrame) -> pd.DataFrame:
    """地区ごと・最寄駅ごとの、面積・建築年・駅距離の順位（0〜1）を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        順位の列の表。
    """
    out = pd.DataFrame(index=df.index)
    for name, keys in LOCATION_KEYS.items():
        groups = df.groupby(keys, observed=True)
        for col in LOCATION_COLS:
            out[f"{name}内順位_{col}"] = groups[col].rank(pct=True)
    return out


def make_year_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """取引年ごとの、面積・建築年の z スコアを作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        z スコアの列の表。
    """
    return _zscore(df, ["取引年"], YEAR_COLS, "取引年内")


def _zscore(
    df: pd.DataFrame, keys: list[str], cols: list[str], prefix: str
) -> pd.DataFrame:
    """グループごとに (値 − 平均) / 標準偏差 を計算する。

    Args:
        df: 前処理済みの表。
        keys: グループに分ける列。
        cols: z スコアにする列。
        prefix: 列名の頭に付ける文字列。

    Returns:
        z スコアの列の表。1 行だけのグループや、値がすべて同じグループは NaN。
    """
    out = pd.DataFrame(index=df.index)
    # observed=True: カテゴリ型のキーで、出てこない組み合わせまで作らない
    groups = df.groupby(keys, observed=True)
    for col in cols:
        std = groups[col].transform("std")
        out[f"{prefix}z_{col}"] = (df[col] - groups[col].transform("mean")) / std.where(
            std > 0
        )
    return out
