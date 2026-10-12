"""特徴量をまとめて作る。

特徴量はテーマごとにファイルを分けて make(df) として書き（base.py を参照）、MAKERS に足す。
"""

from collections.abc import Callable

import pandas as pd

from src.features import base

# 上から順に作る。後の関数は、前の関数が足した列を使える。
# 新しい特徴量は、import に足してから、ここに 1 行足す（例: location.make,）
MAKERS: list[Callable[[pd.DataFrame], pd.DataFrame]] = []


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """特徴量を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        特徴量を足した表と、モデルに入れる列名の一覧。
    """
    features = list(base.FEATURES)
    for make in MAKERS:
        new = make(df)
        # 特徴量を足した後の表にもう一度かけると、同じ名前の列が二重にできる
        duplicated = new.columns[new.columns.isin(df.columns)]
        assert duplicated.empty, f"同じ名前の列がすでにある: {list(duplicated)}"
        df = pd.concat([df, new], axis=1)
        features += list(new.columns)
    return df, features
