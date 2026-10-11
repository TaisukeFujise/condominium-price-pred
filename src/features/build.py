"""特徴量をまとめて作る。

特徴量はテーマごとにファイルを分けて書き（base.py を参照）、build_features() に足す。
"""

import pandas as pd

from src.features import base


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """特徴量を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        特徴量を足した表と、モデルに入れる列名の一覧。
    """
    features = list(base.FEATURES)

    # 新しい特徴量は、次の形で足す:
    #   new = location.make(df)
    #   df = pd.concat([df, new], axis=1)
    #   features += list(new.columns)

    return df, features
