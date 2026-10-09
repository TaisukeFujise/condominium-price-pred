"""特徴量エンジニアリング。

特徴量はテーマごとにファイルを分けて書き（base.py を参照）、build_features() に足す。
"""

import pandas as pd

from src.features import base


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """特徴量を足した表と、モデルに入れる列の一覧を返す。"""
    features = list(base.FEATURES)

    # 新しい特徴量は、次の形で足す:
    #   new = location.make(df)
    #   df = pd.concat([df, new], axis=1)
    #   features += list(new.columns)

    return df, features
