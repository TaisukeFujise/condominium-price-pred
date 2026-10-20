"""特徴量をまとめて作る（工程 2）。

特徴量は、テーマごとに src/features/<テーマ>.py の make(df) として書き、MAKERS に足す。
make は前処理済みの表を受け取り、新しい列だけの表を返す（例は building.py）。
モデルにそのまま入れる前処理済みの列は、base.py の FEATURES に並べてある。
"""

from collections.abc import Callable

import pandas as pd

from src.features import (
    base,
    building,
    district_latlon,
    group_aggregates,
    major_station,
)

# 上から順に作る。後の関数は、前の関数が足した列を使える。
# 新しい特徴量は、import に足してから、ここに 1 行足す（例: location.make,）
MAKERS: list[Callable[[pd.DataFrame], pd.DataFrame]] = [
    district_latlon.make,
    # 主要駅までの距離は、district_latlon が足した緯度・経度から計算するので後ろに置く
    major_station.make,
    group_aggregates.make,
    building.make,
]


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
        # 同じ名前の列が二つあると df[features] が両方を返し、モデルに入る列がずれる。
        # assert は python -O で消えるので、例外を投げる
        duplicated = new.columns[
            new.columns.isin(df.columns) | new.columns.duplicated()
        ]
        if not duplicated.empty:
            raise ValueError(f"同じ名前の列が二つできる: {list(duplicated)}")
        df = pd.concat([df, new], axis=1)
        features += list(new.columns)
    return df, features
