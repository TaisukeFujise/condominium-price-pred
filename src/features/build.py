"""特徴量をまとめて作る。

特徴量はテーマごとにファイルを分けて make(df) として書き（base.py を参照）、MAKERS に足す。
価格を使う特徴量（target encoding など）は、make(tr_used, y, pred_rows) として書き、FOLD_MAKERS に足す。
"""

from collections.abc import Callable

import pandas as pd

from src.features import base, district_latlon, major_station, target_encoding

# 上から順に作る。後の関数は、前の関数が足した列を使える。
# 新しい特徴量は、import に足してから、ここに 1 行足す（例: location.make,）
MAKERS: list[Callable[[pd.DataFrame], pd.DataFrame]] = [
    district_latlon.make,
    # 主要駅までの距離は、district_latlon が足した緯度・経度から計算するので後ろに置く
    major_station.make,
]

# 価格を使う特徴量。fold に分けた後、src/pipeline.py から fold ごとに上から順に呼ばれる。
# 関数は (学習に使う行, その目的変数, 予測する行) を受け取り、(学習に使う行に足す列, 予測する行に足す列) を返す
FOLD_MAKERS: list[
    Callable[[pd.DataFrame, pd.Series, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]
] = [
    target_encoding.make,
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


def build_fold_features(
    tr_used: pd.DataFrame, y: pd.Series, pred_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """価格を使う特徴量を、学習に使う行だけから計算して作る。

    Args:
        tr_used: 学習に使う行。CV では、その fold の学習データから選んだ行。
        y: tr_used の目的変数。
        pred_rows: 予測する行。

    Returns:
        列を足した tr_used と pred_rows、足した列名の一覧。
    """
    new_features = []
    for make in FOLD_MAKERS:
        tr_new, pred_new = make(tr_used, y, pred_rows)
        tr_used = pd.concat([tr_used, tr_new], axis=1)
        pred_rows = pd.concat([pred_rows, pred_new], axis=1)
        new_features += list(tr_new.columns)
    return tr_used, pred_rows, new_features
