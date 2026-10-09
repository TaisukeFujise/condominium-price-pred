"""目的変数の設計。モデルに何を予測させるか。

make_target と to_price は対になっている。to_price(モデルの予測) で円の価格に戻せるようにする。
"""

import numpy as np
import pandas as pd

from src.config import TARGET


def make_target(train: pd.DataFrame) -> pd.Series:
    """学習に使う目的変数を返す。

    評価指標は RMSLE なので、log1p(価格) を二乗誤差で学習すれば RMSLE を直接小さくできる。
    """
    return np.log1p(train[TARGET])


def to_price(pred: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """モデルの予測を円の価格に戻す。rows は予測した行（㎡単価を予測する場合などに使う）。"""
    return np.expm1(pred)
