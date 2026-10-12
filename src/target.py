"""目的変数の設計。モデルに何を予測させるか。

make_target と to_price は対になっている。to_price(モデルの予測) で円の価格に戻せるようにする。
"""

import numpy as np
import pandas as pd

from src.config import TARGET


def make_target(rows: pd.DataFrame) -> pd.Series:
    """学習に使う目的変数を作る。

    評価指標は RMSLE なので、log1p(価格) を二乗誤差で学習すれば RMSLE を直接小さくできる。

    Args:
        rows: 価格の列を含む行（学習データ、または early stopping 用の検証データ）。

    Returns:
        目的変数（log1p した価格）。
    """
    return np.log1p(rows[TARGET])


def to_price(pred: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """モデルの予測を円の価格に戻す。make_target の逆変換。

    Args:
        pred: モデルの予測。
        rows: 予測した行。㎡単価を予測する場合などに、面積を掛け戻すのに使う。

    Returns:
        円の価格。
    """
    return np.expm1(pred)
