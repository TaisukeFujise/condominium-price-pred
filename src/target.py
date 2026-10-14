"""目的変数の設計。モデルに何を予測させるか。

to_target と to_price は対になっている。to_price(モデルの予測) で円の価格に戻せるようにする。
"""

import numpy as np
import pandas as pd

from src.config import TARGET


def to_target(rows: pd.DataFrame) -> pd.Series:
    """学習に使う目的変数を作る。

    総額を面積で割ることで、物件規模を除いた㎡単価を学習する。

    Args:
        rows: 価格の列を含む行（学習データ、または early stopping 用の検証データ）。

    Returns:
        目的変数（log した㎡単価）。
    """
    unit_price = rows[TARGET] / rows["面積_数値"]
    return np.log(unit_price)


def to_price(pred: np.ndarray, pred_rows: pd.DataFrame) -> np.ndarray:
    """モデルの予測した㎡単価に面積を掛けて総額に戻す。

    Args:
        pred: モデルの予測。
        pred_rows: 面積_数値列を含む予測対象の行。

    Returns:
        円の価格。
    """
    unit_price = np.exp(pred)
    return unit_price * pred_rows["面積_数値"].to_numpy()
