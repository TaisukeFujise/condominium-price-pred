"""アンサンブル。複数モデルの予測（円）を混ぜる。

今はモデルが 1 つなので、渡された予測をそのまま返すのと同じ。
"""

import numpy as np


def blend(preds: list[np.ndarray]) -> np.ndarray:
    """複数モデルの予測を平均する。

    Args:
        preds: モデルごとの予測（円）。

    Returns:
        混ぜた予測（円）。
    """
    return np.mean(preds, axis=0)
