"""アンサンブル。複数モデルの予測（円）を混ぜる。

今はモデルが 1 つなので、渡された予測をそのまま返すのと同じ。
"""

import numpy as np


def blend(preds: list[np.ndarray]) -> np.ndarray:
    return np.mean(preds, axis=0)
