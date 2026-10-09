"""後処理。予測した価格（円）を補正する。

CV では検証データの予測にも同じ補正をかけるので、補正の効果も CV で測れる。
補正に使ってよいのは train の情報だけ（CV では、その fold の学習データだけ）。
"""

import numpy as np
import pandas as pd


def postprocess(price: np.ndarray, rows: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """rows は予測した行、train はそのときの学習データ。今は何もしない。"""
    return price
