"""後処理。予測した価格（円）を補正する。

CV では検証データの予測にも同じ補正をかけるので、補正の効果も CV で測れる。
補正に使ってよいのは学習データ（tr）の情報だけ（CV では、その fold の学習データだけ）。
"""

import numpy as np
import pandas as pd


def postprocess(
    price: np.ndarray, pred_rows: pd.DataFrame, tr: pd.DataFrame
) -> np.ndarray:
    """予測した価格を補正する。今は何もしない。

    Args:
        price: 予測した価格（円）。
        pred_rows: 予測した行。
        tr: そのときの学習データ。補正に使ってよいのはこれだけ。

    Returns:
        補正した価格（円）。
    """
    return price
