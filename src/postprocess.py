"""後処理。予測した価格（円）を補正する。

CV では検証データの予測にも同じ補正をかけるので、補正の効果も CV で測れる。
補正に使ってよいのは学習データ（tr）の情報だけ（CV では、その fold の学習データだけ）。
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src import target, validation
from src.models.lgbm import LGBM


def _fit_calibrator(tr: pd.DataFrame, features: list[str]) -> IsotonicRegression:
    """予測値（log）から実際値（log）への isotonic 較正を、学習データの中だけで学習する。

    tr の中にさらに内側のホールドアウト（直近 3 四半期）を作り、そこでの予測と実際の値を対応させる。
    安い物件を過大予測し、高い物件を過小予測するという価格帯による系統的なズレ
    （regression to the mean）を、この対応表で補正する。

    Args:
        tr: そのときの学習データ（CV では fold の学習データ、本番では train 全体）。
        features: モデルに入れる列名の一覧。

    Returns:
        学習済みの IsotonicRegression。
    """
    inner_tr_index, inner_va_index = validation.time_series_folds(tr)[0]
    inner_tr, inner_va = tr.iloc[inner_tr_index], tr.iloc[inner_va_index]

    model = LGBM().fit_with_validation(
        inner_tr[features],
        target.to_target(inner_tr),
        inner_va[features],
        target.to_target(inner_va),
        None,
    )
    pred_log = model.predict(inner_va[features])
    actual_log = target.to_target(inner_va).to_numpy()

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pred_log, actual_log)
    return calibrator


def adjust(
    price: np.ndarray, pred_rows: pd.DataFrame, tr: pd.DataFrame, features: list[str]
) -> np.ndarray:
    """予測した価格を、isotonic 較正で補正する。

    Args:
        price: 予測した価格（円）。
        pred_rows: 予測した行。
        tr: そのときの学習データ。補正に使ってよいのはこれだけ。
        features: モデルに入れる列名の一覧。

    Returns:
        補正した価格（円）。
    """
    calibrator = _fit_calibrator(tr, features)
    pred_log = np.log1p(price)
    calibrated_log = calibrator.predict(pred_log)
    return np.expm1(calibrated_log)
