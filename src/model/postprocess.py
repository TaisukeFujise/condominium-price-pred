"""後処理。予測した価格（円）を補正する。

CV では検証データの予測にも同じ補正をかけるので、補正の効果も CV で測れる。
補正に使ってよいのは train の情報だけ（CV では、その fold の学習データだけ）。
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src import validation
from src.features.build import build_features
from src.model import lgbm, target


def _fit_calibrator(train: pd.DataFrame) -> IsotonicRegression:
    """train の中にさらに内側のホールドアウト（直近 3 四半期）を作り、
    予測値（log）から実際値（log）への isotonic 較正を学習する。

    安い物件を過大予測し、高い物件を過小予測するという価格帯による系統的なズレ
    （regression to the mean）を、この対応表で補正する。

    Args:
        train: 学習データ（その時点の fold の学習データ、または本番用の全データ）。

    Returns:
        学習済みの IsotonicRegression。
    """
    _, features = build_features(train)
    # time_series_folds は位置（flatnonzero）を返すので、loc で引けるよう index をリセットする
    train = train.reset_index(drop=True)
    inner_train_idx, inner_valid_idx = validation.time_series_folds(train)[0]
    inner_train, inner_valid = train.loc[inner_train_idx], train.loc[inner_valid_idx]

    y_train = target.make_target(inner_train)
    model = lgbm.fit_with_validation(
        inner_train[features],
        y_train,
        inner_valid[features],
        target.make_target(inner_valid),
    )
    pred_log = lgbm.predict(model, inner_valid[features])
    actual_log = target.make_target(inner_valid).to_numpy()

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pred_log, actual_log)
    return calibrator


def postprocess(
    price: np.ndarray, rows: pd.DataFrame, train: pd.DataFrame
) -> np.ndarray:
    """予測した価格を、isotonic 較正で補正する。

    Args:
        price: 予測した価格（円）。
        rows: 予測した行。
        train: そのときの学習データ。補正に使ってよいのはこれだけ。

    Returns:
        補正した価格（円）。
    """
    calibrator = _fit_calibrator(train)
    pred_log = np.log1p(price)
    calibrated_log = calibrator.predict(pred_log)
    return np.expm1(calibrated_log)
