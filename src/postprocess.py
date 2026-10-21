"""後処理（工程 8）。混ぜた予測の価格（円）を補正する。今は補正していない（adjust）。

CV でも検証データの予測に同じ補正をかけるので、補正の効果を CV で測れる。
補正に使ってよいのは、学習データ（tr）の情報だけ。CV では fold の学習データになる。
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src import target, validation
from src.config import TARGET
from src.models.lgbm import LGBM


def _fit_calibrator(tr: pd.DataFrame, features: list[str]) -> IsotonicRegression:
    """予測価格の log を実際の価格の log に対応させる isotonic 較正を、学習データだけで学習する。

    tr を時系列 CV と同じ方法で分け、いちばん新しい検証期間（直近 3 四半期）で、予測と実際の値を対応させる。
    この対応表で、安い物件を高く、高い物件を安く予測するずれ（平均への回帰）を補正する。

    Args:
        tr: そのときの学習データ（CV では fold の学習データ、本番では train 全体）。
        features: モデルに入れる列名の一覧。

    Returns:
        学習済みの IsotonicRegression。
    """
    inner_tr_pos, inner_va_pos = validation.time_series_folds(tr)[0]
    inner_tr, inner_va = tr.iloc[inner_tr_pos], tr.iloc[inner_va_pos]

    inner_trend = target.fit_trend(inner_tr)
    model = LGBM().fit_with_validation(
        inner_tr[features],
        target.to_target(inner_tr, inner_trend),
        inner_va[features],
        target.to_target(inner_va, inner_trend),
        None,
    )
    # adjust には円の価格が来るので、対応表も円の価格の log で作る。
    # モデルの出力（to_target の世界）のまま作ると、目的変数を変えたときに目盛りがずれる
    pred_price = target.to_price(
        model.predict(inner_va[features]), inner_va, inner_trend
    )
    pred_log = np.log1p(pred_price)
    actual_log = np.log1p(inner_va[TARGET].to_numpy())

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pred_log, actual_log)
    return calibrator


def _calibrate(price: np.ndarray, tr: pd.DataFrame, features: list[str]) -> np.ndarray:
    """予測した価格を、isotonic 較正で補正する。

    Args:
        price: 予測した価格（円）。
        tr: そのときの学習データ。補正に使ってよいのはこれだけ。
        features: モデルに入れる列名の一覧。

    Returns:
        補正した価格（円）。
    """
    calibrator = _fit_calibrator(tr, features)
    pred_log = np.log1p(price)
    calibrated_log = calibrator.predict(pred_log)
    return np.expm1(calibrated_log)


def adjust(
    price: np.ndarray, pred_rows: pd.DataFrame, tr: pd.DataFrame, features: list[str]
) -> np.ndarray:
    """予測した価格を補正する。今は何もせず、そのまま返す。

    isotonic 較正（_calibrate）は使わない（CV は良くなるが、Public LB で悪くなった。#22）。
    試すときは、戻り値を `_calibrate(price, tr, features)` にする。

    Args:
        price: 予測した価格（円）。
        pred_rows: 予測した行。
        tr: そのときの学習データ。補正に使ってよいのはこれだけ。
        features: モデルに入れる列名の一覧。

    Returns:
        補正した価格（円）。
    """
    return price
