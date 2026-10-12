"""学習データの選択から後処理まで（工程 3〜8）をつなぎ、価格を予測する。

CV（fit_and_predict_valid）と本番（fit_and_predict_test）の違いは、繰り返しの回数（木の本数など）の決め方だけ。
それ以外の工程は _training_data と _predict_price にまとめ、両方が同じものを通るようにする。
どちらも、渡された学習データ（CV では fold の学習データ）だけを使って動く。
"""

import numpy as np
import pandas as pd

from src import ensemble, postprocess, selection, target
from src.models.base import Model


def fit_and_predict_valid(
    tr: pd.DataFrame, va: pd.DataFrame, features: list[str], model: Model
) -> tuple[np.ndarray, Model]:
    """CV 用。va のスコアで木の本数を決め（early stopping）、va を予測する。

    Args:
        tr: fold の学習データ。
        va: fold の検証データ。
        features: モデルに入れる列名の一覧。
        model: 学習前のモデル。

    Returns:
        va の予測価格（円）と、学習済みのモデル。best_iteration に繰り返しの回数が入る。
    """
    tr_used, weight, y = _training_data(tr)
    model = model.fit_with_validation(
        tr_used[features], y, va[features], target.to_target(va), weight
    )
    price = _predict_price(model, va, tr_used, features)
    return price, model


def fit_and_predict_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model: Model,
    num_boost_round: int | None,
) -> tuple[np.ndarray, Model]:
    """本番用。train 全体を決まった繰り返しの回数で学習し、test を予測する。

    Args:
        train: コンペの学習データ全体。
        test: コンペのテストデータ。
        features: モデルに入れる列名の一覧。
        model: 学習前のモデル。
        num_boost_round: 繰り返しの回数（木の本数など）。

    Returns:
        test の予測価格（円）と、学習済みのモデル。
    """
    tr_used, weight, y = _training_data(train)
    model = model.fit(tr_used[features], y, num_boost_round, weight)
    price = _predict_price(model, test, tr_used, features)
    return price, model


def _training_data(
    tr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series]:
    """学習に使う行・重み・目的変数を決める（工程 3・4）。

    Args:
        tr: 学習データ。

    Returns:
        学習に使う行、行ごとの重み、目的変数。
    """
    tr_used, weight = selection.select(tr)
    return tr_used, weight, target.to_target(tr_used)


def _predict_price(
    model: Model,
    pred_rows: pd.DataFrame,
    tr_used: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """予測して円に戻し、アンサンブルと後処理をかける（工程 7・8）。

    Args:
        model: 学習済みのモデル。
        pred_rows: 予測する行。
        tr_used: 学習に使った行。後処理に渡す。
        features: モデルに入れる列名の一覧。

    Returns:
        pred_rows の予測価格（円）。
    """
    price = target.to_price(model.predict(pred_rows[features]), pred_rows)
    price = ensemble.blend([price])
    return postprocess.adjust(price, pred_rows, tr_used)
