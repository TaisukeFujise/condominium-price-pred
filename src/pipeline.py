"""学習データの選択から後処理まで（工程 3〜8）をつなぎ、価格を予測する。

CV（fit_and_predict_valid）と本番（fit_and_predict_test）の違いは、繰り返しの回数（木の本数など）の決め方だけ。
モデルはクラスのリスト（main.py の MODELS）で受け取り、それぞれの予測を ensemble.blend で混ぜる。
それ以外の工程は _training_data と _predict_price にまとめ、両方が同じものを通るようにする。
どちらも、渡された学習データ（CV では fold の学習データ）だけを使って動く。
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src import ensemble, postprocess, selection, target
from src.models.base import Model


def fit_and_predict_valid(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    features: list[str],
    model_classes: list[Callable[[], Model]],
) -> tuple[np.ndarray, list[Model]]:
    """CV 用。va のスコアで繰り返しの回数を決め（early stopping）、va を予測する。

    Args:
        tr: fold の学習データ。
        va: fold の検証データ。
        features: モデルに入れる列名の一覧。
        model_classes: 使うモデルのクラス（main.py の MODELS）。

    Returns:
        va の予測価格（円）と、学習済みのモデル。best_iteration に繰り返しの回数が入る。
    """
    tr_used, weight, y = _training_data(tr)
    models = [
        model.fit_with_validation(
            tr_used[features], y, va[features], target.to_target(va), weight
        )
        for model in _new_models(model_classes)
    ]
    price = _predict_price(models, va, tr_used, features)
    return price, models


def fit_and_predict_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_classes: list[Callable[[], Model]],
    cv_models: list[list[Model]],
) -> tuple[np.ndarray, list[Model]]:
    """本番用。train 全体で学習し、test を予測する。

    Args:
        train: コンペの学習データ全体。
        test: コンペのテストデータ。
        features: モデルに入れる列名の一覧。
        model_classes: 使うモデルのクラス（main.py の MODELS）。
        cv_models: [i] は model_classes[i] を CV の fold ごとに学習したもの。
            本番は検証データがないので、繰り返しの回数などはここから決める。

    Returns:
        test の予測価格（円）と、学習済みのモデル。
    """
    tr_used, weight, y = _training_data(train)
    models = [
        model.fit(tr_used[features], y, weight, cv)
        for model, cv in zip(_new_models(model_classes), cv_models, strict=True)
    ]
    price = _predict_price(models, test, tr_used, features)
    return price, models


def _new_models(model_classes: list[Callable[[], Model]]) -> list[Model]:
    """学習前のモデルを 1 つずつ作る。

    Args:
        model_classes: 使うモデルのクラス。

    Returns:
        学習前のモデル。model_classes と同じ順に並ぶ。
    """
    return [model_class() for model_class in model_classes]


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
    models: list[Model],
    pred_rows: pd.DataFrame,
    tr_used: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """モデルごとに予測して円に戻し、アンサンブルと後処理をかける（工程 7・8）。

    Args:
        models: 学習済みのモデル。
        pred_rows: 予測する行。
        tr_used: 学習に使った行。後処理に渡す。
        features: モデルに入れる列名の一覧。

    Returns:
        pred_rows の予測価格（円）。
    """
    preds = [
        target.to_price(model.predict(pred_rows[features]), pred_rows)
        for model in models
    ]
    price = ensemble.blend(preds)
    return postprocess.adjust(price, pred_rows, tr_used, features)
