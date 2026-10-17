"""学習データの選択から後処理まで（工程 3〜8）をつなぎ、価格を予測する。

CV（fit_and_predict_valid）と本番（fit_and_predict_test）の違いは、繰り返しの回数（木の本数など）の決め方と、
混ぜる重みをいつ渡すか。モデルはクラスのリスト（main.py の MODELS）で受け取り、それぞれの予測を ensemble.blend で混ぜる。
CV では重みを全 fold の予測がそろってから決めるので、fit_and_predict_valid はモデルごとの予測を返し、
混ぜて後処理するのは main.py から blend_and_adjust を呼んで行う。本番は重みを先に受け取る。
それ以外の工程は _training_data・_predict_each・blend_and_adjust にまとめ、両方が同じものを通るようにする。
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
) -> tuple[list[np.ndarray], list[Model]]:
    """CV 用。va のスコアで繰り返しの回数を決め（early stopping）、va をモデルごとに予測する。

    混ぜる重みはほかの fold の予測から決めるので、ここでは混ぜない（blend_and_adjust で混ぜる）。

    Args:
        tr: fold の学習データ。
        va: fold の検証データ。
        features: モデルに入れる列名の一覧。
        model_classes: 使うモデルのクラス（main.py の MODELS）。

    Returns:
        va のモデルごとの予測価格（円）と、学習済みのモデル。best_iteration に繰り返しの回数が入る。
    """
    tr_used, weight, trend = _training_data(tr)
    models = [
        model.fit_with_validation(
            tr_used[features],
            target.to_target(tr_used, trend, model.per_area),
            va[features],
            target.to_target(va, trend, model.per_area),
            weight,
        )
        for model in _new_models(model_classes)
    ]
    return _predict_each(models, va, features, trend), models


def fit_and_predict_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_classes: list[Callable[[], Model]],
    cv_models: list[list[Model]],
    weights: np.ndarray,
) -> tuple[np.ndarray, list[Model]]:
    """本番用。train 全体で学習し、test を予測する。

    Args:
        train: コンペの学習データ全体。
        test: コンペのテストデータ。
        features: モデルに入れる列名の一覧。
        model_classes: 使うモデルのクラス（main.py の MODELS）。
        cv_models: [i] は model_classes[i] を CV の fold ごとに学習したもの。
            本番は検証データがないので、繰り返しの回数などはここから決める。
        weights: モデルごとの混ぜる重み。CV の全 fold の予測から決めたもの。

    Returns:
        test の予測価格（円）と、学習済みのモデル。
    """
    tr_used, weight, trend = _training_data(train)
    models = [
        model.fit(
            tr_used[features],
            target.to_target(tr_used, trend, model.per_area),
            weight,
            cv,
        )
        for model, cv in zip(_new_models(model_classes), cv_models, strict=True)
    ]
    preds = _predict_each(models, test, features, trend)
    return blend_and_adjust(preds, weights, test, train, features), models


def blend_and_adjust(
    preds: list[np.ndarray],
    weights: np.ndarray | None,
    pred_rows: pd.DataFrame,
    tr: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """モデルごとの予測を混ぜ、後処理をかける（工程 7・8）。

    Args:
        preds: pred_rows のモデルごとの予測価格（円）。
        weights: モデルごとの混ぜる重み。None なら単純平均。
        pred_rows: 予測した行。
        tr: 学習データ（CV では fold の学習データ）。後処理に渡す。
        features: モデルに入れる列名の一覧。

    Returns:
        pred_rows の予測価格（円）。
    """
    tr_used, _ = selection.select(tr)
    price = ensemble.blend(preds, weights)
    return postprocess.adjust(price, pred_rows, tr_used, features)


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
) -> tuple[pd.DataFrame, pd.Series | None, target.Trend]:
    """学習に使う行・重み・相場の指数を決める（工程 3・4）。

    目的変数はモデルごとに違ってよい（Model.per_area）ので、ここでは作らず、モデルごとに to_target で作る。

    Args:
        tr: 学習データ。

    Returns:
        学習に使う行、行ごとの重み、相場の指数。
    """
    tr_used, weight = selection.select(tr)
    trend = target.fit_trend(tr_used)
    return tr_used, weight, trend


def _predict_each(
    models: list[Model],
    pred_rows: pd.DataFrame,
    features: list[str],
    trend: target.Trend,
) -> list[np.ndarray]:
    """モデルごとに予測して円に戻す。

    Args:
        models: 学習済みのモデル。
        pred_rows: 予測する行。
        features: モデルに入れる列名の一覧。
        trend: 予測価格を円に戻す相場の指数。

    Returns:
        pred_rows のモデルごとの予測価格（円）。models と同じ順に並ぶ。
    """
    return [
        target.to_price(
            model.predict(pred_rows[features]), pred_rows, trend, model.per_area
        )
        for model in models
    ]
