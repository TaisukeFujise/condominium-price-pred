"""学習データの選択から後処理まで（工程 3〜8）をつなぎ、価格を予測する。

CV（fit_and_predict_valid）と本番（fit_and_predict_test）の違いは、繰り返しの回数（木の本数など）の決め方と、
混ぜ方（2 段目）をいつ渡すか。モデルはクラスのリスト（main.py の MODELS）で受け取り、それぞれの予測を ensemble.blend で混ぜる。
CV では 2 段目を全 fold の予測がそろってから学習するので、fit_and_predict_valid は 2 段目の入力を返し、
混ぜて後処理するのは main.py から blend_and_adjust を呼んで行う。本番は学習済みの 2 段目を先に受け取る。
それ以外の工程は _training_data・_stack_inputs・blend_and_adjust にまとめ、両方が同じものを通るようにする。
どちらも、渡された学習データ（CV では fold の学習データ）だけを使って動く。
"""

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import ensemble, postprocess, selection, target
from src.models.base import Model


def fit_and_predict_valid(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    features: list[str],
    model_classes: list[Callable[[], Model]],
) -> tuple[pd.DataFrame, list[Model]]:
    """CV 用。va のスコアで繰り返しの回数を決め（early stopping）、va をモデルごとに予測する。

    2 段目はほかの fold の予測から学習するので、ここでは混ぜない（blend_and_adjust で混ぜる）。

    Args:
        tr: fold の学習データ。
        va: fold の検証データ。
        features: モデルに入れる列名の一覧。
        model_classes: 使うモデルのクラス（main.py の MODELS）。

    Returns:
        va の 2 段目の入力（ensemble.make_inputs）と、学習済みのモデル。best_iteration に繰り返しの回数が入る。
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
    return _stack_inputs(models, va, features, trend), models


def fit_and_predict_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_classes: list[Callable[[], Model]],
    cv_models: list[list[Model]],
    stacker: lgb.Booster,
) -> tuple[np.ndarray, list[Model]]:
    """本番用。train 全体で学習し、test を予測する。

    Args:
        train: コンペの学習データ全体。
        test: コンペのテストデータ。
        features: モデルに入れる列名の一覧。
        model_classes: 使うモデルのクラス（main.py の MODELS）。
        cv_models: [i] は model_classes[i] を CV の fold ごとに学習したもの。
            本番は検証データがないので、繰り返しの回数などはここから決める。
        stacker: 2 段目。CV の全 fold の予測から学習したもの。

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
    inputs = _stack_inputs(models, test, features, trend)
    return blend_and_adjust(inputs, stacker, test, train, features), models


def blend_and_adjust(
    inputs: pd.DataFrame,
    stacker: lgb.Booster | None,
    pred_rows: pd.DataFrame,
    tr: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """モデルごとの予測を 2 段目で混ぜ、後処理をかける（工程 7・8）。

    Args:
        inputs: pred_rows の 2 段目の入力（ensemble.make_inputs）。
        stacker: 2 段目。None なら土台（全行で学習したモデルの log の平均）のまま。
        pred_rows: 予測した行。
        tr: 学習データ（CV では fold の学習データ）。後処理に渡す。
        features: モデルに入れる列名の一覧。

    Returns:
        pred_rows の予測価格（円）。
    """
    tr_used, _ = selection.select(tr)
    price = ensemble.blend(inputs, stacker)
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


def _stack_inputs(
    models: list[Model],
    pred_rows: pd.DataFrame,
    features: list[str],
    trend: target.Trend,
) -> pd.DataFrame:
    """モデルごとに予測して円に戻し、2 段目の入力にする。

    Args:
        models: 学習済みのモデル。
        pred_rows: 予測する行。
        features: モデルに入れる列名の一覧。
        trend: 予測価格を円に戻す相場の指数。

    Returns:
        pred_rows の 2 段目の入力（ensemble.make_inputs）。
    """
    preds = [
        target.to_price(
            model.predict(pred_rows[features]), pred_rows, trend, model.per_area
        )
        for model in models
    ]
    in_base = [model.subset is None and not model.stack_only for model in models]
    return ensemble.make_inputs(preds, in_base, pred_rows, trend)
