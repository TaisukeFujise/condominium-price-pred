"""学習データの選択から後処理まで（工程 3〜8）をつなぎ、価格を予測する。

CV 用の fit_and_predict_valid と、本番用の fit_and_predict_test がある。
どちらも、渡された学習データ（CV では fold の学習データ）だけを使う。

CV 用と本番用で違うのは、次の 2 点だけ。それ以外の工程は同じ関数を通るので、CV と本番で処理がずれない。
- 繰り返しの回数（木の本数など）：CV 用は検証データのスコアで決める（early stopping）。本番用は CV の結果から決める。
- 2 段目（予測の混ぜ方。src/ensemble.py）：CV 用は 2 段目の入力を返すだけで、混ぜない。
  CV の 2 段目は、全 fold の予測がそろってから学習するため。混ぜて後処理するのは、main.py が blend_and_adjust を呼んで行う。
  本番用は、学習済みの 2 段目を受け取って、混ぜて後処理するところまで行う。
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
    """CV 用。モデルごとに tr で学習して va を予測し、2 段目の入力を作る。

    繰り返しの回数は、va のスコアで決める（early stopping）。
    予測はここでは混ぜない。混ぜるのは、main.py が 2 段目を学習してから blend_and_adjust で行う。

    Args:
        tr: fold の学習データ。
        va: fold の検証データ。
        features: モデルに入れる列名の一覧。
        model_classes: 使うモデルのクラス（main.py の MODELS）。

    Returns:
        va の 2 段目の入力（ensemble.make_inputs）と、学習済みのモデル。
        モデルの best_iteration に、early stopping で決まった繰り返しの回数が入る。
    """
    tr_used, weight, trend = _training_data(tr)
    fold_models: list[Model] = []
    for model in _new_models(model_classes):
        x_tr, y_tr, weight_tr = _apply_subset(
            model,
            tr_used[features],
            target.to_target(tr_used, trend, model.per_area),
            weight,
        )
        x_va, y_va, _ = _apply_subset(
            model, va[features], target.to_target(va, trend, model.per_area), None
        )
        fold_models.append(model.fit_with_validation(x_tr, y_tr, x_va, y_va, weight_tr))
    return _stack_inputs(fold_models, va, features, trend), fold_models


def fit_and_predict_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_classes: list[Callable[[], Model]],
    cv_models: list[list[Model]],
    stacker: lgb.Booster,
) -> tuple[np.ndarray, list[Model]]:
    """本番用。モデルごとに train 全体で学習して test を予測し、2 段目で混ぜて後処理する。

    Args:
        train: コンペの学習データ全体。
        test: コンペのテストデータ。
        features: モデルに入れる列名の一覧。
        model_classes: 使うモデルのクラス（main.py の MODELS）。
        cv_models: [i][fold] は、model_classes[i] を CV のその fold で学習したもの。
            本番は検証データがなく early stopping できないので、繰り返しの回数などはここから決める。
        stacker: 2 段目。CV の全 fold の予測から学習したもの。

    Returns:
        test の予測価格（円）と、学習済みのモデル。
    """
    train_used, weight, trend = _training_data(train)
    test_models: list[Model] = []
    for model, cv in zip(_new_models(model_classes), cv_models, strict=True):
        x_train, y_train, weight_train = _apply_subset(
            model,
            train_used[features],
            target.to_target(train_used, trend, model.per_area),
            weight,
        )
        test_models.append(model.fit(x_train, y_train, weight_train, cv))
    test_stack_inputs = _stack_inputs(test_models, test, features, trend)
    return (
        blend_and_adjust(test_stack_inputs, stacker, test, train, features),
        test_models,
    )


def blend_and_adjust(
    stack_inputs: pd.DataFrame,
    stacker: lgb.Booster | None,
    pred_rows: pd.DataFrame,
    tr: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """モデルごとの予測を 2 段目で混ぜ、後処理をかける（工程 7・8）。

    Args:
        stack_inputs: pred_rows の 2 段目の入力（ensemble.make_inputs）。
        stacker: 2 段目。None なら混ぜずに、土台（全行で学習したモデルの予測の log の平均）を使う。
        pred_rows: 予測した行。
        tr: 学習データ（CV では fold の学習データ）。学習に使う行を選んで（selection.select）から後処理に渡す。
        features: モデルに入れる列名の一覧。

    Returns:
        pred_rows の予測価格（円）。
    """
    tr_used, _ = selection.select(tr)
    price = ensemble.blend(stack_inputs, stacker)
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
    """学習に使う行と重み（工程 3）、相場の指数（工程 4）を決める。

    目的変数はここでは作らない。㎡単価か総額か（Model.per_area）がモデルごとに違うので、
    各モデルを学習する直前に target.to_target で作る。

    Args:
        tr: 学習データ。

    Returns:
        学習に使う行、行ごとの重み、相場の指数。
    """
    tr_used, weight = selection.select(tr)
    trend = target.fit_trend(tr_used)
    return tr_used, weight, trend


def _apply_subset(
    model: Model, x: pd.DataFrame, y: pd.Series, weight: pd.Series | None
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """モデルが学習に使う行（Model.subset）だけに絞る。

    Args:
        model: 学習前のモデル。
        x: 特徴量。
        y: 目的変数。
        weight: 行ごとの重み。None なら全行同じ重み。

    Returns:
        絞った特徴量・目的変数・重み。subset が None ならそのまま。
    """
    if model.subset is None:
        return x, y, weight
    rows = model.subset(x)
    return x[rows], y[rows], None if weight is None else weight[rows]


def _stack_inputs(
    models: list[Model],
    pred_rows: pd.DataFrame,
    features: list[str],
    trend: target.Trend,
) -> pd.DataFrame:
    """モデルごとに pred_rows を予測して円に戻し、2 段目の入力を作る。

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
