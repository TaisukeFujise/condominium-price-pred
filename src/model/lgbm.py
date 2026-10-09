"""LightGBM の学習と予測。パラメータは params.py。"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model.params import PARAMS


def fit_with_validation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    weight: pd.Series | None = None,
) -> lgb.Booster:
    """CV 用に学習する。検証データのスコアが 100 ラウンド改善しなくなったら止める。

    Args:
        x_train: 学習データの特徴量。
        y_train: 学習データの目的変数。
        x_valid: 検証データの特徴量。
        y_valid: 検証データの目的変数。
        weight: 学習データの行ごとの重み。None なら全行同じ重み。

    Returns:
        学習済みのモデル。best_iteration に最良の木の本数が入る。
    """
    return lgb.train(
        PARAMS,
        lgb.Dataset(x_train, y_train, weight=weight),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(x_valid, y_valid)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def fit(
    x: pd.DataFrame,
    y: pd.Series,
    num_boost_round: int,
    weight: pd.Series | None = None,
) -> lgb.Booster:
    """本番用に学習する。検証データがないので、木の本数は CV で決めた値を使う。

    Args:
        x: 特徴量。
        y: 目的変数。
        num_boost_round: 木の本数。
        weight: 行ごとの重み。None なら全行同じ重み。

    Returns:
        学習済みのモデル。
    """
    return lgb.train(PARAMS, lgb.Dataset(x, y, weight=weight), num_boost_round=num_boost_round)


def best_iteration(model: lgb.Booster) -> int:
    """early stopping で決まった木の本数を返す。"""
    return model.best_iteration


def predict(model: lgb.Booster, x: pd.DataFrame) -> np.ndarray:
    """予測する。fit_with_validation のモデルは、最良の本数までの木だけを使う。"""
    # best_iteration がない（fit() で作った）モデルでは、全部の木を使う
    return model.predict(x, num_iteration=model.best_iteration or None)


def importance(model: lgb.Booster, features: list[str]) -> pd.Series:
    """特徴量の重要度を計算する。

    Args:
        model: 学習済みのモデル。
        features: モデルに入れた列名の一覧（学習時と同じ順番）。

    Returns:
        特徴量ごとの gain の割合（合計 1）。大きい順に並ぶ。
    """
    gain = pd.Series(model.feature_importance(importance_type="gain"), index=features)
    return (gain / gain.sum()).sort_values(ascending=False)
