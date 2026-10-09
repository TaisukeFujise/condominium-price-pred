"""工程 5：モデル。学習と予測。パラメータは step6_params.py。"""

import lightgbm as lgb
import pandas as pd

from src.step6_params import PARAMS


def fit_with_validation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    weight: pd.Series | None = None,
) -> lgb.Booster:
    """CV 用。検証データのスコアが 100 ラウンド改善しなくなったら止める。"""
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
    """本番用。検証データがないので、木の本数は CV で決めた値を使う。"""
    return lgb.train(PARAMS, lgb.Dataset(x, y, weight=weight), num_boost_round=num_boost_round)


def best_iteration(model: lgb.Booster) -> int:
    return model.best_iteration


def predict(model: lgb.Booster, x: pd.DataFrame) -> pd.Series:
    # best_iteration がない（fit() で作った）モデルでは、全部の木を使う
    return model.predict(x, num_iteration=model.best_iteration or None)


def importance(model: lgb.Booster, features: list[str]) -> pd.Series:
    """特徴量の重要度（gain の割合）。"""
    gain = pd.Series(model.feature_importance(importance_type="gain"), index=features)
    return (gain / gain.sum()).sort_values(ascending=False)
