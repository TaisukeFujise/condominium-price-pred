"""LightGBM の学習と予測。パラメータは params.py、メソッドの決まりは base.py。"""

from typing import Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import Model
from src.models.params import LGBM_PARAMS


class LGBM(Model):
    """LightGBM のモデル。1 つのインスタンスは 1 回だけ学習する。"""

    def __init__(self, params: dict = LGBM_PARAMS) -> None:
        """学習前のモデルを作る。

        Args:
            params: LightGBM のパラメータ。省略すると params.py の LGBM_PARAMS。
        """
        # 初期値の LGBM_PARAMS は全インスタンスで共有されるので、seed だけ変えるときなどに
        # 書き換えが他のモデルへ広がらないようコピーする
        self.params = dict(params)
        self.booster: lgb.Booster | None = None
        self.features: list[str] = []

    def fit_with_validation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        weight: pd.Series | None,
    ) -> Self:
        """CV 用に学習する。検証データのスコアが 100 ラウンド改善しなくなったら止める。

        Args:
            x_train: 学習データの特徴量。
            y_train: 学習データの目的変数。
            x_valid: 検証データの特徴量。
            y_valid: 検証データの目的変数。
            weight: 学習データの行ごとの重み。None なら全行同じ重み。

        Returns:
            学習済みのモデル（自分自身）。
        """
        self.features = list(x_train.columns)
        self.booster = lgb.train(
            self.params,
            lgb.Dataset(x_train, y_train, weight=weight),
            num_boost_round=10000,
            valid_sets=[lgb.Dataset(x_valid, y_valid)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return self

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        num_boost_round: int | None,
        weight: pd.Series | None,
    ) -> Self:
        """本番用に学習する。検証データがないので、木の本数は CV で決めた値を使う。

        Args:
            x: 特徴量。
            y: 目的変数。
            num_boost_round: 木の本数。
            weight: 行ごとの重み。None なら全行同じ重み。

        Returns:
            学習済みのモデル（自分自身）。
        """
        self.features = list(x.columns)
        self.booster = lgb.train(
            self.params,
            lgb.Dataset(x, y, weight=weight),
            num_boost_round=num_boost_round,
        )
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """予測する。fit_with_validation のモデルは、最良の本数までの木だけを使う。

        Args:
            x: 特徴量。

        Returns:
            目的変数の予測。
        """
        # best_iteration がない（fit() で作った）モデルでは、全部の木を使う
        return self.booster.predict(
            x, num_iteration=self.booster.best_iteration or None
        )

    @property
    def best_iteration(self) -> int | None:
        """early stopping で決まった木の本数。"""
        return self.booster.best_iteration

    def importance(self) -> pd.Series:
        """特徴量の重要度を計算する。

        Returns:
            特徴量ごとの gain の割合（合計 1）。大きい順に並ぶ。
        """
        gain = pd.Series(
            self.booster.feature_importance(importance_type="gain"), index=self.features
        )
        return (gain / gain.sum()).sort_values(ascending=False)
