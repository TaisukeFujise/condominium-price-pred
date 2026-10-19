"""LightGBM の学習と予測。パラメータは params.py、メソッドの決まりは base.py。"""

from collections.abc import Callable
from typing import Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import Model
from src.models.params import LGBM_PARAMS


class LGBM(Model):
    """LightGBM のモデル。1 つのインスタンスは 1 回だけ学習する。"""

    def __init__(
        self,
        params: dict = LGBM_PARAMS,
        per_area: bool = True,
        subset: Callable[[pd.DataFrame], np.ndarray] | None = None,
    ) -> None:
        """学習前のモデルを作る。

        Args:
            params: LightGBM のパラメータ。省略すると params.py の LGBM_PARAMS。
            per_area: 目的変数を㎡単価にするか（False なら総額）。
            subset: 学習に使う行を絞る条件（base.py の Model.subset）。None なら全行。
        """
        # 初期値の LGBM_PARAMS は全インスタンスで共有されるので、seed だけ変えるときなどに
        # 書き換えが他のモデルへ広がらないようコピーする
        self.params = dict(params)
        self.per_area = per_area
        self.subset = subset
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
        weight: pd.Series | None,
        cv_models: list[Self],
    ) -> Self:
        """本番用に学習する。木の本数は、CV の fold ごとの best_iteration の平均を使う。

        Args:
            x: 特徴量。
            y: 目的変数。
            weight: 行ごとの重み。None なら全行同じ重み。
            cv_models: CV で同じモデルを fold ごとに学習したもの。

        Returns:
            学習済みのモデル（自分自身）。
        """
        # fold ごとに学習データの量が違うので、1 つの fold に合わせず平均を使う
        num_boost_round = round(np.mean([m.best_iteration for m in cv_models]))
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
        return self._trained_booster().predict(x, num_iteration=self.best_iteration)

    @property
    def best_iteration(self) -> int | None:
        """early stopping で決まった木の本数。fit() で作ったモデルでは None。"""
        # LightGBM は early stopping しなかったとき 0 を返すが、Model の決まりでは None
        return self._trained_booster().best_iteration or None

    def importance(self) -> pd.Series:
        """特徴量の重要度を計算する。

        Returns:
            特徴量ごとの gain の割合（合計 1）。大きい順に並ぶ。
        """
        gain = pd.Series(
            self._trained_booster().feature_importance(importance_type="gain"),
            index=self.features,
        )
        return (gain / gain.sum()).sort_values(ascending=False)

    def _trained_booster(self) -> lgb.Booster:
        """学習済みの booster を返す。

        Returns:
            学習済みの booster。学習前なら ValueError を投げる。
        """
        # 学習前に呼ぶと 'NoneType' has no attribute という分かりにくいエラーになるので、先に止める
        if self.booster is None:
            raise ValueError(
                "まだ学習していない。先に fit か fit_with_validation を呼ぶ"
            )
        return self.booster
