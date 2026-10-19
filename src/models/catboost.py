"""CatBoost の学習と予測。パラメータは params.py、メソッドの決まりは base.py。"""

from collections.abc import Callable
from typing import Self

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.models.base import Model, fill_missing_category
from src.models.params import CATBOOST_PARAMS


class CatBoost(Model):
    """CatBoost のモデル。1 つのインスタンスは 1 回だけ学習する。"""

    def __init__(
        self,
        params: dict = CATBOOST_PARAMS,
        per_area: bool = True,
        subset: Callable[[pd.DataFrame], np.ndarray] | None = None,
    ) -> None:
        """学習前のモデルを作る。

        Args:
            params: CatBoost のパラメータ。省略すると params.py の CATBOOST_PARAMS。
            per_area: 目的変数を㎡単価にするか（False なら総額）。
            subset: 学習に使う行を絞る条件（base.py の Model.subset）。None なら全行。
        """
        # 初期値の CATBOOST_PARAMS は全インスタンスで共有されるので、書き換えが広がらないようコピーする
        self.params = dict(params)
        self.per_area = per_area
        self.subset = subset
        self.model: CatBoostRegressor | None = None
        self.features: list[str] = []
        self._best_iteration: int | None = None
        self._offset = 0.0

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
        # CatBoost の Huber は 0 から学習を始め（boost_from_average が使えない）、1 回に動ける幅が
        # 小さいので、目的変数の水準に届くまでに木を使い切る。LightGBM と同じく平均から始める
        self._offset = _weighted_mean(y_train, weight)
        self.model = CatBoostRegressor(**self.params, iterations=10000)
        self.model.fit(
            _pool(x_train, y_train - self._offset, weight),
            eval_set=_pool(x_valid, y_valid - self._offset, None),
            early_stopping_rounds=100,
            use_best_model=True,
        )
        # CatBoost の best_iteration_ は 0 始まりの番号なので、木の本数にするには 1 を足す
        self._best_iteration = self.model.get_best_iteration() + 1
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
        iterations = round(np.mean([m.best_iteration for m in cv_models]))
        self.features = list(x.columns)
        self._offset = _weighted_mean(y, weight)
        self.model = CatBoostRegressor(**self.params, iterations=iterations)
        self.model.fit(_pool(x, y - self._offset, weight))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """予測する。

        Args:
            x: 特徴量。

        Returns:
            目的変数の予測。
        """
        return self._trained_model().predict(_pool(x, None, None)) + self._offset

    @property
    def best_iteration(self) -> int | None:
        """early stopping で決まった木の本数。fit() で作ったモデルでは None。"""
        return self._best_iteration

    def importance(self) -> pd.Series:
        """特徴量の重要度を計算する。

        Returns:
            特徴量ごとの重要度（PredictionValuesChange）の割合（合計 1）。大きい順に並ぶ。
        """
        values = pd.Series(
            self._trained_model().get_feature_importance(), index=self.features
        )
        return (values / values.sum()).sort_values(ascending=False)

    def _trained_model(self) -> CatBoostRegressor:
        """学習済みのモデルを返す。

        Returns:
            学習済みのモデル。学習前なら ValueError を投げる。
        """
        if self.model is None:
            raise ValueError(
                "まだ学習していない。先に fit か fit_with_validation を呼ぶ"
            )
        return self.model


def _weighted_mean(y: pd.Series, weight: pd.Series | None) -> float:
    """目的変数の（重み付き）平均を計算する。

    Args:
        y: 目的変数。
        weight: 行ごとの重み。None なら全行同じ重み。

    Returns:
        目的変数の平均。
    """
    return float(np.average(y, weights=weight))


def _pool(x: pd.DataFrame, y: pd.Series | None, weight: pd.Series | None) -> Pool:
    """カテゴリ列を CatBoost が受け取れる形にして Pool を作る。

    Args:
        x: 特徴量。
        y: 目的変数。予測のときは None。
        weight: 行ごとの重み。None なら全行同じ重み。

    Returns:
        CatBoost に渡す Pool。
    """
    # 文字列にするとカテゴリ列の型が消えるので、列名は先に取っておく
    cat_features = list(x.select_dtypes("category").columns)
    return Pool(fill_missing_category(x), y, cat_features=cat_features, weight=weight)
