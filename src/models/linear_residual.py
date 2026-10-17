"""線形回帰（Ridge）で粗く当て、残差を LightGBM で学習するモデル。

GBDT 同士（LightGBM・CatBoost）は残差相関が 0.9 超えになりやすく、混ぜても伸びにくい
（Issue #38）。線形回帰は GBDT と間違え方の系統が違うので、MLP のような前処理コスト
（標準化・カテゴリの埋め込み・欠損処理を一から作る）をかけずに安く多様性を作れる。
"""

from typing import Self

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.base import Model
from src.models.params import LGBM_PARAMS

# 地区名・最寄駅：名称はカテゴリ数が多すぎる（1000〜600 種類）ので線形側では使わない。
# 緯度・経度・主要駅までの距離が数値特徴量としてすでにあり、位置情報は失われない。
# 残差側の LightGBM にはどちらもそのまま渡す
_LINEAR_CAT_MAX_CARDINALITY = 60
_MISSING = "欠損"


class LinearResidual(Model):
    """Ridge の残差を LightGBM で学習するモデル。1 つのインスタンスは 1 回だけ学習する。"""

    def __init__(
        self, params: dict = LGBM_PARAMS, per_area: bool = True, alpha: float = 10.0
    ) -> None:
        """学習前のモデルを作る。

        Args:
            params: 残差側 LightGBM のパラメータ。省略すると params.py の LGBM_PARAMS。
            per_area: 目的変数を㎡単価にするか（False なら総額）。
            alpha: Ridge の正則化の強さ。
        """
        self.params = dict(params)
        self.per_area = per_area
        self.alpha = alpha
        self._linear: Pipeline | None = None
        self._linear_cols: list[str] = []
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
        """CV 用に学習する。検証データの残差が 100 ラウンド改善しなくなったら止める。

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
        resid_train, resid_valid = self._fit_linear(
            x_train, y_train, x_valid, y_valid, weight
        )
        self.booster = lgb.train(
            self.params,
            lgb.Dataset(x_train[self.features], resid_train, weight=weight),
            num_boost_round=10000,
            valid_sets=[lgb.Dataset(x_valid[self.features], resid_valid)],
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
        self.features = list(x.columns)
        self._linear_cols = _linear_columns(x)
        self._linear = _build_linear(x[self._linear_cols], self.alpha)
        self._linear.fit(_prepare(x[self._linear_cols]), y, ridge__sample_weight=weight)
        resid = y - self._linear.predict(_prepare(x[self._linear_cols]))
        num_boost_round = round(np.mean([m.best_iteration for m in cv_models]))
        self.booster = lgb.train(
            self.params,
            lgb.Dataset(x[self.features], resid, weight=weight),
            num_boost_round=num_boost_round,
        )
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """予測する（線形回帰の予測 + 残差 LightGBM の予測）。

        Args:
            x: 特徴量。

        Returns:
            目的変数の予測。
        """
        linear_pred = self._linear.predict(_prepare(x[self._linear_cols]))
        resid_pred = self._trained_booster().predict(
            x[self.features], num_iteration=self.best_iteration
        )
        return linear_pred + resid_pred

    @property
    def best_iteration(self) -> int | None:
        """early stopping で決まった木の本数。fit() で作ったモデルでは None。"""
        return self._trained_booster().best_iteration or None

    def importance(self) -> pd.Series:
        """残差側 LightGBM の特徴量重要度を計算する。

        Returns:
            特徴量ごとの gain の割合（合計 1）。大きい順に並ぶ。線形回帰側の寄与は含まない。
        """
        gain = pd.Series(
            self._trained_booster().feature_importance(importance_type="gain"),
            index=self.features,
        )
        return (gain / gain.sum()).sort_values(ascending=False)

    def _fit_linear(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        weight: pd.Series | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """線形回帰を学習し、train・valid の残差（目的変数 − 線形回帰の予測）を返す。"""
        self._linear_cols = _linear_columns(x_train)
        self._linear = _build_linear(x_train[self._linear_cols], self.alpha)
        self._linear.fit(
            _prepare(x_train[self._linear_cols]), y_train, ridge__sample_weight=weight
        )
        resid_train = y_train - self._linear.predict(
            _prepare(x_train[self._linear_cols])
        )
        resid_valid = y_valid - self._linear.predict(
            _prepare(x_valid[self._linear_cols])
        )
        return resid_train, resid_valid

    def _trained_booster(self) -> lgb.Booster:
        """学習済みの booster を返す。"""
        if self.booster is None:
            raise ValueError(
                "まだ学習していない。先に fit か fit_with_validation を呼ぶ"
            )
        return self.booster


def _linear_columns(x: pd.DataFrame) -> list[str]:
    """線形回帰に使う列を選ぶ。カテゴリ数が多すぎる列は除く。"""
    cat_cols = x.select_dtypes("category").columns
    num_cols = [c for c in x.columns if c not in cat_cols]
    low_card_cat_cols = [
        c for c in cat_cols if x[c].nunique() <= _LINEAR_CAT_MAX_CARDINALITY
    ]
    return num_cols + low_card_cat_cols


def _prepare(x: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ列の欠損を文字列の「欠損」にする（CatBoost と同じ扱い）。"""
    cat_cols = x.select_dtypes("category").columns
    return x.assign(
        **{c: x[c].astype(object).fillna(_MISSING).astype(str) for c in cat_cols}
    )


def _build_linear(x: pd.DataFrame, alpha: float) -> Pipeline:
    """数値列は欠損埋め+標準化、カテゴリ列は one-hot にして Ridge へつなぐ。"""
    cat_cols = x.select_dtypes("category").columns.tolist()
    num_cols = [c for c in x.columns if c not in cat_cols]
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_cols,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=alpha))])
