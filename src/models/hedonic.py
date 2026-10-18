"""ヘドニック回帰（地区・駅などの固定効果と、築年数・面積などの曲線を足し合わせる線形モデル）。

単独の CV は 0.231 と弱いが、決定木を通らないので、LightGBM との残差の相関が 0.83 と低い
（決定木どうしは 0.96〜0.98）。平均の土台には入れず、src/ensemble.py の 2 段目の入力にだけ使う（#72）。
LinearResidual の線形の部分と違い、地区名・駅のような種類の多いカテゴリも one-hot で入れる。
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from src.models.base import Model
from src.models.params import HEDONIC_PARAMS

# 価格との関係が直線でない列。スプラインで曲線にする
_CURVE_COLUMNS = [
    "築年数",
    "面積_数値",
    "駅距離_分",
    "緯度",
    "経度",
    "東京駅までの距離_km",
]
_MISSING = "欠損"


class Hedonic(Model):
    """ヘドニック回帰のモデル。1 つのインスタンスは 1 回だけ学習する。"""

    stack_only = True

    def __init__(self, params: dict = HEDONIC_PARAMS) -> None:
        """学習前のモデルを作る。

        Args:
            params: params.py の HEDONIC_PARAMS の形の設定。
        """
        self.params = dict(params)
        self.model: Pipeline | None = None

    def fit_with_validation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        weight: pd.Series | None,
    ) -> Self:
        """CV 用に学習する。繰り返しの回数がないので、検証データは使わない。

        Args:
            x_train: 学習データの特徴量。
            y_train: 学習データの目的変数。
            x_valid: 検証データの特徴量。
            y_valid: 検証データの目的変数。
            weight: 学習データの行ごとの重み。None なら全行同じ重み。

        Returns:
            学習済みのモデル（自分自身）。
        """
        return self.fit(x_train, y_train, weight, [])

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        weight: pd.Series | None,
        cv_models: list[Self],
    ) -> Self:
        """本番用に学習する。CV と同じ設定なので、cv_models は使わない。

        Args:
            x: 特徴量。
            y: 目的変数。
            weight: 行ごとの重み。None なら全行同じ重み。
            cv_models: CV で同じモデルを fold ごとに学習したもの。

        Returns:
            学習済みのモデル（自分自身）。
        """
        x = _prepare(x)
        self.model = _build(x, self.params)
        self.model.fit(x, y, ridge__sample_weight=weight)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """予測する。

        Args:
            x: 特徴量。

        Returns:
            目的変数の予測。
        """
        if self.model is None:
            raise ValueError(
                "まだ学習していない。先に fit か fit_with_validation を呼ぶ"
            )
        return self.model.predict(_prepare(x))

    @property
    def best_iteration(self) -> None:
        """繰り返しの回数という考え方がないので None。"""
        return None

    def importance(self) -> None:
        """one-hot の列が数千あり、元の列ごとの重要度にまとめにくいので None を返す。"""
        return None


def _prepare(x: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ列の欠損を文字列の「欠損」にする（LinearResidual と同じ扱い）。

    Args:
        x: 特徴量。

    Returns:
        カテゴリ列を文字列にした特徴量。
    """
    cat_cols = x.select_dtypes("category").columns
    return x.assign(
        **{c: x[c].astype(object).fillna(_MISSING).astype(str) for c in cat_cols}
    )


def _build(x: pd.DataFrame, params: dict) -> Pipeline:
    """数値列は標準化、曲線の列はスプライン、カテゴリ列は one-hot にして Ridge へつなぐ。

    Args:
        x: _prepare 済みの学習データの特徴量。
        params: HEDONIC_PARAMS の形の設定。

    Returns:
        学習前のモデル。
    """
    cat_cols = x.select_dtypes("object").columns.tolist()
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
                "curve",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("spline", SplineTransformer(n_knots=params["n_knots"])),
                    ]
                ),
                _CURVE_COLUMNS,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=params["min_frequency"]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline(
        [("preprocess", preprocess), ("ridge", Ridge(alpha=params["alpha"]))]
    )
