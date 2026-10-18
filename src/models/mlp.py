"""ニューラルネット（scikit-learn の MLPRegressor）の学習と予測。パラメータは params.py。

GBDT（LightGBM・CatBoost）や、残差を GBDT で学習するモデル（LinearResidual）は、
どれもブレンドとの残差相関が 0.93〜0.98 と高く、足しても伸びにくい（Issue #38）。
木を使わないモデルなら間違え方の系統が変わるので、単体が少し弱くても混ぜると伸びる見込みがある。
"""

import copy
from typing import Self

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from src.models.base import Model
from src.models.params import MLP_PARAMS

_MISSING = "欠損"
# 検証データのスコアが何エポック改善しなかったら止めるか
_PATIENCE = 5
_MAX_EPOCHS = 200


class MLP(Model):
    """MLPRegressor のモデル。1 つのインスタンスは 1 回だけ学習する。"""

    def __init__(self, params: dict = MLP_PARAMS) -> None:
        """学習前のモデルを作る。

        Args:
            params: MLPRegressor のパラメータ。省略すると params.py の MLP_PARAMS。
        """
        self.params = dict(params)
        self._encoder: ColumnTransformer | None = None
        self._net: MLPRegressor | None = None
        self._best_iteration: int | None = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit_with_validation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        weight: pd.Series | None,
    ) -> Self:
        """CV 用に学習する。検証データのスコアが 5 エポック改善しなくなったら止める。

        Args:
            x_train: 学習データの特徴量。
            y_train: 学習データの目的変数。
            x_valid: 検証データの特徴量。
            y_valid: 検証データの目的変数。
            weight: 学習データの行ごとの重み。None なら全行同じ重み。

        Returns:
            学習済みのモデル（自分自身）。
        """
        x, y = self._start(x_train, y_train)
        x_va = self._encoder.transform(_prepare(x_valid))
        best_score, best_net, best_epoch = np.inf, None, 0
        for epoch in range(1, _MAX_EPOCHS + 1):
            self._net.partial_fit(x, y, sample_weight=_to_array(weight))
            score = np.sqrt(np.mean((self._predict_encoded(x_va) - y_valid) ** 2))
            if score < best_score:
                best_score, best_net, best_epoch = (
                    score,
                    copy.deepcopy(self._net),
                    epoch,
                )
            elif epoch - best_epoch >= _PATIENCE:
                break
        self._net = best_net
        self._best_iteration = best_epoch
        return self

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        weight: pd.Series | None,
        cv_models: list[Self],
    ) -> Self:
        """本番用に学習する。エポック数は、CV の fold ごとの best_iteration の平均を使う。

        Args:
            x: 特徴量。
            y: 目的変数。
            weight: 行ごとの重み。None なら全行同じ重み。
            cv_models: CV で同じモデルを fold ごとに学習したもの。

        Returns:
            学習済みのモデル（自分自身）。
        """
        # fold ごとに学習データの量が違うので、1 つの fold に合わせず平均を使う
        epochs = round(np.mean([m.best_iteration for m in cv_models]))
        x_enc, y_scaled = self._start(x, y)
        for _ in range(epochs):
            self._net.partial_fit(x_enc, y_scaled, sample_weight=_to_array(weight))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """予測する。

        Args:
            x: 特徴量。

        Returns:
            目的変数の予測。
        """
        if self._encoder is None or self._net is None:
            raise ValueError(
                "まだ学習していない。先に fit か fit_with_validation を呼ぶ"
            )
        return self._predict_encoded(self._encoder.transform(_prepare(x)))

    @property
    def best_iteration(self) -> int | None:
        """early stopping で決まったエポック数。fit() で作ったモデルでは None。"""
        return self._best_iteration

    def importance(self) -> None:
        """ニューラルネットは重要度を出せないので None を返す。"""
        return None

    def _start(self, x: pd.DataFrame, y: pd.Series) -> tuple[object, np.ndarray]:
        """列の変換を学習データで学習し、学習前のネットを作る。

        Args:
            x: 学習データの特徴量。
            y: 学習データの目的変数。

        Returns:
            変換した特徴量と、標準化した目的変数。
        """
        x = _prepare(x)
        self._encoder = _build_encoder(x)
        x_enc = self._encoder.fit_transform(x)
        # 目的変数は 0 付近にない（log(㎡単価) の水準）ので、そのままだと最初の数エポックを
        # 水準合わせに使ってしまう。平均 0・分散 1 にして学習し、予測で戻す
        self._y_mean, self._y_std = float(y.mean()), float(y.std())
        self._net = MLPRegressor(**self.params)
        return x_enc, ((y - self._y_mean) / self._y_std).to_numpy()

    def _predict_encoded(self, x_enc: object) -> np.ndarray:
        """変換済みの特徴量から、目的変数の予測（標準化を戻した値）を返す。

        Args:
            x_enc: _encoder で変換した特徴量。

        Returns:
            目的変数の予測。
        """
        return self._net.predict(x_enc) * self._y_std + self._y_mean


def _to_array(weight: pd.Series | None) -> np.ndarray | None:
    """重みを MLPRegressor が受け取れる配列にする。

    Args:
        weight: 行ごとの重み。None なら全行同じ重み。

    Returns:
        重みの配列。None ならそのまま。
    """
    return None if weight is None else weight.to_numpy()


def _prepare(x: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ列の欠損を文字列の「欠損」にする（CatBoost・LinearResidual と同じ扱い）。

    Args:
        x: 特徴量。

    Returns:
        カテゴリ列を文字列にした特徴量。
    """
    cat_cols = x.select_dtypes("category").columns
    return x.assign(
        **{c: x[c].astype(object).fillna(_MISSING).astype(str) for c in cat_cols}
    )


def _build_encoder(x: pd.DataFrame) -> ColumnTransformer:
    """数値列は分位点で正規分布に変換し、カテゴリ列は one-hot にする変換を作る。

    Args:
        x: _prepare 済みの学習データの特徴量。

    Returns:
        学習前の変換。
    """
    cat_cols = x.select_dtypes("object").columns.tolist()
    num_cols = [c for c in x.columns if c not in cat_cols]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        # 欠損かどうかも情報なので、埋めた値とは別に 0/1 の列を足す
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        # 面積・件数などは裾が長く、標準化だけだと外れ値でネットが不安定になる
                        (
                            "scale",
                            QuantileTransformer(output_distribution="normal"),
                        ),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                # 地区名・駅名もそのまま one-hot にする。1 層目の重みが地区ごとの値の役をする。
                # 件数の少ない値は 1 つにまとめ、数件の取引に合わせすぎないようにする
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10),
                cat_cols,
            ),
        ]
    )
