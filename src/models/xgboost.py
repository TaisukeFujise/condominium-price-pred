"""XGBoost の学習と予測。パラメータは params.py、メソッドの決まりは base.py。"""

from typing import Self

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import Model
from src.models.params import XGBOOST_PARAMS


class XGBoost(Model):
    """XGBoost のモデル。1 つのインスタンスは 1 回だけ学習する。"""

    def __init__(self, params: dict = XGBOOST_PARAMS) -> None:
        """学習前のモデルを作る。

        Args:
            params: XGBoost のパラメータ。省略すると params.py の XGBOOST_PARAMS。
        """
        # 初期値の XGBOOST_PARAMS は全インスタンスで共有されるので、seed だけ変えるときなどに
        # 書き換えが他のモデルへ広がらないようコピーする
        self.params = dict(params)
        self.booster: xgb.Booster | None = None
        self._best_iteration: int | None = None

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
        self.booster = xgb.train(
            _with_base_score(self.params, y_train),
            _matrix(x_train, y_train, weight),
            num_boost_round=10000,
            evals=[(_matrix(x_valid, y_valid), "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        # XGBoost の best_iteration は 0 始まりなので、LightGBM と同じ「木の本数」にそろえる
        self._best_iteration = self.booster.best_iteration + 1
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
        self.booster = xgb.train(
            _with_base_score(self.params, y),
            _matrix(x, y, weight),
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
        # early stopping 後も booster には最良の本数より後の木が残っているので、使う範囲を指定する
        iteration_range = (0, self._best_iteration or 0)
        return self._trained_booster().predict(
            _matrix(x), iteration_range=iteration_range
        )

    @property
    def best_iteration(self) -> int | None:
        """early stopping で決まった木の本数。fit() で作ったモデルでは None。"""
        return self._best_iteration

    def importance(self) -> pd.Series:
        """特徴量の重要度を計算する。

        Returns:
            特徴量ごとの gain の割合（合計 1）。大きい順に並ぶ。
        """
        booster = self._trained_booster()
        # 一度も分岐に使われなかった特徴量は get_score に出てこないので 0 で埋める
        gain = pd.Series(booster.get_score(importance_type="total_gain")).reindex(
            booster.feature_names, fill_value=0.0
        )
        return (gain / gain.sum()).sort_values(ascending=False)

    def _trained_booster(self) -> xgb.Booster:
        """学習済みの booster を返す。

        Returns:
            学習済みの booster。学習前なら ValueError を投げる。
        """
        if self.booster is None:
            raise ValueError(
                "まだ学習していない。先に fit か fit_with_validation を呼ぶ"
            )
        return self.booster


def _with_base_score(params: dict, y: pd.Series) -> dict:
    """木を足し始める前の予測（base_score）に、目的変数の中央値を入れたパラメータを返す。

    Args:
        params: XGBoost のパラメータ。
        y: 学習データの目的変数。

    Returns:
        base_score を足したパラメータ。params に base_score があれば、そちらを使う。
    """
    # pseudo-Huber では XGBoost が自動で決める base_score が大きく外れ（目的変数は 14 前後なのに
    # 6 万を超える）、勾配が huber_slope で頭打ちになるため木で戻しきれない。
    # Huber は外れ値に強いので、平均ではなく中央値から始める
    return {"base_score": float(np.median(y)), **params}


def _matrix(
    x: pd.DataFrame, y: pd.Series | None = None, weight: pd.Series | None = None
) -> xgb.DMatrix:
    """XGBoost に渡す形にする。category 型の列はカテゴリとして扱う。

    Args:
        x: 特徴量。
        y: 目的変数。予測のときは None。
        weight: 行ごとの重み。None なら全行同じ重み。

    Returns:
        XGBoost の DMatrix。
    """
    # LightGBM と同じく category 型の列を種類ごとに分岐させる。指定しないとエラーになる
    return xgb.DMatrix(x, label=y, weight=weight, enable_categorical=True)
