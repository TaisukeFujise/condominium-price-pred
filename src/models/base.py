"""モデルが揃えるメソッドの決まり。src/models/ のモデルは、すべて Model を継承する。

pipeline.py と main.py はこのメソッドだけを使うので、Model を継承したクラスを書けば
main.py の MODEL を書き換えるだけでモデルを付け替えられる（例は lgbm.py）。
メソッドが 1 つでも足りないと、インスタンスを作った時点でエラーになる。
"""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd


class Model(ABC):
    """モデルが揃えるメソッド。1 つのインスタンスは 1 回だけ学習する。"""

    @abstractmethod
    def fit_with_validation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        weight: pd.Series | None,
    ) -> Self:
        """CV 用に学習する。検証データを見て、繰り返しの回数（木の本数など）を決める。

        Args:
            x_train: 学習データの特徴量。
            y_train: 学習データの目的変数。
            x_valid: 検証データの特徴量。
            y_valid: 検証データの目的変数。
            weight: 学習データの行ごとの重み。None なら全行同じ重み。

        Returns:
            学習済みのモデル（自分自身）。
        """

    @abstractmethod
    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        num_boost_round: int | None,
        weight: pd.Series | None,
    ) -> Self:
        """本番用に学習する。繰り返しの回数は CV で決めた値を使う。

        Args:
            x: 特徴量。
            y: 目的変数。
            num_boost_round: 繰り返しの回数。回数という考え方がないモデルでは None。
            weight: 行ごとの重み。None なら全行同じ重み。

        Returns:
            学習済みのモデル（自分自身）。
        """

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """予測する。

        Args:
            x: 特徴量。

        Returns:
            目的変数の予測。
        """

    @property
    @abstractmethod
    def best_iteration(self) -> int | None:
        """fit_with_validation で決まった繰り返しの回数。回数という考え方がないモデルでは None。"""

    @abstractmethod
    def importance(self) -> pd.Series | None:
        """特徴量の重要度を返す。

        Returns:
            特徴量ごとの重要度（合計 1）。大きい順に並ぶ。出せないモデルでは None。
        """
