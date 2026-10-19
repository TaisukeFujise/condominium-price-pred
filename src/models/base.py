"""モデルが揃えるメソッドの決まり。src/models/ のモデルは、すべて Model を継承する。

pipeline.py と main.py はこのメソッドだけを使うので、Model を継承したクラスを書けば
main.py の MODELS に足すだけでモデルを足したり付け替えたりできる（例は lgbm.py）。
メソッドが 1 つでも足りないと、インスタンスを作った時点でエラーになる。
複数のモデルで同じ前処理（fill_missing_category）もここに置く。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self

import numpy as np
import pandas as pd


class Model(ABC):
    """モデルが揃えるメソッド。1 つのインスタンスは 1 回だけ学習する。

    Attributes:
        per_area: 目的変数を㎡単価にするか（False なら総額）。src/target.py の to_target に渡す。
        subset: 学習に使う行を絞る条件（特徴量の表 → 行ごとの True / False）。None なら全行で学習する。
            行を絞るのは src/pipeline.py なので、モデルの側では何もしなくてよい。
            絞ったモデルも予測は全行に出すが、平均の土台には入れず、src/ensemble.py の 2 段目の入力にだけ使う。
        stack_only: True なら、全行で学習しても平均の土台には入れず、2 段目の入力にだけ使う。
            単独では弱いが、決定木と違う間違え方をするモデル（hedonic.py など）に使う。
    """

    per_area: bool = True
    subset: Callable[[pd.DataFrame], np.ndarray] | None = None
    stack_only: bool = False

    @abstractmethod
    def fit_with_validation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        weight: pd.Series | None,
    ) -> Self:
        """CV 用に学習する。検証データを見て、繰り返しの回数（木の本数など）を決めてよい。

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
        weight: pd.Series | None,
        cv_models: list[Self],
    ) -> Self:
        """本番用に学習する。検証データがないので、繰り返しの回数などは CV の結果から決める。

        決め方はモデルによって違うので、各モデルの fit に書く（LightGBM は fold の平均）。

        Args:
            x: 特徴量。
            y: 目的変数。
            weight: 行ごとの重み。None なら全行同じ重み。
            cv_models: CV で同じモデルを fold ごとに学習したもの。

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


def fill_missing_category(x: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ列の欠損を「欠損」という 1 つの値にして、文字列の列にする。

    Args:
        x: 特徴量。

    Returns:
        カテゴリ列を文字列にした特徴量。ほかの列はそのまま。
    """
    # CatBoost や one-hot はカテゴリ列の欠損を受け取れない。LightGBM も欠損を独立した値として
    # 分岐するので、扱いがそろう。astype(str) は pandas の版によって欠損を NaN のまま残すので、先に埋める
    cat_columns = x.select_dtypes("category").columns
    return x.assign(
        **{c: x[c].astype(object).fillna("欠損").astype(str) for c in cat_columns}
    )
