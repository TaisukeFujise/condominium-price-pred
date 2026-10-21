"""疑似建物・地区ごとの過去の価格の平均（target encoding）を、学習のたびに作って足す LightGBM。

同じ建物・同じ地区の過去の取引の価格は、その物件の価格の強い手がかりになる。ただし価格を使うので、
src/features/ で全データから作ると検証データの答えが混ざる（AGENTS.md）。このモデルは、渡された学習データ
（CV では fold の学習データ）だけで平均を作るので、fold ごとに作る決まりを自然に守れる。
学習データの行自身の平均は、学習データを 5 つに分けた残りの行から作る。自分の価格が自分の平均に入ると、
木がその列を信じすぎるため。
平均の土台には入れず、src/ensemble.py の 2 段目の入力にだけ使う（#72。土台に入れるより CV が良かった）。
"""

from collections.abc import Callable
from typing import Self

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import SEED
from src.features.building import BUILDING_KEYS
from src.models.lgbm import LGBM
from src.models.params import LGBM_PARAMS

# 平均を作る単位。地区×建築年は、建物の列の一部が欠けていても同じ建物をまとめられるように足す
UNITS = {
    "建物": BUILDING_KEYS,
    "地区": ["市区町村名", "地区名"],
    "地区×建築年": ["市区町村名", "地区名", "建築年_数値"],
}
# 駅と駅×建築年も足した単位（#80）。駅の単位だけに置き換えると効かず、上の単位と別のモデルとして
# 並べると 2 段目が良くなった。地区とは違う切り口の平均を持つモデルが 1 本増える効果と考えている
STATION_UNITS = {
    **UNITS,
    "駅": ["最寄駅：名称"],
    "駅×建築年": ["最寄駅：名称", "建築年_数値"],
}
# 件数の少ない単位の平均を、1 つ上の親の階層の平均に寄せる強さ（何件分の親の平均を混ぜるか）
SMOOTHING = 3.0
_N_SPLITS = 5
# 親から子の順に平均を作るための階層定義。市区町村は親の事前分布としてだけ使い、列としては足さない
_CITY_UNIT: dict[str, list[str]] = {"市区町村": ["市区町村名"]}
_HIERARCHY_ORDER: tuple[str, ...] = (
    "市区町村",
    "地区",
    "地区×建築年",
    "建物",
    "駅",
    "駅×建築年",
)
_PARENT_UNIT: dict[str, str] = {
    "地区": "市区町村",
    "地区×建築年": "地区",
    "建物": "地区×建築年",
    "駅": "市区町村",
    "駅×建築年": "駅",
}


class TargetEncodedLGBM(LGBM):
    """target encoding の列を足して学習する LightGBM。1 つのインスタンスは 1 回だけ学習する。"""

    stack_only = True

    def __init__(
        self,
        params: dict = LGBM_PARAMS,
        per_area: bool = True,
        subset: Callable[[pd.DataFrame], np.ndarray] | None = None,
        units: dict[str, list[str]] = UNITS,
    ) -> None:
        """学習前のモデルを作る。

        Args:
            params: LightGBM のパラメータ。省略すると params.py の LGBM_PARAMS。
            per_area: 目的変数を㎡単価にするか（False なら総額）。
            subset: 学習に使う行を絞る条件（base.py の Model.subset）。None なら全行。
            units: 平均を作る単位（単位の名前 → 単位を決める列）。省略すると UNITS。
        """
        super().__init__(params, per_area, subset)
        self.units = units
        self._tables: dict[str, pd.DataFrame] = {}
        self._prior = 0.0

    def fit_with_validation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        weight: pd.Series | None,
    ) -> Self:
        """CV 用に学習する。検証データの平均は、学習データだけから作る。

        Args:
            x_train: 学習データの特徴量。
            y_train: 学習データの目的変数。
            x_valid: 検証データの特徴量。
            y_valid: 検証データの目的変数。
            weight: 学習データの行ごとの重み。None なら全行同じ重み。

        Returns:
            学習済みのモデル（自分自身）。
        """
        self._tables, self._prior = _fit_tables(x_train, y_train, self.units)
        return super().fit_with_validation(
            _add_out_of_fold(x_train, y_train, self.units),
            y_train,
            _add(x_valid, self._tables, self._prior, self.units),
            y_valid,
            weight,
        )

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        weight: pd.Series | None,
        cv_models: list[Self],
    ) -> Self:
        """本番用に学習する。木の本数は LGBM.fit と同じく CV の平均を使う。

        Args:
            x: 特徴量。
            y: 目的変数。
            weight: 行ごとの重み。None なら全行同じ重み。
            cv_models: CV で同じモデルを fold ごとに学習したもの。

        Returns:
            学習済みのモデル（自分自身）。
        """
        self._tables, self._prior = _fit_tables(x, y, self.units)
        return super().fit(_add_out_of_fold(x, y, self.units), y, weight, cv_models)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """学習データから作った平均の列を足して予測する。

        Args:
            x: 特徴量。

        Returns:
            目的変数の予測。
        """
        return super().predict(_add(x, self._tables, self._prior, self.units))


def _keys(x: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """行ごとに、単位を表す文字列（列の値を | でつないだもの）を作る。

    Args:
        x: 特徴量。
        columns: 単位を決める列。

    Returns:
        行ごとの単位の文字列。
    """
    key = x[columns[0]].astype(str)
    for column in columns[1:]:
        key = key + "|" + x[column].astype(str)
    return key.to_numpy()


def _fit_tables(
    x: pd.DataFrame, y: pd.Series | np.ndarray, units: dict[str, list[str]]
) -> tuple[dict[str, pd.DataFrame], float]:
    """単位ごとに、目的変数の合計と件数を数える。

    Args:
        x: 学習データの特徴量。
        y: 学習データの目的変数。
        units: 平均を作る単位。

    Returns:
        単位の名前 → 単位の文字列ごとの合計（sum）と件数（count）の表、と全体の平均。
    """
    y_array = np.asarray(y)
    all_units = {**_CITY_UNIT, **units}
    tables = {
        name: pd.DataFrame({"key": _keys(x, columns), "y": y_array})
        .groupby("key")["y"]
        .agg(["sum", "count"])
        for name, columns in all_units.items()
    }
    return tables, float(y_array.mean())


def _add(
    x: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    prior: float,
    units: dict[str, list[str]],
) -> pd.DataFrame:
    """単位ごとの平均（1 つ上の親の階層の平均に寄せたもの）と件数の列を足す。

    件数の少ない建物や地区を東京都全体の平均へ寄せると、都心の希少な物件が安く見え、
    郊外の物件が高く見えるため、「全体 → 市区町村 → 地区/駅 → 地区×建築年/駅×建築年 → 建物」の順に
    親の階層の平均へスムージングする。

    Args:
        x: 特徴量。
        tables: _fit_tables で数えた表。
        prior: 全体の平均。
        units: 平均を作る単位。_fit_tables に渡したものと同じ。

    Returns:
        列を足した特徴量。
    """
    out = x.copy()
    all_units = {**_CITY_UNIT, **units}
    computed: dict[str, np.ndarray] = {}
    for name in _HIERARCHY_ORDER:
        if name not in all_units:
            continue
        keys = pd.Series(_keys(x, all_units[name]))
        total = keys.map(tables[name]["sum"]).fillna(0).to_numpy()
        count = keys.map(tables[name]["count"]).fillna(0).to_numpy()
        parent_name = _PARENT_UNIT.get(name)
        parent_prior = computed[parent_name] if parent_name in computed else prior
        computed[name] = (total + parent_prior * SMOOTHING) / (count + SMOOTHING)
    for name, columns in units.items():
        keys = pd.Series(_keys(x, columns))
        count = keys.map(tables[name]["count"]).fillna(0).to_numpy()
        out[f"TE_{name}"] = computed[name]
        out[f"TE_{name}_件数"] = count
    return out


def _add_out_of_fold(
    x: pd.DataFrame, y: pd.Series, units: dict[str, list[str]]
) -> pd.DataFrame:
    """学習データの行に、その行を含まない残りの行から作った平均の列を足す。

    Args:
        x: 学習データの特徴量。
        y: 学習データの目的変数。
        units: 平均を作る単位。

    Returns:
        列を足した特徴量。
    """
    y_array = np.asarray(y)
    parts = []
    for rest, part in KFold(_N_SPLITS, shuffle=True, random_state=SEED).split(x):
        tables, prior = _fit_tables(x.iloc[rest], y_array[rest], units)
        parts.append(_add(x.iloc[part], tables, prior, units))
    # KFold は行を並べ替えるので、元の順に戻す
    return pd.concat(parts).loc[x.index]
