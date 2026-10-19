"""src/models/target_encoding.py の TargetEncodedLGBM を、単位（units）と寄せる強さ（smoothing）を
引数で変えられるようにしたもの（Issue #80）。平均の作り方は元と同じ。"""

from collections.abc import Callable
from typing import Self

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import SEED
from src.features.building import BUILDING_KEYS
from src.models.lgbm import LGBM
from src.models.params import LGBM_PARAMS

BASE_UNITS = {
    "建物": BUILDING_KEYS,
    "地区": ["市区町村名", "地区名"],
    "地区×建築年": ["市区町村名", "地区名", "建築年_数値"],
}
STATION_UNITS = {
    **BASE_UNITS,
    "駅": ["最寄駅：名称"],
    "駅×建築年": ["最寄駅：名称", "建築年_数値"],
}
_N_SPLITS = 5


class ParamTE(LGBM):
    """単位と寄せる強さを指定できる target encoding 付き LightGBM。2 段目の入力にだけ使う。"""

    stack_only = True

    def __init__(
        self,
        units: dict[str, list[str]] = BASE_UNITS,
        smoothing: float = 3.0,
        params: dict = LGBM_PARAMS,
        per_area: bool = True,
        subset: Callable[[pd.DataFrame], np.ndarray] | None = None,
    ) -> None:
        super().__init__(params, per_area, subset)
        self.units = units
        self.smoothing = smoothing
        self._tables: dict[str, pd.DataFrame] = {}
        self._prior = 0.0

    def fit_with_validation(self, x_train, y_train, x_valid, y_valid, weight) -> Self:
        self._tables, self._prior = self._fit_tables(x_train, y_train)
        return super().fit_with_validation(
            self._add_out_of_fold(x_train, y_train),
            y_train,
            self._add(x_valid, self._tables, self._prior),
            y_valid,
            weight,
        )

    def fit(self, x, y, weight, cv_models) -> Self:
        self._tables, self._prior = self._fit_tables(x, y)
        return super().fit(self._add_out_of_fold(x, y), y, weight, cv_models)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return super().predict(self._add(x, self._tables, self._prior))

    @staticmethod
    def _keys(x: pd.DataFrame, columns: list[str]) -> np.ndarray:
        key = x[columns[0]].astype(str)
        for column in columns[1:]:
            key = key + "|" + x[column].astype(str)
        return key.to_numpy()

    def _fit_tables(self, x, y):
        y = np.asarray(y)
        tables = {
            name: pd.DataFrame({"key": self._keys(x, columns), "y": y})
            .groupby("key")["y"]
            .agg(["sum", "count"])
            for name, columns in self.units.items()
        }
        return tables, float(y.mean())

    def _add(self, x, tables, prior):
        out = x.copy()
        for name, columns in self.units.items():
            keys = pd.Series(self._keys(x, columns))
            total = keys.map(tables[name]["sum"]).fillna(0).to_numpy()
            count = keys.map(tables[name]["count"]).fillna(0).to_numpy()
            out[f"TE_{name}"] = (total + prior * self.smoothing) / (
                count + self.smoothing
            )
            out[f"TE_{name}_件数"] = count
        return out

    def _add_out_of_fold(self, x, y):
        y = np.asarray(y)
        parts = []
        for rest, part in KFold(_N_SPLITS, shuffle=True, random_state=SEED).split(x):
            tables, prior = self._fit_tables(x.iloc[rest], y[rest])
            parts.append(self._add(x.iloc[part], tables, prior))
        return pd.concat(parts).loc[x.index]
