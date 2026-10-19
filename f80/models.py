"""oof_bank.py add に渡す工場（引数なしで呼ぶと Model を返す）。名前は f80_ で始める。"""

from functools import partial

import numpy as np
import pandas as pd

from f80.te import BASE_UNITS, STATION_UNITS, ParamTE
from src.models.lgbm import LGBM
from src.models.params import LGBM_PARAMS

# 1. 駅・駅×建築年の単位を足す（寄せる強さは今の 3 のまま）
station = partial(ParamTE, units=STATION_UNITS, smoothing=3.0)
# 2. 総額の目的変数（単位は今のまま）
total = partial(ParamTE, units=BASE_UNITS, smoothing=3.0, per_area=False)
# 2'. 総額の目的変数で、寄せる強さ 1（PR #77 のコメントの kataokake さんの構成 #12 の総額側）
total_s1 = partial(ParamTE, units=BASE_UNITS, smoothing=1.0, per_area=False)
# 3. 寄せる強さを変える（単位は今のまま。1 が効いたら駅入りに差し替える）
s1 = partial(ParamTE, units=BASE_UNITS, smoothing=1.0)
s10 = partial(ParamTE, units=BASE_UNITS, smoothing=10.0)
s30 = partial(ParamTE, units=BASE_UNITS, smoothing=30.0)
station_s1 = partial(ParamTE, units=STATION_UNITS, smoothing=1.0)
station_s10 = partial(ParamTE, units=STATION_UNITS, smoothing=10.0)
station_s30 = partial(ParamTE, units=STATION_UNITS, smoothing=30.0)
# 今の x72_te と同じ設定（作り直しの確認用。普段は使わない）
same = partial(ParamTE, units=BASE_UNITS, smoothing=3.0)


# Issue #82：部分集合の LightGBM の min_data_in_leaf を 50・100 にした版（num_threads=4、heavy.py なしで add する）
def _built_over_30(x: pd.DataFrame) -> np.ndarray:
    """築 30 年を超える行を選ぶ（main.py と同じ）。"""
    return (x["築年数"] > 30).to_numpy()


def _area_30_or_less(x: pd.DataFrame) -> np.ndarray:
    """面積 30㎡ 以下の行を選ぶ（main.py と同じ）。"""
    return (x["面積_数値"] <= 30).to_numpy()


def _sub_params(min_data_in_leaf: int) -> dict:
    return {**LGBM_PARAMS, "min_data_in_leaf": min_data_in_leaf, "num_threads": 4}


over30_mdl50 = partial(LGBM, _sub_params(50), subset=_built_over_30)
over30_mdl100 = partial(LGBM, _sub_params(100), subset=_built_over_30)
area30_mdl50 = partial(LGBM, _sub_params(50), subset=_area_30_or_less)
area30_mdl100 = partial(LGBM, _sub_params(100), subset=_area_30_or_less)
