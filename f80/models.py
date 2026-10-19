"""oof_bank.py add に渡す工場（引数なしで呼ぶと Model を返す）。名前は f80_ で始める。"""

from functools import partial

from f80.te import BASE_UNITS, STATION_UNITS, ParamTE

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
