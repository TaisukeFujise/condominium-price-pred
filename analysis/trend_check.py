"""検証期間の本当の指数（検証データも入れて推定）と、直線で延ばした指数を比べる。"""

import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from load import load

from src import target, validation

pd.set_option("display.width", 200)
d, train, features = load()
meta = pickle.load(open("/Users/fujisetaisuke/condo-overnight/bank/_meta.pkl", "rb"))
folds = validation.time_series_folds(train)
rows = []
for k, (tr_i, va_i) in enumerate(folds):
    tr, va = train.iloc[tr_i], train.iloc[va_i]
    t = meta["trends"][k]
    both = pd.concat([tr, va])
    group = both["市区町村名"].astype(object).map(t.city_group).fillna(target.OVERALL)
    for g in sorted(t.index_by_group):
        sub = both if g == target.OVERALL else both[group == g]
        true_idx = target._market_index(sub)  # 最後(検証の最後)を 0 にした指数
        true_idx = true_idx - true_idx.loc[t.last_quarter]  # 学習の最後を 0 に
        fitted = t.index_by_group[g]
        for q in range(t.last_quarter - 7, t.last_quarter + 4):
            rows.append(
                dict(
                    fold=k,
                    tier=g,
                    q=q,
                    ahead=q - t.last_quarter,
                    true=true_idx.get(q, np.nan),
                    fit=fitted.get(q, np.nan),
                    extrap=t.slope[g] * max(q - t.last_quarter, 0)
                    if q > t.last_quarter
                    else np.nan,
                    n=int((sub["経過四半期"] == q).sum()),
                )
            )
r = pd.DataFrame(rows)
r["ずれ(延ばし−本当)"] = r.extrap - r.true
print(r.round(4).to_string())
print(
    "\n=== 傾き（1 四半期あたり、log）: 学習の直近 4 四半期の直線 vs 本当の検証期間の平均の傾き"
)
for k in range(3):
    t = meta["trends"][k]
    for g in sorted(t.index_by_group):
        rr = r[(r.fold == k) & (r.tier == g) & (r.ahead > 0)]
        print(
            f"fold {k} tier {g}: slope {t.slope[g]:+.4f}  本当(3期平均) {rr.true.iloc[-1] / 3:+.4f}  ずれ(先1,2,3) {' '.join(f'{x:+.4f}' for x in rr['ずれ(延ばし−本当)'])}"
        )
r.to_pickle("analysis/trend_check.pkl")
