"""2 段目の学習行から極端な外れ値を除く案と、補正の平均を先の四半期数ごとに引く案。"""

import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import ensemble, validation
from src.config import TARGET
from src.models import params as P

sys.path.insert(0, "analysis")
from load import load

d, train, features = load()
meta = pickle.load(open("/Users/fujisetaisuke/condo-overnight/bank/_meta.pkl", "rb"))
trends = meta["trends"]
folds = validation.time_series_folds(train)
NAMES = [
    "lgbm",
    "lgbm_total",
    "catboost",
    "linear_residual",
    "lgbm_over30",
    "lgbm_area30",
    "x72_seiyaku",
    "x72_torihiki",
    "x72_not23",
    "x72_hedonic",
    "x72_te",
]
IN_BASE = [True] * 4 + [False] * 7
inputs, prices = [], []
for k, (_, va_i) in enumerate(folds):
    va = train.iloc[va_i]
    dk = d[d.fold == k]
    inputs.append(
        ensemble.make_inputs(
            [dk[f"pred_{n}"].to_numpy() for n in NAMES], IN_BASE, va, trends[k]
        )
    )
    prices.append(va[TARGET].to_numpy())


def fit(fi, fp, cut=None):
    x, base = ensemble._split(pd.concat(fi, ignore_index=True))
    y = np.log1p(np.concatenate(fp))
    if cut is not None:
        keep = np.abs(y - base) < cut
        x, base, y = x[keep], base[keep], y[keep]
    return lgb.train(
        P.STACK_PARAMS,
        lgb.Dataset(x, y, init_score=base, categorical_feature=ensemble.CATEGORIES),
        ensemble.STACK_ROUNDS,
    )


def blend(inp, st, per_ahead=False):
    x, base = ensemble._split(inp)
    c = st.predict(x)
    if per_ahead:
        c = (
            c
            - pd.Series(c)
            .groupby(x["先の四半期数"].to_numpy())
            .transform("mean")
            .to_numpy()
        )
    else:
        c = c - c.mean()
    return np.expm1(base + c)


def cv(cut=None, per_ahead=False):
    out = []
    for k in range(3):
        st = fit(
            [inputs[j] for j in range(3) if j != k],
            [prices[j] for j in range(3) if j != k],
            cut,
        )
        out.append(validation.rmsle(prices[k], blend(inputs[k], st, per_ahead)))
    return out


base = None
for label, kw in [
    ("基準", {}),
    ("2 段目の学習行から |log誤差|>1 を除く", dict(cut=1.0)),
    ("同 >0.7", dict(cut=0.7)),
    ("同 >1.5", dict(cut=1.5)),
    ("補正の平均を先の四半期数ごとに引く", dict(per_ahead=True)),
]:
    s = cv(**kw)
    if base is None:
        base = np.array(s)
    dd = np.array(s) - base
    print(
        f"{label:36s} {' '.join(f'{x:.5f}' for x in s)} CV {np.mean(s):.5f} 差 {np.mean(s) - base.mean():+.5f} 3fold良い={bool((dd < 0).all())}",
        flush=True,
    )
