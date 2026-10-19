"""2 段目まわりの候補を OOF 上で試算する（再学習なし）。"""

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
seed1 = pickle.load(
    open("/Users/fujisetaisuke/condo-overnight/bank/x72_seed1.pkl", "rb")
)


def fit_stacker_seeds(fold_inputs, fold_prices, seeds):
    x, base = ensemble._split(pd.concat(fold_inputs, ignore_index=True))
    log_true = np.log1p(np.concatenate(fold_prices))
    boosters = []
    for s in seeds:
        ds = lgb.Dataset(
            x,
            log_true,
            init_score=base,
            categorical_feature=[c for c in ensemble.CATEGORIES if c in x.columns],
        )
        boosters.append(
            lgb.train({**P.STACK_PARAMS, "seed": s}, ds, ensemble.STACK_ROUNDS)
        )
    return boosters


def blend_seeds(inputs, boosters):
    x, base = ensemble._split(inputs)
    corr = np.mean([b.predict(x) for b in boosters], axis=0)
    return np.expm1(base + corr - corr.mean())


def cv(preds_fn=None, drop=(), seeds=(42,), label=""):
    inputs, prices = [], []
    for k, (_, va_i) in enumerate(folds):
        va = train.iloc[va_i]
        dk = d[d.fold == k]
        preds = [dk[f"pred_{n}"].to_numpy() for n in NAMES]
        if preds_fn:
            preds = preds_fn(preds, k)
        inp = ensemble.make_inputs(preds, IN_BASE, va, trends[k]).drop(
            columns=list(drop)
        )
        inputs.append(inp)
        prices.append(va[TARGET].to_numpy())
    out = []
    for k, (_, va_i) in enumerate(folds):
        va = train.iloc[va_i]
        bs = fit_stacker_seeds(
            [inputs[j] for j in range(3) if j != k],
            [prices[j] for j in range(3) if j != k],
            seeds,
        )
        out.append(validation.rmsle(va[TARGET], blend_seeds(inputs[k], bs)))
    return out


def avg_lgbm(preds, k):
    preds = list(preds)
    preds[0] = np.sqrt(preds[0] * np.asarray(seed1["va"][k]))  # log の平均 = 幾何平均
    return preds


base = None
for label, kw in [
    ("基準（#77 の構成）", {}),
    ("lgbm を seed 42・1 の 2 本平均に", dict(preds_fn=avg_lgbm)),
    ("2 段目を 5 seed で平均", dict(seeds=(42, 1, 2, 3, 4))),
    ("2 段目から 先の四半期数 を外す", dict(drop=("先の四半期数",))),
    (
        "2 段目から 先の四半期数 を外し 5 seed 平均",
        dict(drop=("先の四半期数",), seeds=(42, 1, 2, 3, 4)),
    ),
    ("2 段目から 価格帯 を外す", dict(drop=("価格帯",))),
    ("2 段目から 価格情報区分 を外す", dict(drop=("価格情報区分",))),
]:
    s = cv(**kw)
    if base is None:
        base = np.array(s)
    dd = np.array(s) - base
    print(
        f"{label:40s} {' '.join(f'{x:.5f}' for x in s)} CV {np.mean(s):.5f} 差 {np.mean(s) - base.mean():+.5f} 3fold良い={bool((dd < 0).all())}",
        flush=True,
    )
