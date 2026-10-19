"""指数の延ばし方を変えたときの CV を、再学習なしで OOF 上で試算する。
延ばした分は検証行の予測にだけ足されるので、予測に exp(新 − 旧) を掛け、2 段目だけ学習し直す。"""

import pickle
import sys

import numpy as np
import pandas as pd

from src import ensemble, pipeline, target, validation
from src.config import TARGET

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
IN_BASE = [True, True, True, True] + [False] * 7


def slope_of(qi: pd.Series, window: int, method: str = "ols") -> float:
    recent = qi[qi.index > qi.index.max() - window]
    if method == "ols":
        return float(np.polyfit(recent.index, recent.to_numpy(), 1)[0])
    if method == "endpoints":  # 両端の差を期間で割る
        return float(
            (recent.iloc[-1] - recent.iloc[0]) / (recent.index[-1] - recent.index[0])
        )


def new_log_index(t: target.Trend, rows: pd.DataFrame, cfg: dict) -> np.ndarray:
    """cfg に従って延ばした指数（検証行は全部 last_quarter より先）。"""
    group = rows["市区町村名"].astype(object).map(t.city_group).fillna(target.OVERALL)
    ahead = (rows[target.QUARTER] - t.last_quarter).to_numpy().astype(float)
    out = np.zeros(len(rows))
    overall_slope = slope_of(
        t.index_by_group[target.OVERALL], cfg["window"], cfg.get("method", "ols")
    )
    for g in group.unique():
        m = (group == g).to_numpy()
        qi = t.index_by_group[g]
        s = slope_of(qi, cfg["window"], cfg.get("method", "ols"))
        s = cfg.get("tier_w", 1.0) * s + (1 - cfg.get("tier_w", 1.0)) * overall_slope
        s *= cfg.get("shrink", 1.0)
        level = 0.0
        if (
            cfg.get("anchor") == "line"
        ):  # 直線の最後の点を起点にする（最後の四半期のぶれを消す）
            recent = qi[qi.index > qi.index.max() - cfg["window"]]
            a, b = np.polyfit(recent.index, recent.to_numpy(), 1)
            level = a * t.last_quarter + b
        elif cfg.get("anchor") == "mean2":  # 最後の 2 四半期の平均 + 半期分の傾き
            level = qi.iloc[-2:].mean() + 0.5 * s
        out[m] = level + s * ahead[m]
    return out


def cv(cfg: dict) -> list[float]:
    inputs, prices = [], []
    for k, (_, va_i) in enumerate(folds):
        va = train.iloc[va_i]
        t = trends[k]
        dk = d[d.fold == k]
        assert (dk["row"].to_numpy() == va_i).all()
        delta = new_log_index(t, va, cfg) - t.log_index(va)
        preds = [dk[f"pred_{n}"].to_numpy() * np.exp(delta) for n in NAMES]
        inputs.append(ensemble.make_inputs(preds, IN_BASE, va, t))
        prices.append(va[TARGET].to_numpy())
    out = []
    for k, (tr_i, va_i) in enumerate(folds):
        tr, va = train.iloc[tr_i], train.iloc[va_i]
        st = ensemble.fit_stacker(
            [inputs[j] for j in range(3) if j != k],
            [prices[j] for j in range(3) if j != k],
        )
        out.append(
            validation.rmsle(
                va[TARGET], pipeline.blend_and_adjust(inputs[k], st, va, tr, features)
            )
        )
    return out


if __name__ == "__main__":
    cfgs = [dict(window=4)]  # 今の設定（再現の確認）
    cfgs += [dict(window=w) for w in (2, 3, 6, 8, 12, 20)]
    cfgs += [
        dict(window=4, tier_w=0.0),
        dict(window=4, tier_w=0.5),
        dict(window=8, tier_w=0.0),
    ]
    cfgs += [dict(window=4, shrink=s) for s in (0.5, 0.75, 1.25, 1.5)]
    cfgs += [
        dict(window=4, anchor="line"),
        dict(window=8, anchor="line"),
        dict(window=4, anchor="mean2"),
    ]
    cfgs += [dict(window=4, method="endpoints"), dict(window=8, method="endpoints")]
    base = None
    for cfg in cfgs:
        s = cv(cfg)
        if base is None:
            base = np.array(s)
        dd = np.array(s) - base
        print(
            f"{cfg!s:55s} {' '.join(f'{x:.5f}' for x in s)} CV {np.mean(s):.5f} 差 {np.mean(s) - base.mean():+.5f} 3fold良い={bool((dd < 0).all())}",
            flush=True,
        )
