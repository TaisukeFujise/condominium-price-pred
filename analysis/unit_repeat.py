"""同じ住戸（疑似建物＋面積＋間取り）の過去の取引（fold の学習データだけ）が、OOF の残差を説明するか。"""

import pickle
import sys

import numpy as np
import pandas as pd

from src import validation
from src.features.building import BUILDING_KEYS

sys.path.insert(0, "analysis")
from load import load

pd.set_option("display.width", 200)
d, train, features = load()
meta = pickle.load(open("/Users/fujisetaisuke/condo-overnight/bank/_meta.pkl", "rb"))
trends = meta["trends"]
folds = validation.time_series_folds(train)
d["先"] = d["経過四半期"] - d["fold"].map(
    {k: t.last_quarter for k, t in enumerate(trends)}
)
d["res"] = d.err - d.groupby(["fold", "先"]).err.transform("mean")


def key(df, cols):
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + df[c].astype(str)
    return k.to_numpy()


UNITS = {
    "建物": BUILDING_KEYS,
    "建物+面積": [*BUILDING_KEYS, "面積_数値"],
    "建物+面積+間取り": [*BUILDING_KEYS, "面積_数値", "間取り"],
}
out = []
for k, (tr_i, va_i) in enumerate(folds):
    tr = train.iloc[tr_i]
    va = train.iloc[va_i]
    dk = d[d.fold == k].copy()
    t = trends[k]
    adj_tr = np.log(tr["取引価格（総額）"] / tr["面積_数値"]).to_numpy() - t.log_index(
        tr
    )
    adj_pred = np.log(dk.pred_stack / dk["面積_数値"]).to_numpy() - t.log_index(va)
    for name, cols in UNITS.items():
        g = pd.DataFrame(
            {"key": key(tr, cols), "y": adj_tr, "q": tr["経過四半期"].to_numpy()}
        ).groupby("key")
        stats = g["y"].agg(["mean", "count"])
        last_q = g["q"].max()
        kv = pd.Series(key(va, cols))
        dk[f"{name}_mean"] = kv.map(stats["mean"]).to_numpy()
        dk[f"{name}_n"] = kv.map(stats["count"]).fillna(0).to_numpy()
        dk[f"{name}_diff"] = dk[f"{name}_mean"] - adj_pred
        dk[f"{name}_gap"] = t.last_quarter - kv.map(last_q).to_numpy()
    out.append(dk)
d2 = pd.concat(out)
for name in UNITS:
    m = d2[f"{name}_n"] > 0
    print(
        f"=== {name}: 学習データに同じ単位がある検証行 {m.mean():.1%}（{m.sum()} 行）"
    )
    sub = d2[m]
    print(
        f"  残差と diff（単位の平均 − 予測）の相関 {np.corrcoef(sub.res, sub[f'{name}_diff'])[0, 1]:.3f}"
    )
    # 件数帯ごと
    for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 10**6)]:
        s = sub[(sub[f"{name}_n"] >= lo) & (sub[f"{name}_n"] <= hi)]
        if len(s) == 0:
            continue
        print(
            f"  件数 {lo}-{hi}: {len(s)} 行, 相関 {np.corrcoef(s.res, s[f'{name}_diff'])[0, 1]:.3f}, 残差 rmse {np.sqrt((s.res**2).mean()):.4f}"
        )
    # 他 fold で係数を決めて、残差をどれだけ減らせるか（全行ベースの RMSE）
    for k in range(3):
        o = d2[(d2.fold != k) & m]
        v = d2[(d2.fold == k)]
        x = o[f"{name}_diff"].to_numpy()
        beta = float((x * o.res).sum() / (x * x).sum())
        pred = np.where(v[f"{name}_n"] > 0, beta * v[f"{name}_diff"].fillna(0), 0.0)
        e0 = np.sqrt((v.res**2).mean())
        e1 = np.sqrt(((v.res - pred) ** 2).mean())
        print(
            f"  fold {k}: beta {beta:.3f}, 全体 rmse {e0:.5f} → {e1:.5f} (差 {e1 - e0:+.5f})"
        )
d2.to_pickle("analysis/unit_repeat.pkl")
