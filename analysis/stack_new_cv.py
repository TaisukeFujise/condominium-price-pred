"""保管庫の予測で 2 段目だけを学習し直し、新しい測り方（2 段目は前の期間の fold だけで学習。fold 2 は 2 段目なし）で
採点する。Issue #81（2 段目のパラメータ）・#83（2 段目に渡す列）・#82（部分集合モデルの置き換え）用。
feat-overnight の直下で: PYTHONPATH=.:<このブランチ> uv run python <このファイル> <81|83|82>
土台は #77 の 11 モデル ＋ 候補 B の 2 つ（f80_te_station・f80_te_total）の 13 モデル。
"""

import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "condo-overnight"))
import oof_bank as B

from src import ensemble, validation
from src.config import TARGET
from src.models.params import STACK_PARAMS, STACK_ROUNDS

BASE13 = [
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
    "f80_te_station",
    "f80_te_total",
]
EXTRA_NUMERIC = {"駅距離_分": "駅距離_分"}
EXTRA_CATEGORICAL = {
    "取引の事情等": "取引の事情等",
    "間取り": "間取り",
    "建物の構造": "建物の構造",
}

train, test, features = B._data()
folds = validation.time_series_folds(train)
meta = B._meta(train, test, features, folds)
_rec_cache: dict[str, dict] = {}


def rec(name: str) -> dict:
    if name not in _rec_cache:
        _rec_cache[name] = pickle.load(open(B.BANK / f"{name}.pkl", "rb"))
    return _rec_cache[name]


def make_inputs(
    names: list[str], extra: list[str] = ()
) -> tuple[list[pd.DataFrame], list[np.ndarray], list[str]]:
    recs = [rec(n) for n in names]
    in_base = [r["in_base"] for r in recs]
    inputs, prices = [], []
    cats = list(ensemble.CATEGORIES)
    for k, (_, va_i) in enumerate(folds):
        va = train.iloc[va_i]
        inp = ensemble.make_inputs(
            [r["va"][k] for r in recs], in_base, va, meta["trends"][k]
        )
        for col in extra:
            if col in EXTRA_NUMERIC:
                inp[col] = va[EXTRA_NUMERIC[col]].to_numpy()
            else:
                inp[col] = va[EXTRA_CATEGORICAL[col]].cat.codes.to_numpy()
                if col not in cats:
                    cats.append(col)
        inputs.append(inp)
        prices.append(va[TARGET].to_numpy())
    return inputs, prices, cats


def fit(
    fold_inputs, fold_prices, params: dict, rounds: int, cats: list[str]
) -> lgb.Booster:
    x, base = ensemble._split(pd.concat(fold_inputs, ignore_index=True))
    y = np.log1p(np.concatenate(fold_prices))
    ds = lgb.Dataset(
        x, y, init_score=base, categorical_feature=[c for c in cats if c in x.columns]
    )
    return lgb.train(params, ds, rounds)


def score(
    names=BASE13, extra=(), drop=(), params_over=None, rounds=STACK_ROUNDS
) -> dict:
    inputs, prices, cats = make_inputs(names, extra)
    inputs = [inp.drop(columns=list(drop)) for inp in inputs]
    params = {**STACK_PARAMS, "num_threads": 4, **(params_over or {})}
    out = {}
    for k in range(3):
        _, base = ensemble._split(inputs[k])
        base_price = np.expm1(base)
        earlier = [j for j in range(3) if j > k]
        if earlier:
            st = fit(
                [inputs[j] for j in earlier],
                [prices[j] for j in earlier],
                params,
                rounds,
                cats,
            )
            x, _ = ensemble._split(inputs[k])
            corr = st.predict(x)
            price = np.expm1(base + corr - corr.mean())
        else:
            price = base_price
        out[f"stack{k}"] = validation.rmsle(prices[k], price)
        out[f"base{k}"] = validation.rmsle(prices[k], base_price)
    out["effect"] = np.mean([out[f"stack{k}"] - out[f"base{k}"] for k in (0, 1)])
    out["cv"] = np.mean([out[f"stack{k}"] for k in range(3)])
    return out


def show(label: str, s: dict, ref: dict | None) -> None:
    d0 = d1 = de = ""
    if ref is not None:
        d0, d1, de = (
            f"{s['stack0'] - ref['stack0']:+.5f}",
            f"{s['stack1'] - ref['stack1']:+.5f}",
            f"{s['effect'] - ref['effect']:+.5f}",
        )
    print(
        f"{label:52s} fold0 {s['stack0']:.5f} ({d0}) fold1 {s['stack1']:.5f} ({d1}) fold2(2段目なし) {s['stack2']:.5f} CV {s['cv']:.5f} 効果 {s['effect']:+.5f} ({de})",
        flush=True,
    )


if __name__ == "__main__":
    which = sys.argv[1]
    ref = score()
    show("基準（13 モデル、今の 2 段目）", ref, None)
    if which == "81":
        trials = [
            ("STACK_ROUNDS 200", dict(rounds=200)),
            ("STACK_ROUNDS 800", dict(rounds=800)),
            ("num_leaves 15", dict(params_over={"num_leaves": 15})),
            ("num_leaves 7, max_depth -1", dict(params_over={"max_depth": -1})),
            ("min_data_in_leaf 300", dict(params_over={"min_data_in_leaf": 300})),
            ("min_data_in_leaf 3000", dict(params_over={"min_data_in_leaf": 3000})),
            ("lambda_l2 1", dict(params_over={"lambda_l2": 1.0})),
            ("lambda_l2 100", dict(params_over={"lambda_l2": 100.0})),
            (
                "learning_rate 0.015, ROUNDS 800",
                dict(params_over={"learning_rate": 0.015}, rounds=800),
            ),
            (
                "learning_rate 0.06, ROUNDS 200",
                dict(params_over={"learning_rate": 0.06}, rounds=200),
            ),
        ]
    elif which == "83":
        trials = [
            (f"＋ {c}", dict(extra=[c]))
            for c in ["駅距離_分", "取引の事情等", "間取り", "建物の構造"]
        ]
        trials += [
            (f"− {c}", dict(drop=[c]))
            for c in ["面積", "築年数", "先の四半期数", "価格帯", "価格情報区分"]
        ]
    elif which == "82":
        trials = []
        for sub, var in [
            ("lgbm_over30", "f82_over30_mdl50"),
            ("lgbm_over30", "f82_over30_mdl100"),
            ("lgbm_area30", "f82_area30_mdl50"),
            ("lgbm_area30", "f82_area30_mdl100"),
        ]:
            if (B.BANK / f"{var}.pkl").exists():
                trials.append(
                    (
                        f"{sub} → {var}",
                        dict(names=[var if n == sub else n for n in BASE13]),
                    )
                )
        if (B.BANK / "f82_over30_mdl50.pkl").exists() and (
            B.BANK / "f82_area30_mdl50.pkl"
        ).exists():
            trials.append(
                (
                    "両方 → mdl50",
                    dict(
                        names=[
                            {
                                "lgbm_over30": "f82_over30_mdl50",
                                "lgbm_area30": "f82_area30_mdl50",
                            }.get(n, n)
                            for n in BASE13
                        ]
                    ),
                )
            )
        if (B.BANK / "f82_over30_mdl100.pkl").exists() and (
            B.BANK / "f82_area30_mdl100.pkl"
        ).exists():
            trials.append(
                (
                    "両方 → mdl100",
                    dict(
                        names=[
                            {
                                "lgbm_over30": "f82_over30_mdl100",
                                "lgbm_area30": "f82_area30_mdl100",
                            }.get(n, n)
                            for n in BASE13
                        ]
                    ),
                )
            )
    else:
        trials = [(a, dict()) for a in sys.argv[2:]]
    for label, kw in trials:
        show(label, score(**kw), ref)
    print(f"試した回数: {len(trials)}（基準 1 回を除く）")
