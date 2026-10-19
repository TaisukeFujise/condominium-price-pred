"""LightGBM 1 つだけの CV で、学習データの重み付け（工程 3）の候補をふるい分ける。
使い方: PYTHONPATH=. uv run python analysis/lgbm_only.py <設定名> [seed]
設定名: base / recent8 / decay12 / decay24 / seiyaku15 / no_time（経過四半期・取引四半期を外す）/ no_season（取引四半期を外す）
"""

import sys
import time

import numpy as np
import pandas as pd

from src import pipeline, preprocess, selection, validation
from src.config import TARGET
from src.features.build import build_features
from src.models.lgbm import LGBM
from src.models.params import LGBM_PARAMS

name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42


def weight_fn(tr: pd.DataFrame) -> pd.Series | None:
    last = tr["経過四半期"].max()
    age = last - tr["経過四半期"]
    if name in ("base", "no_time", "no_season"):
        return None
    if name == "recent8":
        return pd.Series(np.where(age < 8, 2.0, 1.0), index=tr.index)
    if name == "decay12":
        return pd.Series(np.exp(-age / 12), index=tr.index)
    if name == "decay24":
        return pd.Series(np.exp(-age / 24), index=tr.index)
    if name == "seiyaku15":
        return pd.Series(
            np.where(tr["価格情報区分"] == "成約価格情報", 1.5, 1.0), index=tr.index
        )
    raise ValueError(name)


selection.select = lambda tr: (tr, weight_fn(tr))
params = {**LGBM_PARAMS, "num_threads": 4, "seed": seed}


def factory() -> LGBM:
    return LGBM(params)


df = preprocess.load()
df, features = build_features(df)
train = df[df["is_train"]].reset_index(drop=True)
DROP = {"no_time": ["経過四半期", "取引四半期"], "no_season": ["取引四半期"]}.get(
    name, []
)
features = [f for f in features if f not in DROP]
scores = []
t0 = time.time()
for k, (tr_i, va_i) in enumerate(validation.time_series_folds(train)):
    tr, va = train.iloc[tr_i], train.iloc[va_i]
    inputs, models = pipeline.fit_and_predict_valid(tr, va, features, [factory])
    price = pipeline.blend_and_adjust(inputs, None, va, tr, features)
    scores.append(validation.rmsle(va[TARGET], price))
    print(
        f"{name} seed{seed} fold {k}: {scores[-1]:.5f} best_it={models[0].best_iteration} ({time.time() - t0:.0f}s)",
        flush=True,
    )
print(
    f"RESULT {name} seed{seed}: {' '.join(f'{s:.5f}' for s in scores)} CV {np.mean(scores):.5f}",
    flush=True,
)
