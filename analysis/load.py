"""分析用。OOF（#77 の構成）と train の特徴量を結び付けて返す。"""

import pandas as pd

from src import preprocess
from src.features.build import build_features


def load():
    oof = pd.read_pickle("/Users/fujisetaisuke/condo-overnight/oof77.pkl")
    df = preprocess.load()
    df, features = build_features(df)
    train = df[df["is_train"]].reset_index(drop=True)
    assert (train.loc[oof["row"], "id"].to_numpy() == oof["id"].to_numpy()).all()
    cols = [c for c in train.columns if c not in oof.columns]
    out = pd.concat(
        [
            oof.reset_index(drop=True),
            train.loc[oof["row"], cols].reset_index(drop=True),
        ],
        axis=1,
    )
    return out, train, features
