"""アンサンブル。複数モデルの予測（円）を、2 段目の LightGBM で混ぜる（stacking）。

土台は、全行で学習したモデルの予測の log の平均。2 段目は、土台からのずれ（log）を
各モデルの予測と土台との差・面積・築年数・市区町村の価格帯などから予測する。
一部の行だけで学習したモデル（Model.subset）は単独では弱く、平均に入れると悪くなるので、2 段目の入力にだけ使う。
土台の値そのものは 2 段目に入れない。入れると CV はさらに良くなるが、それは予測の高さごとの補正で、
CV では良くなり Public LB では悪くなった isotonic 較正（src/postprocess.py）と同じ種類の補正だから。
2 段目は価格（答え）から学習するので、学習に使った行のスコアは良く見えすぎる。
CV では、ある fold の 2 段目はそれより前の期間の fold の予測だけで学習する（main.py の _cross_validate）。
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import target
from src.models.params import STACK_PARAMS, STACK_ROUNDS

BASE = "土台"
CATEGORIES = ["価格帯", "価格情報区分"]


def make_inputs(
    preds: list[np.ndarray],
    in_base: list[bool],
    pred_rows: pd.DataFrame,
    trend: target.Trend,
) -> pd.DataFrame:
    """2 段目の入力を作る。

    Args:
        preds: pred_rows のモデルごとの予測価格（円）。
        in_base: モデルごとに、土台の平均に入れるか（全行で学習したモデルなら True）。
        pred_rows: 予測する行。
        trend: その予測に使った相場の指数。市区町村の価格帯と学習期間の最後の四半期を引く。

    Returns:
        2 段目の入力。BASE 列は土台（log）で、2 段目の特徴量には入れない。
    """
    log_preds = np.log1p(np.array(preds))
    base = log_preds[np.array(in_base)].mean(axis=0)
    inputs = pd.DataFrame(
        {f"差{i}": log_pred - base for i, log_pred in enumerate(log_preds)}
    )
    inputs["面積"] = pred_rows["面積_数値"].to_numpy()
    inputs["築年数"] = pred_rows["築年数"].to_numpy()
    # 学習期間の何四半期先か。test も CV の検証データも 1〜3 になる
    inputs["先の四半期数"] = (pred_rows[target.QUARTER] - trend.last_quarter).to_numpy()
    city = pred_rows["市区町村名"].astype(object)
    inputs["価格帯"] = city.map(trend.city_group).fillna("0").astype(int).to_numpy()
    inputs["価格情報区分"] = pred_rows["価格情報区分"].cat.codes.to_numpy()
    inputs[BASE] = base
    return inputs


def fit_stacker(
    cv_stack_inputs: list[pd.DataFrame], cv_true_prices: list[np.ndarray]
) -> lgb.Booster:
    """渡した fold の予測から、土台からのずれ（log）を予測する 2 段目を学習する。

    Args:
        cv_stack_inputs: fold ごとの、make_inputs で作った検証データの入力。
        cv_true_prices: fold ごとの、検証データの正解の価格（円）。

    Returns:
        学習済みの 2 段目。
    """
    x, base = _split(pd.concat(cv_stack_inputs, ignore_index=True))
    log_true = np.log1p(np.concatenate(cv_true_prices))
    # 土台を初期値にして、ずれだけを木に学習させる
    dataset = lgb.Dataset(x, log_true, init_score=base, categorical_feature=CATEGORIES)
    return lgb.train(STACK_PARAMS, dataset, STACK_ROUNDS)


def blend(inputs: pd.DataFrame, stacker: lgb.Booster | None) -> np.ndarray:
    """2 段目で混ぜた予測を返す。

    Args:
        inputs: make_inputs で作った入力。
        stacker: 学習済みの 2 段目。None なら土台（log の平均）をそのまま使う。

    Returns:
        混ぜた予測（円）。
    """
    x, base = _split(inputs)
    if stacker is None:
        return np.expm1(base)
    return np.expm1(base + stacker.predict(x))


def _split(inputs: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """入力を、2 段目の特徴量と土台（log）に分ける。

    Args:
        inputs: make_inputs で作った入力。

    Returns:
        2 段目の特徴量と、土台（log）。
    """
    return inputs.drop(columns=BASE), inputs[BASE].to_numpy()
