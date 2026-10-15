"""地区・最寄駅ごとの目的変数（log(㎡単価) − 相場の指数）の平均（target encoding）。

価格を使うので、MAKERS（fold に分ける前）ではなく FOLD_MAKERS に入れ、fold ごとにその fold の学習データだけで計算する。
目的変数で平均するのは、取引時期の違い（相場の上昇）が平均に混ざらないようにするため。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import SEED

# 件数の少ない地区・駅を全体の平均に寄せる強さ。件数がこの数のとき、地区の平均と全体の平均を半々に混ぜる
SMOOTHING = 20
# 学習データ自身の値を作るときに分ける数
N_SPLITS = 5


def _keys(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """平均を取る単位ごとに、行のキーを作る。

    Args:
        rows: 市区町村名・地区名・最寄駅：名称の列を含む行。

    Returns:
        作る列名 → 行ごとのキー。キーが欠損の行は NaN。
    """
    # 地区名は別の市区町村に同じ名前があるので、市区町村名と組にする
    district = rows["市区町村名"].astype(object) + "/" + rows["地区名"].astype(object)
    return {
        "地区_TE": district,
        "最寄駅_TE": rows["最寄駅：名称"].astype(object),
    }


def _smoothed_mean(
    key_fit: pd.Series, y_fit: np.ndarray, key_apply: pd.Series
) -> np.ndarray:
    """key_fit ごとの y_fit の平均を、件数に応じて全体の平均に寄せてから key_apply に付ける。

    Args:
        key_fit: 平均を計算する行のキー。
        y_fit: 平均を計算する行の目的変数。
        key_apply: 値を付ける行のキー。

    Returns:
        key_apply の行ごとの値。キーが欠損の行は NaN、学習データにないキーは全体の平均。
    """
    prior = float(np.mean(y_fit))
    stats = pd.Series(y_fit).groupby(key_fit.to_numpy()).agg(["sum", "count"])
    total = key_apply.map(stats["sum"]).fillna(0).to_numpy()
    count = key_apply.map(stats["count"]).fillna(0).to_numpy()
    encoded = (total + SMOOTHING * prior) / (count + SMOOTHING)
    return np.where(key_apply.isna().to_numpy(), np.nan, encoded)


def make(
    tr_used: pd.DataFrame, y: pd.Series, pred_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """地区・最寄駅ごとの目的変数の平均の列を、学習に使う行と予測する行に作る。

    学習に使う行の値は、学習に使う行を N_SPLITS 個に分け、自分を含まない残りから計算する（out-of-fold）。
    自分の価格が自分の特徴量に入ると、学習データでだけ当たりすぎるため。
    予測する行の値は、学習に使う行すべてから計算する。

    Args:
        tr_used: 学習に使う行。
        y: tr_used の目的変数。
        pred_rows: 予測する行（CV では検証データ、本番では test）。

    Returns:
        tr_used に足す列の表と、pred_rows に足す列の表。
    """
    y_values = y.to_numpy()
    tr_keys, pred_keys = _keys(tr_used), _keys(pred_rows)
    tr_out = pd.DataFrame(index=tr_used.index)
    pred_out = pd.DataFrame(index=pred_rows.index)
    splits = list(
        KFold(N_SPLITS, shuffle=True, random_state=SEED).split(np.arange(len(tr_used)))
    )
    for name, key in tr_keys.items():
        oof = np.empty(len(tr_used))
        for fit_index, apply_index in splits:
            oof[apply_index] = _smoothed_mean(
                key.iloc[fit_index], y_values[fit_index], key.iloc[apply_index]
            )
        tr_out[name] = oof
        pred_out[name] = _smoothed_mean(key, y_values, pred_keys[name])
    return tr_out, pred_out
