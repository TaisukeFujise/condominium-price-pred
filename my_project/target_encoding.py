"""高カード性カテゴリ（地区名・最寄駅：名称）のtarget encoding。

生カテゴリのままだと希少カテゴリ（地区名1,209種、最寄駅630種）の分割が不安定になりやすいので、
log価格の平均（smoothing付き）を数値特徴量として追加する。
学習データに対しては必ずout-of-foldで計算し、自分自身の価格が自分の特徴量に混ざるリークを防ぐ。
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold

TARGET_ENCODE_COLUMNS = ["地区名", "最寄駅：名称"]
SMOOTHING = 20  # 疑似サンプル数。小さいカテゴリほど全体平均に寄せる強さ


def _smoothed_mean(group_sum: pd.Series, group_count: pd.Series, global_mean: float, smoothing: float) -> pd.Series:
    return (group_sum + global_mean * smoothing) / (group_count + smoothing)


def fit_mapping(category: pd.Series, y_log: pd.Series, smoothing: float = SMOOTHING) -> tuple[pd.Series, float]:
    """カテゴリ→smoothed平均 のマッピングを作る（test/validへの適用用）。"""
    global_mean = float(y_log.mean())
    agg = y_log.groupby(category, observed=True).agg(["sum", "count"])
    mapping = _smoothed_mean(agg["sum"], agg["count"], global_mean, smoothing)
    return mapping, global_mean


def transform(category: pd.Series, mapping: pd.Series, global_mean: float) -> pd.Series:
    # category dtypeのままmapすると結果もCategoricalのままになり、未知の値でfillnaできないことがある
    return category.astype(object).map(mapping).fillna(global_mean).astype(float)


def oof_encode(
    category: pd.Series, y_log: pd.Series, n_folds: int = 5, smoothing: float = SMOOTHING, seed: int = 42
) -> pd.Series:
    """学習データ内のout-of-fold target encoding。"""
    oof = pd.Series(index=category.index, dtype=float)
    global_mean = float(y_log.mean())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(category):
        tr_cat, tr_y = category.iloc[tr_idx], y_log.iloc[tr_idx]
        mapping, _ = fit_mapping(tr_cat, tr_y, smoothing)
        va_cat = category.iloc[va_idx]
        oof.iloc[va_idx] = transform(va_cat, mapping, global_mean).values
    return oof


def apply_encoding(category: dict[str, pd.Series], mappings: dict[str, tuple[pd.Series, float]]) -> pd.DataFrame:
    """fit_mapping済みのmappingsを、別データ（valid/test）のカテゴリ列に適用する。

    `mappings`は {列名: (mapping, global_mean)}。`category`は{列名: Series}の辞書。
    """
    out = {}
    for col, (mapping, global_mean) in mappings.items():
        out[f"{col}_target_enc"] = transform(category[col], mapping, global_mean).values
    return pd.DataFrame(out, index=next(iter(category.values())).index)
