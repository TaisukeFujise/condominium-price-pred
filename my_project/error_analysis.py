"""モデルの誤差分析：どこで外しているかをエリア・価格帯・時期・価格情報区分別に見る。

直近3四半期（`time_series_split`のfold、本番に一番近い条件）で学習・予測し、
残差（log誤差）をいろいろな軸で集計する。外部データを足す前に、ここを見てから狙いを定める。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from my_project.io import TARGET_COLUMN, load_train
from my_project.model import _train_fold, time_series_split
from my_project.preprocessing import build_features, fit_categories
from my_project.target_encoding import TARGET_ENCODE_COLUMNS, apply_encoding


def build_error_table(train_df: pd.DataFrame) -> pd.DataFrame:
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    train_mask, valid_mask = time_series_split(features)
    model, score, mappings = _train_fold(features, target, train_mask, valid_mask, verbose=True)
    print(f"\n(参考) このfoldのRMSLE: {score:.5f}")

    X_valid = features[valid_mask].copy()
    valid_te = apply_encoding({col: X_valid[col] for col in TARGET_ENCODE_COLUMNS}, mappings)
    X_valid = pd.concat([X_valid, valid_te], axis=1)

    pred = np.clip(np.expm1(model.predict(X_valid, num_iteration=model.best_iteration)), a_min=1, a_max=None)
    actual = target[valid_mask]
    log_error = np.log1p(pred) - np.log1p(actual)  # 正=過大予測、負=過小予測

    err = train_df.loc[valid_mask, ["市区町村名", "地区名", "価格情報区分", "取引時期"]].copy()
    err["actual"] = actual.values
    err["pred"] = pred
    err["log_error"] = log_error.values
    err["sq_log_error"] = log_error.values ** 2
    err["area_sqm"] = features.loc[valid_mask, "area_sqm"].values
    err["txn_quarter_index"] = features.loc[valid_mask, "txn_quarter_index"].values
    return err


def summarize(err: pd.DataFrame, by, min_count: int = 30) -> pd.DataFrame:
    g = err.groupby(by, observed=True).agg(
        n=("sq_log_error", "size"),
        rmsle=("sq_log_error", lambda s: np.sqrt(s.mean())),
        bias=("log_error", "mean"),
    )
    return g[g["n"] >= min_count].sort_values("rmsle", ascending=False)


if __name__ == "__main__":
    train_df = load_train()
    err = build_error_table(train_df)

    overall_rmsle = np.sqrt(err["sq_log_error"].mean())
    print(f"\n全体RMSLE(valid): {overall_rmsle:.5f}  bias(log_error平均): {err['log_error'].mean():+.5f}")

    print("\n=== 市区町村名別（上位15、RMSLEが悪い順, n>=50）===")
    print(summarize(err, "市区町村名", min_count=50).head(15))

    print("\n=== 価格帯別（実際価格の10分位）===")
    err["price_decile"] = pd.qcut(err["actual"], 10, duplicates="drop")
    print(summarize(err, "price_decile", min_count=1))

    print("\n=== 面積帯別（5分位）===")
    err["area_quintile"] = pd.qcut(err["area_sqm"], 5, duplicates="drop")
    print(summarize(err, "area_quintile", min_count=1))

    print("\n=== 四半期別 ===")
    print(summarize(err, "txn_quarter_index", min_count=1))

    print("\n=== 価格情報区分別 ===")
    print(summarize(err, "価格情報区分", min_count=1))

    print("\n=== 残差が大きい行 上位20（過大・過小予測含む） ===")
    worst = err.reindex(err["sq_log_error"].sort_values(ascending=False).index).head(20)
    print(worst[["市区町村名", "地区名", "価格情報区分", "取引時期", "actual", "pred", "log_error"]].to_string())
