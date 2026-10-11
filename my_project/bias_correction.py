"""直近ほど過小予測が拡大する傾向（`error_analysis.py`で確認）への補正。

誤差分析で、ホールドアウト内の「何四半期目か」(horizon=1,2,3)が進むほどlog誤差(bias)が
悪化することが分かった（2024Q4: -0.010 → 2025Q1: -0.041 → 2025Q2: -0.062）。
旧2段階学習（Stage1から時間特徴量を丸ごと除く設計）はシングルステージに届かなかったが、
「トレンドを外挿する」という発想自体は誤差分析で裏付けられたので、
物件評価は普通のシングルステージモデルに任せたまま、horizon別のbias補正だけを後付けする
軽量な方式を試す。

bias補正値は、各foldのtrain内でさらに1つ内側のcutoff（outer trainの最後の3四半期をホールドアウト）
を切って推定する。outer validのデータは一切使わない。
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from my_project.io import TARGET_COLUMN, load_test, load_train, write_submission
from my_project.model import LGB_PARAMS, N_CV_FOLDS, _train_fold, rmsle, time_series_cv_splits
from my_project.preprocessing import CATEGORICAL_COLUMNS, build_features, fit_categories
from my_project.target_encoding import TARGET_ENCODE_COLUMNS, apply_encoding, fit_mapping, oof_encode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def _horizon(quarter_index: pd.Series, valid_start: int) -> pd.Series:
    """valid_startを1とした何四半期目か（1, 2, 3, ...）。"""
    return quarter_index - valid_start + 1


def _predict_log(model, mappings, features_subset: pd.DataFrame) -> np.ndarray:
    X = features_subset.copy()
    if mappings:
        te = apply_encoding({c: X[c] for c in TARGET_ENCODE_COLUMNS}, mappings)
        X = pd.concat([X, te], axis=1)
    raw_only_cols = [c for c in TARGET_ENCODE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    X = X.drop(columns=raw_only_cols, errors="ignore")
    return model.predict(X, num_iteration=model.best_iteration)


def estimate_horizon_bias(
    features: pd.DataFrame, target: pd.Series, train_mask: pd.Series, params: dict | None = None
) -> tuple[dict[int, float], int]:
    """outer train内でさらに1つ内側のcutoffを切り、horizonごとの平均log誤差(bias)を推定する。

    ついでにこの内側学習のbest_iterationも返す（本番モデルのboosting回数の目安に使い回す）。
    """
    inner_features = features[train_mask]
    inner_target = target[train_mask]
    inner_train_mask, inner_valid_mask = time_series_cv_splits(inner_features, n_folds=1)[0]

    model, _, mappings = _train_fold(
        inner_features, inner_target, inner_train_mask, inner_valid_mask, verbose=False, params=params
    )

    pred_log = _predict_log(model, mappings, inner_features[inner_valid_mask])
    actual = inner_target[inner_valid_mask]
    log_error = pred_log - np.log1p(actual)  # 正=過大予測、負=過小予測

    valid_start = int(inner_features.loc[inner_valid_mask, "txn_quarter_index"].min())
    horizon = _horizon(inner_features.loc[inner_valid_mask, "txn_quarter_index"], valid_start)

    bias_map = pd.Series(log_error.values, index=horizon.values).groupby(level=0).mean().to_dict()
    return bias_map, model.best_iteration


def evaluate_with_bias_correction(train_df: pd.DataFrame, n_folds: int = N_CV_FOLDS) -> dict:
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    raw_scores, corrected_scores = [], []
    for i, (train_mask, valid_mask) in enumerate(time_series_cv_splits(features, n_folds=n_folds)):
        bias_map, _ = estimate_horizon_bias(features, target, train_mask)
        print(f"fold {i + 1} horizon別bias推定(内側cutoff): { {k: round(v, 5) for k, v in bias_map.items()} }")

        model, raw_score, mappings = _train_fold(features, target, train_mask, valid_mask, verbose=False)

        pred_log = _predict_log(model, mappings, features[valid_mask])
        valid_start = int(features.loc[valid_mask, "txn_quarter_index"].min())
        horizon = _horizon(features.loc[valid_mask, "txn_quarter_index"], valid_start)
        correction = horizon.map(bias_map).fillna(0.0).values

        corrected_pred = np.clip(np.expm1(pred_log - correction), a_min=1, a_max=None)
        actual = target[valid_mask]
        corrected_score = rmsle(actual, corrected_pred)

        print(f"fold {i + 1}: raw RMSLE={raw_score:.5f} -> corrected RMSLE={corrected_score:.5f}")
        raw_scores.append(raw_score)
        corrected_scores.append(corrected_score)

    raw_arr, corr_arr = np.array(raw_scores), np.array(corrected_scores)
    print(f"\nraw CV:       mean={raw_arr.mean():.5f} ± {raw_arr.std():.5f}")
    print(f"corrected CV: mean={corr_arr.mean():.5f} ± {corr_arr.std():.5f}")
    return {
        "raw_scores": raw_scores,
        "corrected_scores": corrected_scores,
        "raw_mean": float(raw_arr.mean()),
        "corrected_mean": float(corr_arr.mean()),
    }


def train_production_model(train_df: pd.DataFrame, params: dict | None = None) -> dict:
    """提出用の本番モデル一式を作る。

    bias補正とboosting回数は、学習データ全体を内側ホールドアウト（直近3Q）で評価して決め、
    モデル自体はホールドアウト分も含めた学習データ全体で学習し直す（データを無駄にしない）。
    """
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    all_mask = pd.Series(True, index=features.index)
    bias_map, best_iteration = estimate_horizon_bias(features, target, all_mask, params=params)
    print(
        f"production用 horizon別bias推定: { {k: round(v, 5) for k, v in bias_map.items()} }, "
        f"boosting回数={best_iteration}"
    )

    y_log = np.log1p(target)
    train_features = features.copy()
    mappings = {}
    for col in TARGET_ENCODE_COLUMNS:
        train_features[f"{col}_target_enc"] = oof_encode(train_features[col], y_log)
        mappings[col] = fit_mapping(train_features[col], y_log)

    raw_only_cols = [c for c in TARGET_ENCODE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    train_features = train_features.drop(columns=raw_only_cols, errors="ignore")

    train_set = lgb.Dataset(train_features, label=y_log, categorical_feature=CATEGORICAL_COLUMNS)
    model = lgb.train(params or LGB_PARAMS, train_set, num_boost_round=best_iteration)

    return {
        "model": model,
        "mappings": mappings,
        "categories": categories,
        "bias_map": bias_map,
        "last_train_quarter": int(features["txn_quarter_index"].max()),
    }


def predict_test_with_correction(train_df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    prod = train_production_model(train_df, params=params)

    test_df = load_test()
    test_features = build_features(test_df, categories=prod["categories"])

    pred_log = _predict_log(prod["model"], prod["mappings"], test_features)
    horizon = _horizon(test_features["txn_quarter_index"], prod["last_train_quarter"] + 1)
    correction = horizon.map(prod["bias_map"])
    if correction.isna().any():
        missing = sorted(horizon[correction.isna()].unique())
        print(f"警告: bias_mapに無いhorizonがtestに存在: {missing}（0補正で扱う）")
        correction = correction.fillna(0.0)

    pred = np.clip(np.expm1(pred_log - correction.values), a_min=1, a_max=None)
    return pd.DataFrame({"id": test_df["id"], "price": pred})


if __name__ == "__main__":
    import sys

    train_df = load_train()
    if "--submit" in sys.argv:
        submission = predict_test_with_correction(train_df)
        out_path = SUBMISSIONS_DIR / "bias_corrected_lightgbm.csv"
        write_submission(submission["id"], submission["price"], out_path)
        print(f"saved submission: {out_path}")
    else:
        evaluate_with_bias_correction(train_df)
