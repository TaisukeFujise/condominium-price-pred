"""価格帯による予測の系統的なズレ（誤差分析で確認：安い方は過大予測、高い方は過小予測）への較正。

予測値(log)と実際値(log)の関係にisotonic回帰をあてて、予測を較正する
（regression-to-the-meanの典型パターンなので、単調な較正曲線で補正できるはず）。

`bias_correction.py`のhorizon補正と同じ設計：outer train内にさらに1つ内側のcutoffを切り、
そこで較正曲線を学習する。outer validのデータには一切触れない。
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from my_project.bias_correction import _predict_log
from my_project.io import TARGET_COLUMN, load_test, load_train, write_submission
from my_project.model import LGB_PARAMS, N_CV_FOLDS, _train_fold, rmsle, time_series_cv_splits
from my_project.preprocessing import CATEGORICAL_COLUMNS, build_features, fit_categories
from my_project.target_encoding import TARGET_ENCODE_COLUMNS, fit_mapping, oof_encode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def fit_price_calibrator(
    features: pd.DataFrame, target: pd.Series, train_mask: pd.Series, params: dict | None = None
) -> tuple[IsotonicRegression, int]:
    """outer train内でさらに1つ内側のcutoffを切り、予測値(log)→実際値(log)のisotonic較正を学習する。

    ついでにこの内側学習のbest_iterationも返す（本番モデルのboosting回数の目安に使い回す）。
    """
    inner_features = features[train_mask]
    inner_target = target[train_mask]
    inner_train_mask, inner_valid_mask = time_series_cv_splits(inner_features, n_folds=1)[0]

    model, _, mappings = _train_fold(
        inner_features, inner_target, inner_train_mask, inner_valid_mask, verbose=False, params=params
    )

    pred_log = _predict_log(model, mappings, inner_features[inner_valid_mask])
    actual_log = np.log1p(inner_target[inner_valid_mask].values)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pred_log, actual_log)
    return calibrator, model.best_iteration


def evaluate_with_price_calibration(train_df: pd.DataFrame, n_folds: int = N_CV_FOLDS) -> dict:
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    raw_scores, calibrated_scores = [], []
    for i, (train_mask, valid_mask) in enumerate(time_series_cv_splits(features, n_folds=n_folds)):
        calibrator, _ = fit_price_calibrator(features, target, train_mask)

        model, raw_score, mappings = _train_fold(features, target, train_mask, valid_mask, verbose=False)
        pred_log = _predict_log(model, mappings, features[valid_mask])
        calibrated_log = calibrator.predict(pred_log)

        calibrated_pred = np.clip(np.expm1(calibrated_log), a_min=1, a_max=None)
        actual = target[valid_mask]
        calibrated_score = rmsle(actual, calibrated_pred)

        print(f"fold {i + 1}: raw RMSLE={raw_score:.5f} -> calibrated RMSLE={calibrated_score:.5f}")
        raw_scores.append(raw_score)
        calibrated_scores.append(calibrated_score)

    raw_arr, cal_arr = np.array(raw_scores), np.array(calibrated_scores)
    print(f"\nraw CV:        mean={raw_arr.mean():.5f} ± {raw_arr.std():.5f}")
    print(f"calibrated CV: mean={cal_arr.mean():.5f} ± {cal_arr.std():.5f}")
    return {
        "raw_scores": raw_scores,
        "calibrated_scores": calibrated_scores,
        "raw_mean": float(raw_arr.mean()),
        "calibrated_mean": float(cal_arr.mean()),
    }


def train_production_model(train_df: pd.DataFrame, params: dict | None = None) -> dict:
    """提出用の本番モデル一式を作る（isotonic較正版）。

    較正曲線とboosting回数は、学習データ全体を内側ホールドアウト（直近3Q）で評価して決め、
    モデル自体はホールドアウト分も含めた学習データ全体で学習し直す（データを無駄にしない）。
    """
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    all_mask = pd.Series(True, index=features.index)
    calibrator, best_iteration = fit_price_calibrator(features, target, all_mask, params=params)
    print(f"production用 boosting回数={best_iteration}")

    y_log = np.log1p(target)
    train_features = features.copy()
    mappings = {}
    for col in TARGET_ENCODE_COLUMNS:
        train_features[f"{col}_target_enc"] = oof_encode(train_features[col], y_log)
        mappings[col] = fit_mapping(train_features[col], y_log)

    train_set = lgb.Dataset(train_features, label=y_log, categorical_feature=CATEGORICAL_COLUMNS)
    model = lgb.train(params or LGB_PARAMS, train_set, num_boost_round=best_iteration)

    return {"model": model, "mappings": mappings, "categories": categories, "calibrator": calibrator}


def predict_test_with_calibration(train_df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    prod = train_production_model(train_df, params=params)

    test_df = load_test()
    test_features = build_features(test_df, categories=prod["categories"])

    pred_log = _predict_log(prod["model"], prod["mappings"], test_features)
    calibrated_log = prod["calibrator"].predict(pred_log)
    pred = np.clip(np.expm1(calibrated_log), a_min=1, a_max=None)
    return pd.DataFrame({"id": test_df["id"], "price": pred})


if __name__ == "__main__":
    import sys

    train_df = load_train()
    if "--submit" in sys.argv:
        submission = predict_test_with_calibration(train_df)
        out_path = SUBMISSIONS_DIR / "calibrated_lightgbm.csv"
        write_submission(submission["id"], submission["price"], out_path)
        print(f"saved submission: {out_path}")
    else:
        evaluate_with_price_calibration(train_df)
