"""horizon別bias補正とisotonic価格較正を両方試し、組み合わせの効果を比較する一時検証スクリプト。"""

from __future__ import annotations

import numpy as np

from my_project.bias_correction import _predict_log, estimate_horizon_bias, _horizon
from my_project.calibration import fit_price_calibrator
from my_project.io import TARGET_COLUMN, load_train
from my_project.model import N_CV_FOLDS, _train_fold, rmsle, time_series_cv_splits
from my_project.preprocessing import build_features, fit_categories


def run(train_df, n_folds: int = N_CV_FOLDS):
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    raw_scores, horizon_scores, iso_scores, both_scores = [], [], [], []
    for i, (train_mask, valid_mask) in enumerate(time_series_cv_splits(features, n_folds=n_folds)):
        bias_map, _ = estimate_horizon_bias(features, target, train_mask)
        calibrator, _ = fit_price_calibrator(features, target, train_mask)

        model, raw_score, mappings = _train_fold(features, target, train_mask, valid_mask, verbose=False)
        pred_log = _predict_log(model, mappings, features[valid_mask])
        actual = target[valid_mask]

        valid_start = int(features.loc[valid_mask, "txn_quarter_index"].min())
        horizon = _horizon(features.loc[valid_mask, "txn_quarter_index"], valid_start)
        correction = horizon.map(bias_map).fillna(0.0).values

        horizon_pred = np.clip(np.expm1(pred_log - correction), a_min=1, a_max=None)
        iso_pred = np.clip(np.expm1(calibrator.predict(pred_log)), a_min=1, a_max=None)
        both_log = calibrator.predict(pred_log - correction)
        both_pred = np.clip(np.expm1(both_log), a_min=1, a_max=None)

        s_raw, s_hz, s_iso, s_both = (
            raw_score,
            rmsle(actual, horizon_pred),
            rmsle(actual, iso_pred),
            rmsle(actual, both_pred),
        )
        print(f"fold {i + 1}: raw={s_raw:.5f}  horizon={s_hz:.5f}  isotonic={s_iso:.5f}  both={s_both:.5f}")
        raw_scores.append(s_raw)
        horizon_scores.append(s_hz)
        iso_scores.append(s_iso)
        both_scores.append(s_both)

    for name, scores in [("raw", raw_scores), ("horizon", horizon_scores), ("isotonic", iso_scores), ("both", both_scores)]:
        arr = np.array(scores)
        print(f"{name:<10} mean={arr.mean():.5f} ± {arr.std():.5f}")


if __name__ == "__main__":
    run(load_train())
