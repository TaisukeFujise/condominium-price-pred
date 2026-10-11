"""3fold時系列CVのOOF予測（isotonic較正後）を1ファイルにまとめる。

以後の残差分析（地図化、疑似建物ID分析、特徴量×残差プロット等）は全てこのファイルから行う。
毎回モデルを再学習しなくて済むようにするのが目的。

列: 元の生カラム一式 + area_sqm等の主要な加工後特徴量 + actual/pred/calibrated_pred/
    log_error(較正前)/log_error_calibrated(較正後)/fold/horizon
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from my_project.bias_correction import _predict_log
from my_project.calibration import fit_price_calibrator
from my_project.io import TARGET_COLUMN, load_train
from my_project.model import N_CV_FOLDS, _train_fold, time_series_cv_splits
from my_project.preprocessing import build_features, fit_categories

OUT_PATH = Path(__file__).resolve().parent / "analysis" / "oof_residuals.csv"

# 残差分析に使う主な加工後特徴量（build_featuresの出力から持ってくる）
EXTRA_FEATURE_COLUMNS = [
    "area_sqm",
    "built_year",
    "building_age",
    "station_distance_min",
    "station_lat",
    "station_lon",
    "station_ridership_total",
    "layout_rooms",
    "layout_has_L",
    "layout_has_D",
    "layout_has_K",
    "layout_has_S",
    "txn_quarter_index",
]

# 残差分析に使う生カラム
RAW_COLUMNS = [
    "市区町村名",
    "地区名",
    "最寄駅：名称",
    "価格情報区分",
    "建物の構造",
    "都市計画",
    "間取り",
    "建築年",
    "最寄駅：距離（分）",
    "取引時期",
]


def build_oof_residuals(train_df: pd.DataFrame, n_folds: int = N_CV_FOLDS) -> pd.DataFrame:
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    rows = []
    for fold_id, (train_mask, valid_mask) in enumerate(time_series_cv_splits(features, n_folds=n_folds)):
        calibrator, _ = fit_price_calibrator(features, target, train_mask)
        model, raw_score, mappings = _train_fold(features, target, train_mask, valid_mask, verbose=False)

        pred_log = _predict_log(model, mappings, features[valid_mask])
        calibrated_log = calibrator.predict(pred_log)
        actual = target[valid_mask]
        actual_log = np.log1p(actual.values)

        valid_start = int(features.loc[valid_mask, "txn_quarter_index"].min())
        horizon = features.loc[valid_mask, "txn_quarter_index"] - valid_start + 1

        chunk = pd.DataFrame(index=features.index[valid_mask])
        chunk["fold"] = fold_id + 1
        chunk["horizon"] = horizon.values
        chunk["actual"] = actual.values
        chunk["pred"] = np.expm1(pred_log)
        chunk["calibrated_pred"] = np.expm1(calibrated_log)
        chunk["log_error_raw"] = pred_log - actual_log
        chunk["log_error_calibrated"] = calibrated_log - actual_log

        for col in EXTRA_FEATURE_COLUMNS:
            chunk[col] = features.loc[valid_mask, col].values
        for col in RAW_COLUMNS:
            chunk[col] = train_df.loc[valid_mask, col].values

        print(f"fold {fold_id + 1}: {len(chunk)}行")
        rows.append(chunk)

    return pd.concat(rows, axis=0)


if __name__ == "__main__":
    train_df = load_train()
    oof = build_oof_residuals(train_df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(OUT_PATH, index=True, index_label="row_id")
    print(f"\nsaved: {OUT_PATH} ({len(oof)}行)")
    print(oof[["fold", "horizon", "log_error_raw", "log_error_calibrated"]].groupby("fold").agg(["mean", "std", "count"]))
