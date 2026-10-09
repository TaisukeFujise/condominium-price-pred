"""最小構成のLightGBMモデル。

- 目的変数は log1p(price) で学習（RMSLE ≒ log変換後のRMSE）。
- 検証は時系列split：学習データの直近2四半期（2025年第1四半期以降）をホールドアウトにする。
  テスト期間（2025年第3四半期以降）は学習データに含まれないため、この境目が最もテスト条件に近い。
- カテゴリ特徴量の集合は学習データで固定し、test側にも同じcategory codeを適用する
  （固定しないとtrain/testでcode対応がずれてpredictが壊れる）。
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from my_project.io import TARGET_COLUMN, load_test, load_train, write_submission
from my_project.preprocessing import CATEGORICAL_COLUMNS, build_features, fit_categories
from my_project.target_encoding import TARGET_ENCODE_COLUMNS, apply_encoding, fit_mapping, oof_encode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# 2025年第1四半期(=2025*4+1)以降を検証用ホールドアウトにする
VALID_QUARTER_INDEX_THRESHOLD = 2025 * 4 + 1

SEED = 42

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}


def time_series_split(features: pd.DataFrame, threshold: int = VALID_QUARTER_INDEX_THRESHOLD):
    is_valid = features["txn_quarter_index"] >= threshold
    return ~is_valid, is_valid


def rmsle(y_true, y_pred) -> float:
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def _train_fold(
    features: pd.DataFrame,
    target: pd.Series,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    verbose: bool = True,
):
    """1つのtrain/valid分割でtarget encoding→学習→検証スコア算出までを行う（単一split・CV共通）。"""
    X_train, X_valid = features[train_mask].copy(), features[valid_mask].copy()
    y_train, y_valid = target[train_mask], target[valid_mask]
    y_train_log = np.log1p(y_train)

    # target encoding（地区名・最寄駅：名称）: train側はOOF、valid/testはtrain全体でfitしたmappingを適用
    mappings = {}
    for col in TARGET_ENCODE_COLUMNS:
        X_train[f"{col}_target_enc"] = oof_encode(X_train[col], y_train_log)
        mappings[col] = fit_mapping(X_train[col], y_train_log)
    valid_te = apply_encoding({col: X_valid[col] for col in TARGET_ENCODE_COLUMNS}, mappings)
    X_valid = pd.concat([X_valid, valid_te], axis=1)

    train_set = lgb.Dataset(X_train, label=y_train_log, categorical_feature=CATEGORICAL_COLUMNS)
    valid_set = lgb.Dataset(
        X_valid, label=np.log1p(y_valid), categorical_feature=CATEGORICAL_COLUMNS, reference=train_set
    )

    callbacks = [lgb.early_stopping(100, verbose=False)]
    if verbose:
        callbacks.append(lgb.log_evaluation(100))

    model = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    valid_pred = np.expm1(model.predict(X_valid, num_iteration=model.best_iteration))
    score = rmsle(y_valid, valid_pred)
    return model, score, mappings


def train_lightgbm(train_df: pd.DataFrame):
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    train_mask, valid_mask = time_series_split(features)
    print(f"train: {train_mask.sum()}件, valid(2025Q1以降): {valid_mask.sum()}件")

    model, score, mappings = _train_fold(features, target, train_mask, valid_mask, verbose=True)
    print(f"\nValidation RMSLE: {score:.5f} (best_iteration={model.best_iteration})")

    return model, score, categories, mappings


def time_series_cv_splits(
    features: pd.DataFrame,
    n_folds: int = 3,
    valid_quarters: int = 2,
    base_threshold: int = VALID_QUARTER_INDEX_THRESHOLD,
) -> list[tuple[pd.Series, pd.Series]]:
    """expanding window：学習期間を伸ばしながら、その直後の`valid_quarters`四半期を検証にする分割をn_folds個作る。"""
    splits = []
    for k in range(n_folds):
        threshold = base_threshold - k * valid_quarters
        train_mask = features["txn_quarter_index"] < threshold
        valid_mask = (features["txn_quarter_index"] >= threshold) & (
            features["txn_quarter_index"] < threshold + valid_quarters
        )
        splits.append((train_mask, valid_mask))
    return splits


def cross_validate(train_df: pd.DataFrame, n_folds: int = 3, valid_quarters: int = 2) -> dict:
    """複数の時系列foldで学習・評価し、RMSLEの平均とばらつき（std）を見る。

    単一ホールドアウトだとスコア差が本物の改善か誤差か判断できないため、
    改善施策の効果判定はこのCVのmean/stdを基準にする（docs/experiments.md参照）。
    """
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    scores = []
    for i, (train_mask, valid_mask) in enumerate(
        time_series_cv_splits(features, n_folds=n_folds, valid_quarters=valid_quarters)
    ):
        quarters = sorted(features.loc[valid_mask, "txn_quarter_index"].unique())
        _, score, _ = _train_fold(features, target, train_mask, valid_mask, verbose=False)
        print(
            f"fold {i + 1}/{n_folds}: valid quarters={quarters}, "
            f"train={int(train_mask.sum())}件, valid={int(valid_mask.sum())}件 -> RMSLE={score:.5f}"
        )
        scores.append(score)

    scores_arr = np.array(scores)
    print(f"\nCV RMSLE: mean={scores_arr.mean():.5f} ± {scores_arr.std():.5f}  (各fold: {[round(s, 5) for s in scores]})")
    return {"scores": scores, "mean": float(scores_arr.mean()), "std": float(scores_arr.std())}


def predict_test(model, categories: dict, mappings: dict) -> pd.DataFrame:
    test_df = load_test()
    features = build_features(test_df, categories=categories)
    te = apply_encoding({col: features[col] for col in TARGET_ENCODE_COLUMNS}, mappings)
    features = pd.concat([features, te], axis=1)
    pred = np.expm1(model.predict(features, num_iteration=model.best_iteration))
    pred = np.clip(pred, a_min=1, a_max=None)
    return pd.DataFrame({"id": test_df["id"], "price": pred})


def main() -> None:
    train_df = load_train()
    model, score, categories, mappings = train_lightgbm(train_df)

    importance = pd.Series(
        model.feature_importance(importance_type="gain"), index=model.feature_name()
    ).sort_values(ascending=False)
    print("\n特徴量重要度 (gain) 上位15:")
    print(importance.head(15))

    submission = predict_test(model, categories, mappings)
    out_path = SUBMISSIONS_DIR / "baseline_lightgbm.csv"
    write_submission(submission["id"], submission["price"], out_path)
    print(f"\nsaved submission: {out_path}")


if __name__ == "__main__":
    import sys

    if "--cv" in sys.argv:
        cross_validate(load_train())
    else:
        main()
