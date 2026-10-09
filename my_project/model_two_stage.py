"""2段階学習（detrending）の実験。

Stage1: 時間に関する特徴量（取引年・四半期・築年数）を除いた物件特徴量だけでLightGBMを学習
        → 「物件そのものの価値」を時間を無視して評価する。
Stage2: Stage1のout-of-fold残差を四半期ごとに集計し、線形回帰で市場トレンドを抽出・外挿
        → Stage1予測 + トレンド補正 を最終予測にする。

検証は model.py と同じ時系列split（2025年Q1以降をホールドアウト）で行い、
シングルステージのRMSLE(0.210)と比較できるようにする。
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from my_project.io import TARGET_COLUMN, load_test, load_train, write_submission
from my_project.model import LGB_PARAMS, SEED, rmsle, time_series_split
from my_project.preprocessing import CATEGORICAL_COLUMNS, build_features, fit_categories

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Stage1では「暦時間そのもの」だけを落とす。building_ageは物件属性（築年数）として残す
# （txn_year/txn_quarterを残すと暦日付をそのまま学習してトレンド分離の意味がなくなるため除外）
TIME_COLUMNS = ["txn_year", "txn_quarter", "txn_quarter_index"]

N_FOLDS = 5
STAGE1_ROUNDS = 800
TREND_WINDOW_QUARTERS = 8  # トレンドの傾きを直近何四半期から推定するか（Noneなら全期間）


def _stage1_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in TIME_COLUMNS]


def _train_stage1(X: pd.DataFrame, y_log: pd.Series) -> lgb.Booster:
    train_set = lgb.Dataset(X, label=y_log, categorical_feature=CATEGORICAL_COLUMNS)
    return lgb.train(LGB_PARAMS, train_set, num_boost_round=STAGE1_ROUNDS)


def _stage1_oof(X: pd.DataFrame, y_log: pd.Series, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """物件特徴量だけのLightGBMをK-foldで回し、out-of-fold予測を返す（残差推定の過学習バイアスを防ぐ）。"""
    oof = np.zeros(len(X))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        train_set = lgb.Dataset(
            X.iloc[tr_idx], label=y_log.iloc[tr_idx], categorical_feature=CATEGORICAL_COLUMNS
        )
        model = lgb.train(LGB_PARAMS, train_set, num_boost_round=STAGE1_ROUNDS)
        oof[va_idx] = model.predict(X.iloc[va_idx])
        print(f"  stage1 fold {fold + 1}/{n_folds} done")
    return oof


def _fit_trend(
    quarter_index: pd.Series, residual: pd.Series, window: int | None = TREND_WINDOW_QUARTERS
) -> tuple[LinearRegression, pd.Series]:
    """四半期ごとの残差中央値に線形回帰をあてて市場トレンドを抽出する。"""
    trend = (
        pd.DataFrame({"quarter_index": quarter_index.values, "residual": residual.values})
        .groupby("quarter_index")["residual"]
        .median()
    )
    fit_target = trend.iloc[-window:] if window is not None else trend
    reg = LinearRegression()
    reg.fit(fit_target.index.values.reshape(-1, 1), fit_target.values)
    return reg, trend


def evaluate_two_stage(train_df: pd.DataFrame) -> dict:
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target_log = np.log1p(train_df[TARGET_COLUMN])

    train_mask, valid_mask = time_series_split(features)
    stage1_cols = _stage1_columns(features)

    X_train = features.loc[train_mask, stage1_cols]
    X_valid = features.loc[valid_mask, stage1_cols]
    y_train = target_log[train_mask]

    print(f"train: {len(X_train)}件, valid(2025Q1以降): {len(X_valid)}件")
    print("Stage1 OOF予測（学習データ内, 5-fold）...")
    oof_pred = _stage1_oof(X_train, y_train)
    residual = pd.Series(y_train.values - oof_pred, index=X_train.index)

    print("Stage1 最終モデル学習（学習データ全体）...")
    stage1_model = _train_stage1(X_train, y_train)

    print("Stage2: 四半期トレンドを線形回帰で抽出...")
    trend_reg, trend_series = _fit_trend(features.loc[train_mask, "txn_quarter_index"], residual)

    stage1_valid_pred = stage1_model.predict(X_valid)
    valid_quarter_index = features.loc[valid_mask, "txn_quarter_index"].values.reshape(-1, 1)
    trend_correction = trend_reg.predict(valid_quarter_index)
    two_stage_pred = np.expm1(stage1_valid_pred + trend_correction)
    stage1_only_pred = np.expm1(stage1_valid_pred)

    y_valid_raw = train_df.loc[valid_mask, TARGET_COLUMN]
    score_two_stage = rmsle(y_valid_raw, two_stage_pred)
    score_stage1_only = rmsle(y_valid_raw, stage1_only_pred)

    print(f"\n四半期ごとの残差中央値（log尺度、直近{TREND_WINDOW_QUARTERS}四半期をトレンド推定に使用）:")
    print(trend_series.tail(TREND_WINDOW_QUARTERS + 2))
    print(f"\nStage1単体のRMSLE（トレンド補正なし）: {score_stage1_only:.5f}")
    print(f"2段階（Stage1+トレンド補正）のRMSLE   : {score_two_stage:.5f}")
    print("（参考）シングルステージモデルのRMSLE  : 0.21020")

    return {
        "score_stage1_only": score_stage1_only,
        "score_two_stage": score_two_stage,
        "stage1_model": stage1_model,
        "trend_reg": trend_reg,
        "categories": categories,
        "stage1_cols": stage1_cols,
    }


def predict_test_two_stage(result: dict) -> pd.DataFrame:
    test_df = load_test()
    features = build_features(test_df, categories=result["categories"])
    X_test = features[result["stage1_cols"]]
    stage1_pred = result["stage1_model"].predict(X_test)
    trend_correction = result["trend_reg"].predict(features["txn_quarter_index"].values.reshape(-1, 1))
    pred = np.expm1(stage1_pred + trend_correction)
    pred = np.clip(pred, a_min=1, a_max=None)
    return pd.DataFrame({"id": test_df["id"], "price": pred})


if __name__ == "__main__":
    train_df = load_train()
    result = evaluate_two_stage(train_df)

    if result["score_two_stage"] < result["score_stage1_only"]:
        print("\nトレンド補正ありの方が良いので、このtrend_regでtest提出ファイルを作成します。")
    submission = predict_test_two_stage(result)
    out_path = SUBMISSIONS_DIR / "two_stage_lightgbm.csv"
    write_submission(submission["id"], submission["price"], out_path)
    print(f"saved submission: {out_path}")
