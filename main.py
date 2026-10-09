"""学習から提出ファイルの作成までの流れ。各工程の中身は src/ にある。

精度改善は、該当する工程のファイルを直してこのファイルを実行し、
CV が良くなったら PR を出す（docs/experiment-guide.md）。

実行: uv run python main.py
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import validation
from src.config import SUBMISSION_DIR, TARGET
from src.data import preprocess, selection
from src.features import build_features
from src.model import ensemble, lgbm, postprocess, target


def train_and_predict(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    features: list[str],
    num_boost_round: int | None = None,
    valid: pd.DataFrame | None = None,
) -> tuple[np.ndarray, lgb.Booster]:
    """学習データの選択から後処理までを通して、価格を予測する。

    valid を渡すと CV 用（early stopping で木の本数を決める）、
    num_boost_round を渡すと本番用（その本数で学習する）。

    Args:
        train: 学習データ。
        rows: 予測する行。
        features: モデルに入れる列名の一覧。
        num_boost_round: 本番用の木の本数。
        valid: CV 用の検証データ。

    Returns:
        rows の予測価格（円）と、学習済みのモデル。
    """
    train, weight = selection.select(train)
    y = target.make_target(train)
    if valid is not None:
        model = lgbm.fit_with_validation(
            train[features], y, valid[features], target.make_target(valid), weight
        )
    else:
        model = lgbm.fit(train[features], y, num_boost_round, weight)
    price = target.to_price(lgbm.predict(model, rows[features]), rows)
    price = ensemble.blend([price])
    price = postprocess.postprocess(price, rows, train)
    return price, model


# 前処理と特徴量（train と test をまとめて処理する）
df = preprocess.load()
df, features = build_features(df)
train = df[df["is_train"]].reset_index(drop=True)
test = df[~df["is_train"]].reset_index(drop=True)

# 検証：時系列 CV（src/validation.py。チームの合意なしに変えない）
scores, best_iterations = [], []
for i, (train_idx, valid_idx) in enumerate(validation.time_series_folds(train)):
    fold_train, fold_valid = train.loc[train_idx], train.loc[valid_idx]
    price, model = train_and_predict(fold_train, fold_valid, features, valid=fold_valid)
    score = validation.rmsle(fold_valid[TARGET], price)
    scores.append(score)
    best_iterations.append(lgbm.best_iteration(model))
    print(
        f"fold {i}（検証 {validation.describe_period(fold_valid)}）: "
        f"RMSLE {score:.5f}（best_iteration={best_iterations[-1]}）"
    )
print(
    f"CV RMSLE（{validation.N_FOLDS} fold の平均、検証 {validation.VERSION}）: {np.mean(scores):.5f}"
)

# 本番：全データで学習し直して test を予測する。
# fold ごとに学習データの量が違うので、木の本数は 1 つの fold に合わせず平均を使う
num_boost_round = round(np.mean(best_iterations))
price, model = train_and_predict(train, test, features, num_boost_round=num_boost_round)

print("特徴量の重要度（gain）:")
print(lgbm.importance(model, features).round(3).to_string())

SUBMISSION_DIR.mkdir(exist_ok=True)
submission_path = SUBMISSION_DIR / "submission.csv"
pd.DataFrame({"id": test["id"], "price": price}).to_csv(submission_path, index=False)
print(f"保存しました: submissions/submission.csv（{len(test)} 行）")
