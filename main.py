"""学習から提出ファイルの作成までの流れ。各工程の中身は src/step*.py にある。

工程の番号は docs/workflow.md と同じ。精度改善は、該当する工程のファイルを直して
このファイルを実行し、CV が良くなったら PR を出す（docs/experiment-guide.md）。

実行: uv run python main.py
"""

import numpy as np
import pandas as pd

from src import (
    step1_preprocess,
    step2_features,
    step3_selection,
    step4_target,
    step5_model,
    step7_ensemble,
    step8_postprocess,
    validation,
)
from src.config import SUBMISSION_DIR, TARGET


def train_and_predict(train, rows, features, num_boost_round=None, valid=None):
    """工程 3〜8 を通して、rows の価格（円）を予測する。

    valid を渡すと CV 用（early stopping で木の本数を決める）、
    num_boost_round を渡すと本番用（その本数で学習する）。
    """
    train, weight = step3_selection.select(train)
    y = step4_target.make_target(train)
    if valid is not None:
        model = step5_model.fit_with_validation(
            train[features], y, valid[features], step4_target.make_target(valid), weight
        )
    else:
        model = step5_model.fit(train[features], y, num_boost_round, weight)
    price = step4_target.to_price(step5_model.predict(model, rows[features]), rows)
    price = step7_ensemble.blend([price])
    price = step8_postprocess.postprocess(price, rows, train)
    return price, model


# 工程 1・2：前処理と特徴量（train と test をまとめて処理する）
df = step1_preprocess.load()
df, features = step2_features.build(df)
train = df[df["is_train"]].reset_index(drop=True)
test = df[~df["is_train"]].reset_index(drop=True)

# 検証：時系列 CV（src/validation.py。チームの合意なしに変えない）
scores, best_iterations = [], []
for i, (train_idx, valid_idx) in enumerate(validation.time_series_folds(train)):
    fold_train, fold_valid = train.loc[train_idx], train.loc[valid_idx]
    price, model = train_and_predict(fold_train, fold_valid, features, valid=fold_valid)
    score = validation.rmsle(fold_valid[TARGET], price)
    scores.append(score)
    best_iterations.append(step5_model.best_iteration(model))
    print(
        f"fold {i}（検証 {validation.describe_period(fold_valid)}）: "
        f"RMSLE {score:.5f}（best_iteration={best_iterations[-1]}）"
    )
print(f"CV RMSLE（{validation.N_FOLDS} fold の平均、検証 {validation.VERSION}）: {np.mean(scores):.5f}")

# 本番：全データで学習し直して test を予測する。
# fold ごとに学習データの量が違うので、木の本数は 1 つの fold に合わせず平均を使う
num_boost_round = int(round(np.mean(best_iterations)))
price, model = train_and_predict(train, test, features, num_boost_round=num_boost_round)

print("特徴量の重要度（gain）:")
print(step5_model.importance(model, features).round(3).to_string())

SUBMISSION_DIR.mkdir(exist_ok=True)
submission_path = SUBMISSION_DIR / "submission.csv"
pd.DataFrame({"id": test["id"], "price": price}).to_csv(submission_path, index=False)
print(f"保存しました: submissions/submission.csv（{len(test)} 行）")
