"""学習から提出ファイルの作成までの流れ。各工程の中身は src/ にある。

工程 3〜8（学習データの選択から後処理まで）のつなぎ方は src/pipeline.py。

精度改善は、該当する工程のファイルを直してこのファイルを実行し、
CV が良くなったら PR を出す（docs/experiment-guide.md）。

実行: uv run python main.py
"""

import numpy as np
import pandas as pd

from src import pipeline, preprocess, validation
from src.config import SUBMISSION_DIR, TARGET
from src.features.build import build_features
from src.models import lgbm


def main() -> None:
    """前処理から CV の表示、提出ファイルの作成までを順に実行する。"""
    # 前処理と特徴量（train と test をまとめて処理する）
    df = preprocess.load()
    df, features = build_features(df)
    train = df[df["is_train"]].reset_index(drop=True)
    test = df[~df["is_train"]].reset_index(drop=True)

    # 検証：時系列 CV（src/validation.py。チームの合意なしに変えない）
    scores, best_iterations = [], []
    for i, (tr_index, va_index) in enumerate(validation.time_series_folds(train)):
        tr, va = train.iloc[tr_index], train.iloc[va_index]
        price, model = pipeline.fit_and_predict_valid(tr, va, features)
        score = validation.rmsle(va[TARGET], price)
        scores.append(score)
        best_iterations.append(lgbm.best_iteration(model))
        print(
            f"fold {i}（検証 {validation.describe_period(va)}）: "
            f"RMSLE {score:.5f}（best_iteration={best_iterations[-1]}）"
        )
    print(
        f"CV RMSLE（{validation.N_FOLDS} fold の平均、検証 {validation.VERSION}）: {np.mean(scores):.5f}"
    )

    # 本番：全データで学習し直して test を予測する。
    # fold ごとに学習データの量が違うので、木の本数は 1 つの fold に合わせず平均を使う
    num_boost_round = round(np.mean(best_iterations))
    price, model = pipeline.fit_and_predict_test(train, test, features, num_boost_round)

    print("特徴量の重要度（gain）:")
    print(lgbm.importance(model, features).round(3).to_string())

    SUBMISSION_DIR.mkdir(exist_ok=True)
    submission_path = SUBMISSION_DIR / "submission.csv"
    pd.DataFrame({"id": test["id"], "price": price}).to_csv(
        submission_path, index=False
    )
    print(f"保存しました: submissions/submission.csv（{len(test)} 行）")


if __name__ == "__main__":
    main()
