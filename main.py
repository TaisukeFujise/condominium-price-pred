"""学習から提出ファイルの作成までの流れ。各工程の中身は src/ にある。

工程 3〜8（学習データの選択から後処理まで）のつなぎ方は src/pipeline.py。

精度改善は、該当する工程のファイルを直してこのファイルを実行し、
CV が良くなったら PR を出す（docs/experiment-guide.md）。

実行: uv run python main.py
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src import pipeline, preprocess, validation
from src.config import SUBMISSION_DIR, TARGET
from src.features.build import build_features
from src.models.base import Model
from src.models.catboost import CatBoost
from src.models.lgbm import LGBM

# 使うモデル。src/models/base.py の Model を継承したクラスを並べる。
# 2 つ以上並べると、予測を src/ensemble.py で混ぜる（例: [LGBM, CatBoost]）
MODELS: list[Callable[[], Model]] = [LGBM, CatBoost]


def main() -> None:
    """前処理から CV の表示、提出ファイルの作成までを順に実行する。"""
    # 前処理と特徴量（train と test をまとめて処理する）
    df = preprocess.load()
    df, features = build_features(df)
    train = df[df["is_train"]].reset_index(drop=True)
    test = df[~df["is_train"]].reset_index(drop=True)

    # 検証：時系列 CV（src/validation.py。チームの合意なしに変えない）
    cv_models = _cross_validate(train, features)

    # 本番：全データで学習し直して test を予測する
    price, test_models = pipeline.fit_and_predict_test(
        train, test, features, MODELS, cv_models
    )
    _print_importance(test_models)
    _save_submission(test, price)


def _cross_validate(train: pd.DataFrame, features: list[str]) -> list[list[Model]]:
    """時系列 CV で学習・検証し、fold ごとと平均の RMSLE を表示する。

    Args:
        train: コンペの学習データ全体。
        features: モデルに入れる列名の一覧。

    Returns:
        fold ごとに学習したモデル。[i] は MODELS[i] を fold ごとに学習したもの。
        本番の木の本数などを決めるのに使う。
    """
    scores = []
    cv_models = [[] for _ in MODELS]
    for fold, (tr_index, va_index) in enumerate(validation.time_series_folds(train)):
        tr, va = train.iloc[tr_index], train.iloc[va_index]
        price, fold_models = pipeline.fit_and_predict_valid(tr, va, features, MODELS)
        score = validation.rmsle(va[TARGET], price)
        scores.append(score)
        for model_index, model in enumerate(fold_models):
            cv_models[model_index].append(model)
        best_iterations = [model.best_iteration for model in fold_models]
        print(
            f"fold {fold}（検証 {validation.describe_period(va)}）: "
            f"RMSLE {score:.5f}（best_iteration={best_iterations}）"
        )
    print(
        f"CV RMSLE（{validation.N_FOLDS} fold の平均、検証 {validation.VERSION}）: {np.mean(scores):.5f}"
    )
    return cv_models


def _print_importance(test_models: list[Model]) -> None:
    """本番のモデルごとに、特徴量の重要度を表示する。

    Args:
        test_models: 本番用に学習したモデル。
    """
    for model in test_models:
        importance = model.importance()
        if importance is not None:
            print(f"特徴量の重要度（{type(model).__name__}）:")
            print(importance.round(3).to_string())


def _save_submission(test: pd.DataFrame, price: np.ndarray) -> None:
    """test の予測価格を、提出用の CSV に保存する。

    Args:
        test: コンペのテストデータ。
        price: test の予測価格（円）。
    """
    SUBMISSION_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"id": test["id"], "price": price}).to_csv(
        SUBMISSION_DIR / "submission.csv", index=False
    )
    print(f"保存しました: submissions/submission.csv（{len(test)} 行）")


if __name__ == "__main__":
    main()
