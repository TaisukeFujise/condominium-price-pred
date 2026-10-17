"""学習から提出ファイルの作成までの流れ。各工程の中身は src/ にある。

工程 3〜8（学習データの選択から後処理まで）のつなぎ方は src/pipeline.py。

精度改善は、該当する工程のファイルを直してこのファイルを実行し、
CV が良くなったら PR を出す（docs/experiment-guide.md）。

実行: uv run python main.py
"""

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd

from src import ensemble, pipeline, preprocess, validation
from src.config import SUBMISSION_DIR, TARGET
from src.features.build import build_features
from src.models.base import Model
from src.models.catboost import CatBoost
from src.models.lgbm import LGBM
from src.models.linear_residual import LinearResidual

# 使うモデル。src/models/base.py の Model を継承したクラスを並べる。
# 2 つ以上並べると、予測を src/ensemble.py で混ぜる（例: [LGBM, CatBoost]）。重みは CV の予測から自動で決まる
# 目的変数を総額にしたモデルは per_area=False を渡す（例: partial(LGBM, per_area=False)）
MODELS: list[Callable[[], Model]] = [
    LGBM,
    partial(LGBM, per_area=False),
    CatBoost,
    LinearResidual,
]


def main() -> None:
    """前処理から CV の表示、提出ファイルの作成までを順に実行する。"""
    # 前処理と特徴量（train と test をまとめて処理する）
    df = preprocess.load()
    df, features = build_features(df)
    train = df[df["is_train"]].reset_index(drop=True)
    test = df[~df["is_train"]].reset_index(drop=True)

    # 検証：時系列 CV（src/validation.py。チームの合意なしに変えない）
    cv_models, weights = _cross_validate(train, features)

    # 本番：全データで学習し直して test を予測する
    price, test_models = pipeline.fit_and_predict_test(
        train, test, features, MODELS, cv_models, weights
    )
    _print_importance(test_models)
    _save_submission(test, price)


def _cross_validate(
    train: pd.DataFrame, features: list[str]
) -> tuple[list[list[Model]], np.ndarray]:
    """時系列 CV で学習・検証し、fold ごとと平均の RMSLE を表示する。

    混ぜる重みを検証データの答えで決めると CV が良く見えすぎるので、
    ある fold の重みは残りの fold の予測だけで決めてから、その fold のスコアを測る。

    Args:
        train: コンペの学習データ全体。
        features: モデルに入れる列名の一覧。

    Returns:
        fold ごとに学習したモデルと、本番で混ぜる重み。
        モデルの [i] は MODELS[i] を fold ごとに学習したもの。本番の木の本数などを決めるのに使う。
        重みは全 fold の予測から決める。
    """
    folds = validation.time_series_folds(train)
    fold_preds = []
    cv_models = [[] for _ in MODELS]
    for tr_index, va_index in folds:
        tr, va = train.iloc[tr_index], train.iloc[va_index]
        preds, fold_models = pipeline.fit_and_predict_valid(tr, va, features, MODELS)
        fold_preds.append(preds)
        for model_index, model in enumerate(fold_models):
            cv_models[model_index].append(model)

    scores, mean_scores = [], []
    for fold, (tr_index, va_index) in enumerate(folds):
        tr, va = train.iloc[tr_index], train.iloc[va_index]
        others = [k for k in range(len(folds)) if k != fold]
        weights = _fit_weights(train, folds, fold_preds, others)
        price = pipeline.blend_and_adjust(fold_preds[fold], weights, va, tr, features)
        mean_price = pipeline.blend_and_adjust(fold_preds[fold], None, va, tr, features)
        scores.append(validation.rmsle(va[TARGET], price))
        mean_scores.append(validation.rmsle(va[TARGET], mean_price))
        best_iterations = [models[fold].best_iteration for models in cv_models]
        print(
            f"fold {fold}（検証 {validation.describe_period(va)}）: "
            f"RMSLE {scores[-1]:.5f}（単純平均 {mean_scores[-1]:.5f}、"
            f"重み={weights.round(3).tolist()}、best_iteration={best_iterations}）"
        )
    print(
        f"CV RMSLE（{validation.N_FOLDS} fold の平均、検証 {validation.VERSION}）: {np.mean(scores):.5f}"
        f"（単純平均 {np.mean(mean_scores):.5f}）"
    )
    weights = _fit_weights(train, folds, fold_preds, list(range(len(folds))))
    print(f"本番の重み（全 fold の予測から決めた）: {weights.round(3).tolist()}")
    return cv_models, weights


def _fit_weights(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_preds: list[list[np.ndarray]],
    use_folds: list[int],
) -> np.ndarray:
    """指定した fold の検証データの予測をつなげて、混ぜる重みを決める。

    Args:
        train: コンペの学習データ全体。
        folds: validation.time_series_folds の戻り値。
        fold_preds: [k][i] は fold k の検証データに対する MODELS[i] の予測（円）。
        use_folds: 重みを決めるのに使う fold の番号。

    Returns:
        モデルごとの重み。
    """
    preds = [
        np.concatenate([fold_preds[k][i] for k in use_folds])
        for i in range(len(MODELS))
    ]
    va_index = np.concatenate([folds[k][1] for k in use_folds])
    return ensemble.fit_weights(preds, train[TARGET].to_numpy()[va_index])


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
