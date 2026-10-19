"""学習から提出ファイルの作成までの流れ。各工程の中身は src/ にある。

工程 3〜8（学習データの選択から後処理まで）のつなぎ方は src/pipeline.py。

精度改善は、該当する工程のファイルを直してこのファイルを実行し、
CV が良くなったら PR を出す（docs/experiment-guide.md）。

実行: uv run python main.py
"""

from collections.abc import Callable
from functools import partial

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import ensemble, pipeline, preprocess, validation
from src.config import SUBMISSION_DIR, TARGET
from src.features.build import build_features
from src.models.base import Model
from src.models.catboost import CatBoost
from src.models.hedonic import Hedonic
from src.models.lgbm import LGBM
from src.models.linear_residual import LinearResidual
from src.models.target_encoding import TargetEncodedLGBM

# Issue #80 の確認用（駅・駅×建築年の単位を足した target encoding）
from f80.te import STATION_UNITS, ParamTE


def _built_over_30(x: pd.DataFrame) -> np.ndarray:
    """築 30 年を超える行を選ぶ。

    Args:
        x: 特徴量。

    Returns:
        行ごとの True / False。
    """
    return (x["築年数"] > 30).to_numpy()


def _area_30_or_less(x: pd.DataFrame) -> np.ndarray:
    """面積 30㎡ 以下の行を選ぶ。

    Args:
        x: 特徴量。

    Returns:
        行ごとの True / False。
    """
    return (x["面積_数値"] <= 30).to_numpy()


def _contract_price(x: pd.DataFrame) -> np.ndarray:
    """成約価格情報の行を選ぶ。

    Args:
        x: 特徴量。

    Returns:
        行ごとの True / False。
    """
    return (x["価格情報区分"] == "成約価格情報").to_numpy()


def _transaction_price(x: pd.DataFrame) -> np.ndarray:
    """不動産取引価格情報（成約価格情報でない）の行を選ぶ。

    Args:
        x: 特徴量。

    Returns:
        行ごとの True / False。
    """
    return (x["価格情報区分"] == "不動産取引価格情報").to_numpy()


def _outside_23_wards(x: pd.DataFrame) -> np.ndarray:
    """23 区以外（市町村）の行を選ぶ。

    Args:
        x: 特徴量。

    Returns:
        行ごとの True / False。
    """
    return (~x["市区町村名"].astype(str).str.endswith("区")).to_numpy()


# 使うモデル。src/models/base.py の Model を継承したクラスを並べる。
# 2 つ以上並べると、予測を src/ensemble.py の 2 段目で混ぜる（例: [LGBM, CatBoost]）。混ぜ方は CV の予測から自動で学習する
# 目的変数を総額にしたモデルは per_area=False を渡す（例: partial(LGBM, per_area=False)）
# 一部の行だけで学習するモデルは subset に行を選ぶ関数を渡す（例: partial(LGBM, subset=_built_over_30)）。
# その予測は平均には入れず、2 段目の入力にだけ使う。行を選ぶ条件には test でも分かる列だけを使う
# 全行で学習しても平均に入れないモデル（例: Hedonic・TargetEncodedLGBM）は、クラスに stack_only = True を書く
MODELS: list[Callable[[], Model]] = [
    LGBM,
    partial(LGBM, per_area=False),
    CatBoost,
    LinearResidual,
    partial(LGBM, subset=_built_over_30),
    partial(LGBM, subset=_area_30_or_less),
    partial(LGBM, subset=_contract_price),
    partial(LGBM, subset=_transaction_price),
    partial(LGBM, subset=_outside_23_wards),
    Hedonic,
    TargetEncodedLGBM,
    # Issue #80：駅入りの単位の TE と、総額の目的変数の TE を 2 段目の入力に足す
    partial(ParamTE, units=STATION_UNITS),
    partial(TargetEncodedLGBM, per_area=False),
]


def main() -> None:
    """前処理から CV の表示、提出ファイルの作成までを順に実行する。"""
    # 前処理と特徴量（train と test をまとめて処理する）
    df = preprocess.load()
    df, features = build_features(df)
    train = df[df["is_train"]].reset_index(drop=True)
    test = df[~df["is_train"]].reset_index(drop=True)

    # 検証：時系列 CV（src/validation.py。チームの合意なしに変えない）
    cv_models, stacker = _cross_validate(train, features)

    # 本番：全データで学習し直して test を予測する
    price, test_models = pipeline.fit_and_predict_test(
        train, test, features, MODELS, cv_models, stacker
    )
    _print_importance(test_models)
    _save_submission(test, price)


def _cross_validate(
    train: pd.DataFrame, features: list[str]
) -> tuple[list[list[Model]], lgb.Booster]:
    """時系列 CV で学習・検証し、fold ごとと平均の RMSLE を表示する。

    2 段目（混ぜ方）を検証データの答えで学習すると CV が良く見えすぎるので、
    ある fold の 2 段目は残りの fold の予測だけで学習してから、その fold のスコアを測る。

    Args:
        train: コンペの学習データ全体。
        features: モデルに入れる列名の一覧。

    Returns:
        fold ごとに学習したモデルと、本番で使う 2 段目。
        モデルの [i] は MODELS[i] を fold ごとに学習したもの。本番の木の本数などを決めるのに使う。
        2 段目は全 fold の予測から学習する。
    """
    folds = validation.time_series_folds(train)
    fold_inputs, fold_prices = [], []
    cv_models = [[] for _ in MODELS]
    for tr_index, va_index in folds:
        tr, va = train.iloc[tr_index], train.iloc[va_index]
        inputs, fold_models = pipeline.fit_and_predict_valid(tr, va, features, MODELS)
        fold_inputs.append(inputs)
        fold_prices.append(va[TARGET].to_numpy())
        for model_index, model in enumerate(fold_models):
            cv_models[model_index].append(model)

    scores, base_scores = [], []
    for fold, (tr_index, va_index) in enumerate(folds):
        tr, va = train.iloc[tr_index], train.iloc[va_index]
        others = [k for k in range(len(folds)) if k != fold]
        stacker = ensemble.fit_stacker(
            [fold_inputs[k] for k in others], [fold_prices[k] for k in others]
        )
        price = pipeline.blend_and_adjust(fold_inputs[fold], stacker, va, tr, features)
        base_price = pipeline.blend_and_adjust(
            fold_inputs[fold], None, va, tr, features
        )
        scores.append(validation.rmsle(va[TARGET], price))
        base_scores.append(validation.rmsle(va[TARGET], base_price))
        best_iterations = [models[fold].best_iteration for models in cv_models]
        print(
            f"fold {fold}（検証 {validation.describe_period(va)}）: "
            f"RMSLE {scores[-1]:.5f}（2 段目なし {base_scores[-1]:.5f}、"
            f"best_iteration={best_iterations}）"
        )
    print(
        f"CV RMSLE（{validation.N_FOLDS} fold の平均、検証 {validation.VERSION}）: {np.mean(scores):.5f}"
        f"（2 段目なし {np.mean(base_scores):.5f}）"
    )
    return cv_models, ensemble.fit_stacker(fold_inputs, fold_prices)


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
