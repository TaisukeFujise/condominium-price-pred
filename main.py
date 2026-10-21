"""前処理から提出ファイルの作成までを、工程の順に呼ぶ。

各工程の中身は src/ にある。工程の番号と一覧は docs/workflow.md。
工程 3〜8（学習データの選択から後処理まで）は fold ごとに動くので、src/pipeline.py がまとめて呼ぶ。

精度を改善するときは、工程のファイルを直してこのファイルを実行し、CV が良くなったら PR を出す
（docs/experiment-guide.md）。

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
from src.models.hedonic import Hedonic
from src.models.lgbm import LGBM
from src.models.linear_residual import LinearResidual
from src.models.target_encoding import TargetEncodedLGBM


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
# モデルが 1 つでも 2 段目はかかり、面積・築年数などで予測を補正する。2 段目なしの単体の CV と比べるときは、表示の「2 段目なし」の値を使う
# 目的変数を総額にしたモデルは per_area=False を渡す（例: partial(LGBM, per_area=False)）
# 一部の行だけで学習するモデルは subset に行を選ぶ関数を渡す（例: partial(LGBM, subset=_built_over_30)。CatBoost なども同じ）。
# その予測は平均には入れず、2 段目の入力にだけ使う。行を選ぶ関数には特徴量（features）の列しか渡らないので、条件には特徴量の列を使う
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
]


def main() -> None:
    """前処理から CV の表示、提出ファイルの作成までを順に実行する。"""
    # 前処理と特徴量（train と test をまとめて処理する）
    df = preprocess.load()
    df, features = build_features(df)
    train = df[df["is_train"]].reset_index(drop=True)
    test = df[~df["is_train"]].reset_index(drop=True)

    # 検証：時系列 CV（src/validation.py。チームの合意なしに変えない）
    cv_models, cv_stack_inputs, cv_true_prices = _cross_val_predict(train, features)
    _print_cv_scores(train, features, cv_models, cv_stack_inputs, cv_true_prices)

    # 本番：2 段目を全 fold の予測で学習し、全データで学習し直したモデルの test の予測を混ぜる
    stacker = ensemble.fit_stacker(cv_stack_inputs, cv_true_prices)
    price, test_models = pipeline.fit_and_predict_test(
        train, test, features, MODELS, cv_models, stacker
    )
    _print_importance(test_models)
    _save_submission(test, price)


def _cross_val_predict(
    train: pd.DataFrame, features: list[str]
) -> tuple[list[list[Model]], list[pd.DataFrame], list[np.ndarray]]:
    """時系列 CV の fold ごとに、各モデル（MODELS）を学習データで学習し、検証データを予測する。

    戻り値は、本番の学習の材料になる。RMSLE は計算しない（_print_cv_scores で計算する）。

    Args:
        train: コンペの学習データ全体。
        features: モデルに入れる列名の一覧。

    Returns:
        次の 3 つ。2 段目の入力と正解は、本番の 2 段目（src/ensemble.py）の学習データになる。
        - モデル：[i][fold] は MODELS[i] をその fold で学習したもの。本番の木の本数などを決めるのに使う。
        - 2 段目の入力：fold ごとの、検証データに対する各モデルの予測。
        - 正解：fold ごとの、検証データの価格（円）。
    """
    folds = validation.time_series_folds(train)
    cv_stack_inputs: list[pd.DataFrame] = []
    cv_true_prices: list[np.ndarray] = []
    cv_models: list[list[Model]] = [[] for _ in MODELS]
    for tr_pos, va_pos in folds:
        tr, va = train.iloc[tr_pos], train.iloc[va_pos]
        va_stack_inputs, fold_models = pipeline.fit_and_predict_valid(
            tr, va, features, MODELS
        )
        cv_stack_inputs.append(va_stack_inputs)
        cv_true_prices.append(va[TARGET].to_numpy())
        for model_index, model in enumerate(fold_models):
            cv_models[model_index].append(model)
    return cv_models, cv_stack_inputs, cv_true_prices


def _print_cv_scores(
    train: pd.DataFrame,
    features: list[str],
    cv_models: list[list[Model]],
    cv_stack_inputs: list[pd.DataFrame],
    cv_true_prices: list[np.ndarray],
) -> None:
    """fold ごとに 2 段目を学習して検証データの RMSLE を計算し、fold ごとの値と平均を表示する。

    表示するだけで、本番には何も渡さない。

    2 段目の学習には、RMSLE を計算する fold より前の期間の fold の予測だけを使う。
    RMSLE を計算する fold の正解で学習すると、答えを見て混ぜ方を決めることになり、CV が良く見えすぎるため。
    いちばん前の期間の fold には、それより前の fold がないので、2 段目を使わずに計算する。

    Args:
        train: コンペの学習データ全体。
        features: モデルに入れる列名の一覧。
        cv_models: _cross_val_predict で学習したモデル。表示する best_iteration を取り出す。
        cv_stack_inputs: _cross_val_predict で作った、fold ごとの 2 段目の入力。
        cv_true_prices: _cross_val_predict で作った、fold ごとの正解の価格（円）。
    """
    folds = validation.time_series_folds(train)
    scores: list[float] = []
    base_scores: list[float] = []
    stack_gains: list[float] = []
    for fold, (tr_pos, va_pos) in enumerate(folds):
        tr, va = train.iloc[tr_pos], train.iloc[va_pos]
        # 本番の 2 段目は test より前の期間の予測しか見られないので、CV でも後の期間の fold は使わない。
        # folds は新しい期間から順に並ぶので、前の期間の fold は番号が大きい方
        earlier_folds = [k for k in range(len(folds)) if k > fold]
        base_price = pipeline.blend_and_adjust(
            cv_stack_inputs[fold], None, va, tr, features
        )
        if earlier_folds:
            fold_stacker = ensemble.fit_stacker(
                [cv_stack_inputs[k] for k in earlier_folds],
                [cv_true_prices[k] for k in earlier_folds],
            )
            price = pipeline.blend_and_adjust(
                cv_stack_inputs[fold], fold_stacker, va, tr, features
            )
        else:
            price = base_price
        scores.append(validation.rmsle(va[TARGET], price))
        base_scores.append(validation.rmsle(va[TARGET], base_price))
        if earlier_folds:
            stack_gains.append(scores[-1] - base_scores[-1])
        best_iterations = [per_fold[fold].best_iteration for per_fold in cv_models]
        print(
            f"fold {fold}（検証 {validation.describe_period(va)}）: "
            f"RMSLE {scores[-1]:.5f}（2 段目なし {base_scores[-1]:.5f}、"
            f"2 段目の学習に使った fold {earlier_folds}、"
            f"best_iteration={best_iterations}）"
        )
    print(
        f"CV RMSLE（{validation.N_FOLDS} fold の平均、検証 {validation.VERSION}）: {np.mean(scores):.5f}"
        f"（2 段目なし {np.mean(base_scores):.5f}）"
    )
    # 2 段目なしで測る fold があるので、3 fold の平均では 2 段目の効果が薄まる。
    # 2 段目を変える実験は、2 段目がある fold だけのこの値で比べる
    print(
        f"2 段目の効果（2 段目がある {len(stack_gains)} fold の、2 段目あり − なしの平均）: "
        f"{np.mean(stack_gains):+.5f}"
    )


def _print_importance(test_models: list[Model]) -> None:
    """本番のモデルごとに、特徴量の重要度を表示する。

    Args:
        test_models: 本番用に学習したモデル。
    """
    for model in test_models:
        importance = model.importance()
        if importance is not None:
            print(f"特徴量の重要度（{_describe(model)}）:")
            print(importance.round(3).to_string())


def _describe(model: Model) -> str:
    """同じクラスのモデルを見分けられるよう、クラス名に設定を添えた名前を作る。

    Args:
        model: モデル。

    Returns:
        クラス名と、既定から変えた設定（例: LGBM・総額、LGBM・_built_over_30）。
    """
    names = [type(model).__name__]
    if not model.per_area:
        names.append("総額")
    if model.subset is not None:
        names.append(model.subset.__name__)
    return "・".join(names)


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
