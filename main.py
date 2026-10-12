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
from src.features.build import build_features
from src.model import ensemble, lgbm, postprocess, target


def train_and_predict(
    tr: pd.DataFrame,
    pred_rows: pd.DataFrame,
    features: list[str],
    num_boost_round: int | None = None,
    va: pd.DataFrame | None = None,
) -> tuple[np.ndarray, lgb.Booster]:
    """学習データの選択から後処理までを通して、価格を予測する。

    va を渡すと CV 用（early stopping で木の本数を決める）、
    num_boost_round を渡すと本番用（その本数で学習する）。

    Args:
        tr: 学習データ。CV では fold の学習データ、本番では train 全体。
        pred_rows: 予測する行。
        features: モデルに入れる列名の一覧。
        num_boost_round: 本番用の木の本数。
        va: CV 用の検証データ。

    Returns:
        pred_rows の予測価格（円）と、学習済みのモデル。
    """
    tr_used, weight = selection.select(tr)
    y = target.make_target(tr_used)
    if va is not None:
        model = lgbm.fit_with_validation(
            tr_used[features], y, va[features], target.make_target(va), weight
        )
    else:
        model = lgbm.fit(tr_used[features], y, num_boost_round, weight)
    price = target.to_price(lgbm.predict(model, pred_rows[features]), pred_rows)
    price = ensemble.blend([price])
    price = postprocess.postprocess(price, pred_rows, tr_used)
    return price, model


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
        # time_series_folds は位置を返す。train は reset_index 済みなので loc で引ける
        tr, va = train.loc[tr_index], train.loc[va_index]
        price, model = train_and_predict(tr, va, features, va=va)
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
    price, model = train_and_predict(
        train, test, features, num_boost_round=num_boost_round
    )

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
