"""Optuna で LightGBM のハイパーパラメータを探す（#40）。

目的関数は main.py と同じ時系列 CV（src/validation.py、3 fold の平均 RMSLE）。
見つけた値は自動では反映しない。表示された値を src/models/params.py の LGBM_PARAMS に手で書く。

探すのは木の形と正則化だけ。learning_rate・cat_smooth・Huber の alpha は params.py の値に固定する
（learning_rate を下げると 1 回の CV が遅くなり、試せる回数が減るため）。
1 回目の試行は今の LGBM_PARAMS そのものにするので、同じ実行の中で今の値と比べられる。

試行の結果は tuning/ の SQLite に残る。同じ --study-name で実行し直すと続きから探す。
特徴量やモデルが変わったら、別の --study-name で最初から探し直す（前の結果は使えない）。

実行:
    uv run python -m scripts.tune_lgbm --n-trials 40 --timeout 10800
"""

import argparse
import time
from functools import partial

import numpy as np
import optuna
import pandas as pd

from src import pipeline, preprocess, validation
from src.config import ROOT, TARGET
from src.features.build import build_features
from src.models.lgbm import LGBM
from src.models.params import LGBM_PARAMS

TUNING_DIR = ROOT / "tuning"


def main() -> None:
    """コマンドラインの指定どおりに探索し、結果を表示する。"""
    args = _parse_args()
    train, features = _load_train()
    TUNING_DIR.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{TUNING_DIR / 'lgbm.db'}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        # fold ごとに途中経過を報告し、最初の fold が明らかに悪い試行は残りの fold を回さない
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(_current_values())
    study.optimize(
        partial(
            _objective, train=train, features=features, num_threads=args.num_threads
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    _print_result(study)


def _parse_args() -> argparse.Namespace:
    """コマンドラインの引数を読む。

    Returns:
        読んだ引数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40, help="試行の回数")
    parser.add_argument(
        "--timeout", type=int, default=None, help="この秒数を過ぎたら打ち切る"
    )
    parser.add_argument(
        "--study-name",
        default="lgbm",
        help="結果を残す名前。特徴量などが変わったら新しい名前にする",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="LightGBM のスレッド数。0 はすべてのコア。ほかの実験と同時に回すときは減らす",
    )
    parser.add_argument("--seed", type=int, default=0, help="探索（TPE）の乱数の seed")
    return parser.parse_args()


def _load_train() -> tuple[pd.DataFrame, list[str]]:
    """main.py と同じ前処理と特徴量で、学習データを作る。

    Returns:
        コンペの学習データ全体と、モデルに入れる列名の一覧。
    """
    df = preprocess.load()
    df, features = build_features(df)
    train = df[df["is_train"]].reset_index(drop=True)
    return train, features


def _suggest(trial: optuna.Trial) -> dict:
    """探す範囲からパラメータを 1 組選ぶ。

    Args:
        trial: Optuna の試行。

    Returns:
        LGBM_PARAMS に上書きするパラメータ。
    """
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        # bagging_fraction は bagging_freq が 0（既定）だと効かない
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


def _current_values() -> dict:
    """今の LGBM_PARAMS を、探す範囲のパラメータの形で返す。

    Returns:
        パラメータ名と値。書いていないものは LightGBM の既定値。
    """
    # 既定値の lambda は 0 だが、log の範囲には 0 を入れられないので下限で近づける
    return {
        "num_leaves": LGBM_PARAMS.get("num_leaves", 31),
        "min_data_in_leaf": LGBM_PARAMS.get("min_data_in_leaf", 20),
        "feature_fraction": LGBM_PARAMS.get("feature_fraction", 1.0),
        "bagging_fraction": LGBM_PARAMS.get("bagging_fraction", 1.0),
        "lambda_l1": max(LGBM_PARAMS.get("lambda_l1", 0.0), 1e-3),
        "lambda_l2": max(LGBM_PARAMS.get("lambda_l2", 0.0), 1e-3),
    }


def _objective(
    trial: optuna.Trial, train: pd.DataFrame, features: list[str], num_threads: int
) -> float:
    """選んだパラメータで時系列 CV を回し、平均 RMSLE を返す。

    Args:
        trial: Optuna の試行。
        train: コンペの学習データ全体。
        features: モデルに入れる列名の一覧。
        num_threads: LightGBM のスレッド数。

    Returns:
        3 fold の平均 RMSLE。
    """
    params = {**LGBM_PARAMS, **_suggest(trial), "num_threads": num_threads}
    model_class = partial(LGBM, params)
    start = time.time()
    scores = []
    for fold, (tr_pos, va_pos) in enumerate(validation.time_series_folds(train)):
        tr, va = train.iloc[tr_pos], train.iloc[va_pos]
        va_stack_inputs, _ = pipeline.fit_and_predict_valid(
            tr, va, features, [model_class]
        )
        price = pipeline.blend_and_adjust(va_stack_inputs, None, va, tr, features)
        scores.append(validation.rmsle(va[TARGET], price))
        # fold ごとに RMSLE の大きさが違うので、途中経過は「ここまでの平均」ではなく fold の値をそのまま比べる
        trial.report(scores[-1], fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("fold_scores", scores)
    print(
        f"trial {trial.number}: CV {np.mean(scores):.5f} "
        f"fold {[round(s, 5) for s in scores]}（{time.time() - start:.0f} 秒）",
        # ログをファイルに書き出すときも、試行ごとにすぐ見えるようにする
        flush=True,
    )
    return float(np.mean(scores))


def _print_result(study: optuna.Study) -> None:
    """今の値と最良の値の CV を並べ、params.py に書く値を表示する。

    Args:
        study: 探索の結果。
    """
    done = study.trials_dataframe(attrs=("number", "value", "state"))
    print(f"完了 {int((done['state'] == 'COMPLETE').sum())} / 試行 {len(done)}")
    first = study.trials[0]
    if first.value is not None:
        print(
            f"今の LGBM_PARAMS: CV {first.value:.5f} fold {first.user_attrs['fold_scores']}"
        )
    best = study.best_trial
    print(
        f"最良（trial {best.number}）: CV {best.value:.5f} fold {best.user_attrs['fold_scores']}"
    )
    print("params.py に書く値:")
    for name, value in best.params.items():
        print(
            f'    "{name}": {round(value, 4) if isinstance(value, float) else value},'
        )
    print('    "bagging_freq": 1,')


if __name__ == "__main__":
    main()
