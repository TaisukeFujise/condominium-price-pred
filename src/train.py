"""実験の実行：CV で学習・評価 → 全データで再学習 → 提出ファイルと実験記録を保存（main.py のセクション 4〜8）。"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import validation
from src.config import EXPERIMENTS_DIR, ROOT, SEED, SUBMISSION_DIR, TARGET

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "seed": SEED,
    "verbose": -1,
}


def run_experiment(
    script_path: str,
    df: pd.DataFrame,
    features: list[str],
    description: str,
    params: dict | None = None,
) -> dict:
    """実験を 1 つ実行し、記録を返す。

    script_path には実験スクリプトの __file__ を渡す。実験 ID はそのパスから決まる
    （experiments/fujise/exp001_xxx.py → "fujise/exp001_xxx"）。
    """
    script = Path(script_path).resolve()
    exp_id = script.relative_to(EXPERIMENTS_DIR).with_suffix("").as_posix()
    params = {**DEFAULT_PARAMS, **(params or {})}
    print(f"=== {exp_id}: {description} ===")

    train = df[df["is_train"]].reset_index(drop=True)
    test = df[~df["is_train"]].reset_index(drop=True)
    y = np.log1p(train[TARGET])

    fold_results = []
    for i, (train_idx, valid_idx) in enumerate(validation.time_series_folds(train)):
        model = lgb.train(
            params,
            lgb.Dataset(train.loc[train_idx, features], y.iloc[train_idx]),
            num_boost_round=10000,
            valid_sets=[lgb.Dataset(train.loc[valid_idx, features], y.iloc[valid_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = model.predict(train.loc[valid_idx, features], num_iteration=model.best_iteration)
        result = {
            "fold": i,
            "valid_period": validation.describe_period(train.loc[valid_idx]),
            "rmsle": validation.rmsle(y.iloc[valid_idx], pred),
            "best_iteration": model.best_iteration,
        }
        fold_results.append(result)
        print(
            f"fold {i}（検証 {result['valid_period']}）: "
            f"RMSLE {result['rmsle']:.5f}（best_iteration={result['best_iteration']}）"
        )

    cv_rmsle = float(np.mean([r["rmsle"] for r in fold_results]))
    print(f"CV RMSLE（{validation.N_FOLDS} fold の平均）: {cv_rmsle:.5f}")

    # fold ごとに学習データの量が違うので、木の本数は 1 つの fold に合わせず平均を使う
    num_boost_round = int(round(np.mean([r["best_iteration"] for r in fold_results])))
    final_model = lgb.train(
        params, lgb.Dataset(train[features], y), num_boost_round=num_boost_round
    )

    importance = pd.Series(
        final_model.feature_importance(importance_type="gain"), index=features
    ).sort_values(ascending=False)
    print("特徴量の重要度（gain）:")
    print((importance / importance.sum()).round(3).to_string())

    SUBMISSION_DIR.mkdir(exist_ok=True)
    submission_path = SUBMISSION_DIR / f"{exp_id.replace('/', '__')}.csv"
    pd.DataFrame(
        {"id": test["id"], "price": np.expm1(final_model.predict(test[features]))}
    ).to_csv(submission_path, index=False)

    record = {
        "exp_id": exp_id,
        "description": description,
        "cv_rmsle": cv_rmsle,
        # 提出したら手で書き込む
        "lb_public": None,
        "validation_version": validation.VERSION,
        "folds": fold_results,
        "num_boost_round": num_boost_round,
        "features": features,
        "params": params,
        "git_commit": _git("rev-parse", "HEAD"),
        # True なら、記録のスコアは git_commit の時点のコードとは一致しない
        "git_dirty": _is_dirty(script),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    record_path = script.with_suffix(".json")
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    print(f"提出ファイル: {submission_path.relative_to(ROOT)}")
    print(f"実験記録:     {record_path.relative_to(ROOT)}")
    return record


def _git(*args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=check
    ).stdout.strip()


def _is_dirty(script: Path) -> bool:
    # 関係ない未追跡ファイルは無視する。ただし実験スクリプト自体が未コミットなら dirty とみなす
    has_changes = bool(_git("status", "--porcelain", "--untracked-files=no"))
    script_tracked = bool(_git("ls-files", str(script), check=False))
    return has_changes or not script_tracked
