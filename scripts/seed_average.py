"""seed だけを変えて main.py を何回か動かし、test の予測を平均した提出ファイルを作る（seed 平均）。

LightGBM・CatBoost・2 段目は、列や行をランダムに選んで木を作るので、seed を変えると予測が変わる。
test の予測は seed どうしで log の差の RMS が約 0.02 あり、モデルを 5 つ足したとき（#76）の変化と
同じくらい揺れる。平均すると、この揺れが打ち消し合う。各 seed の CV もそれぞれ表示される。

seed は src/config.py の SEED を、ほかのモジュールを読む前に書き換えて変える。params.py などは
読み込んだときの SEED で値を決めるので、seed ごとに別のプロセスで main.py を動かす。
RMSLE は log をとってから誤差を測るので、平均も log の空間でとる。

実行（seed 1 つにつき main.py 1 回分、3 つで 2 時間ほど）:
    uv run python -m scripts.seed_average
    uv run python -m scripts.seed_average --seeds 42 1 2 3 4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SUBMISSION_DIR

DEFAULT_SEEDS = [42, 1, 2]


def main() -> None:
    """seed ごとに main.py を動かし、予測を平均して submissions/submission.csv に保存する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    # 子のプロセスとして 1 つの seed だけを動かすときに使う（直接は指定しない）
    parser.add_argument("--one", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one is not None:
        _run_one(args.one)
        return

    paths = []
    for seed in args.seeds:
        print(f"===== seed {seed}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "scripts.seed_average", "--one", str(seed)],
            check=True,
        )
        paths.append(_seed_path(seed))
    _average(paths).to_csv(SUBMISSION_DIR / "submission.csv", index=False)
    print(f"保存しました: submissions/submission.csv（{len(paths)} つの seed の平均）")


def _average(paths: list[Path]) -> pd.DataFrame:
    """提出ファイルの予測を、log の空間で平均する。

    Args:
        paths: 平均する提出ファイル。id の並びはどれも同じであること。

    Returns:
        平均した提出用の表（id, price）。
    """
    subs = [pd.read_csv(path) for path in paths]
    for sub in subs[1:]:
        if not (sub["id"].to_numpy() == subs[0]["id"].to_numpy()).all():
            raise ValueError("提出ファイルどうしで id の並びが違う")
    log_price = np.mean([np.log1p(sub["price"].to_numpy()) for sub in subs], axis=0)
    return pd.DataFrame({"id": subs[0]["id"], "price": np.expm1(log_price)})


def _run_one(seed: int) -> None:
    """SEED を書き換えてから main.py を動かし、提出ファイルを seed ごとの名前で残す。

    Args:
        seed: 使う seed。
    """
    import src.config

    src.config.SEED = seed
    # SEED を書き換えた後に読み込む（先に読むと、params.py が元の SEED で値を決めてしまう）
    import main as main_module

    main_module.main()
    (SUBMISSION_DIR / "submission.csv").replace(_seed_path(seed))


def _seed_path(seed: int) -> Path:
    """seed ごとの提出ファイルの置き場所を返す。

    Args:
        seed: seed。

    Returns:
        submissions/submission_seed<seed>.csv
    """
    return SUBMISSION_DIR / f"submission_seed{seed}.csv"


if __name__ == "__main__":
    main()
