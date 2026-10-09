"""全員共通の検証方法。チームで合意せずに変えない。

変えるときは VERSION を上げる。VERSION が違う実験どうしの CV スコアは比べられない。
v1 は暫定。数回の提出で CV と LB が連動するかを確かめ、ずれる場合はチームで見直す。
この方法を選んだ理由と見直しの手順は docs/validation.md。

v1: 直近 9 四半期を 3 四半期ずつ 3 つの fold に分ける。
    各 fold は、検証期間より前のデータだけで学習する（未来のデータで過去を予測しない）。
    test は学習データの 1〜3 四半期先なので、検証期間も 3 四半期にそろえる。

    fold 0: 学習 2019Q1〜2024Q3 → 検証 2024Q4〜2025Q2
    fold 1: 学習 2019Q1〜2023Q4 → 検証 2024Q1〜2024Q3
    fold 2: 学習 2019Q1〜2023Q1 → 検証 2023Q2〜2023Q4
"""

import numpy as np
import pandas as pd

VERSION = "v1"
N_FOLDS = 3
VALID_QUARTERS = 3


def time_series_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """時系列 CV の fold を作る。

    Args:
        train: 学習データ。経過四半期の列が必要。

    Returns:
        (学習に使う行番号, 検証に使う行番号) の組の一覧。新しい期間の fold から順に並ぶ。
    """
    quarter = train["経過四半期"]
    last = quarter.max()
    folds = []
    for k in range(N_FOLDS):
        valid_end = last - k * VALID_QUARTERS
        valid_start = valid_end - VALID_QUARTERS + 1
        train_idx = np.flatnonzero(quarter < valid_start)
        valid_idx = np.flatnonzero(quarter.between(valid_start, valid_end))
        folds.append((train_idx, valid_idx))
    return folds


def rmsle(price_true: np.ndarray, price_pred: np.ndarray) -> float:
    """RMSLE（コンペの評価指標）を計算する。

    Args:
        price_true: 正解の価格（円）。
        price_pred: 予測した価格（円）。

    Returns:
        RMSLE。小さいほど良い。
    """
    diff = np.log1p(np.asarray(price_pred)) - np.log1p(np.asarray(price_true))
    return float(np.sqrt(np.mean(diff**2)))


def describe_period(rows: pd.DataFrame) -> str:
    """取引時期の範囲を表示用の文字列にする。

    Args:
        rows: 取引年・取引四半期の列を含む行。

    Returns:
        "2024Q4〜2025Q2" の形の文字列。
    """
    quarters = (
        rows["取引年"].astype(str) + "Q" + rows["取引四半期"].astype(str)
    ).sort_values()
    return f"{quarters.iloc[0]}〜{quarters.iloc[-1]}"
