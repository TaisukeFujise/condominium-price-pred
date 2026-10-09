"""全員共通の検証方法。チームで合意せずに変えない。

変えるときは VERSION を上げる。VERSION が違う実験どうしの CV スコアは比べられない。

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
    """(学習に使う行番号, 検証に使う行番号) の組を、新しい期間の fold から順に返す。"""
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


def rmsle(y_log: np.ndarray, pred_log: np.ndarray) -> float:
    """log1p した正解と予測から RMSLE を計算する（対数価格での RMSE ＝ 元の価格での RMSLE）。"""
    return float(np.sqrt(np.mean((np.asarray(pred_log) - np.asarray(y_log)) ** 2)))


def describe_period(rows: pd.DataFrame) -> str:
    """行の取引時期の範囲を "2024Q4〜2025Q2" の形で返す。"""
    quarters = (rows["取引年"].astype(str) + "Q" + rows["取引四半期"].astype(str)).sort_values()
    return f"{quarters.iloc[0]}〜{quarters.iloc[-1]}"
