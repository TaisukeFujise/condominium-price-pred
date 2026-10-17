"""アンサンブル。複数モデルの予測（円）を、定数の重みで混ぜる。

重みは非負・合計 1 で、RMSLE が最小になるように fit_weights で決める。
混ぜるのは円の空間（重み付きの算術平均）。log の空間で混ぜる（幾何平均）より予測が少し高めに出るが、
単純平均のころから円で混ぜていて、Issue #38 で重みを測ったのも円で混ぜた予測なので、それに合わせる。
重みは価格（答え）から決めるので、決めるのに使った行のスコアは良く見えすぎる。
CV では、ある fold の重みは残りの fold の予測だけで決める（main.py の _cross_validate）。
モデルが 1 つ（main.py の MODELS が 1 つ）なら、重みは 1 で、渡された予測をそのまま返すのと同じ。
"""

import numpy as np
from scipy.optimize import minimize


def fit_weights(preds: list[np.ndarray], price_true: np.ndarray) -> np.ndarray:
    """混ぜた予測の RMSLE が最小になる、非負・合計 1 の重みを決める。

    Args:
        preds: モデルごとの予測（円）。
        price_true: 正解の価格（円）。

    Returns:
        モデルごとの重み。preds と同じ順に並ぶ。
    """
    n_models = len(preds)
    log_true = np.log1p(price_true)

    def rmsle(weights: np.ndarray) -> float:
        diff = np.log1p(blend(preds, weights)) - log_true
        return float(np.sqrt(np.mean(diff**2)))

    result = minimize(
        rmsle,
        np.full(n_models, 1 / n_models),
        method="SLSQP",
        bounds=[(0, 1)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
    )
    # SLSQP は境界をわずかに越えることがあるので、非負・合計 1 に戻す
    weights = np.clip(result.x, 0, None)
    return weights / weights.sum()


def blend(preds: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    """複数モデルの予測を、重み付きで平均する。

    Args:
        preds: モデルごとの予測（円）。
        weights: モデルごとの重み（合計 1）。None なら単純平均。

    Returns:
        混ぜた予測（円）。
    """
    return np.average(preds, axis=0, weights=weights)
