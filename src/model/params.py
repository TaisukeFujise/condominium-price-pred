"""ハイパーパラメータ。損失関数以外は LightGBM のデフォルトのまま。"""

from src.config import SEED

PARAMS = {
    # 相場から極端にずれた取引（アンケート側の共有持分・入力ミスなど）に引っ張られないよう、
    # log 空間で 0.2（約 1.2 倍）より大きいずれは二乗ではなく線形にしか効かせない
    "objective": "huber",
    "alpha": 0.2,
    # 目的変数は log1p(価格) なので、rmse がそのまま RMSLE になる
    "metric": "rmse",
    "seed": SEED,
    "verbose": -1,
}
