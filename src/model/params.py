"""ハイパーパラメータ。今は LightGBM のデフォルトのまま。"""

from src.config import SEED

PARAMS = {
    "objective": "regression",
    # 目的変数は log1p(価格) なので、rmse がそのまま RMSLE になる
    "metric": "rmse",
    "seed": SEED,
    "verbose": -1,
}
