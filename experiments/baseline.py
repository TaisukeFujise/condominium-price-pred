"""ベースライン：main.py と同じ特徴量・パラメータを、共通の時系列 CV で評価する。

実行: uv run python -m experiments.baseline
"""

from src.data import load
from src.features import base
from src.train import run_experiment

df = load()
run_experiment(
    __file__,
    df,
    features=base.FEATURES,
    description="main.py と同じ特徴量・パラメータ",
)
