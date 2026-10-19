"""ハイパーパラメータ（工程 6）。モデルを足したら、<モデル名>_PARAMS をここに並べる。

LightGBM の損失関数は Huber（#6）。木の形と正則化は、Optuna で探した値（#40、scripts/tune_lgbm.py）。
ほかの値を選んだ理由は、値の横のコメントに書く。
"""

from typing import Any

from src.config import SEED

LGBM_PARAMS: dict[str, Any] = {
    # 相場から極端にずれた取引（アンケート側の共有持分・入力ミスなど）に引っ張られないよう、
    # log 空間で 0.2（約 1.2 倍）より大きいずれは二乗ではなく線形にしか効かせない
    "objective": "huber",
    "alpha": 0.2,
    # 目的変数（log(㎡単価) − 指数）の誤差は log(総額) の誤差と同じなので、rmse がほぼ RMSLE になる
    "metric": "rmse",
    # 地区名・駅のように種類の多いカテゴリで、件数の少ない値に合わせすぎないよう、
    # 値ごとの平均を全体の平均に強めに寄せる（デフォルト 10）。30〜100 で同じくらい良く、300 では悪くなる
    "cat_smooth": 50,
    # デフォルトの 0.1 より小さくして木を増やす。学習時間はおよそ 2 倍になる
    "learning_rate": 0.05,
    # 木の形と正則化は Optuna で探した（#40、scripts/tune_lgbm.py）。特徴量が大きく変わったら探し直す。
    # 上位の試行はどれも「列を半分以下に絞り、葉を多く細かくする」組み合わせで、細かい値の違いは
    # CV にほとんど効かない（上位 8 試行の差は 0.0004）
    "num_leaves": 168,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.44,
    "bagging_fraction": 0.91,
    "bagging_freq": 1,
    "lambda_l1": 0.0076,
    "lambda_l2": 0.30,
    "seed": SEED,
    "verbose": -1,
}

# 目的変数・損失・評価指標は LGBM_PARAMS とそろえる。学習率と木の形は CatBoost 単体の CV で選んだ
CATBOOST_PARAMS: dict[str, Any] = {
    # LGBM_PARAMS の huber（alpha=0.2）と同じ。0.2 より大きいずれは線形にしか効かせない
    "loss_function": "Huber:delta=0.2",
    "eval_metric": "RMSE",
    # 0.05 では木の上限（10000 本）を使い切ってもまだ良くなっていたので上げる
    "learning_rate": 0.1,
    # デフォルトの 6 より深くする。8 では CatBoost 単体の CV が 0.0018 良くなった
    "depth": 8,
    # デフォルトの Newton 法だと depth 8 で 2 本目の木から発散する。Huber は大きくずれた行の
    # 2 階微分がほぼ 0 なので、そうした行ばかりの葉の値が極端に大きくなるため
    "leaf_estimation_method": "Gradient",
    "random_seed": SEED,
    "verbose": False,
    # 学習の記録（catboost_info/）をリポジトリの中に書かせない
    "allow_writing_files": False,
}

# 2 段目（src/ensemble.py の stacking）。学習できるのは OOF のある数四半期だけなので、
# 浅い木・葉ごとに多くの行・強い L2 にして合わせすぎを防ぐ
STACK_PARAMS: dict[str, Any] = {
    "objective": "huber",
    "alpha": 0.2,
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 1000,
    "lambda_l2": 10.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "verbose": -1,
}
# 50〜500 本で CV はほぼ同じ（差 0.0001）。一番古い fold は本数を増やすと悪くなるので少なめにする
STACK_ROUNDS = 50
