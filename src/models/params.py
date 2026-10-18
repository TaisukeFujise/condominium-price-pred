"""ハイパーパラメータ。モデルを足したら、<モデル名>_PARAMS をここに並べる。

LightGBM は、特徴量の実験の前に理由のあるものだけ見直した（#12）。損失関数は Huber（#6）。
木の形と正則化は Optuna で探した（#40）。
"""

from src.config import SEED

LGBM_PARAMS = {
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

# 目的変数・損失・評価指標は LGBM_PARAMS とそろえ、ほかは CatBoost のデフォルトのまま
CATBOOST_PARAMS = {
    # LGBM_PARAMS の huber（alpha=0.2）と同じ。0.2 より大きいずれは線形にしか効かせない
    "loss_function": "Huber:delta=0.2",
    "eval_metric": "RMSE",
    "learning_rate": 0.05,
    "random_seed": SEED,
    "verbose": False,
    # 学習の記録（catboost_info/）をリポジトリの中に書かせない
    "allow_writing_files": False,
}

# ほかのモデルと間違え方を変えるためのモデルなので、層の数・幅は小さめの既定値から始めた
MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128),
    # 重みの L2 正則化。地区名など one-hot の列が多いので、既定の 1e-4 より強めにする
    "alpha": 1e-3,
    "batch_size": 512,
    "learning_rate_init": 1e-3,
    "random_state": SEED,
}
