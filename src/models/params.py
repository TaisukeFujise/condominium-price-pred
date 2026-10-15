"""ハイパーパラメータ。モデルを足したら、<モデル名>_PARAMS をここに並べる。

LightGBM は、特徴量の実験の前に理由のあるものだけ見直した（#12）。損失関数は Huber（#6）。
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
    "seed": SEED,
    "verbose": -1,
}

XGBOOST_PARAMS = {
    # LGBM_PARAMS の huber と同じく、log 空間で 0.2 より大きいずれを線形にしか効かせない。
    # XGBoost に huber はないので、滑らかに近づけた pseudo-Huber を使う
    "objective": "reg:pseudohubererror",
    "huber_slope": 0.2,
    "eval_metric": "rmse",
    # LightGBM と同じく、ヒストグラムで分岐を探す（category 型の列を使うのにも必要）
    "tree_method": "hist",
    "learning_rate": 0.05,
    "seed": SEED,
}
