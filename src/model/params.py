"""ハイパーパラメータ。特徴量の実験の前に、理由のあるものだけ見直した（#12）。"""

from src.config import SEED

PARAMS = {
    "objective": "regression",
    # 目的変数は log1p(価格) なので、rmse がそのまま RMSLE になる
    "metric": "rmse",
    # 地区名・駅のように種類の多いカテゴリで、件数の少ない値に合わせすぎないよう、
    # 値ごとの平均を全体の平均に強めに寄せる（デフォルト 10）。30〜100 で同じくらい良く、300 では悪くなる
    "cat_smooth": 50,
    # デフォルトの 0.1 より小さくして木を増やす。学習時間はおよそ 2 倍になる
    "learning_rate": 0.05,
    "seed": SEED,
    "verbose": -1,
}
