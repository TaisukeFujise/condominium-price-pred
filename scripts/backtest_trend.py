"""過去の時点ごとに、相場の指数（src/target.py）を延ばしたとき、1〜3 四半期先でどれだけ外れたかを測る。

各時点 T について、T までの学習データだけで fit_trend を作り、T から先へ延ばした指数の動きと、
全データで作った指数での T からの実際の動きを比べる。外れ = 延ばした動き − 実際の動き（プラスなら高く見積もった）。
scripts/lb_probe.py の RATIO（1 四半期先の外れに対する 2・3 四半期先の外れの比）は、この結果から出した。
test と同じ割合で 1〜3 四半期先を重み付けした外れと、その時点の傾きの関係も表示する。

モデルは学習しないので数分で終わる。

実行:
    uv run python -m scripts.backtest_trend
"""

import numpy as np
import pandas as pd

from src import preprocess, target

_MIN_QUARTERS = 5


def main() -> None:
    """時点ごとの外れを表示し、2・3 四半期先の外れと 1 四半期先の外れの比を出す。"""
    df = preprocess.load()
    train = df[df["is_train"]].reset_index(drop=True)
    test = df[~df["is_train"]].reset_index(drop=True)
    quarter = target.QUARTER
    last = int(train[quarter].max())
    full = target.fit_trend(train)
    test_share = (test[quarter] - last).value_counts(normalize=True)

    rows = []
    for t in range(int(train[quarter].min()) + _MIN_QUARTERS - 1, last - 2):
        trend = target.fit_trend(train[train[quarter] <= t])
        future = train[(train[quarter] > t) & (train[quarter] <= t + 3)]
        at_t = future.assign(**{quarter: t})
        error = (trend.log_index(future) - trend.log_index(at_t)) - (
            full.log_index(future) - full.log_index(at_t)
        )
        # 取引のない市区の区分があると指数が NaN になり、平均から黙って抜けて比が偏る
        if np.isnan(error).any():
            raise ValueError(
                f"時点 {t} で指数が NaN の行が {int(np.isnan(error).sum())} 行ある"
            )
        by_ahead = pd.Series(error).groupby((future[quarter] - t).to_numpy()).mean()
        rows.append(
            {
                "時点": t,
                "傾き": trend.slope[target.OVERALL],
                **{f"{h}四半期先": by_ahead[h] for h in (1, 2, 3)},
                "test の重み": sum(test_share[h] * by_ahead[h] for h in (1, 2, 3)),
            }
        )
    result = pd.DataFrame(rows)
    print(result.round(4).to_string(index=False))

    first = result["1四半期先"]
    for h in (2, 3):
        later = result[f"{h}四半期先"]
        ratio = float((first * later).sum() / (first**2).sum())
        print(
            f"{h} 四半期先 ≒ {ratio:.2f} × 1 四半期先（相関 {first.corr(later):.2f}）"
        )
    weighted = result["test の重み"]
    print(
        f"test の重みでの外れ：平均 {weighted.mean():+.4f}、"
        f"二乗平均の平方根 {np.sqrt((weighted**2).mean()):.4f}、"
        f"傾きとの相関 {result['傾き'].corr(weighted):.2f}。"
        f"今の傾き {full.slope[target.OVERALL]:+.4f}"
    )


if __name__ == "__main__":
    main()
