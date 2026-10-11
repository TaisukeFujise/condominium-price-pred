"""目的変数の設計。モデルに何を予測させるか。

make_target と to_price は対になっている。to_price(モデルの予測) で円の価格に戻せるようにする。

決定木は学習期間の最後の水準より先へ予測を伸ばせないので、相場の上昇分を目的変数から切り離す。
学習データから四半期ごとの相場の指数を作り（fit_trend）、目的変数は log1p(価格) から指数を引いたもの、
予測は指数を学習期間より先へ直線で延ばした値を足して戻す。
指数は価格から作るので、CV では fold ごとにその fold の学習データだけで fit_trend を呼ぶ。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import TARGET

# 延ばす速さを決める期間。"since_2022"（2022 年以降の直線）/ "last_year"（直近 4 四半期の直線）/
# None（延ばさない。学習期間の最後の水準のまま）
GROWTH = "last_year"
# 指数の分け方。"all"（全体 1 本）/ "ward_tier"（市区町村の価格帯 5 つ）
GROUPING = "ward_tier"

QUARTER = "経過四半期"
FIRST_QUARTER_2022 = 12
N_TIERS = 5
N_ITER = 30


@dataclass
class Trend:
    """学習データから作った相場の指数（log）と、学習期間より先へ延ばす速さ。

    Attributes:
        level: (グループ, 経過四半期) ごとの指数（log）。学習期間の最後の四半期を 0 とする。
        slope: グループごとの 1 四半期あたりの上がる速さ（log）。
        last_quarter: 学習期間の最後の経過四半期。
        ward_group: 市区町村名 → グループ。ここにない市区町村は "全体" の指数を使う。
    """

    level: dict[str, pd.Series]
    slope: dict[str, float]
    last_quarter: int
    ward_group: dict[str, str]

    def log_index(self, rows: pd.DataFrame) -> np.ndarray:
        """行ごとに、その四半期の指数（log）を返す。学習期間より先は延ばした値。

        Args:
            rows: 経過四半期と市区町村名の列を含む行。

        Returns:
            行ごとの指数（log）。
        """
        group = rows["市区町村名"].astype(object).map(self.ward_group).fillna("全体")
        out = np.zeros(len(rows))
        for g in group.unique():
            mask = (group == g).to_numpy()
            quarter = rows.loc[mask, QUARTER].to_numpy()
            level = self.level[g]
            ahead = np.maximum(quarter - self.last_quarter, 0)
            within = level.reindex(np.minimum(quarter, self.last_quarter)).to_numpy()
            out[mask] = within + self.slope[g] * ahead
        return out


def market_index(train: pd.DataFrame) -> pd.Series:
    """log(㎡単価) = 組の効果 + 四半期の効果 として、四半期の効果を指数（log）にする。

    組は 価格情報区分 × 地区名 × 面積帯 × 築年数帯（notebooks/fujise_detail.ipynb の Q10）。
    組ごとに含まれる期間が違う（成約価格情報は 2021 年から）ので、2 つの効果を交互に推定する。

    Args:
        train: 価格・面積・築年数・経過四半期の列を含む学習データ。

    Returns:
        経過四半期ごとの指数（log）。最後の四半期を 0 とする。
    """
    log_unit = np.log(train[TARGET] / train["面積_数値"])
    area_band = pd.cut(
        train["面積_数値"],
        [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 2000],
    )
    age_band = pd.cut(train["築年数"], range(-5, 80, 5))
    # 欠損（築年数が分からないなど）は code -1 になり、それも 1 つの組として扱う
    cell = (
        pd.DataFrame(
            {
                "kubun": train["価格情報区分"].cat.codes,
                "district": train["地区名"].cat.codes,
                "area": area_band.cat.codes,
                "age": age_band.cat.codes,
            }
        )
        .groupby(["kubun", "district", "area", "age"])
        .ngroup()
    )
    quarter = train[QUARTER]
    quarter_effect = pd.Series(0.0, index=train.index)
    for _ in range(N_ITER):
        cell_effect = (log_unit - quarter_effect).groupby(cell).transform("mean")
        quarter_effect = (log_unit - cell_effect).groupby(quarter).transform("mean")
    level = quarter_effect.groupby(quarter).first().sort_index()
    return level - level.iloc[-1]


def growth_slope(level: pd.Series) -> float:
    """指数（log）を直線で当てはめ、1 四半期あたりの上がる速さを返す。

    Args:
        level: 経過四半期ごとの指数（log）。

    Returns:
        1 四半期あたりの上がる速さ（log）。GROWTH が None なら 0。
    """
    if GROWTH is None:
        return 0.0
    if GROWTH == "since_2022":
        recent = level[level.index >= FIRST_QUARTER_2022]
    elif GROWTH == "last_year":
        recent = level[level.index > level.index.max() - 4]
    else:
        raise ValueError(GROWTH)
    return float(np.polyfit(recent.index, recent.to_numpy(), 1)[0])


def ward_tiers(train: pd.DataFrame) -> dict[str, str]:
    """市区町村を、2021 年より前の㎡単価の中央値で 5 つの価格帯に分ける。

    上がる速さそのものではなく水準で分けるのは、たまたまのぶれに合わせたグループにしないため。

    Args:
        train: 学習データ。

    Returns:
        市区町村名 → 価格帯（"1"〜"5"、5 が一番高い）。
    """
    early = train[train["取引年"] < 2021]
    ward = early["市区町村名"].astype(object)
    ward_level = np.log(early[TARGET] / early["面積_数値"]).groupby(ward).median()
    # 境目は件数で重みづけした分位点にする（notebooks/fujise_detail.ipynb の Q11 と同じ）
    tier = pd.qcut(ward.map(ward_level), N_TIERS, labels=False) + 1
    return tier.groupby(ward).first().astype(str).to_dict()


def fit_trend(train: pd.DataFrame) -> Trend:
    """学習データから相場の指数と延ばす速さを決める。

    Args:
        train: 学習データ。CV では、その fold の学習データ。

    Returns:
        make_target と to_price に渡す指数。
    """
    ward_group = ward_tiers(train) if GROUPING == "ward_tier" else {}
    if GROUPING not in ("all", "ward_tier"):
        raise ValueError(GROUPING)
    group = train["市区町村名"].astype(object).map(ward_group).fillna("全体")
    level = {"全体": market_index(train)}
    for g in set(ward_group.values()):
        level[g] = market_index(train[group == g])
    slope = {g: growth_slope(lv) for g, lv in level.items()}
    return Trend(level, slope, int(train[QUARTER].max()), ward_group)


def make_target(rows: pd.DataFrame, trend: Trend) -> pd.Series:
    """学習に使う目的変数を作る。

    評価指標は RMSLE なので、log1p(価格) を二乗誤差で学習すれば RMSLE を直接小さくできる。
    そこから相場の指数を引き、決定木には「その時期の相場に比べて高いか安いか」だけを予測させる。

    Args:
        rows: 価格の列を含む行（学習データ、または early stopping 用の検証データ）。
        trend: fit_trend で学習データから作った指数。

    Returns:
        目的変数（log1p した価格 − 指数）。
    """
    return np.log1p(rows[TARGET]) - trend.log_index(rows)


def to_price(pred: np.ndarray, rows: pd.DataFrame, trend: Trend) -> np.ndarray:
    """モデルの予測を円の価格に戻す。make_target の逆変換。

    Args:
        pred: モデルの予測。
        rows: 予測した行。
        trend: make_target と同じ指数。

    Returns:
        円の価格。
    """
    return np.expm1(pred + trend.log_index(rows))
