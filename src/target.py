"""目的変数の設計。モデルに何を予測させるか。

to_target と to_price は対になっている。to_price(モデルの予測) で円の価格に戻せるようにする。

決定木は学習期間の最後の水準より先へ予測を伸ばせないので、相場の上昇分を目的変数から切り離す。
学習データから四半期ごとの相場の指数を作り（fit_trend）、目的変数は log(㎡単価) から指数を引いたもの、
予測は指数を学習期間より先へ直線で延ばした値を足して戻す。
指数は価格から作るので、CV では fold ごとにその fold の学習データだけで fit_trend を呼ぶ。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import TARGET

# 延ばす速さを決める期間。"last_year"（直近 4 四半期の直線）
GROWTH = "last_year"
# 指数の分け方。"ward_tier"（市区町村の価格帯 5 つ）
GROUPING = "ward_tier"

QUARTER = "経過四半期"
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
            quarter = rows[QUARTER].to_numpy()[mask]
            level = self.level[g]
            ahead = np.maximum(quarter - self.last_quarter, 0)
            within = (
                level.reindex(np.minimum(quarter, self.last_quarter))
                .ffill()
                .bfill()
                .to_numpy()
            )
            out[mask] = within + self.slope[g] * ahead
        return out


def _market_index(tr: pd.DataFrame) -> pd.Series:
    """log(㎡単価) = 組の効果 + 四半期の効果 として、四半期の効果を指数（log）にする。

    組は 価格情報区分 × 地区名 × 面積帯 × 築年数帯。
    組ごとに含まれる期間が違う（成約価格情報は 2021 年から）ので、2 つの効果を交互に推定する。

    Args:
        tr: 価格・面積・築年数・経過四半期の列を含む学習データ。

    Returns:
        経過四半期ごとの指数（log）。最後の四半期を 0 とする。
    """
    log_unit = np.log(tr[TARGET] / tr["面積_数値"])
    area_band = pd.cut(
        tr["面積_数値"],
        [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 2000],
    )
    age_band = pd.cut(tr["築年数"], range(-5, 80, 5))
    # 欠損（築年数が分からないなど）は code -1 になり、それも 1 つの組として扱う
    cell = (
        pd.DataFrame(
            {
                "kubun": tr["価格情報区分"].cat.codes,
                "district": tr["地区名"].cat.codes,
                "area": area_band.cat.codes,
                "age": age_band.cat.codes,
            }
        )
        .groupby(["kubun", "district", "area", "age"])
        .ngroup()
    )
    quarter = tr[QUARTER]
    quarter_effect = pd.Series(0.0, index=tr.index)
    for _ in range(N_ITER):
        cell_effect = (log_unit - quarter_effect).groupby(cell).transform("mean")
        quarter_effect = (log_unit - cell_effect).groupby(quarter).transform("mean")
    level = quarter_effect.groupby(quarter).first().sort_index()
    return level - level.iloc[-1]


def _growth_slope(level: pd.Series) -> float:
    """指数（log）を直線で当てはめ、1 四半期あたりの上がる速さを返す。

    Args:
        level: 経過四半期ごとの指数（log）。

    Returns:
        1 四半期あたりの上がる速さ（log）。
    """
    if GROWTH == "last_year":
        recent = level[level.index > level.index.max() - 4]
    else:
        raise ValueError(GROWTH)
    return float(np.polyfit(recent.index, recent.to_numpy(), 1)[0])


def _ward_tiers(tr: pd.DataFrame) -> dict[str, str]:
    """市区町村を、2021 年より前の㎡単価の中央値で 5 つの価格帯に分ける。

    上がる速さそのものではなく水準で分けるのは、たまたまのぶれに合わせたグループにしないため。

    Args:
        tr: 学習データ。

    Returns:
        市区町村名 → 価格帯（"1"〜"5"、5 が一番高い）。
    """
    early = tr[tr["取引年"] < 2021]
    ward = early["市区町村名"].astype(object)
    ward_level = np.log(early[TARGET] / early["面積_数値"]).groupby(ward).median()
    tier = pd.qcut(ward.map(ward_level), N_TIERS, labels=False) + 1
    return tier.groupby(ward).first().astype(str).to_dict()


def fit_trend(tr: pd.DataFrame) -> Trend:
    """学習データから相場の指数と延ばす速さを決める。

    Args:
        tr: 学習データ。CV では、その fold の学習データ。

    Returns:
        to_target と to_price に渡す指数。
    """
    ward_group = _ward_tiers(tr) if GROUPING == "ward_tier" else {}
    if GROUPING not in ("all", "ward_tier"):
        raise ValueError(GROUPING)
    group = tr["市区町村名"].astype(object).map(ward_group).fillna("全体")
    level = {"全体": _market_index(tr)}
    for g in set(ward_group.values()):
        level[g] = _market_index(tr[group == g])
    slope = {g: _growth_slope(lv) for g, lv in level.items()}
    return Trend(level, slope, int(tr[QUARTER].max()), ward_group)


def to_target(rows: pd.DataFrame, trend: Trend | None = None) -> pd.Series:
    """学習に使う目的変数を作る。

    総額を面積で割ることで、物件規模を除いた㎡単価を学習する。
    trend が渡された場合、相場の指数を引き、決定木にはその時期の相場に比べた㎡単価を予測させる。

    Args:
        rows: 価格の列を含む行（学習データ、または early stopping 用の検証データ）。
        trend: fit_trend で学習データから作った指数。

    Returns:
        目的変数（log した㎡単価、または指数を引いた値）。
    """
    unit_price = rows[TARGET] / rows["面積_数値"]
    log_unit = np.log(unit_price)
    if trend is not None:
        return log_unit - trend.log_index(rows)
    return log_unit


def to_price(
    pred: np.ndarray, pred_rows: pd.DataFrame, trend: Trend | None = None
) -> np.ndarray:
    """モデルの予測した㎡単価に面積を掛けて総額に戻す。to_target の逆変換。

    Args:
        pred: モデルの予測。
        pred_rows: 面積_数値列を含む予測対象の行。
        trend: to_target と同じ指数。

    Returns:
        円の価格。
    """
    pred_log = pred
    if trend is not None:
        pred_log = pred + trend.log_index(pred_rows)
    unit_price = np.exp(pred_log)
    return unit_price * pred_rows["面積_数値"].to_numpy()
