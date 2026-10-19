"""目的変数の設計（工程 4）。モデルに何を予測させるか。

目的変数は「log(㎡単価) − 相場の指数」（per_area=False のモデルは「log(総額) − 相場の指数」）。
次の 3 つの関数で作り、円に戻す。
- fit_trend：学習データから、四半期ごとの相場の指数を作る。
- to_target：価格を目的変数にする。
- to_price：モデルの予測を円の価格に戻す（to_target の逆）。

相場の指数を引くのは、決定木が、学習期間の最後の水準より先へ予測を伸ばせないため。
学習期間より先の四半期の指数は、直近の上がる速さで直線に延ばす。

指数は価格から作るので、CV では fold ごとに、その fold の学習データだけで fit_trend を呼ぶ。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import TARGET

# 直近何四半期の直線で延ばす速さを決めるか
GROWTH_QUARTERS = 4
# 指数の分け方。"city_tier"（市区町村の価格帯 5 つ）
GROUPING = "city_tier"
# 全部の取引で作った指数のキー。価格帯のない市区町村（2021 年より前に取引がない）にも、この指数を使う
OVERALL = "全体"

QUARTER = "経過四半期"
N_TIERS = 5
# 市区町村の価格帯を分ける基準年（この年より前の取引の㎡単価で分ける）
TIER_BASE_YEAR = 2021
N_ITER = 30


@dataclass
class Trend:
    """学習データから作った相場の指数（log）と、学習期間より先へ延ばす速さ。

    Attributes:
        index_by_group: (グループ, 経過四半期) ごとの指数（log）。学習期間の最後の四半期を 0 とする。
        slope: グループごとの 1 四半期あたりの上がる速さ（log）。
        last_quarter: 学習期間の最後の経過四半期。
        city_group: 市区町村名 → グループ（価格帯）。ここにない市区町村は OVERALL の指数を使う。
    """

    index_by_group: dict[str, pd.Series]
    slope: dict[str, float]
    last_quarter: int
    city_group: dict[str, str]

    def log_index(self, rows: pd.DataFrame) -> np.ndarray:
        """行ごとに、取引した四半期の指数（log）を返す。学習期間より先の四半期は、直線で延ばした値を返す。

        Args:
            rows: 経過四半期と市区町村名の列を含む行。

        Returns:
            行ごとの指数（log）。
        """
        group = rows["市区町村名"].astype(object).map(self.city_group).fillna(OVERALL)
        out = np.zeros(len(rows))
        for g in group.unique():
            mask = (group == g).to_numpy()
            quarter = rows[QUARTER].to_numpy()[mask]
            quarter_index = self.index_by_group[g]
            quarters_ahead = np.maximum(quarter - self.last_quarter, 0)
            base_index = quarter_index.reindex(
                np.minimum(quarter, self.last_quarter)
            ).to_numpy()
            out[mask] = base_index + self.slope[g] * quarters_ahead
        return out


def _market_index(tr: pd.DataFrame) -> pd.Series:
    """log(㎡単価) = 組の効果 + 四半期の効果 とみなし、四半期の効果を相場の指数（log）にする。

    組は、価格情報区分 × 地区名 × 面積帯 × 築年数帯。
    築年数帯で分けるのは、売れる物件が年々古くなっているため。分けないと、古くなった分の値下がりが相場の下落に見える。
    面積帯・築年数帯の区切りは、細かさを変えても上がる速さがほとんど変わらない。

    組ごとに取引のある期間が違う（成約価格情報は 2021 年から）ので、2 つの効果を N_ITER 回、交互に推定する。

    Args:
        tr: 価格・面積・築年数・経過四半期の列を含む学習データ。

    Returns:
        経過四半期ごとの指数（log）。最後の四半期を 0 とする。
    """
    # np.log(Series) は Series を返すが、pandas-stubs では ndarray になる
    log_unit: pd.Series = np.log(tr[TARGET] / tr["面積_数値"])  # type: ignore[assignment]
    area_band = pd.cut(
        tr["面積_数値"],
        [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 2000],
    )
    age_band = pd.cut(tr["築年数"], range(-5, 80, 5))
    # 欠損（築年数が分からないなど）は code -1 になり、それも 1 つの組として扱う
    segment = (
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
        segment_effect = (log_unit - quarter_effect).groupby(segment).transform("mean")
        quarter_effect = (log_unit - segment_effect).groupby(quarter).transform("mean")
    quarter_index = quarter_effect.groupby(quarter).first().sort_index()
    return quarter_index - quarter_index.iloc[-1]


def _growth_slope(quarter_index: pd.Series) -> float:
    """指数（log）を直線で当てはめ、1 四半期あたりの上がる速さを返す。

    Args:
        quarter_index: 経過四半期ごとの指数（log）。

    Returns:
        1 四半期あたりの上がる速さ（log）。
    """
    recent = quarter_index[
        quarter_index.index > quarter_index.index.max() - GROWTH_QUARTERS
    ]
    return float(np.polyfit(recent.index, recent.to_numpy(), 1)[0])


def _city_tiers(tr: pd.DataFrame) -> dict[str, str]:
    """市区町村を、2021 年より前の㎡単価の中央値で 5 つの価格帯に分ける。

    上がる速さではなく、価格の水準で分ける。上がる速さで分けると、たまたまのぶれに合わせたグループになるため。

    Args:
        tr: 学習データ。

    Returns:
        市区町村名 → 価格帯（"1"〜"5"、5 が一番高い）。
    """
    early = tr[tr["取引年"] < TIER_BASE_YEAR]
    city = early["市区町村名"].astype(object)
    # np.log(Series) は Series を返すが、pandas-stubs では ndarray になる
    city_median = np.log(early[TARGET] / early["面積_数値"]).groupby(city).median()  # type: ignore[attr-defined]
    tier = pd.qcut(city.map(city_median), N_TIERS, labels=False) + 1
    # キーは市区町村名（str）だが、pandas-stubs では to_dict() のキーが Hashable になる
    return tier.groupby(city).first().astype(str).to_dict()  # type: ignore[return-value]


def fit_trend(tr: pd.DataFrame) -> Trend:
    """学習データから、価格帯ごとと全体の相場の指数と、学習期間より先へ延ばす速さを作る。

    Args:
        tr: 学習データ。CV では、その fold の学習データ。

    Returns:
        to_target と to_price に渡す指数。
    """
    city_group = _city_tiers(tr) if GROUPING == "city_tier" else {}
    if GROUPING not in ("all", "city_tier"):
        raise ValueError(GROUPING)
    group = tr["市区町村名"].astype(object).map(city_group)
    index_by_group = {OVERALL: _market_index(tr)}
    for g in set(city_group.values()):
        index_by_group[g] = _market_index(tr[group == g])
    slope = {g: _growth_slope(qi) for g, qi in index_by_group.items()}
    return Trend(index_by_group, slope, int(tr[QUARTER].max()), city_group)


def to_target(rows: pd.DataFrame, trend: Trend, per_area: bool = True) -> pd.Series:
    """学習に使う目的変数を作る。

    per_area=True（既定）では「log(㎡単価) − 相場の指数」にする。
    面積で割るのは、物件の広さの違いを除くため。指数を引くのは、その時期の相場に比べた㎡単価を予測させるため。

    per_area=False では面積で割らず、「log(総額) − 相場の指数」にする。
    ㎡単価のモデルと間違え方が違うので、アンサンブルに混ぜるために使う。

    Args:
        rows: 価格の列を含む行（学習データ、または early stopping 用の検証データ）。
        trend: fit_trend で学習データから作った指数。
        per_area: True なら㎡単価、False なら総額を目的変数にする。

    Returns:
        目的変数（log(㎡単価) または log(総額) から、相場の指数を引いた値）。
    """
    price = rows[TARGET] / rows["面積_数値"] if per_area else rows[TARGET]
    return np.log(price) - trend.log_index(rows)


def to_price(
    pred: np.ndarray, pred_rows: pd.DataFrame, trend: Trend, per_area: bool = True
) -> np.ndarray:
    """モデルの予測に指数を足して、円の価格に戻す。to_target の逆変換。

    Args:
        pred: モデルの予測。
        pred_rows: 予測した行。面積_数値・経過四半期・市区町村名の列が必要。
        trend: to_target と同じ指数。
        per_area: to_target と同じ値。False なら面積を掛けない。

    Returns:
        円の価格。
    """
    price = np.exp(pred + trend.log_index(pred_rows))
    return price * pred_rows["面積_数値"].to_numpy() if per_area else price
