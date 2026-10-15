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

# 直近何四半期の直線で延ばす速さを決めるか
GROWTH_QUARTERS = 4
# 指数の分け方。"city_tier"（市区町村の価格帯 5 つ）
GROUPING = "city_tier"
# 全部の取引で作った指数のキー。価格帯のない市区町村（2021 年より前に取引がない）にも、この指数を使う
OVERALL = "全体"
# 指数を築年数でも分けるときの境目（年）。None なら分けない。
# 築浅ほど速く上がる（築 10 年以内 +9.4%/年、30 年超 +4.1%/年、Issue #7）のに、
# 同じ速さで延ばすと築浅を安く、築古を高く予測しすぎる
AGE_EDGES: list[int] | None = [15, 30]

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
        city_group: 市区町村名 → グループ。ここにない市区町村は OVERALL の指数を使う。
        age_edges: 築年数で分ける境目。None なら分けない。
    """

    index_by_group: dict[str, pd.Series]
    slope: dict[str, float]
    last_quarter: int
    city_group: dict[str, str]
    age_edges: list[int] | None

    def log_index(self, rows: pd.DataFrame) -> np.ndarray:
        """行ごとに、その四半期の指数（log）を返す。学習期間より先は延ばした値。

        Args:
            rows: 経過四半期と市区町村名の列を含む行。

        Returns:
            行ごとの指数（log）。
        """
        group = _group_keys(rows, self.city_group, self.age_edges, self.index_by_group)
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
    """log(㎡単価) = 組の効果 + 四半期の効果 として、四半期の効果を指数（log）にする。

    組は 価格情報区分 × 地区名 × 面積帯 × 築年数帯。
    築年数をそろえるのは、売れる物件が年々古くなっていて、そろえないと古くなった分の値下がりが相場の下落に見えるため。
    面積帯・築年数帯の区切りは EDA 用のものを流用していて、細かさを変えても傾きはほとんど変わらない。
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

    上がる速さそのものではなく水準で分けるのは、たまたまのぶれに合わせたグループにしないため。

    Args:
        tr: 学習データ。

    Returns:
        市区町村名 → 価格帯（"1"〜"5"、5 が一番高い）。
    """
    early = tr[tr["取引年"] < TIER_BASE_YEAR]
    city = early["市区町村名"].astype(object)
    city_median = np.log(early[TARGET] / early["面積_数値"]).groupby(city).median()
    tier = pd.qcut(city.map(city_median), N_TIERS, labels=False) + 1
    return tier.groupby(city).first().astype(str).to_dict()


def _group_keys(
    rows: pd.DataFrame,
    city_group: dict[str, str],
    age_edges: list[int] | None,
    known: dict[str, pd.Series],
) -> pd.Series:
    """行ごとに、どの指数を使うか（グループのキー）を決める。

    築年数で分けるときは「価格帯・築年数帯」のキーにする。築年数が分からない行と、
    指数を作れなかった組み合わせ（known にないキー）は、価格帯だけのキーにする。

    Args:
        rows: 市区町村名と築年数の列を含む行。
        city_group: 市区町村名 → 価格帯。
        age_edges: 築年数で分ける境目。None なら分けない。
        known: 指数を作れたグループ。fit_trend の途中では、まだ空でよい。

    Returns:
        行ごとのグループのキー。
    """
    base = rows["市区町村名"].astype(object).map(city_group).fillna(OVERALL)
    if age_edges is None:
        return base
    band = pd.Series(np.digitize(rows["築年数"], age_edges), index=rows.index)
    with_age = base + "・築" + band.astype(str)
    use_age = rows["築年数"].notna()
    if known:
        use_age &= with_age.isin(known.keys())
    return with_age.where(use_age, base)


def fit_trend(tr: pd.DataFrame) -> Trend:
    """学習データから相場の指数と延ばす速さを決める。

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
    if AGE_EDGES is not None:
        age_group = _group_keys(tr, city_group, AGE_EDGES, {})
        n_quarters = tr[QUARTER].nunique()
        for g in age_group[age_group.str.contains("・築")].unique():
            quarter_index = _market_index(tr[age_group == g])
            # 件数が少なく、取引のない四半期がある組み合わせは、価格帯だけの指数を使う
            if len(quarter_index) == n_quarters:
                index_by_group[g] = quarter_index
    slope = {g: _growth_slope(qi) for g, qi in index_by_group.items()}
    return Trend(index_by_group, slope, int(tr[QUARTER].max()), city_group, AGE_EDGES)


def to_target(rows: pd.DataFrame, trend: Trend) -> pd.Series:
    """学習に使う目的変数を作る。

    総額を面積で割ることで、物件規模を除いた㎡単価を学習する。
    さらに相場の指数を引き、決定木にはその時期の相場に比べた㎡単価を予測させる。

    Args:
        rows: 価格の列を含む行（学習データ、または early stopping 用の検証データ）。
        trend: fit_trend で学習データから作った指数。

    Returns:
        目的変数（log した㎡単価から指数を引いた値）。
    """
    unit_price = rows[TARGET] / rows["面積_数値"]
    return np.log(unit_price) - trend.log_index(rows)


def to_price(pred: np.ndarray, pred_rows: pd.DataFrame, trend: Trend) -> np.ndarray:
    """モデルの予測に指数を足して㎡単価に戻し、面積を掛けて総額にする。to_target の逆変換。

    Args:
        pred: モデルの予測。
        pred_rows: 面積_数値列を含む予測対象の行。
        trend: to_target と同じ指数。

    Returns:
        円の価格。
    """
    unit_price = np.exp(pred + trend.log_index(pred_rows))
    return unit_price * pred_rows["面積_数値"].to_numpy()
