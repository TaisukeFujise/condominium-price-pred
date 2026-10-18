"""間取りから作る特徴量（価格を使わない）。

過去の不動産コンペの上位解法で使われた間取りの使い方を、候補として並べる。
間取りの解析と単身向けの判定は #30 の定義にそろえる。価格を使わないので、
fold に分ける前に train と test をまとめて作る。
"""

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

from src.config import SEED

# 先頭の数字に足す部屋数。SIGNATE 賃料予測 2019 の 3 位の換算式（1K と 1R は実質同じ広さ、
# DK は 1 部屋、LDK は 1.5 部屋、S（納戸）は 1 部屋）
_DK_ROOMS = 1.0
_L_ROOMS = 0.5
_S_ROOMS = 1.0
_LDA_TOPICS = 5


def make(df: pd.DataFrame) -> pd.DataFrame:
    """間取りから、面積との組み合わせ・世帯の型・地域ごとの分布などの列を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        間取りの特徴量の表。間取りが欠損の行は NaN。
    """
    layout = _normalize(df["間取り"])
    out = pd.DataFrame(index=df.index)
    out = out.join(_area_ratio(df, layout))
    out["一部屋あたり面積"] = (df["面積_数値"] / _converted_rooms(layout)).astype(float)
    # 「新しくて広いか」を 1 列で表す。Nishika 2023 夏の 3 位・秋の 1 位で最も効いた列
    out["面積引く築年数"] = df["面積_数値"] - df["築年数"]
    out["世帯型"] = _household_type(layout)
    out = out.join(_split_layout(layout))
    out = out.join(_lda_by_city(df, layout))
    return out


def _normalize(layout: pd.Series) -> pd.Series:
    """間取りを半角・大文字にそろえる（#30 と同じ）。

    Args:
        layout: 元の間取りの列。

    Returns:
        正規化した間取り。欠損は <NA>。
    """
    return layout.astype("string").str.normalize("NFKC").str.upper()


def _leading_rooms(layout: pd.Series) -> pd.Series:
    """間取りの先頭の数字（ワンルームは 1）を取り出す（#30 と同じ）。

    Args:
        layout: 正規化した間取り。

    Returns:
        先頭の部屋数。数字で始まらない間取り（オープンフロアなど）は NaN。
    """
    rooms = pd.to_numeric(
        layout.str.extract(r"^\s*(\d+)", expand=False), errors="coerce"
    )
    return rooms.mask(layout.eq("ワンルーム"), 1)


def _area_ratio(df: pd.DataFrame, layout: pd.Series) -> pd.DataFrame:
    """同じ間取りの平均面積に対する、その行の面積の比を作る。

    Args:
        df: 前処理済みの表。
        layout: 正規化した間取り。

    Returns:
        間取りごと、市区町村×間取りごとの平均面積との比の表。
    """
    out = pd.DataFrame(index=df.index)
    area = df["面積_数値"]
    for name, keys in {
        "間取り": [layout],
        "市区町村間取り": [df["市区町村名"], layout],
    }.items():
        mean = area.groupby(keys, observed=True).transform("mean")
        out[f"間取り面積比_{name}"] = area / mean
    return out


def _converted_rooms(layout: pd.Series) -> pd.Series:
    """SIGNATE 3 位の換算式で、間取りを部屋数に直す（例: 3LDK は 4.5 部屋）。

    Args:
        layout: 正規化した間取り。

    Returns:
        換算した部屋数。数字で始まらない間取りは NaN。
    """
    rest = layout.str.replace(r"^\s*\d+", "", regex=True)
    has_dk = rest.str.contains("DK", regex=False)
    has_l = rest.str.contains("L", regex=False)
    has_s = rest.str.contains("S", regex=False)
    added = (
        has_dk.astype(float) * _DK_ROOMS
        + has_l.astype(float) * _L_ROOMS
        + has_s.astype(float) * _S_ROOMS
    )
    return _leading_rooms(layout) + added


def _household_type(layout: pd.Series) -> pd.Series:
    """間取りを単身用・新婚用・家族用に分ける（SIGNATE 3 位の区分）。

    Args:
        layout: 正規化した間取り。

    Returns:
        世帯の型のカテゴリ。数字で始まらない間取りと欠損は NaN。
    """
    rooms = _leading_rooms(layout)
    single = layout.str.contains(r"^(?:ワンルーム|1(?:R|K|DK))(?:\+.*)?$", na=False)
    newlywed = layout.str.contains(r"^(?:1LDK|2K|2DK)(?:\+.*)?$", na=False)
    household = pd.Series(pd.NA, index=layout.index, dtype="object")
    household[rooms.notna()] = "家族"
    household[newlywed] = "新婚"
    household[single] = "単身"
    return household.astype("category")


def _split_layout(layout: pd.Series) -> pd.DataFrame:
    """間取りを部屋数・LDK の有無・単身向けに分ける（#30 と同じ列）。

    Args:
        layout: 正規化した間取り。

    Returns:
        #30 の 3 列の表。
    """
    out = pd.DataFrame(index=layout.index)
    out["間取り分解_部屋数"] = _leading_rooms(layout).astype(float)
    out["間取り分解_LDKあり"] = layout.str.contains(
        r"(?:LDK|DK)", regex=True, na=False
    ).astype("int8")
    out["間取り分解_単身向け"] = layout.str.contains(
        r"^(?:ワンルーム|1(?:R|K|DK|LDK))(?:\+.*)?$", regex=True, na=False
    ).astype("int8")
    return out


def _lda_by_city(df: pd.DataFrame, layout: pd.Series) -> pd.DataFrame:
    """市区町村を文書、間取りを単語とみなした LDA で、市区町村の間取りの構成を 5 次元にする。

    Nishika 中古マンション価格予測 2022 春の 3 位の方法。単純な割合（単身向けの割合など）は
    地域の平均面積とほぼ同じ情報だったので、間取りの組み合わせの違いまで拾わせる。

    Args:
        df: 前処理済みの表。
        layout: 正規化した間取り。

    Returns:
        市区町村ごとのトピックの割合を、各行に付けた表。
    """
    counts = pd.crosstab(df["市区町村名"], layout)
    lda = LatentDirichletAllocation(n_components=_LDA_TOPICS, random_state=SEED)
    topics = pd.DataFrame(
        lda.fit_transform(counts),
        index=counts.index,
        columns=[f"間取りLDA_{i}" for i in range(_LDA_TOPICS)],
    )
    return df[["市区町村名"]].join(topics, on="市区町村名").drop(columns="市区町村名")
