"""文字列カラムの数値化・分解ロジック（前処理・EDA共通）。

対象: `面積（㎡）`, `建築年`, `最寄駅：距離（分）`, `取引時期`, `間取り`。
値は元データの表記のまま（全角・範囲表記・「戦前」等）なので、ここで数値/フラグに変換する。
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

STATION_DISTANCE_RANGE_MINUTES = {
    "30分～60分": 45,
    "1H～1H30": 75,
    "1H30～2H": 105,
    "2H～": 130,
}

_LAYOUT_SPECIAL = {"オープンフロア", "スタジオ", "メゾネット"}
_LAYOUT_PATTERN = re.compile(r"^(\d+)([LDKSR+]*)$")


def to_halfwidth(series: pd.Series) -> pd.Series:
    return series.map(lambda s: s if pd.isna(s) else unicodedata.normalize("NFKC", s))


def normalize_place_name(name) -> str | None:
    """駅名・地区名の表記ゆれを吸収する（外部データとの結合キー正規化、カテゴリの重複統合にも使う）。

    - NFKC正規化: 互換漢字（例: 塚の異体字）や全角差を統一
    - 末尾の`(東京)`等の括弧（同名駅の disambiguation）を除去
    - `ケ`/`ヶ`の表記ゆれ統一（ひばりケ丘 → ひばりヶ丘 等）
    """
    if pd.isna(name):
        return None
    name = unicodedata.normalize("NFKC", str(name))
    name = re.sub(r"\([^)]*\)$", "", name)
    name = name.replace("ケ", "ヶ")
    return name


def parse_area(series: pd.Series) -> pd.Series:
    """`面積（㎡）`: カンマ除去して先頭の数値を取り出す（`2,000㎡以上`等にも対応）。"""
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    cleaned = cleaned.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_built_year(series: pd.Series) -> pd.DataFrame:
    """`建築年`: `YYYY年`→西暦。`戦前`は年が特定できないので built_year はNaNのまま、is_prewarフラグで情報を残す。"""
    text = series.astype(str)
    year = pd.to_numeric(text.str.extract(r"(\d{4})年", expand=False), errors="coerce")
    is_prewar = text.str.strip() == "戦前"
    return pd.DataFrame({"built_year": year, "is_prewar": is_prewar})


def parse_station_distance(series: pd.Series) -> pd.Series:
    """`最寄駅：距離（分）`: 数字はそのまま、範囲表記は代表値（分）に変換する簡易換算。"""

    def convert(value):
        if pd.isna(value):
            return np.nan
        value = str(value)
        if value in STATION_DISTANCE_RANGE_MINUTES:
            return float(STATION_DISTANCE_RANGE_MINUTES[value])
        if re.fullmatch(r"\d+", value):
            return float(value)
        return np.nan

    return series.map(convert)


def parse_quarter(series: pd.Series) -> pd.DataFrame:
    """`取引時期`: `YYYY年第N四半期` → year, quarter, 通し番号 quarter_index（= year*4+quarter）。"""
    extracted = series.astype(str).str.extract(r"(\d{4})年第(\d)四半期")
    year = pd.to_numeric(extracted[0], errors="coerce")
    quarter = pd.to_numeric(extracted[1], errors="coerce")
    quarter_index = year * 4 + quarter
    return pd.DataFrame({"year": year, "quarter": quarter, "quarter_index": quarter_index})


def parse_multi_category(series: pd.Series, categories: list[str], prefix: str) -> pd.DataFrame:
    """`、`区切りの複合カテゴリ値を複数のフラグ列に分解する（`建物の構造`の`ＳＲＣ、ＲＣ`等）。

    欠損は全フラグ0にする（`間取り`分解と同じ扱い）。
    """
    tokens = series.map(lambda v: set(str(v).split("、")) if pd.notna(v) else set())
    data = {f"{prefix}_{cat}": tokens.map(lambda s, c=cat: int(c in s)) for cat in categories}
    return pd.DataFrame(data, index=series.index)


def parse_layout(series: pd.Series) -> pd.DataFrame:
    """`間取り`: 部屋数とL/D/K/Sの有無に分解する（全角→半角正規化してから解析）。"""
    normalized = to_halfwidth(series)

    def parse_one(value):
        if pd.isna(value):
            return (np.nan, 0, 0, 0, 0, False)
        if value in _LAYOUT_SPECIAL:
            return (np.nan, 0, 0, 0, 0, True)
        match = _LAYOUT_PATTERN.match(value)
        if not match:
            return (np.nan, 0, 0, 0, 0, False)
        rooms = int(match.group(1))
        letters = match.group(2)
        return (
            rooms,
            int("L" in letters),
            int("D" in letters),
            int("K" in letters),
            int("S" in letters),
            False,
        )

    records = normalized.map(parse_one)
    return pd.DataFrame(
        records.tolist(),
        index=series.index,
        columns=[
            "layout_rooms",
            "layout_has_L",
            "layout_has_D",
            "layout_has_K",
            "layout_has_S",
            "layout_is_special",
        ],
    )
