"""生データ（train/test）からモデル入力用の特徴量行列を作る。

欠損値はLightGBMに直接扱わせるため埋めない（`価格情報区分`依存の構造的欠損も含む）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from my_project.parsing import (
    normalize_place_name,
    parse_area,
    parse_built_year,
    parse_layout,
    parse_multi_category,
    parse_quarter,
    parse_station_distance,
)

# 出典: 国土交通省 国土数値情報「駅別乗降客数データ」(S12, 2024年度版, CC BY 4.0)。
# my_project/build_station_features.py で加工。取得日2026-10-09。
STATION_FEATURES_PATH = Path(__file__).resolve().parent / "external_data" / "station_features.csv"

# 表記ゆれ（ケ/ヶ等）を正規化してからカテゴリ化する列。正規化しないと同じ場所が別カテゴリに分裂する
# （例: 地区名の「緑ケ丘」「緑ヶ丘」、docs/experiments.mdの品質チェック参照）
PLACE_NAME_COLUMNS = ["地区名", "最寄駅：名称"]

# 価格情報区分で欠損パターンが決まる列（※1, docs/columns.md）も含めてそのままカテゴリに入れる。
# `建物の構造`・`用途`は複合値（`ＳＲＣ、ＲＣ`等）をフラグに分解するのでここには含めない
CATEGORICAL_COLUMNS = [
    "価格情報区分",
    "市区町村名",
    "地区名",
    "最寄駅：名称",
    "今後の利用目的",
    "都市計画",
    "改装",
    "取引の事情等",
]

NUMERIC_PASSTHROUGH_COLUMNS = ["建ぺい率（％）", "容積率（％）"]

STRUCTURE_CATEGORIES = ["ＲＣ", "ＳＲＣ", "鉄骨造", "軽量鉄骨造", "木造", "ブロック造"]
USAGE_CATEGORIES = ["住宅", "事務所", "店舗", "工場", "倉庫", "駐車場", "作業場", "その他"]


def _normalize_place_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in PLACE_NAME_COLUMNS:
        df[col] = df[col].map(normalize_place_name)
    return df


@lru_cache(maxsize=1)
def _load_station_features() -> pd.DataFrame:
    return pd.read_csv(STATION_FEATURES_PATH).set_index("station_name")


def _join_station_features(station_names: pd.Series) -> pd.DataFrame:
    station_feats = _load_station_features()
    keys = station_names.map(normalize_place_name)
    joined = station_feats.reindex(keys)
    joined.index = station_names.index
    return joined


def fit_categories(df: pd.DataFrame, columns: list[str] = CATEGORICAL_COLUMNS) -> dict[str, pd.Index]:
    """学習データからカテゴリの集合を固定する（train/testでcategory codeを揃えるため）。"""
    df = _normalize_place_columns(df)
    return {col: pd.Index(df[col].dropna().unique()) for col in columns}


def _apply_categorical(df: pd.DataFrame, columns: list[str], categories: dict[str, pd.Index] | None) -> pd.DataFrame:
    out = {}
    for col in columns:
        cats = categories[col] if categories is not None else None
        out[col] = pd.Categorical(df[col], categories=cats)
    return pd.DataFrame(out, index=df.index)


def build_features(df: pd.DataFrame, categories: dict[str, pd.Index] | None = None) -> pd.DataFrame:
    df = _normalize_place_columns(df)
    out = pd.DataFrame(index=df.index)

    out["area_sqm"] = parse_area(df["面積（㎡）"])

    built_year = parse_built_year(df["建築年"])
    out["built_year"] = built_year["built_year"]
    out["is_prewar"] = built_year["is_prewar"]

    out["station_distance_min"] = parse_station_distance(df["最寄駅：距離（分）"])

    quarter = parse_quarter(df["取引時期"])
    out["txn_year"] = quarter["year"]
    out["txn_quarter"] = quarter["quarter"]
    out["txn_quarter_index"] = quarter["quarter_index"]

    # 取引時点での築年数（戦前物件や建築年欠損はNaNのまま）
    out["building_age"] = out["txn_year"] - out["built_year"]

    layout = parse_layout(df["間取り"])
    out = pd.concat([out, layout], axis=1)

    structure = parse_multi_category(df["建物の構造"], STRUCTURE_CATEGORIES, prefix="structure")
    out = pd.concat([out, structure], axis=1)

    usage = parse_multi_category(df["用途"], USAGE_CATEGORIES, prefix="usage")
    out = pd.concat([out, usage], axis=1)

    station_feats = _join_station_features(df["最寄駅：名称"])
    out = pd.concat([out, station_feats], axis=1)

    for col in NUMERIC_PASSTHROUGH_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    cat_df = _apply_categorical(df, CATEGORICAL_COLUMNS, categories)
    out = pd.concat([out, cat_df], axis=1)

    return out
