"""駅別乗降客数データから、駅名ごとの特徴量テーブルを作る（前処理とは別に1回だけ実行するバッチ）。

出典: 国土交通省 国土数値情報「駅別乗降客数データ」（S12, 2024年度版, CC BY 4.0）
      https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-2024.html
      取得日: 2026-10-09
      生データ: external_data/raw/S12-25_NumberOfPassengers.geojson（gitignore対象、再ダウンロードで再現可）
      加工結果: my_project/external_data/station_features.csv（コミット対象、小さいので）
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from my_project.parsing import normalize_place_name

RAW_GEOJSON = (
    Path(__file__).resolve().parent.parent
    / "external_data" / "raw" / "S12-25_NumberOfPassengers.geojson"
)
OUT_CSV = Path(__file__).resolve().parent / "external_data" / "station_features.csv"

# 東京都のおおよそのbounding box。同名駅が他県にもあるため混同を避けるフィルタ
TOKYO_BBOX = {"lat_min": 35.45, "lat_max": 35.95, "lon_min": 138.85, "lon_max": 139.95}

# 新しい年度から順に探索し、最初に値がある年を採用する（年度別コードはS12-2024のドキュメント参照）
RIDERSHIP_FIELD_BY_YEAR = {
    2024: "S12_061", 2023: "S12_057", 2022: "S12_053", 2021: "S12_049",
    2020: "S12_045", 2019: "S12_041", 2018: "S12_037", 2017: "S12_033",
    2016: "S12_029", 2015: "S12_025", 2014: "S12_021", 2013: "S12_017",
    2012: "S12_013", 2011: "S12_009",
}


def _centroid(geometry: dict) -> tuple[float, float] | None:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "LineString":
        points = coords
    elif geom_type == "Point":
        points = [coords]
    else:
        return None
    if not points:
        return None
    lats = [p[1] for p in points]
    lons = [p[0] for p in points]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _latest_ridership(props: dict) -> tuple[float | None, int | None]:
    for year, field in RIDERSHIP_FIELD_BY_YEAR.items():
        value = props.get(field)
        if value is not None:
            return float(value), year
    return None, None


def _load_station_records() -> pd.DataFrame:
    with open(RAW_GEOJSON, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for feature in data["features"]:
        props = feature["properties"]
        centroid = _centroid(feature["geometry"])
        if centroid is None:
            continue
        lat, lon = centroid
        if not (
            TOKYO_BBOX["lat_min"] <= lat <= TOKYO_BBOX["lat_max"]
            and TOKYO_BBOX["lon_min"] <= lon <= TOKYO_BBOX["lon_max"]
        ):
            continue
        ridership, year = _latest_ridership(props)
        records.append({
            "station_name": props.get("S12_001"),
            "operator": props.get("S12_002"),
            "line": props.get("S12_003"),
            "lat": lat,
            "lon": lon,
            "ridership": ridership,
            "ridership_year": year,
        })
    return pd.DataFrame(records)


def build_station_features() -> pd.DataFrame:
    """駅名で集約: 座標は平均、乗降客数は路線・事業者をまたいで合計（駅全体の利用者数の目安）。

    駅名は取引データ側（`最寄駅：名称`）との結合キーに合わせて正規化する。
    """
    records = _load_station_records().dropna(subset=["station_name"])
    records["station_name"] = records["station_name"].map(normalize_place_name)

    grouped = records.groupby("station_name").agg(
        station_lat=("lat", "mean"),
        station_lon=("lon", "mean"),
        station_ridership_total=("ridership", lambda s: s.fillna(0).sum()),
        station_line_count=("line", "nunique"),
        station_operator_count=("operator", "nunique"),
    ).reset_index()
    return grouped


if __name__ == "__main__":
    features = build_station_features()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_CSV, index=False)
    print(f"saved: {OUT_CSV} ({len(features)}駅、東京都bbox内)")
    print(features.sort_values("station_ridership_total", ascending=False).head(15))
