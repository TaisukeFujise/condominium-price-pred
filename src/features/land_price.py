"""2025 年版の地価公示から、地区近傍の年別地価特徴量を作る。

出典：「国土数値情報（地価公示データ）」（国土交通省）
(https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L01-2025.html )を加工して作成。
2025 年版の地価公示は 2025 年 1 月 1 日時点の情報で、取引価格データではない。
2026 年の取引には 2025 年の地価を使い、テスト期間の情報は参照しない。
"""

import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.config import EXTERNAL_DIR

LAND_PRICE_URL = "https://nlftp.mlit.go.jp/ksj/gml/data/L01/L01-25/L01-25_13_GML.zip"
LAND_PRICE_ZIP = EXTERNAL_DIR / "land_price_2025" / "L01-25_13_GML.zip"

FIRST_YEAR = 2019
LAST_YEAR = 2025
YEAR_COLUMNS = {
    year: f"L01_{98 + year - FIRST_YEAR:03d}"
    for year in range(FIRST_YEAR, LAST_YEAR + 1)
}
RESIDENTIAL_CODE = "000"
EXCLUDED_USE_CODES = {"013"}
N_NEIGHBORS = 5
MIN_DISTANCE_M = 50.0
EARTH_RADIUS_M = 6_371_000.0
MIN_LATITUDE = 35.4


def _load_land_points() -> pd.DataFrame:
    """地価公示の地点と 2019〜2025 年の価格を読み込む。

    Returns:
        用途区分、緯度、経度、年別価格の表。
    """
    if not LAND_PRICE_ZIP.exists():
        LAND_PRICE_ZIP.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LAND_PRICE_URL, LAND_PRICE_ZIP)

    with zipfile.ZipFile(LAND_PRICE_ZIP) as archive:
        geojson_name = next(
            name for name in archive.namelist() if name.endswith(".geojson")
        )
        geojson = json.loads(archive.read(geojson_name))

    columns = ["L01_002", *YEAR_COLUMNS.values()]
    records = []
    for feature in geojson["features"]:
        properties = feature["properties"]
        longitude, latitude = feature["geometry"]["coordinates"][:2]
        if latitude < MIN_LATITUDE:
            continue
        record = {
            "用途区分": str(properties["L01_002"]).zfill(3),
            "緯度": latitude,
            "経度": longitude,
        }
        record.update({column: properties[column] for column in columns[1:]})
        records.append(record)

    points = pd.DataFrame.from_records(records)
    for column in YEAR_COLUMNS.values():
        points[column] = pd.to_numeric(points[column], errors="coerce")
        points.loc[points[column] <= 0, column] = np.nan
    return points


def _nearest(
    points: pd.DataFrame,
    price: np.ndarray,
    query_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """有効な地価地点から、各地区に近い地点と距離重みを求める。

    Args:
        points: 地価地点の表。
        price: 地点ごとの当該年価格。
        query_coordinates: 地区の緯度・経度（度）。

    Returns:
        最近傍地点の行番号、距離（m）、距離重み。該当地点がない場合は None。
    """
    valid = np.isfinite(price)
    point_positions = np.flatnonzero(valid)
    if len(point_positions) == 0:
        return None

    point_coordinates = points.loc[valid, ["緯度", "経度"]].to_numpy()
    tree = BallTree(np.radians(point_coordinates), metric="haversine")
    n_neighbors = min(N_NEIGHBORS, len(point_positions))
    distance_radians, local_positions = tree.query(
        np.radians(query_coordinates), k=n_neighbors
    )
    neighbor_positions = point_positions[local_positions]
    distance_m = distance_radians * EARTH_RADIUS_M
    inverse_distance = 1 / np.maximum(distance_m, MIN_DISTANCE_M)
    weights = inverse_distance / inverse_distance.sum(axis=1, keepdims=True)
    return neighbor_positions, distance_m, weights


def _weighted_log_price(
    price: np.ndarray,
    neighbor_positions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """距離重み付きの log 地価を計算する。

    Args:
        price: 地点ごとの地価。
        neighbor_positions: 各地区に近い地点の行番号。
        weights: 地点ごとの距離重み。

    Returns:
        地区ごとの距離重み付き log 地価。
    """
    log_price = np.log(price[neighbor_positions])
    return np.sum(log_price * weights, axis=1)


def _features_for_year(
    points: pd.DataFrame,
    year: int,
    query_coordinates: np.ndarray,
) -> np.ndarray:
    """指定年の近傍地価特徴量を地区ごとに計算する。

    Args:
        points: 地価地点の表。
        year: 使用する地価の年。
        query_coordinates: 地区の緯度・経度（度）。

    Returns:
        住宅地地価、最近傍住宅地地価、距離、住宅地変動率、全用途地価の表。
    """
    price_column = YEAR_COLUMNS[year]
    price = points[price_column].to_numpy(dtype=float)
    result = np.full((len(query_coordinates), 5), np.nan)

    residential = points["用途区分"].eq(RESIDENTIAL_CODE).to_numpy()
    residential_price = price.copy()
    residential_price[~residential] = np.nan
    neighbors = _nearest(points, residential_price, query_coordinates)
    if neighbors is not None:
        positions, distance_m, weights = neighbors
        result[:, 0] = _weighted_log_price(residential_price, positions, weights)
        result[:, 1] = np.log(residential_price[positions[:, 0]])
        result[:, 2] = distance_m[:, 0]

        if year > FIRST_YEAR:
            previous_price = points[YEAR_COLUMNS[year - 1]].to_numpy(dtype=float)
            current_values = residential_price[positions]
            previous_values = previous_price[positions]
            valid_previous = np.isfinite(previous_values)
            change = np.full(previous_values.shape, np.nan)
            change[valid_previous] = (
                current_values[valid_previous] / previous_values[valid_previous] - 1
            )
            weighted = np.where(valid_previous, weights, 0.0)
            denominator = weighted.sum(axis=1)
            has_previous = denominator > 0
            result[has_previous, 3] = (
                np.nansum(change * weighted, axis=1)[has_previous]
                / denominator[has_previous]
            )

    all_uses = ~points["用途区分"].isin(EXCLUDED_USE_CODES).to_numpy()
    all_use_price = price.copy()
    all_use_price[~all_uses] = np.nan
    neighbors = _nearest(points, all_use_price, query_coordinates)
    if neighbors is not None:
        positions, _, weights = neighbors
        result[:, 4] = _weighted_log_price(all_use_price, positions, weights)

    return result


def make(df: pd.DataFrame) -> pd.DataFrame:
    """地区に近い地価公示地点から、年に対応する地価特徴量を作る。

    Args:
        df: 緯度・経度と取引年を含む前処理済みの表。

    Returns:
        近傍地価に関する 5 列。該当する地点がない行は NaN。
    """
    points = _load_land_points()
    coordinates = df[["緯度", "経度"]].astype(float)
    location_codes, unique_locations = pd.MultiIndex.from_frame(coordinates).factorize(
        sort=False
    )
    unique_coordinates = unique_locations.to_frame(index=False).to_numpy(dtype=float)
    valid_locations = np.isfinite(unique_coordinates).all(axis=1) & (
        unique_coordinates[:, 0] >= MIN_LATITUDE
    )
    location_features = np.full((len(unique_coordinates), 5), np.nan)
    query_positions = np.flatnonzero(valid_locations)
    query_coordinates = unique_coordinates[query_positions]

    trade_year = df["取引年"].to_numpy(dtype=int)
    land_year = np.clip(trade_year, FIRST_YEAR, LAST_YEAR)
    row_features = np.full((len(df), 5), np.nan)
    for year in np.unique(land_year):
        yearly_features = _features_for_year(points, int(year), query_coordinates)
        location_features[query_positions] = yearly_features
        row_positions = np.flatnonzero(land_year == year)
        row_location_codes = location_codes[row_positions]
        valid_rows = row_location_codes >= 0
        row_features[row_positions[valid_rows]] = location_features[
            row_location_codes[valid_rows]
        ]

    return pd.DataFrame(
        row_features,
        columns=[
            "地価_住宅近傍5_log",
            "地価_最近傍住宅_log",
            "地価_最近傍住宅_距離",
            "地価_住宅近傍5_変動率",
            "地価_全用途近傍5_log",
        ],
        index=df.index,
    )
