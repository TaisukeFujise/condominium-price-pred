"""主要駅（業務の中心地）までの距離。地区の緯度・経度（district_latlon.py）から計算する。

主要駅の緯度・経度は、notebooks/fujise_detail.ipynb の値を書いた:
  - 東京: Q13 の TOKYO_STATION
  - 新宿・渋谷・池袋: Q14 の「国土数値情報（駅別乗降客数データ）」（国土交通省）
    （2024 年度版、S12-25）を加工して作成。https://nlftp.mlit.go.jp/ksj/other/agreement.html
  - 品川・六本木・虎ノ門: ノートブックに値がないので、district_latlon.py と同じ位置参照情報
    （2023 年版（17.0b））の、駅がある丁目の代表点（高輪三丁目・六本木六丁目・虎ノ門一丁目）を使った
"""

import numpy as np
import pandas as pd

STATION_POINTS = {
    "東京": (35.681236, 139.767125),
    "新宿": (35.689796, 139.699930),
    "渋谷": (35.658977, 139.701647),
    "池袋": (35.729807, 139.711077),
    "品川": (35.632106, 139.737307),
    "六本木": (35.660329, 139.730276),
    "虎ノ門": (35.668580, 139.749508),
}

# 価格に合わせて選ぶと train に合わせすぎるので、価格を見ずに理由で決めた候補 3 つから選んだ（Issue #10 の候補 3）
MAJOR_STATIONS = ["東京", "新宿", "渋谷", "品川", "六本木", "虎ノ門"]
# True なら一番近い主要駅までの距離 1 列、False なら主要駅ごとの距離
NEAREST_ONLY = False

# 東京都の範囲なら、緯度 1 度・経度 1 度を一定の km とみなしても誤差は小さい
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(35.68))


def make(df: pd.DataFrame) -> pd.DataFrame:
    """地区の緯度・経度から、主要駅までの距離（km）の列を作る。

    Args:
        df: 緯度・経度の列を足した表。

    Returns:
        距離の列の表。緯度・経度が欠損の行は NaN。
    """
    dist = pd.DataFrame(index=df.index)
    for name in MAJOR_STATIONS:
        lat, lon = STATION_POINTS[name]
        dist[f"{name}駅までの距離_km"] = np.hypot(
            (df["緯度"] - lat) * KM_PER_DEG_LAT, (df["経度"] - lon) * KM_PER_DEG_LON
        )
    if NEAREST_ONLY:
        # skipna=False: 緯度・経度が欠損の行を 0 km にしない
        return dist.min(axis=1, skipna=False).to_frame("一番近い主要駅までの距離_km")
    return dist
