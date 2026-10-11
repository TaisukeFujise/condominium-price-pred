"""容積率の補完。

成約価格情報の行は 容積率（％） が 100% 欠損している（docs/columns.md ※1）。
容積率は用途地域ごとに都市計画で決まる値なので、同じ 地区名＋都市計画 の
不動産取引価格情報の行から最頻値を借りて埋める。価格は使わない。
"""

import pandas as pd

COL = "容積率（％）"


def make(df: pd.DataFrame) -> pd.DataFrame:
    """容積率の欠損を、同じ 地区名＋都市計画 の最頻値で埋めた列を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        容積率_補完 の列だけの表。元の値がある行は元の値のまま。
    """
    keys = ["地区名", "都市計画"]
    known = df.dropna(subset=[COL, *keys])
    mode = (
        known.groupby(keys, observed=True)[COL]
        .agg(lambda s: s.mode().iloc[0])
        .rename("mode")
    )
    filled = df[keys].join(mode, on=keys)["mode"]

    out = pd.DataFrame(index=df.index)
    out["容積率_補完"] = df[COL].fillna(filled)
    return out
