"""場所の特徴量。"""

import pandas as pd


def make(df: pd.DataFrame) -> pd.DataFrame:
    """市区町村名と地区名をつないだ地区の列を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        地区の列だけの表。
    """
    out = pd.DataFrame(index=df.index)
    # 地区名だけでは場所が決まらない（「本町」は 12 の市区町村にある）ので、市区町村名とつなぐ
    out["地区"] = (
        df["市区町村名"].astype(str) + "・" + df["地区名"].astype(str)
    ).astype("category")
    return out
