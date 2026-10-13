"""間取りから部屋数と用途の特徴量を作る。"""

import pandas as pd


def make(df: pd.DataFrame) -> pd.DataFrame:
    """間取りを部屋数、LDK有無、単身向けに分解する。

    Args:
        df: 間取り列を含む前処理済みの表。

    Returns:
        間取りから作った特徴量だけの表。
    """
    layout = df["間取り"].astype("string").str.normalize("NFKC").str.upper()

    out = pd.DataFrame(index=df.index)
    rooms = pd.to_numeric(
        layout.str.extract(r"^\s*(\d+)", expand=False), errors="coerce"
    )
    out["部屋数"] = rooms.mask(layout.eq("ワンルーム"), 1)
    out["LDKあり"] = layout.str.contains(r"(?:LDK|DK)", regex=True, na=False).astype(
        "int8"
    )
    out["単身向け"] = layout.str.contains(
        r"^(?:ワンルーム|1(?:R|K|DK|LDK))(?:\+.*)?$", regex=True, na=False
    ).astype("int8")
    return out
