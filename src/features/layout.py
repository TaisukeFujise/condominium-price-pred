"""間取りから作る特徴量（価格を使わない）。

間取りを部屋数に換算し、一部屋あたりの面積を作る（高級な物件ほど一部屋が広い）。
SIGNATE 賃料予測 2019 の 3 位・Nishika 2023 夏の 3 位・ProbSpace 不動産取引価格予測の 1 位で
使われていた。間取りの正規化は #30 の定義にそろえる。価格を使わないので、
fold に分ける前に train と test をまとめて作る。
"""

import pandas as pd

# 先頭の数字に足す部屋数。SIGNATE 賃料予測 2019 の 3 位の換算式（1K と 1R は実質同じ広さ、
# DK は 1 部屋、LDK は 1.5 部屋、S（納戸）は 1 部屋）
_DK_ROOMS = 1.0
_L_ROOMS = 0.5
_S_ROOMS = 1.0


def make(df: pd.DataFrame) -> pd.DataFrame:
    """間取りを部屋数に換算し、一部屋あたりの面積の列を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        一部屋あたりの面積の表。間取りが欠損か、数字で始まらない行（オープンフロアなど）は NaN。
    """
    layout = df["間取り"].astype("string").str.normalize("NFKC").str.upper()
    out = pd.DataFrame(index=df.index)
    out["一部屋あたり面積"] = (df["面積_数値"] / _converted_rooms(layout)).astype(float)
    return out


def _converted_rooms(layout: pd.Series) -> pd.Series:
    """SIGNATE 3 位の換算式で、間取りを部屋数に直す（例: 3LDK は 4.5 部屋）。

    Args:
        layout: 半角・大文字にそろえた間取り。

    Returns:
        換算した部屋数。数字で始まらない間取りは NaN。
    """
    leading = pd.to_numeric(
        layout.str.extract(r"^\s*(\d+)", expand=False), errors="coerce"
    ).mask(layout.eq("ワンルーム"), 1)
    rest = layout.str.replace(r"^\s*\d+", "", regex=True)
    added = (
        rest.str.contains("DK", regex=False).astype(float) * _DK_ROOMS
        + rest.str.contains("L", regex=False).astype(float) * _L_ROOMS
        + rest.str.contains("S", regex=False).astype(float) * _S_ROOMS
    )
    return leading + added
