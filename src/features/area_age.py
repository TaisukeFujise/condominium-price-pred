"""面積と築年数の組み合わせ。

Nishika 中古マンション価格予測 2023秋の 1 位解法で、一番効いた特徴量（#43）。
広さによって古くなったときの値下がりの仕方が違うなら、掛け算の 1 列で渡すと少ない分岐で表せる。
"""

import pandas as pd


def make(df: pd.DataFrame) -> pd.DataFrame:
    """面積 × 築年数 の列を作る。

    Args:
        df: 前処理済みの表。

    Returns:
        面積×築年数 の 1 列の表。築年数が欠損の行は NaN。
    """
    return pd.DataFrame({"面積×築年数": df["面積_数値"] * df["築年数"]}, index=df.index)
