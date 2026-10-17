"""市区町村・地区・駅ごとの間取りの構成（単身向けか家族向けかの目印）。

Nishika 中古マンション価格予測 2022 春の 3 位は、市区町村ごとの間取りの分布を特徴量にして
小幅に効いた。ここではまず単純な割合にする。間取りの解析と単身向けの判定は、
物件自身の間取りを列にした #30 の定義にそろえる。価格を使わないので、fold に分ける前に
train と test をまとめて集計する（group_aggregates.py と同じ扱い）。
"""

import pandas as pd

# 地区名は市区町村をまたいで同じ名前があるので、市区町村名と組にする
GROUP_KEYS = {
    "市区町村": ["市区町村名"],
    "地区": ["市区町村名", "地区名"],
    "駅": ["最寄駅：名称"],
}


def make(df: pd.DataFrame) -> pd.DataFrame:
    """地域ごとに、単身向けの間取り（1R・1K・1DK・1LDK など）と 3 部屋以上の割合を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        割合の列の表。間取りが欠損の行は分母にも入れない。
    """
    # 間取りは全角なので、半角にそろえてから解析する
    layout = df["間取り"].astype("string").str.normalize("NFKC").str.upper()
    rooms = pd.to_numeric(
        layout.str.extract(r"^\s*(\d+)", expand=False), errors="coerce"
    ).mask(layout.eq("ワンルーム"), 1)
    single = layout.str.contains(
        r"^(?:ワンルーム|1(?:R|K|DK|LDK))(?:\+.*)?$", regex=True, na=False
    )
    flags = pd.DataFrame({"単身向け割合": single, "3部屋以上割合": rooms >= 3}).astype(
        float
    )
    flags = flags.where(layout.notna())
    out = pd.DataFrame(index=df.index)
    for name, keys in GROUP_KEYS.items():
        groups = flags.groupby([df[key] for key in keys], observed=True)
        for col in flags.columns:
            out[f"{name}_{col}"] = groups[col].transform("mean")
    return out
