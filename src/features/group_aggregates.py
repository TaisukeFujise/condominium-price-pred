"""地区ごと・最寄駅ごと・地区×建築年の帯ごとの、価格を使わない集計と、その行との差・比。

集計するのは、件数・平均面積・平均建築年など。

価格を使わないので、fold に分ける前に、train と test をまとめて集計する。
件数には test の行も入る（価格は入らない）。
test の行も CV の検証データの行も、自分の期間の行を含めて数えるので、CV と本番で扱いがそろう。
"""

import pandas as pd

# 集計の単位。地区名は市区町村をまたいで同じ名前があるので、市区町村名と組にする
GROUP_KEYS = {
    "地区": ["市区町村名", "地区名"],
    "駅": ["最寄駅：名称"],
    # 同じ地区でも、建った年代が同じ物件の中で広いか・駅に近いかを見る。年代で広さや立地の相場が違うため
    "地区×建築年帯": ["市区町村名", "地区名", "建築年帯"],
}
# 建築年帯は 10 年ごと（1990 年代なら 1990）
ERA_YEARS = 10
# 平均をとり、その行との差を作る列
MEAN_COLS = ["建築年_数値", "駅距離_分"]
# 平均をとり、その行との比を作る列（面積は桁が広いので差より比）
RATIO_COLS = ["面積_数値"]


def make(df: pd.DataFrame) -> pd.DataFrame:
    """地区ごと・最寄駅ごと・地区×建築年の帯ごとに集計し、件数・平均・その行との差や比の列を作る。

    Args:
        df: 前処理済みの表（train と test の両方を含む）。

    Returns:
        集計の列の表。キーや値が欠損の行は NaN。
    """
    out = pd.DataFrame(index=df.index)
    # 建築年帯は集計のキーにだけ使い、モデルには入れない（建築年_数値から分かる）
    df = df.assign(建築年帯=df["建築年_数値"] // ERA_YEARS * ERA_YEARS)
    for name, keys in GROUP_KEYS.items():
        # observed=True: カテゴリ型のキーで、出てこない組み合わせまで作らない
        groups = df.groupby(keys, observed=True)
        out[f"{name}_件数"] = groups["id"].transform("size")
        for col in MEAN_COLS:
            mean = groups[col].transform("mean")
            out[f"{name}_平均_{col}"] = mean
            out[f"{name}_平均との差_{col}"] = df[col] - mean
        for col in RATIO_COLS:
            mean = groups[col].transform("mean")
            out[f"{name}_平均_{col}"] = mean
            out[f"{name}_平均との比_{col}"] = df[col] / mean
    return out
