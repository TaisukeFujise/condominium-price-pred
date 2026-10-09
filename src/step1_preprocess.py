"""工程 1：データの読み込みと前処理。

ここを変えると全員の結果が変わる。変えるときは PR で合意する。
"""

import pandas as pd

from src.config import DATA_DIR

CATEGORICAL_COLS = [
    "価格情報区分",
    "市区町村名",
    "地区名",
    "最寄駅：名称",
    "間取り",
    "建物の構造",
    "用途",
    "今後の利用目的",
    "都市計画",
    "改装",
    "取引の事情等",
]


def load() -> pd.DataFrame:
    """train と test を連結し、前処理した 1 つの表を返す。is_train 列で見分ける。"""
    # 面積は "45" と "2,000㎡以上" が混ざっているので、いったん文字列として読む
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"面積（㎡）": str})
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"面積（㎡）": str})
    train["is_train"] = True
    test["is_train"] = False
    df = pd.concat([train, test], ignore_index=True)

    # 変換できない値は NaN にする。LightGBM は欠損をそのまま扱える。

    # 経過四半期 = 2019年第1四半期から何四半期たったか（train は 0〜25、test は 26〜28）
    period = df["取引時期"].str.extract(r"(\d{4})年第(\d)四半期").astype(int)
    df["取引年"] = period[0]
    df["取引四半期"] = period[1]
    quarter_index = df["取引年"] * 4 + df["取引四半期"]
    df["経過四半期"] = quarter_index - quarter_index.min()

    df["建築年_数値"] = pd.to_numeric(
        df["建築年"].str.replace("年", "").replace("戦前", "1945"), errors="coerce"
    )
    df["築年数"] = df["取引年"] - df["建築年_数値"]

    # 30 分以上は範囲表記なので、範囲の中央値を入れる
    distance_map = {"30分～60分": 45, "1H～1H30": 75, "1H30～2H": 105, "2H～": 120}
    df["駅距離_分"] = pd.to_numeric(
        df["最寄駅：距離（分）"].replace(distance_map), errors="coerce"
    )

    df["面積_数値"] = pd.to_numeric(
        df["面積（㎡）"].replace("2,000㎡以上", "2000"), errors="coerce"
    )

    # 連結した状態で変換するので、train と test で同じカテゴリ番号が付く
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df
