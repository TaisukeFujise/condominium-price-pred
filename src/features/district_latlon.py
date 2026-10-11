"""地区の緯度・経度（外部データ: 大字・町丁目レベル位置参照情報）。

出典: 「大字・町丁目レベル位置参照情報 国土交通省」（2023 年版（17.0b）、東京都、
ファイル名 13000-17.0b.zip、https://nlftp.mlit.go.jp/isj/ ）を加工して作成。
丁目を外して町名ごとに緯度・経度を平均した。利用約款: https://nlftp.mlit.go.jp/isj/agreement.html

2023 年版は基準日が test 期間（2025年第3四半期以降）より前なので、AGENTS.md の規約を満たす。
"""

import io
import urllib.request
import zipfile

import pandas as pd

from src.config import EXTERNAL_DIR

ISJ_URL = "https://nlftp.mlit.go.jp/isj/dls/data/17.0b/13000-17.0b.zip"
ISJ_ZIP = EXTERNAL_DIR / "13000-17.0b.zip"


def load_town_points() -> pd.DataFrame:
    """町名ごとの緯度・経度を読み込む。zip がなければダウンロードする。

    Returns:
        市区町村名・町名・緯度・経度の表。
    """
    if not ISJ_ZIP.exists():
        EXTERNAL_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(ISJ_URL, ISJ_ZIP)

    # 外から取ってきた zip なので展開はせず、CSV だけをメモリ上で読む
    with zipfile.ZipFile(ISJ_ZIP) as z:
        csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
        towns = pd.read_csv(
            io.BytesIO(z.read(csv_name)),
            encoding="cp932",
            usecols=["市区町村名", "大字町丁目名", "緯度", "経度"],
        )

    # 位置参照情報は丁目ごと、地区名は丁目のない町名なので、丁目を外して町名ごとに平均する
    towns["町名"] = towns["大字町丁目名"].str.replace(
        r"[一二三四五六七八九十]+丁目$", "", regex=True
    )
    return towns.groupby(["市区町村名", "町名"], as_index=False)[
        ["緯度", "経度"]
    ].mean()


def make(df: pd.DataFrame) -> pd.DataFrame:
    """市区町村名＋地区名で位置参照情報と結合し、緯度・経度の列を作る。

    Args:
        df: 前処理済みの表。

    Returns:
        緯度・経度の 2 列の表。結合できない地区（昭島市・代官山だけ）は NaN。
    """
    merged = (
        df[["市区町村名", "地区名"]]
        .astype(str)
        .merge(
            load_town_points(),
            left_on=["市区町村名", "地区名"],
            right_on=["市区町村名", "町名"],
            how="left",
        )
    )
    return pd.DataFrame(
        {"緯度": merged["緯度"].to_numpy(), "経度": merged["経度"].to_numpy()},
        index=df.index,
    )
