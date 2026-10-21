"""学習データの選択（工程 3）。どの行を、どの重みで学習に使うか。

このファイルが選ぶのは行。列（特徴量）は src/features/ で選ぶ。
CV では fold ごとに、その fold の学習データだけを受け取る。
"""

import numpy as np
import pandas as pd

# 成約価格情報は 2021 年からしかなく、学習データでは 35〜45% と少ないが、
# 検証期間や test では約 67% を占め、不動産取引価格情報のようなアンケート由来の極端な外れ値も少ない。
# 学習時の構成を検証・test 期間に近づけるため、成約価格情報の行を重く見る
CONTRACT_PRICE = "成約価格情報"
CONTRACT_WEIGHT = 1.5


def select(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """学習に使う行と、行ごとの重みを決める。

    Args:
        tr: 学習データ。CV では、その fold の学習データ。

    Returns:
        学習に使う行と、行ごとの重み。重みが None なら全行同じ重み。
    """
    weight = pd.Series(
        np.where(tr["価格情報区分"] == CONTRACT_PRICE, CONTRACT_WEIGHT, 1.0),
        index=tr.index,
    )
    return tr, weight
