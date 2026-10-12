"""学習データの選択。どの行を、どの重みで学習に使うか。

列（特徴量）の選択は src/features/ で行う。ここは行の選択。

CV では fold ごとに、その fold の学習データだけに対して呼ばれる。
"""

import pandas as pd


def select(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """学習に使う行と、行ごとの重みを決める。今は全行を同じ重みで使う。

    Args:
        tr: 学習データ。CV では、その fold の学習データ。

    Returns:
        学習に使う行と、行ごとの重み。重みが None なら全行同じ重み。
    """
    return tr, None
