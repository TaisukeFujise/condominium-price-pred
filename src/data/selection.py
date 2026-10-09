"""学習データの選択。どの行を、どの重みで学習に使うか。

CV では fold ごとに、その fold の学習データだけに対して呼ばれる。
"""

import pandas as pd


def select(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """学習に使う行と、行ごとの重みを決める。今は全行を同じ重みで使う。

    Args:
        train: 学習データ。CV では、その fold の学習データ。

    Returns:
        学習に使う行と、行ごとの重み。重みが None なら全行同じ重み。
    """
    return train, None
