"""工程 3：学習データの選択。どの行を、どの重みで学習に使うか。

CV では fold ごとに、その fold の学習データだけに対して呼ばれる。
"""

import pandas as pd


def select(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """学習に使う行と、行ごとの重み（None なら全行同じ重み）を返す。

    今は全行を同じ重みで使う。
    """
    return train, None
