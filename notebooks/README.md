# notebooks

EDA や試行のためのノートブックを置く場所。開き方（カーネルの選び方）はリポジトリ直下の README の「ノートブック」を参照。

## 使い方

各自 `<名前>_eda.ipynb` を作り、自由に可視化・分析する。最初のセルは `fujise_overview.ipynb` からコピーするとよい。

## ルール

- 他の人のノートブックは編集しない（ノートブックは git で衝突すると直すのが大変なため）
- 1 つのノートブックが長くなったら、`<名前>_<テーマ>.ipynb` で分けてよい
- 出力（グラフ）は残したままコミットする
- 分かったことは Issue に書く

## 最初のセル

```python
import matplotlib.pyplot as plt
import matplotlib_fontja  # グラフで日本語を表示する
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_csv("../utokyo-mlapp-2026-ex1/train.csv", dtype={"面積（㎡）": str})
test = pd.read_csv("../utokyo-mlapp-2026-ex1/test.csv", dtype={"面積（㎡）": str})
```

- データは元の値のまま読み込む。数値への変換（「戦前」「30分～60分」「2,000㎡以上」などの扱い）や㎡単価などは、必要に応じて各自で作る。良い変換が見つかったら、モデルの前処理に PR で入れる
- `matplotlib_fontja` を import すると、OS に関係なくグラフで日本語が表示される。`sns.set_theme()` を使うとフォントの設定が消えるので、使ったら `matplotlib_fontja.japanize()` を呼ぶ
