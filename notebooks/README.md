# notebooks

データを調べる（EDA）ためのノートブックを置く場所。開き方（カーネルの選び方）はリポジトリ直下の README の「ノートブック」を参照。

## 最初の EDA

1 つのノートブックを 1 人が担当し、先頭に書いてある「対象のカラム」を可視化・分析する。データの読み込みまでは済ませてある。

| ノートブック | 内容 |
|---|---|
| `eda_1_distribution.ipynb` | 価格の分布 |
| `eda_2_trend.ipynb` | 価格の推移 |
| `eda_3_relations.ipynb` | 価格と各列の関係 |
| `eda_4_train_test.ipynb` | train と test の違い |

## ルール

- 他の人のノートブックは編集しない（ノートブックは git で衝突すると直すのが大変なため）
- 新しく調べたいことは `<名前>_<テーマ>.ipynb` で新しく作る
- 出力（グラフ）は残したままコミットする
- 分かったことは Issue に書く

## eda_utils.py

```python
from eda_utils import TARGET, load, setup_plot

setup_plot()  # グラフで日本語が表示されるようにする
train, test = load()  # データを読み込み、時期・面積_数値・築年数・㎡単価などの列を足す
```

`load()` の変換は、EDA で見やすくするためのもの。モデルの前処理とは別。
