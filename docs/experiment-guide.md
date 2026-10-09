# 実験ガイド

全員が同じデータ・同じ検証方法で実験し、結果を比べられるようにするための手順とルール。何に取り組むかの全体像は [`workflow.md`](workflow.md)。

## 構成

`main.py` の処理を、役割ごとにファイルへ分けている。`main.py` は流れをつかむための 1 ファイル版として残してある。

| ファイル | 役割 | main.py との対応 |
|---|---|---|
| `src/config.py` | パス・目的変数の列名・seed | 冒頭の定数 |
| `src/data.py` | 読み込みと、全実験で共通の前処理 | セクション 1〜3 |
| `src/features/base.py` | ベースラインで使う特徴量の一覧 | セクション 3 の後半 |
| `src/features/*.py` | 追加の特徴量（各自が足していく） | — |
| `src/validation.py` | 共通の検証方法（時系列 CV）と RMSLE | セクション 5 |
| `src/train.py` | CV で学習・評価 → 全データで再学習 → 提出ファイルと記録の保存 | セクション 4, 6〜8 |
| `src/summary.py` | 全員の実験記録を一覧にする | — |
| `experiments/` | 実験スクリプトと、その記録（.json） | — |

## 検証方法（v1）

直近 9 四半期を 3 四半期ずつ 3 つの fold に分け、各 fold では検証期間より前のデータだけで学習する。CV スコアは 3 fold の RMSLE の平均。

| fold | 学習 | 検証 |
|---|---|---|
| 0 | 2019Q1〜2024Q3 | 2024Q4〜2025Q2 |
| 1 | 2019Q1〜2023Q4 | 2024Q1〜2024Q3 |
| 2 | 2019Q1〜2023Q1 | 2023Q2〜2023Q4 |

test（2025Q3〜2026Q1）は学習データの 1〜3 四半期先なので、どの fold も「直前までのデータで、その後の 3 四半期を予測する」形にそろえている。

**v1 は暫定。** CV と LB が連動するかを数回の提出で確かめ、ずれる場合はチームで見直す。この方法を選んだ理由、使わない手法（ランダム分割・Purged K-fold など）、見直しの手順は [`validation.md`](validation.md) にまとめている。

## 実験の流れ

1. main から自分のブランチを切る（例：`feat/station-features`）
2. `experiments/<自分の名前>/` に実験スクリプトを作る。ファイル名は `exp<3 桁の連番>_<内容>.py`（例：`experiments/fujise/exp001_station.py`）。連番は自分のフォルダの中だけで数えればよい
3. 実行する（`.py` は付けず、`/` を `.` に変える）

   ```bash
   uv run python -m experiments.fujise.exp001_station
   ```

4. スクリプトの隣に記録（`exp001_station.json`）ができ、`submissions/fujise__exp001_station.csv` に提出ファイルができる
5. スクリプトと記録の .json をコミットする
6. 提出したら、.json の `"lb_public": null` を Public LB のスコアに書き換えてコミットする
7. CV が良くなった変更は PR で main に入れる。PR には、比べた実験 ID と CV（と LB）を書く

全員の結果は次のコマンドで一覧できる。

```bash
uv run python -m src.summary
```

## 実験スクリプトの書き方

`experiments/baseline.py` をコピーして書き換えるのが早い。

### 特徴量を足すとき

特徴量は `src/features/<テーマ>.py` に、**元の表を受け取って、新しい列だけの表を返す関数**として書く。元の表は書き換えない。こうしておけば、他の人の特徴量とファイルも列も衝突しない。

```python
# src/features/station.py
import pandas as pd


def make(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["駅徒歩10分以内"] = (df["駅距離_分"] <= 10).astype(int)
    return out
```

```python
# experiments/fujise/exp001_station.py
import pandas as pd

from src.data import load
from src.features import base, station
from src.train import run_experiment

df = load()
new = station.make(df)
df = pd.concat([df, new], axis=1)
run_experiment(
    __file__,
    df,
    features=base.FEATURES + list(new.columns),
    description="駅徒歩10分以内のフラグを追加",
)
```

- 文字列のカテゴリ列を足すときは、`make` の中で `.astype("category")` にしておく（LightGBM が種類ごとに分岐するようになる）
- `df` には train と test の両方が入っている（`is_train` 列で見分ける）。集計するときに test の行を混ぜてよいかは毎回考える

### パラメータを変えるとき

`run_experiment` の `params` に、変えたいものだけを渡す。渡さなかったものは `src/train.py` の `DEFAULT_PARAMS` が使われる。

```python
run_experiment(__file__, df, features=base.FEATURES, description="学習率を下げる", params={"learning_rate": 0.03})
```

### 目的変数（価格）を使う特徴量

地区ごとの平均価格（target encoding）のように価格を使う特徴量は、`load()` の直後に全データで計算してはいけない。検証期間の価格が学習に漏れて、CV が実際より良く出る。fold ごとに、その fold の学習データだけで計算する必要がある。今の `run_experiment` はこれに対応していないので、必要になったら仕組みを相談して足す。

## ルール

- **テスト期間（2025年第3四半期以降）の取引データを、外部から取得・参照しない**（コンペ規約）
- `src/validation.py` はチームで合意せずに変えない。変えるときは `VERSION` を上げる。バージョンが違う CV どうしは比べられない
- `src/data.py`・`src/train.py` など共通部分を変えると全員の結果が変わる。変えるときは PR に、`experiments/baseline.py` の CV が変わるかどうかを書く
- 実験は必ず `run_experiment` を通して記録を残す。記録の `git_dirty` が `true` のものは、コミットしていないコードで出したスコアなので、PR の根拠にしない
- 他の人の `experiments/<名前>/` は書き換えない
- Kaggle の API トークン（`kaggle.json` など）や認証情報はコミットしない
- main へは PR で入れる。main に直接 push しない
