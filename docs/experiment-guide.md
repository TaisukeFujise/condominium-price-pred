# 実験ガイド

精度改善の実験の進め方とルール。何に取り組むかの全体像は [`workflow.md`](workflow.md)、検証方法は [`validation.md`](validation.md)。

## 構成

`main.py` が工程を順に呼ぶ。各工程の中身は `src/` に工程ごとに 1 ファイルある（工程の番号は [`workflow.md`](workflow.md) と同じ）。ファイルが増えていく工程だけディレクトリにしている（特徴量はテーマごと、モデルは種類ごと）。

工程 1・2 は fold に分ける前に train と test の全行に対して 1 回だけ動くので、価格を使えない。工程 3〜8 は `src/pipeline.py` から fold ごとに呼ばれ、その fold の学習データだけを使う。

| ファイル | 工程 |
|---|---|
| `src/preprocess.py` | 1. データの読み込みと前処理 |
| `src/features/` | 2. 特徴量（テーマごとにファイルを分ける） |
| `src/selection.py` | 3. 学習データの選択（行の除外・重み） |
| `src/target.py` | 4. 目的変数（何を予測させるか、価格への戻し方） |
| `src/models/` | 5. モデル（学習と予測）。揃えるメソッドは `base.py`、LightGBM は `lgbm.py` |
| `src/models/params.py` | 6. ハイパーパラメータ |
| `src/ensemble.py` | 7. アンサンブル |
| `src/postprocess.py` | 8. 後処理（予測値の補正） |
| `src/pipeline.py` | 工程 3〜8 のつなぎ方（CV 用と本番用の入口） |
| `src/validation.py` | 検証（時系列 CV）。チームの合意なしに変えない |
| `src/config.py` | パス・目的変数の列名・seed |

## 実験の流れ

0. やることを Issue にする（「実験・タスク」テンプレート）。担当を決め、Issue を自分に割り当てる
1. 最新の main からブランチを切る（例：`feat/station-features`）
2. 試したい工程のファイルを直す。**一度に変えるのは 1 つだけ**
3. 実行して、CV を main のときと比べる

   ```bash
   uv run python main.py
   ```

4. コードを整形してチェックする

   ```bash
   uv run ruff format .
   uv run ruff check --fix .
   ```

5. 結果に応じて PR を出す

PR を作ると本文にテンプレートが入るので、CV の表とチェック項目を埋める。本文に `Closes #<Issue の番号>` と書くと、マージしたときに Issue も閉じる。

| 結果 | やること |
|---|---|
| CV が良くなった | PR を出す。本文に「main の CV → この PR の CV」を書く。レビュー後にマージ |
| CV が良くならなかった | **それでも PR を出し**、本文に結果を書いて、マージせずに閉じる。`不採用` ラベルを付ける |

不採用の実験も残すのは、同じことを二度試さないため。「土台が変わったら効くかもしれない」ものは、そう書いておく。

### 提出したとき

PR の本文に Public LB のスコアを追記する。CV と LB が連動するかを確かめる材料になる（[`validation.md`](validation.md) の 5 章）。

### 他の人の PR がマージされたら

main が新しい土台になる。自分のブランチを最新の main に合わせ（`git merge main` か `git rebase main`）、**新しい main の CV と比べ直す**。

## 例：特徴量を足す

テーマごとにファイルを作り、「元の表を受け取って、新しい列だけの表を返す関数」として書く。元の表は書き換えない。

```python
# src/features/station.py
import pandas as pd


def make(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["駅徒歩10分以内"] = (df["駅距離_分"] <= 10).astype(int)
    return out
```

`src/features/build.py` の import と `MAKERS` に 1 行ずつ足す。`build_features()` は `MAKERS` の関数を上から順に呼び、返ってきた列を表とモデルに入れる列名の一覧に足す。

```python
from src.features import base, station

MAKERS = [
    station.make,
]
```

- `MAKERS` は上から順に呼ばれる。ほかの特徴量が作った列を使うときは、その後ろに書く
- 文字列のカテゴリ列を足すときは、`make` の中で `.astype("category")` にしておく（LightGBM が種類ごとに分岐するようになる）
- `df` には train と test の両方が入っている（`is_train` 列で見分ける）
- 価格を使う特徴量（target encoding など）はここでは作れない。理由は [`workflow.md`](workflow.md) の「価格を使う特徴量」

## 例：モデルを替える

`src/models/<モデル名>.py` に、`src/models/base.py` の `Model` を継承したクラスを書く（`lgbm.py` を写して直すのが早い）。メソッドが 1 つでも足りないと、`main.py` を実行してインスタンスを作った時点（CV の最初の fold）でエラーになる。パラメータは `src/models/params.py` に `<モデル名>_PARAMS` として並べる。

`main.py` の `MODELS` を書き換えると、CV も本番もそのモデルで動く。

```python
from src.models.catboost import CatBoost

MODELS = [CatBoost]  # 付け替える
MODELS = [LGBM, CatBoost]  # 2 つを混ぜる（混ぜ方は src/ensemble.py）
```

seed だけ変えた LightGBM を混ぜるときは、`functools.partial` でパラメータを渡す。

```python
from functools import partial

from src.models.params import LGBM_PARAMS

MODELS = [LGBM, partial(LGBM, {**LGBM_PARAMS, "seed": 1})]
```

今の `LGBM_PARAMS` は乱数を使う設定（`bagging_fraction`・`feature_fraction`）が 1 未満なので、seed を変えると別のモデルになる。

- `best_iteration` は、early stopping で決まった繰り返しの回数（木の本数・エポック数など）。回数という考え方がないモデル（線形回帰など）では `None` を返す
- 本番の `fit` には、CV で同じモデルを fold ごとに学習したもの（`cv_models`）が渡る。本番は検証データがないので、繰り返しの回数はここから決める（LightGBM は fold の `best_iteration` の平均）。回数がないモデルでは使わなくてよい
- 重要度を出せないモデルでは、`importance()` で `None` を返す

## ルール

- **テスト期間（2025年第3四半期以降）の取引データを、外部から取得・参照しない**（コンペ規約）
- `src/validation.py` はチームで合意せずに変えない。変えるときは `VERSION` を上げる
- 一度に変えるのは 1 つだけ。複数の変更を 1 つの PR に混ぜない
- PR には、比べた main の CV とこの PR の CV を必ず書く。コミットしていないコードで出したスコアは書かない
- Kaggle の API トークン（`kaggle.json` など）や認証情報はコミットしない
- main へは PR で入れる。main に直接 push しない
