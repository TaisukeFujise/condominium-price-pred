# condominium-resale-price-pred

Kaggle コンペ [`utokyo-mlapp-2026-ex1`](https://www.kaggle.com/competitions/utokyo-mlapp-2026-ex1)（中古マンション価格予測）用のリポジトリ。

Python 環境は [uv](https://docs.astral.sh/uv/) で管理する（Python 3.12）。

## セットアップ

### 1. uv をインストール（未導入の場合）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# または: brew install uv
```

### 2. 依存パッケージをインストール

リポジトリのルートで実行する。

```bash
uv sync
```

`uv sync` で Python 3.12 の取得・`.venv/` の作成・`uv.lock` どおりのパッケージ導入まで自動で行われる。

動作確認（ベースラインの学習から提出ファイルの作成まで、数十秒かかる）:

```bash
uv run python main.py
```

最後に `保存しました: submissions/submission.csv（29341 行）` と表示されれば OK。途中に検証 RMSLE と特徴量の重要度も表示される。`submissions/` は git の管理外。

## データ

コンペのデータはリポジトリに含まれている（`utokyo-mlapp-2026-ex1/`）。clone すればそのまま使える。

- `train.csv` / `test.csv` / `sample_submission.csv`
- 各カラムの値のまとめ: [`docs/columns.md`](docs/columns.md)

補足：Kaggle から取り直す場合は `uv run python download_data.py` を実行する。事前に次の準備が必要。

1. [Settings](https://www.kaggle.com/settings) で API トークンを発行する
2. `uv run python -c "import kagglehub; kagglehub.login()"` でトークンを登録する
3. コンペのルールに同意する

## uv とは

Python 本体・仮想環境（`.venv`）・パッケージをまとめて管理するツール。pyenv + venv + pip の役割を 1 つでこなし、pip よりかなり速い。

| ファイル | 役割 | git |
|---|---|---|
| `pyproject.toml` | 使うパッケージの一覧（`uv add` が自動で書く） | コミットする |
| `uv.lock` | 依存も含めた全パッケージの正確なバージョン（自動生成） | コミットする |
| `.python-version` | 使う Python のバージョン | コミットする |
| `.venv/` | 実際にパッケージが入る場所（`uv sync` / `uv run` が自動で作る） | コミットしない |

この 3 ファイルがあれば、誰の PC でも `uv sync` で同じ環境が再現できる。`requirements.txt` は使わない（`uv.lock` が同じ役割）。

## 使い方

### スクリプト

```bash
uv run python <script>.py
```

### ノートブック（VS Code / Cursor）

1. `uv sync` を済ませておく（`.venv/` ができていればよい）
2. `notebooks/` の `.ipynb` を開く
3. 右上の「カーネルを選択」→「Python 環境」を選ぶ
4. 一覧から次の環境を選ぶ（★ が付いていることが多い）

   ```
   condominium-resale-price-pred (3.12.x)   .venv/bin/python
   ```

   - `/opt/homebrew/bin/python3` や `/usr/bin/python3` は PC 全体の Python で、パッケージが入っていない
   - 「Python 環境の作成」は選ばない（`.venv` が二重にできる）
5. 右上に `condominium-resale-price-pred (3.12.x)` と表示されれば OK。次回からは自動で選ばれる

一覧に出ないときは、パスを直接指定する。

1. コマンドパレット（`Cmd+Shift+P`）で「Python: インタープリターを選択」を開く
2. 「インタープリター パスを入力」で `.venv/bin/python` を指定する

### ノートブック（ブラウザ）

```bash
uv run jupyter lab
```

この方法では `.venv` の環境が自動で使われるので、カーネルの選択は不要。

### ノートブックでの注意

- パッケージが足りないときは、ノートブック内で `!pip install` を使わない。ターミナルで `uv add <package>` を実行し、カーネルを再起動する
- データは `../utokyo-mlapp-2026-ex1/train.csv` のように、ノートブックからの相対パスで読む

## パッケージの追加・削除

```bash
uv add <package>            # 追加（pyproject.toml と uv.lock が更新される）
uv add --dev <package>      # 開発用ツールとして追加
uv remove <package>         # 削除
```

変更した `pyproject.toml` と `uv.lock` はコミットする。他の人は `git pull` のあと `uv sync` すれば同じパッケージが入る。

補足（pip に慣れている人向け）：`pip install` の代わりに `uv add` を使う。pip で入れたパッケージは `pyproject.toml` に記録されず、他の人の環境に入らない。`pip freeze` や `source .venv/bin/activate` も不要。

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| LightGBM の import で `libomp` エラー（macOS） | `brew install libomp` |
| 環境がおかしくなった | `rm -rf .venv && uv sync` |
