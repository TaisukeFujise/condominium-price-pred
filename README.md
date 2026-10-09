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

動作確認:

```bash
uv run python main.py
```

`Python: 3.12.x` と、`.venv/bin/python3` のパスが表示されれば OK。

### 3. Kaggle の認証

1. Kaggle の [Settings](https://www.kaggle.com/settings) → **API** → **Create New Token** で `kaggle.json` をダウンロード
2. 配置する

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. コンペページでルールに同意（Join Competition）しておく

### 4. データのダウンロード

```bash
uv run python download_data.py
```

ダウンロード先のパスが表示される。

## 使い方

```bash
uv run python <script>.py   # スクリプト実行
uv run jupyter lab          # Jupyter Lab を起動
```

VS Code / Cursor でノートブックを使う場合は、カーネルに `.venv/bin/python` を選択する。

## パッケージの追加・削除

```bash
uv add <package>            # 追加（pyproject.toml と uv.lock が更新される）
uv add --dev <package>      # 開発用ツールとして追加
uv remove <package>         # 削除
```

`pip install` は使わず、必ず `uv add` で追加する。変更した `pyproject.toml` と `uv.lock` はコミットする。

## トラブルシューティング

- **LightGBM の import で `libomp` エラー（macOS）**: `brew install libomp`
- **環境がおかしくなった**: `rm -rf .venv && uv sync`
