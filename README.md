# condominium-resale-price-pred

東京都の中古マンション等の取引価格（総額）を予測するモデル。東京大学「機械学習応用」（2026 年度）の演習コンペ [`utokyo-mlapp-2026-ex1`](https://www.kaggle.com/competitions/utokyo-mlapp-2026-ex1) に、グループ D として取り組んだ。

## 課題

- 入力：所在地・最寄駅・面積・間取り・築年・都市計画・取引時期など（各列は [`docs/columns.md`](docs/columns.md)）
- 学習：2019 年第 1 四半期〜2025 年第 2 四半期の取引（約 20 万件）
- 予測：2025 年第 3 四半期〜2026 年第 1 四半期の取引（約 2.9 万件）
- 評価：RMSLE

test が train より未来の期間なので、検証は時系列 CV（3 fold）で行う（[`docs/validation.md`](docs/validation.md)）。

## 手法

- **目的変数**：log(㎡単価) から相場の指数を引いた値を学習する。予測には指数を足し戻し、面積を掛けて総額に戻す
- **特徴量**：建物・間取りの分解、地区や駅ごとの集約、地区の緯度・経度と主要駅への近さ（国土交通省の位置参照情報と駅別乗降客数データ）
- **モデル**：LightGBM・CatBoost・線形モデル・ヘドニック回帰、地区や駅ごとの価格の平均（fold ごとに計算する target encoding）を使う LightGBM、築年数・面積・価格情報区分などで行を絞って学習した LightGBM
- **アンサンブル**：上の各モデルの予測を、2 段目の LightGBM で混ぜる（stacking）

工程ごとの打ち手の全体像は [`docs/workflow.md`](docs/workflow.md)。

## 実行

Python 3.12 と [uv](https://docs.astral.sh/uv/) を使う。コンペのデータはリポジトリに含まれている（`utokyo-mlapp-2026-ex1/`）。

```bash
uv sync
uv run python main.py
```

CV のスコアを表示し、`submissions/submission.csv` を作る（数分かかる）。位置参照情報は初回の実行時に `external_data/` へ自動でダウンロードされる。

## 構成

| パス | 内容 |
|---|---|
| `main.py` | 前処理から提出ファイルの作成までを、工程の順に呼ぶ |
| `src/` | 工程ごとの処理（前処理・特徴量・学習データの選択・目的変数・モデル・アンサンブル・後処理） |
| `scripts/` | ハイパーパラメータの探索、seed 平均など |
| `notebooks/` | EDA |
| `docs/` | 列の説明、検証方法、実験の進め方 |
| `slides/` | 発表資料 |

開発に参加する場合は [`docs/development.md`](docs/development.md)（環境構築）と [`docs/experiment-guide.md`](docs/experiment-guide.md)（実験の進め方）を参照。

## 外部データの扱い

コンペの規約により、テスト期間（2025 年第 3 四半期以降）の取引データは使っていない。外部データは、基準日が 2025 年第 2 四半期以前のものだけを使う。
