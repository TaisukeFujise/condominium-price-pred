# AGENTS.md

コーディングエージェント（Claude Code・Codex・Cursor など）向けの指示。人が作業するときも同じルールに従う。

## 禁止事項（コンペ規約）

**テスト期間（2025年第3四半期以降）の取引データを、外部から取得・参照してはいけない。**
人が手作業で行う場合も、エージェントが自動で行う場合も同じ。

- 国土交通省 不動産情報ライブラリなどの API・Web ページから、2025年第3四半期以降の取引価格・成約価格を取得しない。検索やページ閲覧で参照することも禁止
- 外部データ（地価・人口・駅の乗降客数など）を使うときは、テスト期間の取引データが混ざっていないかを確認してから使う。迷ったら使わずに人に確認する
- コンペのデータ（`utokyo-mlapp-2026-ex1/`）以外の取引データを使う提案をするときは、この規約に触れないことを必ず説明する

## コンペの概要

- 東京都の中古マンション等の取引価格（総額）を予測する。詳細は `utokyo-mlapp-2026-ex1/README.md`
- 学習：2019年第1四半期〜2025年第2四半期、テスト：2025年第3四半期〜2026年第1四半期
- 評価指標は RMSLE。目的変数を `log1p` してから二乗誤差で学習し、予測は `expm1` で円に戻す
- 各列の値の分布と注意点は `docs/columns.md`

## 開発環境

- Python 3.12 を uv で管理する。パッケージの追加は `uv add <package>`。`pip install` は使わない
- `main.py` は流れをつかむための 1 ファイル版。実験は `src/` と `experiments/` で行う
- 実験の実行：`uv run python -m experiments.<名前>.<スクリプト名>`（例：`uv run python -m experiments.baseline`）
- 実験結果の一覧：`uv run python -m src.summary`
- `submissions/` は git の管理外

## 実験のルール

手順と書き方の詳細は `docs/experiment-guide.md`、検証方法の理由と見直しの手順は `docs/validation.md`、精度改善の工程と打ち手は `docs/workflow.md`。

- 検証は `src/validation.py` の時系列 CV（3 fold）を使う。チームの合意なしに変えない。変えるときは `VERSION` を上げる。v1 は暫定で、CV と LB が連動するかを確かめてからチームで確定する
- 実験は `src.train.run_experiment` を通して実行し、記録の .json を残す
- 特徴量は `src/features/<テーマ>.py` に「元の表を受け取り、新しい列だけの表を返す関数」として書く
- 他のメンバーの `experiments/<名前>/` は書き換えない
- test は train より未来の期間。ランダム分割で検証しない
- 価格を使う特徴量（target encoding など）は、fold ごとにその fold の学習データだけで計算する
- 認証情報（`kaggle.json` など）はコミットしない。main へは PR で入れる
