"""
ベースライン：読み込み → 前処理 → 検証 → 学習 → 予測 → 提出ファイル作成。

流れを把握するための 1 ファイル版。実験は、これを src/ に分割して検証を 3 fold にした
仕組みで行う（docs/experiment-guide.md）。

実行: uv run python main.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_DIR = Path("utokyo-mlapp-2026-ex1")
SUBMISSION_DIR = Path("submissions")
TARGET = "取引価格（総額）"


# =============================================================================
# 1. データの読み込み
# =============================================================================
# 面積は "45" と "2,000㎡以上" が混ざっているので、いったん文字列として読む
train = pd.read_csv(DATA_DIR / "train.csv", dtype={"面積（㎡）": str})
test = pd.read_csv(DATA_DIR / "test.csv", dtype={"面積（㎡）": str})
print(f"train: {train.shape}, test: {test.shape}")

# train と test に同じ前処理を当てるため、連結して 1 つの表にする
train["is_train"] = True
test["is_train"] = False
df = pd.concat([train, test], ignore_index=True)


# =============================================================================
# 2. 前処理：文字列で入っている数値を数値に直す
# =============================================================================
# 変換できない値は NaN にする。LightGBM は欠損をそのまま扱える。

# 2-1. 取引時期
#      経過四半期 = 2019年第1四半期から何四半期たったか（train は 0〜25、test は 26〜28）。
#      文字列やカテゴリのままだと時間の前後関係をモデルが使えないので、通し番号にする
period = df["取引時期"].str.extract(r"(\d{4})年第(\d)四半期").astype(int)
df["取引年"] = period[0]
df["取引四半期"] = period[1]
quarter_index = df["取引年"] * 4 + df["取引四半期"]
df["経過四半期"] = quarter_index - quarter_index.min()

# 2-2. 建築年。"戦前" は 1945 とみなす
df["建築年_数値"] = pd.to_numeric(
    df["建築年"].str.replace("年", "").replace("戦前", "1945"), errors="coerce"
)
df["築年数"] = df["取引年"] - df["建築年_数値"]

# 2-3. 最寄駅までの距離。30 分以上は範囲表記なので、範囲の中央値を入れる
distance_map = {"30分～60分": 45, "1H～1H30": 75, "1H30～2H": 105, "2H～": 120}
df["駅距離_分"] = pd.to_numeric(
    df["最寄駅：距離（分）"].replace(distance_map), errors="coerce"
)

# 2-4. 面積
df["面積_数値"] = pd.to_numeric(
    df["面積（㎡）"].replace("2,000㎡以上", "2000"), errors="coerce"
)


# =============================================================================
# 3. 前処理：文字列のカテゴリ列を category 型にする
# =============================================================================
# category 型にすると、LightGBM は数値の大小ではなく種類ごとに分岐する。
# 連結した状態で変換するので、train と test で同じカテゴリ番号が付く。
CATEGORICAL_COLS = [
    "価格情報区分",
    "市区町村名",
    "地区名",
    "最寄駅：名称",
    "間取り",
    "建物の構造",
    "用途",
    "今後の利用目的",
    "都市計画",
    "改装",
    "取引の事情等",
]
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype("category")

# モデルに入れる列（特徴量）の一覧。
# 元データのうち、次の列はモデルに入れない:
#   - id                       : 価格とは関係ない
#   - 取引価格（総額）          : 目的変数
#   - 市区町村コード            : 市区町村名と 1:1 で重複。外部データと連結するキーとして使える（docs/columns.md ※2）
#   - 取引時期, 建築年,
#     最寄駅：距離（分）, 面積（㎡）: 文字列のまま。2 で数値に直した列（経過四半期・建築年_数値など）を代わりに使う
#   - is_train                 : 1 で足した目印の列
NUMERIC_COLS = [
    "経過四半期",
    "取引四半期",
    "建築年_数値",
    "築年数",
    "駅距離_分",
    "面積_数値",
    "建ぺい率（％）",
    "容積率（％）",
]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

train = df[df["is_train"]].reset_index(drop=True)
test = df[~df["is_train"]].reset_index(drop=True)


# =============================================================================
# 4. 目的変数の変換：価格 → log(1 + 価格)
# =============================================================================
# 評価指標は RMSLE。対数価格を普通の二乗誤差で学習すれば、RMSLE を直接小さくできる。
# log ではなく log1p（= log(1 + x)）を使うのは、RMSLE の定義が log(1 + 価格) だから。
# 逆変換は expm1（= e^x − 1）。
y = np.log1p(train[TARGET])


# =============================================================================
# 5. 検証データの分割：直近 3 四半期を検証用にする（ホールドアウト）
# =============================================================================
# test は train より未来の期間。ランダムに分けると本番より甘いスコアが出るので、
# 本番と同じく「過去で学習 → 未来を予測」の形で検証する。
# test は学習データの 1〜3 四半期先なので、検証も 3 四半期にして条件を揃える。
VALID_QUARTERS = 3
is_valid = train["経過四半期"] > train["経過四半期"].max() - VALID_QUARTERS

x_train, y_train = train.loc[~is_valid, FEATURES], y[~is_valid]
x_valid, y_valid = train.loc[is_valid, FEATURES], y[is_valid]
print(f"学習用: {len(x_train)} 行, 検証用: {len(x_valid)} 行")


# =============================================================================
# 6. 学習と検証スコアの計算
# =============================================================================
# パラメータはデフォルトのまま（objective はデフォルトで二乗誤差）。調整は後で行う。
# metric のデフォルトは l2（MSE）。対数価格での RMSE ＝ RMSLE なので、rmse にすれば
# 学習ログの値がそのままコンペのスコアと比べられる。
# early stopping で、ちょうど良い木の本数（best_iteration）を決める。7 の再学習で使う。
params = {"verbose": -1, "metric": "rmse"}
model = lgb.train(
    params,
    lgb.Dataset(x_train, y_train),
    num_boost_round=10000,  # 木の本数の上限。実際は early stopping で止まる
    # valid_sets と early_stopping は 2 つで 1 セット:
    # 毎ラウンド x_valid でスコアを測り、100 ラウンド改善しなければ学習を止める。
    valid_sets=[lgb.Dataset(x_valid, y_valid)],
    callbacks=[lgb.early_stopping(100)],
)

# 対数価格での RMSE ＝ 元の価格での RMSLE
y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
rmsle = np.sqrt(np.mean((y_pred - y_valid) ** 2))
print(f"検証 RMSLE: {rmsle:.5f}（best_iteration={model.best_iteration}）")

# gain = その特徴量での分岐によって誤差がどれだけ減ったかの合計
importance = pd.Series(
    model.feature_importance(importance_type="gain"), index=FEATURES
).sort_values(ascending=False)
print("特徴量の重要度（gain）:")
print((importance / importance.sum()).round(3).to_string())


# =============================================================================
# 7. 全データで学習し直して test を予測
# =============================================================================
# 検証用に取っておいた直近のデータは test に一番近い時期なので、本番の学習には含める。
final_model = lgb.train(
    params,
    lgb.Dataset(train[FEATURES], y),
    num_boost_round=model.best_iteration,
)
# モデルの出力は対数価格（4 で log1p した値）なので、expm1 で円に戻す
y_test_pred = final_model.predict(test[FEATURES])
price_pred = np.expm1(y_test_pred)


# =============================================================================
# 8. 提出ファイルの作成
# =============================================================================
SUBMISSION_DIR.mkdir(exist_ok=True)
submission = pd.DataFrame({"id": test["id"], "price": price_pred})
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)
print(f"保存しました: {SUBMISSION_DIR / 'submission.csv'}（{len(submission)} 行）")
