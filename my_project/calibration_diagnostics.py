"""isotonic較正を`価格情報区分`別に分けるべきか判断するための可視化。

内側ホールドアウト（直近3Q）の予測値(log) vs 実際値(log)を価格情報区分別に色分けしてプロットし、
区分ごとに別々にisotonic回帰を当てた曲線と、全体共通の曲線を重ねて比較する。
曲線が区分間で大きく違えば分ける価値があり、ほぼ同じなら分けなくてよい。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from my_project.bias_correction import _predict_log
from my_project.io import TARGET_COLUMN, load_train
from my_project.model import _train_fold, time_series_cv_splits
from my_project.preprocessing import build_features, fit_categories

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED_INK = "#898781"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Hiragino Sans", "sans-serif"],
    "axes.unicode_minus": False,
})


def build_holdout_predictions(train_df: pd.DataFrame) -> pd.DataFrame:
    categories = fit_categories(train_df)
    features = build_features(train_df, categories=categories)
    target = train_df[TARGET_COLUMN]

    train_mask, valid_mask = time_series_cv_splits(features, n_folds=1)[0]
    model, score, mappings = _train_fold(features, target, train_mask, valid_mask, verbose=False)
    print(f"(参考) このホールドアウトのraw RMSLE: {score:.5f}")

    pred_log = _predict_log(model, mappings, features[valid_mask])
    actual_log = np.log1p(target[valid_mask].values)

    return pd.DataFrame({
        "pred_log": pred_log,
        "actual_log": actual_log,
        "価格情報区分": train_df.loc[valid_mask, "価格情報区分"].values,
    })


def plot_calibration_by_category(df: pd.DataFrame, out_dir: Path) -> Path:
    categories = df["価格情報区分"].unique().tolist()
    colors = {categories[0]: BLUE, categories[1]: ORANGE} if len(categories) == 2 else {}

    x_grid = np.linspace(df["pred_log"].min(), df["pred_log"].max(), 200).reshape(-1, 1).ravel()

    combined_iso = IsotonicRegression(out_of_bounds="clip")
    combined_iso.fit(df["pred_log"], df["actual_log"])

    fig, axes = plt.subplots(1, len(categories), figsize=(7 * len(categories), 6), sharex=True, sharey=True)
    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        sub = df[df["価格情報区分"] == cat]
        ax.scatter(sub["pred_log"], sub["actual_log"], s=4, alpha=0.15, color=colors.get(cat, MUTED_INK), linewidths=0)

        cat_iso = IsotonicRegression(out_of_bounds="clip")
        cat_iso.fit(sub["pred_log"], sub["actual_log"])

        lims = [min(df["pred_log"].min(), df["actual_log"].min()), max(df["pred_log"].max(), df["actual_log"].max())]
        ax.plot(lims, lims, color=MUTED_INK, linewidth=1, linestyle="--", label="y=x（較正不要の場合）")
        ax.plot(x_grid, cat_iso.predict(x_grid), color=colors.get(cat, "#0b0b0b"), linewidth=2.5, label=f"{cat}単独の較正曲線")
        ax.plot(x_grid, combined_iso.predict(x_grid), color="#0b0b0b", linewidth=1.5, linestyle=":", label="全体共通の較正曲線")

        ax.set_xlabel("予測値 log1p(price)")
        ax.set_ylabel("実際値 log1p(price)")
        ax.set_title(f"{cat}（n={len(sub)}）")
        ax.legend(fontsize=9, loc="upper left")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("予測値 vs 実際値（log）: 価格情報区分別の較正曲線の比較")
    fig.tight_layout()
    path = out_dir / "calibration_by_price_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    train_df = load_train()
    df = build_holdout_predictions(train_df)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = plot_calibration_by_category(df, FIGURES_DIR)
    print(f"saved: {path}")
