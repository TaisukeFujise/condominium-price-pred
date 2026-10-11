"""説明資料用のグラフ（時期×価格、時期×誤差）。

- quarterly_price_trend.png（既存、eda.pyで生成）: 四半期ごとの価格トレンド
- quarterly_error_before_after.png（新規）: 3fold分のOOF較正前後の誤差を実際の暦四半期で
  連結してプロット。foldをまたいでも暦四半期は重複しないため、1本の連続した時系列として描ける。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
OOF_PATH = Path(__file__).resolve().parent / "analysis" / "oof_residuals.csv"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Hiragino Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "text.color": "#0b0b0b",
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "axes.facecolor": "#fcfcfb",
    "figure.facecolor": "white",
})


def _quarter_label(raw: pd.Series) -> pd.Series:
    ext = raw.str.extract(r"(\d{4})年第(\d)四半期")
    return ext[0] + "Q" + ext[1]


def plot_quarterly_error(oof: pd.DataFrame, out_dir: Path) -> Path:
    df = oof.copy()
    df["quarter_label"] = _quarter_label(df["取引時期"])
    year = df["取引時期"].str.extract(r"(\d{4})")[0].astype(int)
    q = df["取引時期"].str.extract(r"第(\d)四半期")[0].astype(int)
    df["quarter_order"] = year * 4 + q

    g = df.groupby(["quarter_order", "quarter_label"]).agg(
        raw_bias=("log_error_raw", "mean"),
        calibrated_bias=("log_error_calibrated", "mean"),
        n=("log_error_raw", "size"),
    ).reset_index().sort_values("quarter_order")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(0, color=MUTED_INK, linewidth=1, linestyle="--", zorder=1)
    ax.plot(g["quarter_label"], g["raw_bias"], color=ORANGE, linewidth=2.5, marker="o",
            markersize=6, label="較正前（生のLightGBM予測）")
    ax.plot(g["quarter_label"], g["calibrated_bias"], color=BLUE, linewidth=2.5, marker="o",
            markersize=6, label="較正後（isotonic較正）")

    ax.set_ylabel("平均log誤差（bias）\n正=過大予測　負=過小予測")
    ax.set_title("四半期ごとの予測誤差（bias）：isotonic較正の前後比較\n（3foldのOOF検証予測を暦四半期で連結、fold間の重複なし）")
    ax.grid(axis="y", linewidth=0.6, color=GRIDLINE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower left", fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    path = out_dir / "quarterly_error_before_after.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    oof = pd.read_csv(OOF_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = plot_quarterly_error(oof, FIGURES_DIR)
    print(f"saved: {path}")
