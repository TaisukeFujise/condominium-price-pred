"""駅別の平均残差を地図上に可視化する。

地理的にまとまったパターン（同じ方面が過小/過大予測）があれば、位置情報の不足
（都心へのアクセス、路線特性等）の証拠。まだらなら位置以外の要因。
fold1（直近・一番悪いfold）とfold3（一番古い・一番良いfold）を並べて比較し、
「fold1で悪化しているのは特定のエリアか」も見る。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
OOF_PATH = Path(__file__).resolve().parent / "analysis" / "oof_residuals.csv"

# dataviz skillの diverging pair（blue <-> red、グレー中点）
DIVERGING_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list(
    "blue_red_diverging",
    ["#104281", "#2a78d6", "#f0efec", "#e34948", "#7a1f1f"],
)
MIN_N_PER_STATION = 15

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Hiragino Sans", "sans-serif"],
    "axes.unicode_minus": False,
})


def station_level_residuals(oof: pd.DataFrame, fold: int | None = None) -> pd.DataFrame:
    sub = oof if fold is None else oof[oof["fold"] == fold]
    sub = sub.dropna(subset=["station_lat", "station_lon"])
    g = sub.groupby(["最寄駅：名称", "station_lat", "station_lon"]).agg(
        mean_error=("log_error_calibrated", "mean"),
        n=("log_error_calibrated", "size"),
    ).reset_index()
    return g[g["n"] >= MIN_N_PER_STATION]


def plot_map(oof: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    panels = [("fold1（直近・最も悪い）", 1), ("fold3（一番古い・最も良い）", 3), ("全fold合計", None)]

    vmax = max(station_level_residuals(oof, f).mean_error.abs().max() for _, f in panels)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, (title, fold) in zip(axes, panels):
        g = station_level_residuals(oof, fold)
        sc = ax.scatter(
            g["station_lon"], g["station_lat"], c=g["mean_error"], cmap=DIVERGING_CMAP, norm=norm,
            s=np.clip(g["n"] / 3, 8, 120), alpha=0.85, edgecolors="white", linewidths=0.3,
        )
        ax.set_title(f"{title}\n(駅数={len(g)})")
        ax.set_xlabel("経度")
        ax.set_ylabel("緯度")
        ax.set_aspect("equal")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.colorbar(sc, ax=ax, label="平均log誤差(較正後)\n正=過大予測 負=過小予測", shrink=0.8)

    fig.suptitle("駅別の平均残差（較正後）の地理的分布（点の大きさ=件数、最低15件/駅）")
    fig.tight_layout()
    path = out_dir / "residual_map_by_station.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    oof = pd.read_csv(OOF_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = plot_map(oof, FIGURES_DIR)
    print(f"saved: {path}")

    print("\n=== fold1で特に悪い駅 上位15（|平均誤差|降順、n>=15） ===")
    g1 = station_level_residuals(oof, 1).reindex(
        station_level_residuals(oof, 1)["mean_error"].abs().sort_values(ascending=False).index
    )
    print(g1.head(15).to_string(index=False))
