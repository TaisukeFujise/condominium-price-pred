"""価格と特徴量の関係を可視化するEDA。

ここでの文字列→数値変換（面積・築年数・駅距離・取引時期）はグラフ表示用の簡易変換であり、
本番の前処理ロジック（preprocessingモジュール）とは別実装。閾値や欠損の扱いは厳密ではない。
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from my_project.io import TARGET_COLUMN, load_train

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# dataviz skillのreference palette（blue/orange, sequential blue ramp）を使用
BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
BOX_FILL = "#cde2fb"
BOX_EDGE = BLUE
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"],
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Hiragino Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECONDARY_INK,
    "text.color": "#0b0b0b",
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "grid.color": GRIDLINE,
    "axes.facecolor": "#fcfcfb",
    "figure.facecolor": "white",
})


def _parse_area(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_built_year(series: pd.Series) -> pd.Series:
    year = series.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(year, errors="coerce")


def _parse_station_distance(series: pd.Series) -> pd.Series:
    range_map = {
        "30分～60分": 45,
        "1H～1H30": 75,
        "1H30～2H": 105,
        "2H～": 120,
    }

    def convert(value):
        if pd.isna(value):
            return np.nan
        value = str(value)
        if value in range_map:
            return range_map[value]
        match = re.match(r"^\d+$", value)
        return float(match.group()) if match else np.nan

    return series.map(convert)


def _parse_quarter(series: pd.Series) -> pd.DataFrame:
    extracted = series.astype(str).str.extract(r"(\d{4})年第(\d)四半期")
    year = pd.to_numeric(extracted[0], errors="coerce")
    quarter = pd.to_numeric(extracted[1], errors="coerce")
    order = year * 4 + quarter
    label = year.astype("Int64").astype(str) + "Q" + quarter.astype("Int64").astype(str)
    return pd.DataFrame({"order": order, "label": label})


def plot_price_log_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    log_price = np.log10(df[TARGET_COLUMN])
    ax.hist(log_price, bins=80, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("log10(取引価格（総額）)")
    ax.set_ylabel("件数")
    ax.set_title("取引価格の分布（log10）")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = out_dir / "price_log_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_quarterly_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    quarter = _parse_quarter(df["取引時期"])
    tmp = pd.DataFrame({
        "order": quarter["order"],
        "label": quarter["label"],
        "price": df[TARGET_COLUMN],
    }).dropna(subset=["order"])
    trend = (
        tmp.groupby(["order", "label"])["price"]
        .median()
        .reset_index()
        .sort_values("order")
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(trend["label"], trend["price"], color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.set_ylabel("取引価格（総額）中央値")
    ax.set_title("四半期ごとの価格トレンド（学習データ）")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=90)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = out_dir / "quarterly_price_trend.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_scatter_panels(df: pd.DataFrame, out_dir: Path) -> Path:
    area = _parse_area(df["面積（㎡）"])
    built_year = _parse_built_year(df["建築年"])
    age = 2026 - built_year
    distance = _parse_station_distance(df["最寄駅：距離（分）"])
    price = df[TARGET_COLUMN]

    panels = [
        ("面積（㎡）", area, True),
        ("築年数（年）", age, False),
        ("最寄駅：距離（分、簡易換算）", distance, False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (label, x, log_x) in zip(axes, panels):
        mask = x.notna() & price.notna()
        hb = ax.hexbin(
            x[mask], price[mask],
            gridsize=40, cmap=SEQUENTIAL_BLUE, mincnt=1,
            xscale="log" if log_x else "linear", yscale="log",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("取引価格（総額）")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.colorbar(hb, ax=ax, label="件数", shrink=0.8)
    fig.suptitle("特徴量 vs 取引価格")
    fig.tight_layout()
    path = out_dir / "feature_scatter_panels.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _format_category_labels(series: pd.Series, fillna_label: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.map(lambda v: fillna_label if pd.isna(v) else str(int(v)))
    return series.fillna(fillna_label).astype(str)


def plot_price_by_category(
    df: pd.DataFrame,
    column: str,
    filename: str,
    out_dir: Path,
    top_n: int | None = None,
    fillna_label: str = "不明",
) -> Path:
    """カテゴリ列ごとの価格分布（中央値で昇順の横boxplot）。

    magnitudeによる並べ替えが目的なので色はidentityを持たせず単色で統一する。
    """
    log_price = np.log10(df[TARGET_COLUMN])
    labels = _format_category_labels(df[column], fillna_label)
    title_suffix = ""
    if top_n is not None and labels.nunique() > top_n:
        total_categories = labels.nunique()
        top_categories = labels.value_counts().nlargest(top_n).index
        mask = labels.isin(top_categories)
        labels = labels[mask]
        log_price = log_price[mask]
        title_suffix = f"（取引件数上位{top_n}／全{total_categories}種）"

    tmp = pd.DataFrame({"cat": labels, "log_price": log_price})
    order = tmp.groupby("cat")["log_price"].median().sort_values().index
    data = [tmp.loc[tmp["cat"] == c, "log_price"].values for c in order]

    fig_height = max(3.5, 0.3 * len(order) + 1.3)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    box = ax.boxplot(
        data, orientation="horizontal", tick_labels=list(order), patch_artist=True,
        flierprops={"markersize": 2, "markerfacecolor": MUTED_INK, "markeredgecolor": "none"},
        medianprops={"color": "#184f95", "linewidth": 1.5},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(BOX_FILL)
        patch.set_edgecolor(BOX_EDGE)
    ax.set_xlabel("log10(取引価格（総額）)")
    ax.set_title(f"{column}別の価格分布（中央値で昇順）{title_suffix}")
    ax.grid(axis="x", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# 高基数（地区名・最寄駅：名称）は取引件数上位のみに絞って表示する
CATEGORICAL_PLOTS = [
    dict(column="市区町村名", filename="price_by_ward.png"),
    dict(column="間取り", filename="price_by_layout.png"),
    dict(column="建物の構造", filename="price_by_structure.png"),
    dict(column="用途", filename="price_by_usage.png"),
    dict(column="今後の利用目的", filename="price_by_future_use.png"),
    dict(column="都市計画", filename="price_by_city_planning.png"),
    dict(column="建ぺい率（％）", filename="price_by_coverage_ratio.png"),
    dict(column="容積率（％）", filename="price_by_floor_area_ratio.png"),
    dict(column="改装", filename="price_by_renovation.png"),
    dict(column="取引の事情等", filename="price_by_circumstances.png"),
    dict(column="地区名", filename="price_by_district_top30.png", top_n=30),
    dict(column="最寄駅：名称", filename="price_by_station_top30.png", top_n=30),
]


def plot_price_by_price_category(df: pd.DataFrame, out_dir: Path) -> Path:
    log_price = np.log10(df[TARGET_COLUMN])
    categories = df["価格情報区分"].unique().tolist()
    data = [log_price[df["価格情報区分"] == c].values for c in categories]
    colors = [BLUE, ORANGE]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    box = ax.boxplot(
        data, tick_labels=categories, patch_artist=True,
        flierprops={"markersize": 2, "markerfacecolor": MUTED_INK, "markeredgecolor": "none"},
        medianprops={"color": "#0b0b0b", "linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("log10(取引価格（総額）)")
    ax.set_title("価格情報区分別の価格分布")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = out_dir / "price_by_price_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(out_dir: Path = FIGURES_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_train()
    paths = [
        plot_price_log_distribution(df, out_dir),
        plot_quarterly_trend(df, out_dir),
        plot_feature_scatter_panels(df, out_dir),
        plot_price_by_price_category(df, out_dir),
    ]
    for cfg in CATEGORICAL_PLOTS:
        paths.append(plot_price_by_category(df, out_dir=out_dir, **cfg))
    for path in paths:
        print("saved:", path)


if __name__ == "__main__":
    main()
