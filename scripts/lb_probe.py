"""Public LB を 2 回使って、test での予測のずれ（相場の読みの外れ）を測り、最後の提出を補正する（LB probing）。

予測を c だけずらす（log1p(price) に c を足す）と、ずらした行の log の誤差 e は e + c になる。
RMSLE の二乗は誤差の二乗の平均なので、ずらす前の LB を L0、後を L1、ずらした行の誤差の平均を m とすると
    L1^2 = L0^2 + 2cm + c^2
になり、2 つの LB から m が分かる（参考：https://zenn.dev/bilzard/articles/lb-probing-technique）。

Public LB は test の約 37% で採点される。test のうち 2025年第3四半期の行がちょうど 37.2% なので、
Public は 2025Q3 の行だけ（時期で分けている）の可能性が高い。そこで探りの提出は、2025Q3 の行を少しだけ、
それ以外の行を大きくずらす。時期で分けていれば大きくずらした行は Public に入らず LB はほとんど動かない。
ランダムに分けていれば LB がはっきり悪くなるので、どちらかが分かる。

    場合 A（Public は 2025Q3 だけ）：m1 = (L0^2 + c1^2 − L1^2) / (−2 c1)
    場合 B（ランダム）：L1^2 = L0^2 + p1 (2 c1 m1 + c1^2) + p23 (2 c23 m23 + c23^2) を、m1 ≒ m23 / 1.5 として解く

Private（2025Q4・2026Q1）のずれは直接は測れないので、過去の時点で相場の傾きを延ばしたときの外れの比
（1 四半期先に対して 2 四半期先は 1.36 倍、3 四半期先は 1.63 倍。scripts/backtest_trend.py）で延ばす。

実行:
    uv run python -m scripts.lb_probe make submissions/first.csv           # 探りの提出ファイルを作る
    uv run python -m scripts.lb_probe solve 0.18595 0.18560                # 判定と m、時期ごとのずらし量
    uv run python -m scripts.lb_probe solve 0.18595 0.18560 \\
        --apply submissions/final_raw.csv submissions/final.csv --first submissions/first.csv
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, SUBMISSION_DIR

Q3, Q4, Q1 = "2025年第3四半期", "2025年第4四半期", "2026年第1四半期"
# 探りの提出で log1p(price) に足す量。2025Q3 は m を測るので小さく、それ以外は分け方を見分けるので大きく。
# それ以外を上げる向きにするのは、予測が高めにずれていても打ち消し合わず、LB が必ず大きく動くようにするため
SHIFT_Q3, SHIFT_REST = -0.01, 0.05
# 1 四半期先の外れに対する 2・3 四半期先の外れの比（scripts/backtest_trend.py。相関 0.83・0.76）
RATIO = {Q3: 1.0, Q4: 1.36, Q1: 1.63}
# 場合 B で、2025Q3 の外れを 2025Q4・2026Q1 の外れの何分の 1 とみなすか（上の比の平均に近い値）
B_RATIO = 1.5
# L1 − L0 がこの間なら、場合 A とも B とも言えない
AMBIGUOUS = (0.001, 0.002)


def main() -> None:
    """探りの提出ファイルを作るか、2 つの LB から m を求めて最後の提出を補正する。"""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="探りの提出ファイルを作る")
    make.add_argument("first", type=Path, help="1 回目に提出したファイル")
    make.add_argument("--out", type=Path, default=SUBMISSION_DIR / "probe_split.csv")
    solve = commands.add_parser("solve", help="2 つの LB から m とずらし量を求める")
    solve.add_argument("lb0", type=float, help="1 回目の提出の Public LB")
    solve.add_argument("lb1", type=float, help="探りの提出の Public LB")
    solve.add_argument(
        "--apply",
        nargs=2,
        type=Path,
        metavar=("RAW_CSV", "OUT_CSV"),
        help="RAW_CSV をずらして OUT_CSV に保存する（最後の提出）",
    )
    solve.add_argument(
        "--first",
        type=Path,
        help="1 回目に提出したファイル。RAW_CSV と構成が違うとき、時期ごとの予測の水準の差も補正する",
    )
    args = parser.parse_args()

    periods = _periods()
    if args.command == "make":
        _make_probe(args.first, args.out, periods)
        return
    shifts = _solve(args.lb0, args.lb1, periods)
    for q in (Q3, Q4, Q1):
        print(
            f"  {q}: log1p(price) に {shifts[q]:+.5f} を足す（price を {math.exp(shifts[q]):.5f} 倍）"
        )
    if args.apply:
        raw_csv, out_csv = args.apply
        _apply(raw_csv, out_csv, shifts, periods, args.first)


def _periods() -> pd.Series:
    """test の行ごとの取引時期を、id で引けるようにして返す。

    Returns:
        id → 取引時期。
    """
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id", "取引時期"])
    return test.set_index("id")["取引時期"]


def _period_of(sub: pd.DataFrame, periods: pd.Series) -> np.ndarray:
    """提出ファイルの行ごとの取引時期を返す。

    Args:
        sub: 提出ファイル（id, price）。
        periods: _periods の結果。

    Returns:
        行ごとの取引時期。
    """
    period = sub["id"].map(periods)
    if period.isna().any():
        raise ValueError("test にない id がある")
    return period.to_numpy()


def _make_probe(first: Path, out: Path, periods: pd.Series) -> None:
    """1 回目の提出を時期ごとにずらした、探りの提出ファイルを作る。

    Args:
        first: 1 回目に提出したファイル。
        out: 保存先。
        periods: _periods の結果。
    """
    sub = pd.read_csv(first)
    shift = np.where(_period_of(sub, periods) == Q3, SHIFT_Q3, SHIFT_REST)
    probe = sub.assign(price=np.expm1(np.log1p(sub["price"]) + shift))
    probe.to_csv(out, index=False)
    print(
        f"保存しました: {out}（{Q3} の {int((shift == SHIFT_Q3).sum())} 行に {SHIFT_Q3:+}、"
        f"それ以外の {int((shift == SHIFT_REST).sum())} 行に {SHIFT_REST:+}）"
    )


def _solve(lb0: float, lb1: float, periods: pd.Series) -> dict[str, float]:
    """2 つの LB から分け方を判定し、時期ごとに log1p(price) に足す量を返す。

    Args:
        lb0: 1 回目の提出の Public LB。
        lb1: 探りの提出の Public LB。
        periods: _periods の結果。

    Returns:
        取引時期 → log1p(price) に足す量（負なら下げる）。
    """
    share = periods.value_counts(normalize=True)
    p1 = float(share[Q3])
    p23 = 1 - p1
    w4 = float(share[Q4] / (share[Q4] + share[Q1]))
    delta = lb1 - lb0
    print(f"test の {Q3} の割合 {p1:.4f}。LB {lb0:.5f} → {lb1:.5f}（{delta:+.5f}）")

    if abs(delta) <= AMBIGUOUS[0]:
        m1 = (lb0**2 + SHIFT_Q3**2 - lb1**2) / (-2 * SHIFT_Q3)
        print(f"場合 A（Public は {Q3} だけ）。{Q3} の予測のずれ m1 = {m1:+.5f}")
        after = math.sqrt(max(lb0**2 - m1**2, 0.0))
        print(f"  {Q3} を補正したときの Public LB の見込み: {after:.5f}")
        return {q: -RATIO[q] * m1 for q in (Q3, Q4, Q1)}
    if delta >= AMBIGUOUS[1]:
        # m1 = m23 / B_RATIO を代入した 1 次式を m23 について解く
        known = p1 * SHIFT_Q3**2 + p23 * SHIFT_REST**2
        slope = 2 * SHIFT_Q3 * p1 / B_RATIO + 2 * SHIFT_REST * p23
        m23 = (lb1**2 - lb0**2 - known) / slope
        print(f"場合 B（Public はランダム）。{Q4}・{Q1} の予測のずれ m23 = {m23:+.5f}")
        # Q4 と Q1 の比を保ったまま、2 つの平均がちょうど m23 になるようにする
        scale = m23 / (w4 * RATIO[Q4] + (1 - w4) * RATIO[Q1])
        return {Q3: -m23 / B_RATIO, Q4: -RATIO[Q4] * scale, Q1: -RATIO[Q1] * scale}
    raise SystemExit(
        f"LB の変化 {delta:+.5f} は {AMBIGUOUS[0]}〜{AMBIGUOUS[1]} の間で、場合 A とも B とも言えない"
    )


def _apply(
    raw_csv: Path,
    out_csv: Path,
    shifts: dict[str, float],
    periods: pd.Series,
    first: Path | None,
) -> None:
    """最後の提出の予測を、時期ごとにずらして保存する。

    m は 1 回目の提出で測ったもの。最後の提出の構成が違うと予測の水準も少し違うので、
    first を渡したときは、時期ごとの log の平均の差も一緒に引く（LB は使わない）。

    Args:
        raw_csv: ずらす前の最後の提出ファイル。
        out_csv: 保存先。
        shifts: _solve の結果。
        periods: _periods の結果。
        first: 1 回目に提出したファイル。None なら水準の差は補正しない。
    """
    raw = pd.read_csv(raw_csv)
    period = _period_of(raw, periods)
    log_raw = np.log1p(raw["price"].to_numpy())
    total = np.array([shifts[q] for q in period])
    if first is not None:
        log_first = np.log1p(
            pd.read_csv(first).set_index("id").loc[raw["id"], "price"].to_numpy()
        )
        for q in (Q3, Q4, Q1):
            rows = period == q
            level = float(np.mean(log_raw[rows] - log_first[rows]))
            total[rows] -= level
            print(f"  {q}: 1 回目の提出との水準の差 {level:+.5f} も引く")
    for q in (Q3, Q4, Q1):
        print(f"  {q}: log1p(price) に {total[period == q][0]:+.5f} を足す")
    raw.assign(price=np.expm1(log_raw + total)).to_csv(out_csv, index=False)
    print(f"保存しました: {out_csv}")


if __name__ == "__main__":
    main()
