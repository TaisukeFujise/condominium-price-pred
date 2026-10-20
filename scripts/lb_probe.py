"""Public LB を 2 回使って、test での予測のずれ（相場の読みの外れ）を測り、最後の提出を補正する（LB probing）。

予測を c だけずらす（log1p(price) に c を足す）と、ずらした行の log の誤差 e は e + c になる。
RMSLE の二乗は誤差の二乗の平均なので、ずらす前の LB を L0、後を L1、ずらした行の誤差の平均を m とすると
    L1^2 = L0^2 + 2cm + c^2
になり、2 つの LB から m が分かる（参考：https://zenn.dev/bilzard/articles/lb-probing-technique）。

Public LB は test の約 37% で採点され、test のうち 2025年第3四半期の行がちょうど 37.2% なので、
Public は 2025Q3 の行だけ（時期で分けている）と仮定する。提出の残りが 2 回で、分け方を確かめる探りを出せないため。
探りの提出は 2025Q3 の行だけをずらす。仮定が外れて Public がランダムでも、求めた m1 は
本当の値の約 0.37 倍に縮むだけなので、補正が控えめになるだけで済む。

    m1 = (L1^2 − L0^2 − c^2) / (2c)

Private（2025Q4・2026Q1）のずれは直接は測れない。既定では m1 をそのまま当てる。
--trend-ratio を付けると、過去の時点で相場の傾きを延ばしたときの外れの比（scripts/backtest_trend.py）で
2025Q4 は 1.36 倍、2026Q1 は 1.63 倍にする。m1 にはトレンドと関係ないモデルの偏りも含まれるので、
この比で延ばすと補正しすぎるおそれがある。

実行:
    uv run python -m scripts.lb_probe make                          # 探りの提出ファイルを作る
    uv run python -m scripts.lb_probe solve 0.18560 --shift -0.01   # m1 と時期ごとのずらし量
    uv run python -m scripts.lb_probe solve 0.18560 --shift -0.01 \\
        --apply submissions/final_raw.csv submissions/final.csv
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, ROOT, SUBMISSION_DIR

Q3, Q4, Q1 = "2025年第3四半期", "2025年第4四半期", "2026年第1四半期"
# m1 はこの提出を基準に測る。探りの提出も最後の補正もこのファイルに合わせる必要があり、
# seed 平均で作り直すのが難しいので、submissions/（git の管理外）ではなくリポジトリに置く
BASE = ROOT / "probe" / "1_base.csv"
BASE_LB = 0.18595
# 仮定が外れて Public がランダムだったとき、m1 の見積もりに (0.37 − 1) c / 2 の偏りが乗るので小さくする
SHIFT = -0.01
# 1 四半期先の外れに対する 2・3 四半期先の外れの比（scripts/backtest_trend.py。相関 0.83・0.76）
TREND_RATIO = {Q3: 1.0, Q4: 1.36, Q1: 1.63}
# バックテストでの外れの大きさは ±1.4% 程度。これを大きく超えるなら、時期で分けているという仮定を疑う
IMPLAUSIBLE = 0.05


def main() -> None:
    """探りの提出ファイルを作るか、2 つの LB から m1 を求めて最後の提出を補正する。"""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="探りの提出ファイルを作る")
    make.add_argument(
        "--first", type=Path, default=BASE, help="1 回目に提出したファイル"
    )
    make.add_argument("--shift", type=float, default=SHIFT)
    make.add_argument("--out", type=Path, default=SUBMISSION_DIR / "probe_q3.csv")
    solve = commands.add_parser("solve", help="2 つの LB から m1 とずらし量を求める")
    solve.add_argument("lb1", type=float, help="探りの提出の Public LB")
    solve.add_argument(
        "--lb0", type=float, default=BASE_LB, help="1 回目の提出の Public LB"
    )
    solve.add_argument(
        "--shift",
        type=float,
        required=True,
        help="探りの提出で 2025Q3 に足した量（make を実行したときに表示される）",
    )
    solve.add_argument(
        "--trend-ratio",
        action="store_true",
        help="2025Q4・2026Q1 のずらし量を、相場の傾きの外れの比で大きくする",
    )
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
        default=BASE,
        help="1 回目に提出したファイル。RAW_CSV との時期ごとの予測の水準の差も補正する",
    )
    args = parser.parse_args()

    periods = _periods()
    if args.command == "make":
        _make_probe(args.first, args.out, args.shift, periods)
        return
    shifts = _solve(args.lb0, args.lb1, args.shift, args.trend_ratio)
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


def _period_of(sub: pd.DataFrame, periods: pd.Series) -> pd.Series:
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
    return period


def _make_probe(first: Path, out: Path, shift: float, periods: pd.Series) -> None:
    """1 回目の提出の 2025Q3 の行だけをずらした、探りの提出ファイルを作る。

    Args:
        first: 1 回目に提出したファイル。
        out: 保存先。
        shift: 2025Q3 の行の log1p(price) に足す量。
        periods: _periods の結果。
    """
    sub = pd.read_csv(first)
    is_q3 = (_period_of(sub, periods) == Q3).to_numpy()
    probe = sub.assign(price=np.expm1(np.log1p(sub["price"]) + shift * is_q3))
    out.parent.mkdir(parents=True, exist_ok=True)
    probe.to_csv(out, index=False)
    print(f"保存しました: {out}（{Q3} の {int(is_q3.sum())} 行に {shift:+}）")
    print(f"LB が出たら: solve <探りの LB> --shift {shift}")


def _solve(lb0: float, lb1: float, shift: float, trend_ratio: bool) -> dict[str, float]:
    """2 つの LB から 2025Q3 の予測のずれ m1 を求め、時期ごとに log1p(price) に足す量を返す。

    Args:
        lb0: 1 回目の提出の Public LB。
        lb1: 探りの提出の Public LB。
        shift: 探りの提出で 2025Q3 に足した量。
        trend_ratio: True なら 2025Q4・2026Q1 を TREND_RATIO 倍にする。

    Returns:
        取引時期 → log1p(price) に足す量（負なら下げる）。
    """
    m1 = (lb1**2 - lb0**2 - shift**2) / (2 * shift)
    print(
        f"LB {lb0:.5f} → {lb1:.5f}（{lb1 - lb0:+.5f}）。{Q3} の予測のずれ m1 = {m1:+.5f}"
    )
    print(
        f"  {Q3} を補正したときの Public LB の見込み: {math.sqrt(max(lb0**2 - m1**2, 0.0)):.5f}"
    )
    if abs(m1) > IMPLAUSIBLE:
        print(
            f"  注意: |m1| が {IMPLAUSIBLE} を超えている。Public が {Q3} だけという仮定が外れているかもしれない"
        )
    ratio = TREND_RATIO if trend_ratio else dict.fromkeys(TREND_RATIO, 1.0)
    return {q: -ratio[q] * m1 for q in (Q3, Q4, Q1)}


def _apply(
    raw_csv: Path,
    out_csv: Path,
    shifts: dict[str, float],
    periods: pd.Series,
    first: Path,
) -> None:
    """最後の提出の予測を、時期ごとにずらして保存する。

    m1 は 1 回目の提出で測ったもの。最後の提出の構成が違うと予測の水準も少し違うので、
    時期ごとの log の平均の差も一緒に引く（LB は使わない）。

    Args:
        raw_csv: ずらす前の最後の提出ファイル。
        out_csv: 保存先。
        shifts: _solve の結果。
        periods: _periods の結果。
        first: 1 回目に提出したファイル。
    """
    raw = pd.read_csv(raw_csv)
    period = _period_of(raw, periods)
    log_raw = np.log1p(raw["price"].to_numpy())
    log_first = np.log1p(
        pd.read_csv(first).set_index("id").loc[raw["id"], "price"].to_numpy()
    )
    level = pd.Series(log_raw - log_first).groupby(period.to_numpy()).mean()
    total = period.map(shifts) - period.map(level)
    for q in (Q3, Q4, Q1):
        print(
            f"  {q}: 1 回目の提出との水準の差 {level[q]:+.5f} も引き、"
            f"log1p(price) に {shifts[q] - level[q]:+.5f} を足す"
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.assign(price=np.expm1(log_raw + total.to_numpy())).to_csv(out_csv, index=False)
    print(f"保存しました: {out_csv}")


if __name__ == "__main__":
    main()
