"""#77 のコメント（kataokake さん）の構成 #12 を #71 の上で確かめる（Issue #80）。

main.py の MODELS のうち TargetEncodedLGBM（SMOOTHING 3）を、SMOOTHING 1 の TE と総額・SMOOTHING 1 の TE の
2 本に替えて main.main() を動かす。main.py 自体は変えない（seed_average が main.py を読むため）。
実行: python3 ~/condo-overnight/heavy.py uv run python main_s1.py [seed]
引数の seed は src.config.SEED を main.py より先に書き換える（scripts/seed_average.py と同じやり方）。
"""

import sys
from functools import partial

import src.config

if len(sys.argv) > 1:
    src.config.SEED = int(sys.argv[1])

import main  # SEED を書き換えた後に読む
from f80.te import BASE_UNITS, ParamTE
from src.models.target_encoding import TargetEncodedLGBM

main.MODELS = [
    m
    for m in main.MODELS
    if not (
        m is TargetEncodedLGBM
        or (isinstance(m, partial) and m.func is ParamTE)
        or (isinstance(m, partial) and m.func is TargetEncodedLGBM)
    )
] + [
    partial(ParamTE, units=BASE_UNITS, smoothing=1.0),
    partial(ParamTE, units=BASE_UNITS, smoothing=1.0, per_area=False),
]

if __name__ == "__main__":
    print("MODELS:", len(main.MODELS))
    main.main()
