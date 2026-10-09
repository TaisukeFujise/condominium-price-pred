"""全員の実験記録（experiments/ 以下の .json）を一覧にする。

実行: uv run python -m src.summary
"""

import json

import pandas as pd

from src import validation
from src.config import EXPERIMENTS_DIR


def main() -> None:
    records = [json.loads(p.read_text()) for p in sorted(EXPERIMENTS_DIR.rglob("*.json"))]
    if not records:
        print("実験記録がまだありません。")
        return

    table = pd.DataFrame(
        {
            "実験 ID": r["exp_id"],
            "CV": round(r["cv_rmsle"], 5),
            "LB": "-" if r["lb_public"] is None else r["lb_public"],
            "検証": r["validation_version"],
            "内容": r["description"],
        }
        for r in records
    ).sort_values("CV")
    print(table.to_string(index=False))

    old = table[table["検証"] != validation.VERSION]
    if not old.empty:
        print(
            f"\n注意: 検証が {validation.VERSION} でない実験が {len(old)} 件あります。"
            "検証のバージョンが違う CV どうしは比べられません。"
        )


if __name__ == "__main__":
    main()
