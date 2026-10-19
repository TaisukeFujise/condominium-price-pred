#!/bin/sh
# 候補 C（main_s1.py：TE を SMOOTHING 1 に替え、総額・SMOOTHING 1 の TE を足す 12 モデル）を seed 42・1・2 で回す。
cd /Users/fujisetaisuke/.herdr/worktrees/condominium-resale-price-pred/analysis-fable-errors || exit 1
for s in 42 1 2; do
  echo "=== C seed $s 開始 $(date '+%H:%M')"
  python3 ~/condo-overnight/heavy.py uv run python main_s1.py $s > analysis/logs/main_s1_seed$s.log 2>&1
  mv submissions/submission.csv submissions/submission_s1_seed$s.csv
  echo "=== C seed $s 終了 $(date '+%H:%M')"
done
echo "=== ALL DONE $(date '+%H:%M')"
