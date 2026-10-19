#!/bin/sh
# seed 2（駅入り＋総額）を回し、引数が 1 なら続けて寄せる強さ 1 の版（main_s1.py）を seed 42・1・2 で回す。
# 使い方: sh analysis/final_chain.sh <0|1>
cd /Users/fujisetaisuke/.herdr/worktrees/condominium-resale-price-pred/analysis-fable-errors || exit 1
echo "=== seed 2（駅入り＋総額）開始 $(date '+%H:%M')"
python3 ~/condo-overnight/heavy.py uv run python -m scripts.seed_average --one 2 > analysis/logs/main_f80_seed2.log 2>&1
echo "=== seed 2 終了 $(date '+%H:%M')"
if [ "$1" = "1" ]; then
  for s in 42 1 2; do
    echo "=== 寄せる強さ 1 の版 seed $s 開始 $(date '+%H:%M')"
    python3 ~/condo-overnight/heavy.py uv run python main_s1.py $s > analysis/logs/main_s1_seed$s.log 2>&1
    mv submissions/submission.csv submissions/submission_s1_seed$s.csv
    echo "=== seed $s 終了 $(date '+%H:%M')"
  done
fi
echo "=== ALL DONE $(date '+%H:%M')"
