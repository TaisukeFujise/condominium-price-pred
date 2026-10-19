#!/bin/sh
# Issue #80 の 1・2 を、#6 の計算が終わってから保管庫に足して eval する。feat-overnight の直下で動かす。
MINE=/Users/fujisetaisuke/.herdr/worktrees/condominium-resale-price-pred/analysis-fable-errors
cd /Users/fujisetaisuke/.herdr/worktrees/condominium-resale-price-pred/feat-overnight || exit 1
until [ -f "$MINE/submissions/submission_outlier_cut1.0.csv" ]; do sleep 30; done
echo "=== #6 終了を確認 $(date '+%H:%M')"
export PYTHONPATH=.:$MINE
BANK=/Users/fujisetaisuke/condo-overnight/oof_bank.py
BASE="lgbm lgbm_total catboost linear_residual lgbm_over30 lgbm_area30 x72_seiyaku x72_torihiki x72_not23 x72_hedonic"
echo "=== add f80_te_station $(date '+%H:%M')"
python3 ~/condo-overnight/heavy.py uv run python $BANK add f80_te_station f80.models:station 2>&1 | grep -v Warning
echo "=== add f80_te_total $(date '+%H:%M')"
python3 ~/condo-overnight/heavy.py uv run python $BANK add f80_te_total f80.models:total 2>&1 | grep -v Warning
echo "=== eval $(date '+%H:%M')"
for names in "$BASE x72_te" "$BASE x72_te f80_te_station" "$BASE f80_te_station" "$BASE x72_te f80_te_total" "$BASE x72_te f80_te_station f80_te_total" "$BASE f80_te_station f80_te_total"; do
  echo "--- $names"
  uv run python $BANK eval $names 2>&1 | grep -v Warning
done
echo "=== EVAL DONE $(date '+%H:%M')"
