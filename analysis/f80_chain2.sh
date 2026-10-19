#!/bin/sh
# Issue #80 の 3（SMOOTHING 1・10・30、単位は今のまま）を保管庫に足して eval する。feat-overnight の直下で動かす。
MINE=/Users/fujisetaisuke/.herdr/worktrees/condominium-resale-price-pred/analysis-fable-errors
cd /Users/fujisetaisuke/.herdr/worktrees/condominium-resale-price-pred/feat-overnight || exit 1
export PYTHONPATH=.:$MINE
BANK=/Users/fujisetaisuke/condo-overnight/oof_bank.py
BASE="lgbm lgbm_total catboost linear_residual lgbm_over30 lgbm_area30 x72_seiyaku x72_torihiki x72_not23 x72_hedonic"
for s in s1 s10 s30; do
  echo "=== add f80_te_$s $(date '+%H:%M')"
  python3 ~/condo-overnight/heavy.py uv run python $BANK add f80_te_$s f80.models:$s 2>&1 | grep -v "Warning\|true_divide\|log_preds"
done
echo "=== eval $(date '+%H:%M')"
for names in "$BASE x72_te" "$BASE f80_te_s1" "$BASE f80_te_s10" "$BASE f80_te_s30" "$BASE x72_te f80_te_s1" "$BASE x72_te f80_te_s10" "$BASE x72_te f80_te_s30" "$BASE x72_te f80_te_total f80_te_s10" "$BASE x72_te f80_te_total f80_te_s30" "$BASE x72_te f80_te_station f80_te_total f80_te_s30"; do
  echo "--- $names"
  uv run python $BANK eval $names 2>&1 | grep "folds:"
done
echo "=== EVAL DONE $(date '+%H:%M')"
