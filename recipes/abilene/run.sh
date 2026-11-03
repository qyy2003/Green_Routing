#!/usr/bin/env bash
# The second dataset: Abilene (Internet2, 2004), whose traffic matrices come from NetFlow.
# Same forecasting ladder and carbon-rate method as SWITCHlan, with Abilene's own energy
# inventory (Juniper T640 routers, Qwest DWDM wavelengths) and 2004 grid factors (EPA eGRID).
#
#   recipes/abilene/run.sh data      download the archive, clean it, write the two mirrors
#   recipes/abilene/run.sh train     every tier on the links+doors panel and on the total
#   recipes/abilene/run.sh merge     one run directory per panel
#   recipes/abilene/run.sh dense     forecasts at every 5-min origin, both panels
#   recipes/abilene/run.sh carbon    energy, grid, flat and path rates, bills, forecasts of the rate
#   recipes/abilene/run.sh all
#
# Output: $GR_ARTIFACTS_DIR/abilene/{mirror_links, mirror_total, links, total, carbon}.
# The runs keep their own panel cache (panels are keyed by window, not by data source).

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
need gr-benchmark

ROOT="$GR_ARTIFACTS_DIR/abilene"
: "${PARALLEL:=3}"
AB_DATA="--start 2004-05-01 --end 2004-09-11 --min-mean-mbps 1"
AB_SPLIT="--train-end 2004-07-24 --val-start 2004-07-31 --val-end 2004-08-13 --test-start 2004-08-14"
AB_EVAL="--horizons 15min 1h 8h day week --max-origins 160"
AB_TRAIN="--lookback 2016 --epochs 100 --under-weight 2"      # no --calendar: its holiday table is Swiss, 2025-26
AB_FLAGS="$AB_DATA $AB_SPLIT $AB_EVAL $AB_TRAIN"
PARTS=(
  "baselines:1:persistence,seasonal_naive_day,seasonal_naive_week,historical_average"
  "classical:1 2:persistence,holt_winters,sarima"
  "gbdt:1 3:persistence,gbdt"
  "gru:1 4:persistence,gru"
  "tcn:1 4:persistence,tcn"
  "dlinear_patchtst:1 5:persistence,dlinear,patchtst"
)
MODELS="persistence,seasonal_naive_day,seasonal_naive_week,historical_average,holt_winters,sarima,gbdt,gru,tcn,dlinear,patchtst"

env_for() { echo "GR_DATA_SOURCE=local GR_LOCAL_ROOT='$ROOT/mirror_$1' GR_ARTIFACTS_DIR='$ROOT/cache_$1'"; }

step_data() { log "Abilene data"; "$PYTHON" "$RECIPES/abilene/make_data.py"; }

step_train() {
  local args=() name tiers only rest
  for panel in links total; do
    for part in "${PARTS[@]}"; do
      name="${part%%:*}"; rest="${part#*:}"; tiers="${rest%%:*}"; only="${rest#*:}"
      [[ -f "$ROOT/$panel/part_$name/aggregate.csv" ]] && continue
      args+=("${panel}_$name" "$(env_for $panel) gr-benchmark $AB_FLAGS --tiers $tiers --only $only --out '$ROOT/$panel/part_$name'")
    done
  done
  ((${#args[@]})) || { log "all parts complete"; return; }
  pool "$PARALLEL" "$ROOT/logs" "${args[@]}"
}

step_merge() {
  for panel in links total; do
    local M="$ROOT/$panel/merged"
    mkdir -p "$M"/{records,checkpoints,history}
    for part in "${PARTS[@]}"; do
      local p="$ROOT/$panel/part_${part%%:*}"
      cp -f "$p"/records/*.csv "$M/records/"; cp -f "$p"/checkpoints/* "$M/checkpoints/" 2>/dev/null || true
      cp -f "$p"/history/* "$M/history/" 2>/dev/null || true
    done
    eval "$(env_for $panel) gr-benchmark $AB_FLAGS --tiers 1 2 3 4 5 --only $MODELS --resume --out '$M'" > "$ROOT/logs/merge_$panel.out" 2>&1
    tail -n 12 "$ROOT/logs/merge_$panel.out"
  done
}

step_dense() {
  for panel in links total; do
    local M="$ROOT/$panel/merged"
    eval "$(env_for $panel) gr-dense predict --run '$M' --models persistence,seasonal_naive_day,seasonal_naive_week,historical_average,gbdt"
    for m in dlinear patchtst gru tcn; do eval "$(env_for $panel) gr-dense predict --run '$M' --models $m --batch 512"; done
    eval "$(env_for $panel) gr-dense metrics --run '$M' --common"
  done
}

step_carbon() { log "Abilene carbon"; "$PYTHON" "$RECIPES/abilene/carbon.py"; }

case "${1:-}" in
  data|train|merge|dense|carbon) "step_$1" ;;
  all) step_data; step_train; step_merge; step_dense; step_carbon ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
