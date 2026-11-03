#!/usr/bin/env bash
# Two ways to set SWITCHlan's carbon rate (g CO2 per GB), and how well each can be forecast.
#
#   Flat rate:   one rate per hour for every byte = backbone emissions / total traffic.
#                The total traffic is forecast by ONE model trained on the total series;
#                every tier of the forecasting ladder is tried.
#   Path rate:   every router and optical site splits its emissions among the bytes that
#                cross it; a byte pays the sum along its path. Who-talks-to-whom comes from
#                the counters (tomogravity), checked against real flow data (Abilene).
#
#   recipes/carbon_rates/run.sh total     the total series (all doors) as a one-series mirror
#   recipes/carbon_rates/run.sh train     every tier on the total, parts in parallel
#   recipes/carbon_rates/run.sh merge     one run directory with every model's tables
#   recipes/carbon_rates/run.sh dense     forecasts at every 5-min origin (rolling + block)
#   recipes/carbon_rates/run.sh flat      flat-rate intensity forecast errors
#   recipes/carbon_rates/run.sh path      path-based allocation on SWITCHlan (counters)
#   (flow data vs counters on Abilene: recipes/abilene/run.sh, which the report reads)
#   recipes/carbon_rates/run.sh report    $GR_ARTIFACTS_DIR/carbon_rates/report/carbon-rates.html
#   recipes/carbon_rates/run.sh all       every step
#
# Needs recipes/year (flags, dense logs), recipes/carbon (doors, inventory, grid) and, for the
# report, recipes/abilene (the second dataset) first.
# The total-traffic run keeps its own panel cache: panels are keyed by window, not source.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
source "$RECIPES/year/flags.sh"
need gr-benchmark
need gr-carbon

ROOT="$GR_ARTIFACTS_DIR/carbon_rates"
MIRROR="$ROOT/mirror"
FLAT="$ROOT/flat"
MERGED="$FLAT/merged"
: "${PARALLEL:=3}"

# the total-traffic runs read the mirror and cache their panel apart from the year panel
flat_env() { echo "GR_DATA_SOURCE=local GR_LOCAL_ROOT='$MIRROR' GR_ARTIFACTS_DIR='$ROOT'"; }
FLAT_TRAIN="--lookback 2016 --epochs 100 --calendar --under-weight 2"
FLAT_FLAGS="$YEAR_DATA $YEAR_SPLIT $YEAR_EVAL $FLAT_TRAIN"
FLAT_PARTS=(
  "baselines:1:persistence,seasonal_naive_day,seasonal_naive_week,historical_average"
  "classical:1 2:persistence,holt_winters,sarima"
  "gbdt:1 3:persistence,gbdt"
  "gru:1 4:persistence,gru"
  "tcn:1 4:persistence,tcn"
  "dlinear_patchtst:1 5:persistence,dlinear,patchtst"
)
FLAT_MODELS="persistence,seasonal_naive_day,seasonal_naive_week,historical_average,holt_winters,sarima,gbdt,gru,tcn,dlinear,patchtst"

step_total() { log "total series"; "$PYTHON" "$RECIPES/carbon_rates/make_total.py" "$MIRROR"; }

step_train() {
  local args=() name tiers only rest
  for part in "${FLAT_PARTS[@]}"; do
    name="${part%%:*}"; rest="${part#*:}"; tiers="${rest%%:*}"; only="${rest#*:}"
    if [[ -f "$FLAT/part_$name/aggregate.csv" ]]; then log "part_$name complete — skipping"; continue; fi
    args+=("part_$name" "$(flat_env) gr-benchmark $FLAT_FLAGS --tiers $tiers --only $only --out '$FLAT/part_$name'")
  done
  ((${#args[@]})) || { log "all parts complete"; return; }
  pool "$PARALLEL" "$FLAT/logs" "${args[@]}"
}

step_merge() {
  mkdir -p "$MERGED"/{records,checkpoints,history} "$FLAT/logs"
  for part in "${FLAT_PARTS[@]}"; do
    local p="$FLAT/part_${part%%:*}"
    [[ -f "$p/aggregate.csv" ]] || { echo "missing $p — run 'train' first" >&2; exit 1; }
    cp -f "$p"/records/*.csv "$MERGED/records/"
    cp -f "$p"/checkpoints/* "$MERGED/checkpoints/" 2>/dev/null || true
    cp -f "$p"/history/* "$MERGED/history/" 2>/dev/null || true
  done
  log "rebuilding the tables from the merged records (no training)"
  eval "$(flat_env) gr-benchmark $FLAT_FLAGS --tiers 1 2 3 4 5 --only $FLAT_MODELS --resume --out '$MERGED'" \
    > "$FLAT/logs/merge.out" 2>&1
  tail -n 25 "$FLAT/logs/merge.out"
}

step_dense() {
  log "dense logs (holt_winters and sarima have none: they refit per origin)"
  eval "$(flat_env) gr-dense predict --run '$MERGED' --models persistence,seasonal_naive_day,seasonal_naive_week,historical_average,gbdt"
  for m in dlinear patchtst gru tcn; do eval "$(flat_env) gr-dense predict --run '$MERGED' --models $m --batch 512"; done
  eval "$(flat_env) gr-dense metrics --run '$MERGED' --common"
}

step_flat()    { log "flat rate"; "$PYTHON" "$RECIPES/carbon_rates/flat_rate.py"; }
step_path()    { log "path rate (counters)"; "$PYTHON" "$RECIPES/carbon_rates/path_rate.py"; }
step_report()  { log "report"; "$PYTHON" "$RECIPES/carbon_rates/build_report.py"; }

case "${1:-}" in
  total|train|merge|dense|flat|path|report) "step_$1" ;;
  all) step_total; step_train; step_merge; step_dense; step_flat; step_path; step_report ;;
  *) sed -n '2,22p' "$0"; exit 1 ;;
esac
