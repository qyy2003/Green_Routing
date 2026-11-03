#!/usr/bin/env bash
# The second dataset: Internet2 (2025-26), from its public GlobalNOC TSDS telemetry: port
# counters of the 73 core routers, PoP DC power, and US hourly grid factors (EIA-930); plus the
# old network's tunnel mesh (2021) as a measured traffic matrix. Same method as Switch LAN.
#
#   recipes/internet2/run.sh fetch     download (cached, rate-limited; ~3 h cold)
#   recipes/internet2/run.sh data      ports, links, doors, power; 5-minute link mirror
#   recipes/internet2/run.sh train     forecasting ladder on the directed backbone links
#   recipes/internet2/run.sh merge     one run directory
#   recipes/internet2/run.sh dense     forecasts at every 5-min origin + metrics
#   recipes/internet2/run.sh carbon    energy, grid, flat and path rates, bills, day-ahead rate
#   recipes/internet2/run.sh tunnels   the 2021 measured-matrix check
#   recipes/internet2/run.sh pathfc    day-ahead path rates from the dense link logs (after dense, carbon)
#   recipes/internet2/run.sh all
#
# Output: $GR_ARTIFACTS_DIR/internet2/{raw, data, links, carbon, tunnels}. EIA-930 files go to
# $GR_REFERENCE_DIR/eia930 (public domain).

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
need gr-benchmark

ROOT="$GR_ARTIFACTS_DIR/internet2"
: "${PARALLEL:=3}"
I2_DATA="--start 2025-11-15 --end 2026-11-01 --min-mean-mbps 1"
I2_SPLIT="--train-end 2026-06-14 --val-start 2026-06-21 --val-end 2026-07-12 --test-start 2026-07-19"
I2_EVAL="--horizons 15min 1h 8h day week --max-origins 160"
I2_TRAIN="--lookback 2016 --epochs 100 --under-weight 1"     # symmetric loss; no --calendar (Swiss table)
I2_FLAGS="$I2_DATA $I2_SPLIT $I2_EVAL $I2_TRAIN"
PARTS=(
  "baselines:1:persistence,seasonal_naive_day,seasonal_naive_week,historical_average"
  "gbdt:1 3:persistence,gbdt"
  "gru:1 4:persistence,gru"
  "tcn:1 4:persistence,tcn"
  "dlinear_patchtst:1 5:persistence,dlinear,patchtst"
)
MODELS="persistence,seasonal_naive_day,seasonal_naive_week,historical_average,gbdt,gru,tcn,dlinear,patchtst"
ENV="GR_DATA_SOURCE=local GR_LOCAL_ROOT='$ROOT/data/mirror_links' GR_ARTIFACTS_DIR='$ROOT/cache_links'"

step_fetch() {
  local eia; eia="$("$PYTHON" -c 'from green_routing.paths import reference_dir; print(reference_dir())')/eia930"
  mkdir -p "$eia"
  for p in 2024_Jul_Dec 2025_Jan_Jun 2025_Jul_Dec 2026_Jan_Jun 2026_Jul_Dec; do
    for k in BALANCE INTERCHANGE; do
      [[ -s "$eia/EIA930_${k}_$p.csv" ]] || curl -s -f -o "$eia/EIA930_${k}_$p.csv" \
        "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_${k}_$p.csv"
    done
  done
  "$PYTHON" "$RECIPES/internet2/fetch.py" inventory hourly power tunnels fivemin
}
step_data() { "$PYTHON" "$RECIPES/internet2/make_data.py"; }

step_train() {
  local args=() name tiers only rest
  for part in "${PARTS[@]}"; do
    name="${part%%:*}"; rest="${part#*:}"; tiers="${rest%%:*}"; only="${rest#*:}"
    [[ -f "$ROOT/links/part_$name/aggregate.csv" ]] && continue
    args+=("links_$name" "$ENV gr-benchmark $I2_FLAGS --tiers $tiers --only $only --out '$ROOT/links/part_$name'")
  done
  ((${#args[@]})) || { log "all parts complete"; return; }
  pool "$PARALLEL" "$ROOT/logs" "${args[@]}"
}

step_merge() {
  local M="$ROOT/links/merged"
  mkdir -p "$M"/{records,checkpoints,history} "$ROOT/logs"
  for part in "${PARTS[@]}"; do
    local p="$ROOT/links/part_${part%%:*}"
    cp -f "$p"/records/*.csv "$M/records/"; cp -f "$p"/checkpoints/* "$M/checkpoints/" 2>/dev/null || true
    cp -f "$p"/history/* "$M/history/" 2>/dev/null || true
  done
  eval "$ENV gr-benchmark $I2_FLAGS --tiers 1 3 4 5 --only $MODELS --resume --out '$M'" > "$ROOT/logs/merge.out" 2>&1
  tail -n 15 "$ROOT/logs/merge.out"
}

step_dense() {
  local M="$ROOT/links/merged"
  eval "$ENV gr-dense predict --run '$M' --models persistence,seasonal_naive_day,seasonal_naive_week,historical_average,gbdt"
  for m in dlinear patchtst gru tcn; do eval "$ENV gr-dense predict --run '$M' --models $m --batch 512"; done
  eval "$ENV gr-dense metrics --run '$M' --common"
}

step_carbon() { "$PYTHON" "$RECIPES/internet2/carbon.py"; }
step_tunnels() { "$PYTHON" "$RECIPES/internet2/tunnels.py"; }
step_pathfc() { "$PYTHON" "$RECIPES/internet2/path_dayahead.py"; }

case "${1:-}" in
  fetch|data|train|merge|dense|carbon|tunnels|pathfc) "step_$1" ;;
  all) step_fetch; step_data; step_train; step_merge; step_dense; step_carbon; step_tunnels; step_pathfc ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
