#!/usr/bin/env bash
# How to forecast SWITCHlan's total traffic a day ahead, hour by hour: the denominator of
# the flat carbon rate (recipes/carbon_rates). Every lever is scored on the same days.
#
#   recipes/flat_forecast/run.sh history   the total from 2024 on (current doors), for longer training
#   recipes/flat_forecast/run.sh ladder    the 5-minute ladder on the total, symmetric vs
#                                          peak-aware loss, three seeds each (+ dense logs)
#   recipes/flat_forecast/run.sh run       the experiments (run.py): hourly models, prior
#                                          knowledge, training process, foundation models,
#                                          ensembles, post-processing
#   recipes/flat_forecast/run.sh report    $GR_ARTIFACTS_DIR/flat_forecast/report/flat-forecast.html
#   recipes/flat_forecast/run.sh all       every step
#
# Needs recipes/carbon (doors) and recipes/carbon_rates (the total mirror, the original
# ladder run) first. The ladder runs share the total's panel cache in carbon_rates.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
source "$RECIPES/year/flags.sh"
need gr-benchmark
need gr-dense

ROOT="$GR_ARTIFACTS_DIR/flat_forecast"
LADDER="$ROOT/ladder"
MIRROR="$GR_ARTIFACTS_DIR/carbon_rates/mirror"
: "${PARALLEL:=3}"
: "${THREADS_PER_JOB:=4}"

total_env() {
  echo "OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB GR_DATA_SOURCE=local" \
       "GR_LOCAL_ROOT='$MIRROR' GR_ARTIFACTS_DIR='$GR_ARTIFACTS_DIR/carbon_rates'"
}
LADDER_FLAGS="$YEAR_DATA $YEAR_SPLIT $YEAR_EVAL --lookback 2016 --epochs 100 --calendar"
declare -A TIER=([tcn]=4 [gru]=4 [dlinear]=5 [patchtst]=5)

step_history() { log "total from 2024"; "$PYTHON" "$RECIPES/flat_forecast/history.py"; }

step_ladder() {
  local args=() name out
  for m in tcn patchtst gru dlinear; do
    for uw in 1 2; do
      for s in 0 1 2; do
        name="${m}_uw${uw}_s${s}"; out="$LADDER/$name"
        if [[ -f "$out/dense/pred_block_$m.npy" ]]; then log "$name complete — skipping"; continue; fi
        args+=("$name" "$(total_env) gr-benchmark $LADDER_FLAGS --under-weight $uw --seed $s --tiers 1 ${TIER[$m]} \
--only persistence,$m --out '$out' && $(total_env) gr-dense predict --run '$out' --models persistence,$m --batch 512")
      done
    done
  done
  ((${#args[@]})) || { log "all ladder runs complete"; return; }
  pool "$PARALLEL" "$LADDER/logs" "${args[@]}"
}

step_run()    { log "experiments"; "$PYTHON" "$RECIPES/flat_forecast/run.py"; }
step_report() { log "report"; "$PYTHON" "$RECIPES/flat_forecast/build_report.py"; }

case "${1:-}" in
  history|ladder|run|report) "step_$1" ;;
  all) step_history; step_ladder; step_run; step_report ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
