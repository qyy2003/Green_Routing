#!/usr/bin/env bash
# The follow-up experiments on top of the year run (docs/results.md, report in
# docs/reports/month-window-smoothing.html):
#   E1  1-month input window instead of 1 week     (runs month_<model>, matched TCN pair)
#   E2a post-smoothing of the forecast paths        (logs <model>+ma15m / +ma1h / +ma3h)
#   E2b training on a 1-h smoothed target           (runs starget_<model>)
#
#   recipes/window_smoothing/run.sh train     10 training runs (PARALLEL at once, default 3)
#   recipes/window_smoothing/run.sh dense     their prediction logs, into the year run's dense dir
#   recipes/window_smoothing/run.sh analyze   metrics on common targets, paired tests, cost
#   recipes/window_smoothing/run.sh report    rebuild the HTML report
#   recipes/window_smoothing/run.sh all       train dense analyze report
#
# Needs the year run first (recipes/year/run.sh all): the comparisons are against its
# models and the logs are written next to theirs. Runs go to $GR_ARTIFACTS_DIR/exp/.
#
# GPU memory per training job (batch 85): month GRU ~12-20 GB, month TCN ~11 GB, the
# rest <= 3 GB. PARALLEL=3 fits a 46 GB card; on 24 GB use PARALLEL=1 or 2.
# Wall clock on an L40S: ~10 h of training (month TCN is ~9x a week TCN per step,
# hence its matched 20-epoch budget), ~2-3 h of dense logs with the GPU to itself.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
source "$RECIPES/year/flags.sh"
need gr-benchmark

HERE="$RECIPES/window_smoothing"
YEAR="$GR_ARTIFACTS_DIR/pred_year/merged"
EXP="$GR_ARTIFACTS_DIR/exp"
: "${PARALLEL:=3}"

# The baseline's flags with one factor changed. --max-origins 20: the harness backtest
# is only a sanity check here; the comparisons use the dense logs.
EXP_BASE="$YEAR_FLAGS --max-origins 20"
MONTH="--lookback 8640 --batch-size 85 --tcn-max-levels 12 --patchtst-patch 64 --patchtst-stride 32 --gbdt-context month"
STARGET="--lookback 2016 --target-smooth 12"

step_train() {
  [[ -f "$YEAR/aggregate.csv" ]] || { echo "run recipes/year/run.sh first" >&2; exit 1; }
  local args=() run flags only
  # name | extra flags | models (heaviest first, so they start early)
  local spec=(
    "month_tcn|$MONTH --epochs 20|tcn"           # matched budget, see wk20_tcn
    "month_gru|$MONTH|gru"
    "month_patchtst|$MONTH|patchtst"
    "month_dlinear|$MONTH|dlinear"
    "month_gbdt|$MONTH|gbdt"
    "starget_tcn|$STARGET|tcn"
    "starget_gru|$STARGET|gru"
    "starget_patchtst|$STARGET|patchtst"
    "starget_dlinear|$STARGET|dlinear"
    "wk20_tcn|--lookback 2016 --epochs 20|tcn"   # the 1-week control for month_tcn
  )
  for s in "${spec[@]}"; do
    IFS='|' read -r run flags only <<< "$s"
    if [[ -f "$EXP/$run/aggregate.csv" ]]; then log "$run complete — skipping"; continue; fi
    args+=("$run" "gr-benchmark $EXP_BASE $flags --only persistence,$only --out '$EXP/$run'")
  done
  ((${#args[@]})) || { log "all runs complete"; return; }
  pool "$PARALLEL" "$EXP/logs" "${args[@]}"
}

dense_variant() {   # dense_variant <run> <model> <tag> <batch>
  local run="$1" m="$2" tag="$3" b="$4"
  if [[ -f "$YEAR/dense/pred_block_$m$tag.npy" || ( "$m" == gbdt && -f "$YEAR/dense/pred_rolling_$m$tag.npy" ) ]]; then
    log "$m$tag logged — skipping"; return
  fi
  log "dense logs: $m$tag"
  gr-dense predict --run "$EXP/$run" --out "$YEAR/dense" --models "$m" --tag "$tag" \
    --batch "$b" --tf32 --origin-stride 3
  gr-dense verify --run "$EXP/$run" --out "$YEAR/dense" --tag "$tag"
}

step_dense() {
  # TF32 + every 3rd origin, as in the original study: block logs are exact (every
  # horizon is a multiple of 3 steps), rolling logs score every 15 minutes, and the
  # TF32 drift (~3-7e-4 relative) is far below the effects measured.
  dense_variant month_gbdt gbdt @month 2048
  dense_variant month_dlinear dlinear @month 1024
  dense_variant month_patchtst patchtst @month 512
  dense_variant month_gru gru @month 256
  dense_variant month_tcn tcn @month 512
  dense_variant wk20_tcn tcn @wk20 512
  for m in dlinear patchtst gru tcn; do
    dense_variant "starget_$m" "$m" @starget 512
  done
  for m in dlinear patchtst gru tcn; do
    if [[ -f "$YEAR/dense/pred_block_$m+ma3h.npy" ]]; then log "$m post-smoothing logged — skipping"; continue; fi
    log "dense logs: $m post-smoothed (15 min, 1 h, 3 h)"
    gr-dense predict --run "$YEAR" --models "$m" --batch 512 \
      --smooth 3,12,36 --skip-raw --tf32 --origin-stride 3
  done
}

step_analyze() {
  log "metrics for every model and variant on common targets"
  gr-dense metrics --run "$YEAR" --common
  log "paired tests, link traits, daily series"
  "$PYTHON" "$HERE/analyze.py"
  log "cost benchmark (keep the GPU otherwise idle)"
  "$PYTHON" "$HERE/cost.py"
}

step_report() { "$PYTHON" "$HERE/build_report.py" "$@"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  train) step_train ;;
  dense) step_dense ;;
  analyze) step_analyze ;;
  report) step_report "$@" ;;
  all) step_train; step_dense; step_analyze; step_report ;;
  *) sed -n '2,21p' "$0"; exit 1 ;;
esac
