#!/usr/bin/env bash
# Reproduce the year-scale benchmark (pred_year) and cross-check it against the report
# and against the reference run (expected.json).
#
#   recipes/year/run.sh check     data source + panel fingerprint (downloads ~20 GB once)
#   recipes/year/run.sh train     train + backtest the five parts in parallel
#   recipes/year/run.sh merge     one run directory with every model's tables
#   recipes/year/run.sh dense     prediction logs at every 5-min origin + their metrics
#   recipes/year/run.sh verify    compare with the report and expected.json
#   recipes/year/run.sh viewer    serve the results on :8138
#   recipes/year/run.sh all       check train merge dense verify
#
# Output: $GR_ARTIFACTS_DIR/pred_year/{part_*, merged}. Every step can be re-run: a
# finished part is skipped, an interrupted one resumes from its last epoch.
# PARALLEL (default 5) = parts trained at once; on a GPU under ~16 GB use PARALLEL=2.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
source "$RECIPES/year/flags.sh"
need gr-benchmark

YEAR="$GR_ARTIFACTS_DIR/pred_year"
MERGED="$YEAR/merged"
: "${PARALLEL:=5}"

step_check() {
  # The fingerprint is the gate: with a pinned (copied) panel the bucket isn't needed.
  log "data source"
  gr-data check || log "WARNING: data source check failed — fine only with a pinned panel (docs/reproduce.md)"
  log "panel fingerprint (first run downloads and caches the whole subtree)"
  "$PYTHON" "$RECIPES/year/verify.py" --data-only
}

step_train() {
  local args=() name only
  for part in "${YEAR_PARTS[@]}"; do
    name="${part%%:*}"; only="${part#*:}"
    if [[ -f "$YEAR/part_$name/aggregate.csv" ]]; then
      log "part_$name already complete — skipping"; continue
    fi
    args+=("part_$name" "gr-benchmark $YEAR_FLAGS --only $only --out '$YEAR/part_$name'")
  done
  ((${#args[@]})) || { log "all parts complete"; return; }
  pool "$PARALLEL" "$YEAR/logs" "${args[@]}"
}

step_merge() {
  mkdir -p "$MERGED"/{records,checkpoints,history} "$YEAR/logs"
  for part in "${YEAR_PARTS[@]}"; do
    local p="$YEAR/part_${part%%:*}"
    [[ -f "$p/aggregate.csv" ]] || { echo "missing $p — run 'train' first" >&2; exit 1; }
    cp -f "$p"/records/*.csv "$MERGED/records/"
    cp -f "$p"/checkpoints/* "$MERGED/checkpoints/" 2>/dev/null || true
    cp -f "$p"/history/* "$MERGED/history/" 2>/dev/null || true
  done
  log "rebuilding the tables from the merged records (no training)"
  gr-benchmark $YEAR_FLAGS --only "$YEAR_MODELS" --resume --out "$MERGED" \
    > "$YEAR/logs/merge.out" 2>&1
  tail -n 25 "$YEAR/logs/merge.out"
}

step_dense() {
  # Baselines and GBDT run on the CPU, the four networks on the GPU (full fp32, one
  # at a time); every origin, so the logs are complete in both modes.
  log "dense logs: baselines + gbdt (CPU)"
  gr-dense predict --run "$MERGED" \
    --models persistence,seasonal_naive_day,seasonal_naive_week,historical_average,gbdt
  # batch = origins per forward pass; GRU's cuDNN workspace is the big one (~16 GB at
  # 1024). DENSE_BATCH overrides all four, e.g. DENSE_BATCH=128 on a small GPU.
  for spec in dlinear:2048 patchtst:1024 gru:512 tcn:1024; do
    m="${spec%%:*}"
    log "dense logs: $m (GPU)"
    gr-dense predict --run "$MERGED" --models "$m" --batch "${DENSE_BATCH:-${spec#*:}}"
  done
  log "verify the logs against the sampled records"
  gr-dense verify --run "$MERGED"
  log "metrics on identical targets (year models only)"
  # Own copy for verify.py: later experiments rescore dense/<mode>/ with their variants.
  gr-dense metrics --run "$MERGED" --models untagged --common \
    --tables-dir "$MERGED/dense_year_tables"
  for mode in rolling block; do                 # what the viewer reads
    mkdir -p "$MERGED/dense/$mode"
    cp -f "$MERGED/dense_year_tables/$mode/"*.csv "$MERGED/dense/$mode/"
  done
}

step_verify() { "$PYTHON" "$RECIPES/year/verify.py" --run "$MERGED"; }
step_viewer() { gr-viewer "$MERGED" "$@"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  check) step_check ;;
  train) step_train ;;
  merge) step_merge ;;
  dense) step_dense ;;
  verify) step_verify ;;
  viewer) step_viewer "$@" ;;
  all) step_check; step_train; step_merge; step_dense; step_verify ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
