#!/usr/bin/env bash
# The trained per-link baselines (DLinear, GRU, TCN, PatchTST and the rules) on the common periods
# of recipes/paper/periods.py, both networks: one week of 5-minute input, trained from
# TRAIN_START_5MIN (Internet2's 5-minute counters start 2025-11-15) to TRAIN_END, validation VAL
# (the last 4 weeks before the test block), test block TEST. Same flags as the year run
# (recipes/year/flags.sh, symmetric loss) and recipes/internet2/run.sh otherwise.
#
#   recipes/paper/ladder_common.sh [train|merge|dense|all] [switch|i2|both]
#
# Output: $PAPER_COMMON/links/<net>/{part_*,merged}; logs in $PAPER_COMMON/links/logs.
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
need gr-benchmark
eval "$("$PYTHON" "$RECIPES/paper/periods.py" --sh)"
: "${PARALLEL:=3}"
STEP="${1:-all}"; NETS="${2:-both}"
[[ "$NETS" == both ]] && NETS="switch i2"
L="$PAPER_COMMON/links"
mkdir -p "$L/logs"

COMMON="--start $TRAIN_START_5MIN --end $TEST_END --min-mean-mbps 1 \
 --train-end $TRAIN_END --val-start $VAL0 --val-end $VAL_END --test-start $TEST0 \
 --horizons 15min 1h 8h day week --max-origins 160 --lookback 2016 --epochs 100 --under-weight 1"
declare -A EXTRA ENV
EXTRA[switch]="--calendar"
EXTRA[i2]=""
ENV[switch]="GR_DATA_SOURCE=local GR_LOCAL_ROOT=$HOME/.cache/green-routing/s3/network-energy-efficiency-research GR_ARTIFACTS_DIR=$PAPER_COMMON/cache_switch"
ENV[i2]="GR_DATA_SOURCE=local GR_LOCAL_ROOT=$GR_ARTIFACTS_DIR/internet2/data/mirror_links GR_ARTIFACTS_DIR=$PAPER_COMMON/cache_i2"
PARTS=(
  "baselines:1:persistence,seasonal_naive_day,seasonal_naive_week,historical_average"
  "gru:1 4:persistence,gru"
  "tcn:1 4:persistence,tcn"
  "dlinear_patchtst:1 5:persistence,dlinear,patchtst"
)
MODELS="persistence,seasonal_naive_day,seasonal_naive_week,historical_average,gru,tcn,dlinear,patchtst"

step_train() {
  # the baselines part first: it builds each network's panel cache, which the other parts then read
  local args net name tiers only rest first
  for first in 1 0; do
    args=()
    for net in $NETS; do
      for part in "${PARTS[@]}"; do
        name="${part%%:*}"; rest="${part#*:}"; tiers="${rest%%:*}"; only="${rest#*:}"
        [[ ( "$first" == 1 && "$name" != baselines ) || ( "$first" == 0 && "$name" == baselines ) ]] && continue
        [[ -f "$L/$net/part_$name/aggregate.csv" ]] && continue
        args+=("${net}_$name" "${ENV[$net]} gr-benchmark $COMMON ${EXTRA[$net]} --tiers $tiers --only $only --out '$L/$net/part_$name'")
      done
    done
    ((${#args[@]})) && pool "$PARALLEL" "$L/logs" "${args[@]}"
  done
  log "training parts complete"
}

step_merge() {
  local net M p
  for net in $NETS; do
    M="$L/$net/merged"
    mkdir -p "$M"/{records,checkpoints,history}
    for part in "${PARTS[@]}"; do
      p="$L/$net/part_${part%%:*}"
      cp -f "$p"/records/*.csv "$M/records/"; cp -f "$p"/checkpoints/* "$M/checkpoints/" 2>/dev/null || true
      cp -f "$p"/history/* "$M/history/" 2>/dev/null || true
    done
    eval "${ENV[$net]} gr-benchmark $COMMON ${EXTRA[$net]} --tiers 1 4 5 --only $MODELS --resume --out '$M'" \
      > "$L/logs/merge_$net.out" 2>&1
    tail -n 12 "$L/logs/merge_$net.out"
  done
}

step_dense() {
  local net M
  for net in $NETS; do
    M="$L/$net/merged"
    eval "${ENV[$net]} gr-dense predict --run '$M' --models persistence,seasonal_naive_day,seasonal_naive_week,historical_average"
    for m in dlinear patchtst gru tcn; do eval "${ENV[$net]} gr-dense predict --run '$M' --models $m --batch 512"; done
    eval "${ENV[$net]} gr-dense metrics --run '$M' --common"
  done
}

# The 5-minute ladder on SWITCHlan's network total (Internet2 has no 5-minute total): three seeds per
# model, symmetric loss, trained from TRAIN_START_HOURLY (the start of SWITCHlan's current doors) on the
# one-series mirror of total_mirror.py; else as recipes/flat_forecast/run.sh ladder.
step_total() {
  local args=() m s name out TENV
  declare -A TIER=([tcn]=4 [gru]=4 [dlinear]=5 [patchtst]=5)
  [[ -f "$PAPER_COMMON/total_mirror/router-interfaces/total/all.csv" ]] || "$PYTHON" "$RECIPES/paper/total_mirror.py"
  TENV="GR_DATA_SOURCE=local GR_LOCAL_ROOT=$PAPER_COMMON/total_mirror GR_ARTIFACTS_DIR=$PAPER_COMMON/cache_total"
  local TFLAGS="--start $TRAIN_START_HOURLY --end $TEST_END \
 --train-end $TRAIN_END --val-start $VAL0 --val-end $VAL_END --test-start $TEST0 \
 --horizons 15min 1h 8h day week --max-origins 160 --lookback 2016 --epochs 100 --calendar --under-weight 1"
  # build the panel once
  [[ -f "$PAPER_COMMON/total/persistence/aggregate.csv" ]] || \
    eval "$TENV gr-benchmark $TFLAGS --tiers 1 --only persistence --out '$PAPER_COMMON/total/persistence'" > "$L/logs/total_panel.out" 2>&1
  # symmetric loss (the baselines of Table app-models), three seeds; TOTAL_UW="1 2" adds the peak-aware loss
  # (--under-weight 2). The paper's bias comparison of the two losses uses round 1's seeds of both (trained to
  # 2026-02-15, before the test block), rescored on the common test block (appendix.py).
  for uw in ${TOTAL_UW:-1}; do
    for m in tcn patchtst gru dlinear; do
      for s in 0 1 2; do
        name="total_${m}_uw${uw}_s${s}"; out="$PAPER_COMMON/total/${m}_uw${uw}_s${s}"
        [[ -f "$out/dense/pred_block_$m.npy" ]] && continue
        args+=("$name" "$TENV gr-benchmark ${TFLAGS/--under-weight 1/--under-weight $uw} --seed $s --tiers 1 ${TIER[$m]} \
--only persistence,$m --out '$out' && $TENV gr-dense predict --run '$out' --models persistence,$m --batch 512")
      done
    done
  done
  ((${#args[@]})) && pool "$PARALLEL" "$L/logs" "${args[@]}"
  log "total ladder complete"
}

case "$STEP" in
  train) step_train ;; merge) step_merge ;; dense) step_dense ;; total) step_total ;;
  all) step_train; step_merge; step_dense ;;
  *) echo "unknown step $STEP" >&2; exit 1 ;;
esac
