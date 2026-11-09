#!/usr/bin/env bash
# The paper's numbers and figures on one set of periods for both networks (round 2, periods.py):
# test block 2026-04-01 -> 06-30, development year 2025-04-01 -> 2026-03-31, annual window 2025-07-01 ->
# 2026-06-30, later period 2026-07-01 -> 09-30, trained baselines from the later of the two networks' series
# to 2026-03-03 (validation 03-04 -> 03-31). Outputs in $PAPER_COMMON (artifacts/paper_common); the paper
# scripts run with GR_ARTIFACTS_DIR=$PAPER_COMMON/root, an overlay of artifacts/ in which internet2/carbon is
# round 2's setup (i2_carbon_common.py) and paper/ and ladder_diagnosis/ are new folders.
#
#   recipes/paper/run_common.sh setup      Internet2 setup (measured power from 2024-11), overlay root
#   recipes/paper/run_common.sh links      per-link baselines (ladder_common.sh all), Timestral per link
#   recipes/paper/run_common.sh flat       flat rate: forecasts, margin, own share, bill errors, grid table
#   recipes/paper/run_common.sh rates      path rates, the 7-day path correction, SWITCHlan path rates, facts
#   recipes/paper/run_common.sh steering   one pair alone, everyone, controller; the cap's k = 1, 3, none
#   recipes/paper/run_common.sh totals     network totals: 5-minute ladder (ladder_common.sh total), hourly nets
#   recipes/paper/run_common.sh appendix   appendix_runs.sh (hourly, long5m, data, finetune, fm5m), appendix.py
#   recipes/paper/run_common.sh numbers    text_numbers.py, factcheck.py, figures.py
# GPU: links, totals, appendix (hours); the rest is CPU (steering: about an hour per k).
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
eval "$("$PYTHON" "$RECIPES/paper/periods.py" --sh)"
P="$RECIPES/paper"
LOGS="$PAPER_COMMON/logs"
mkdir -p "$LOGS"
OVERLAY="$PAPER_COMMON/root"
S3="$HOME/.cache/green-routing/s3/network-energy-efficiency-research"
py() { "$PYTHON" -W ignore "$@"; }

step_setup() {
  py "$P/i2_carbon_common.py" > "$LOGS/i2_carbon.log" 2>&1
  py "$P/i2_carbon_common.py" pre28 >> "$LOGS/i2_carbon.log" 2>&1
  local A="$GR_ARTIFACTS_DIR" d
  mkdir -p "$OVERLAY/internet2/links" "$OVERLAY/paper" "$OVERLAY/ladder_diagnosis/i2" "$OVERLAY/ladder_diagnosis/switch"
  for d in abilene carbon carbon_granularity carbon_rates carbon_sym flat_forecast link_difficulty link_roles network \
           pred_year pred_year_sym reference; do ln -sfn "$A/$d" "$OVERLAY/$d"; done
  for d in cache cache_links data raw tunnels green_paths; do ln -sfn "$A/internet2/$d" "$OVERLAY/internet2/$d"; done
  ln -sfn "$PAPER_COMMON/i2_carbon" "$OVERLAY/internet2/carbon"
  ln -sfn "$PAPER_COMMON/i2_carbon_pre28" "$OVERLAY/internet2/carbon_pre28"
  for f in "$A"/ladder_diagnosis/i2/*.csv.gz; do ln -sfn "$f" "$OVERLAY/ladder_diagnosis/i2/"; done
}

step_links() {
  "$P/ladder_common.sh" all both
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$A/internet2/data/mirror_links" GR_ARTIFACTS_DIR="$PAPER_COMMON/cache_i2" \
    py "$P/timestral_links.py" i2 main ctx
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$S3" GR_ARTIFACTS_DIR="$PAPER_COMMON/cache_switch" \
    py "$P/timestral_links.py" switch main ctx
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$A/internet2/data/mirror_links" GR_ARTIFACTS_DIR="$PAPER_COMMON/cache_i2" \
    py "$P/timestral_links.py" i2 link
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$S3" GR_ARTIFACTS_DIR="$PAPER_COMMON/cache_switch" \
    py "$P/timestral_links.py" switch link
  "$P/appendix_runs.sh" fm5m
  local n env M
  for n in i2 switch; do
    M="$PAPER_COMMON/links/$n/merged"
    [[ $n == i2 ]] && env="GR_DATA_SOURCE=local GR_LOCAL_ROOT=$A/internet2/data/mirror_links GR_ARTIFACTS_DIR=$PAPER_COMMON/cache_i2" \
                   || env="GR_DATA_SOURCE=local GR_LOCAL_ROOT=$S3 GR_ARTIFACTS_DIR=$PAPER_COMMON/cache_switch"
    ln -sfn ../dense_fm "$M/dense_fm"
    eval "$env gr-dense metrics --run '$M' --out '$PAPER_COMMON/links/$n/dense_fm' --common \
      --models persistence,seasonal_naive_day,seasonal_naive_week,historical_average,dlinear,gru,tcn,patchtst,timestral@1h_24w,timestral@5m_4w \
      --tables-dir '$PAPER_COMMON/links/$n/dense_fm/tables_both'"
    eval "$env gr-dense metrics --run '$M' --out '$PAPER_COMMON/links/$n/dense_fm' --common \
      --models persistence,seasonal_naive_day,seasonal_naive_week,historical_average,dlinear,gru,tcn,patchtst,timestral@1h_24w \
      --tables-dir '$PAPER_COMMON/links/$n/dense_fm/tables_1h_24w'"
  done
}
A="$GR_ARTIFACTS_DIR"

step_flat() {
  for s in forecasts gridq score grid periods; do py "$P/flat_common.py" $s; done > "$LOGS/flat.log" 2>&1
  py - <<'EOF'
import sys; sys.path.insert(0, "recipes/paper")
import pandas as pd, periods as P
f = pd.read_csv(P.ROOT / "flat" / "i2_forecasts.csv", index_col=0, parse_dates=True)
pd.DataFrame({"Timestral-209M, 24 weeks": f.Tf, "Chronos-2, 24 weeks": f.Tf_chronos2}).to_csv(
    P.ROOT / "i2_carbon" / "timestral_total.csv")
EOF
}

step_rates() {
  export GR_ARTIFACTS_DIR="$OVERLAY"
  py "$P/i2_rates.py" > "$LOGS/i2_rates.log" 2>&1
  py "$P/path_correction.py" > "$LOGS/path_correction.log" 2>&1
  py "$P/switch_rates.py" > "$LOGS/switch_rates.log" 2>&1
  py "$P/switch_routing_check.py" > "$LOGS/switch_routing_check.log" 2>&1
  py "$P/facts.py" > "$LOGS/facts.log" 2>&1
}

step_steering() {
  export GR_ARTIFACTS_DIR="$OVERLAY"
  py "$P/steering.py" all > "$LOGS/steering.log" 2>&1
  for k in 1 3 none; do
    py "$P/steering.py" unilateral --k=$k > "$LOGS/steering_uni_k$k.log" 2>&1
    py "$P/steering.py" everyone --k=$k > "$LOGS/steering_every_k$k.log" 2>&1
  done
}

step_totals() {
  "$P/ladder_common.sh" total
  GR_ARTIFACTS_DIR="$OVERLAY" py "$P/total_neural.py" all > "$LOGS/total_neural.log" 2>&1
}

step_appendix() {
  mkdir -p "$PAPER_COMMON/link_difficulty"
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$A/internet2/data/mirror_links" GR_ARTIFACTS_DIR="$PAPER_COMMON/cache_i2" \
    py "$RECIPES/link_difficulty/analyze.py" "$PAPER_COMMON/links/i2/merged" --model patchtst --name i2 \
    --out "$PAPER_COMMON/link_difficulty" > "$LOGS/link_difficulty.log" 2>&1
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$S3" GR_ARTIFACTS_DIR="$PAPER_COMMON/cache_switch" \
    py "$RECIPES/link_difficulty/analyze.py" "$PAPER_COMMON/links/switch/merged" --model patchtst --name switch \
    --out "$PAPER_COMMON/link_difficulty" >> "$LOGS/link_difficulty.log" 2>&1
  "$P/appendix_runs.sh" history
  for s in hourly long5m data finetune; do "$P/appendix_runs.sh" $s > "$LOGS/appendix_$s.log" 2>&1; done
  GR_ARTIFACTS_DIR="$OVERLAY" py "$P/appendix.py" > "$LOGS/appendix.log" 2>&1
}

step_numbers() {
  export GR_ARTIFACTS_DIR="$OVERLAY"
  py "$P/text_numbers.py" > "$LOGS/text_numbers.log" 2>&1
  py "$P/factcheck.py" > "$LOGS/factcheck.log" 2>&1
  py "$P/figures.py" > "$LOGS/figures.log" 2>&1
}

case "${1:-}" in
  setup|links|flat|rates|steering|totals|appendix|numbers) "step_$1" ;;
  all) for s in setup links flat rates steering totals appendix numbers; do "step_$s"; done ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
