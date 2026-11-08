#!/usr/bin/env bash
# Why the trained ladder trails Timestral, and what closes the gap (2026-11-04).
# Log: docs/worklog/2026-11-04-ladder-diagnosis.md. Output: $GR_ARTIFACTS_DIR/ladder_diagnosis.
#
#   recipes/ladder_diagnosis/run.sh baselines   evaluator calibration + multi-week profiles (CPU)
#   recipes/ladder_diagnosis/run.sh history     Switch hourly history 2024-01 -> 2025-07 (CPU, frozen S3 cache)
#   recipes/ladder_diagnosis/run.sh timestral   Timestral zero-shot at 1, 2, 4, 12, 24 weeks of hourly context
#   recipes/ladder_diagnosis/run.sh grids       every training grid in grids/*.txt (GPU; finished tags skipped)
#   recipes/ladder_diagnosis/run.sh finetune    Timestral fine-tuned on I2 (robust pinball loss)
#   recipes/ladder_diagnosis/run.sh table       one table of every run (all_runs.csv)
#   recipes/ladder_diagnosis/run.sh long5m      5-minute input, up to 24 weeks of context (long5m.py; log
#                                               docs/worklog/2026-11-04-long-context-5m.md): Switch 5-minute
#                                               history, references, grids l5 / l5b / l5c / l5d (GPU, hours)
#   recipes/ladder_diagnosis/run.sh why         why long context does not help trained models (log
#                                               docs/worklog/2026-11-05-why-long-context.md): I2 hourly
#                                               2021 -> 2024 download, preprocessing / saved-model /
#                                               architecture / data-size / door grids, context ablation,
#                                               Timestral fine-tunes (GPU, about a day on one GPU)
#
# Needs the run directories of the year run (pred_year_sym/merged, with dense_fm/) and of
# I2 (internet2/links/merged, with dense_fm/), whose block logs are the references.
# Every training run is short (seconds to ~20 min on one GPU); grids run one after another.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
HERE="$RECIPES/ladder_diagnosis"
OUT="$GR_ARTIFACTS_DIR/ladder_diagnosis"
mkdir -p "$OUT"
cd "$HERE"

step_baselines() { for d in i2 switch; do "$PYTHON" -W ignore baselines.py "$d"; done; }
step_history() {
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$HOME/.cache/green-routing/s3/network-energy-efficiency-research" \
    OMP_NUM_THREADS=1 "$PYTHON" -W ignore switch_history.py
}
step_timestral() { for d in i2 switch; do "$PYTHON" -W ignore timestral_ctx.py "$d" 168 336 672 2016 4032; done; }
step_grids() {
  for g in i2_a i2_b i2_c sw_ab sw_b2 sw_d e_prof e2 g_joint; do
    log "grid $g"; "$PYTHON" sweep.py run "grids/$g.txt" > "$OUT/$g.log" 2>&1
  done
}
step_finetune() {
  for lr in 3e-6 1e-5; do
    "$PYTHON" -W ignore ft_timestral.py --ds i2 --hist ext --ctx 4032 --bs 16 --lr "$lr" \
      --steps 300 --epochs 8 --tag "F_ft_209M_c4032_ext_lr${lr}_robust"
  done
}
step_table() { for d in i2 switch; do "$PYTHON" sweep.py table "$d"; done; }
step_why() {
  "$PYTHON" -W ignore i2_history.py fetch && "$PYTHON" -W ignore i2_history.py build
  for g in l5e_i2 l5e_sw l5f; do log "grid $g"; "$PYTHON" sweep.py run "grids/$g.txt" long5m.py > "$OUT/$g.log" 2>&1; done
  for g in h_arch h_data h_doors; do log "grid $g"; "$PYTHON" sweep.py run "grids/$g.txt" > "$OUT/$g.log" 2>&1; done
  for d in i2 switch; do
    "$PYTHON" -W ignore uses_context.py "$d" timestral L5s_seg_multires_4w_s3 L5s_seg_multires_24w_s3 \
      L5s_seg_coarse_24w_s3 L5s_profmlp_24w_s3
    h=$([ "$d" = i2 ] && echo ext21 || echo ext)
    for lr in 1e-5 3e-5; do
      "$PYTHON" -W ignore ft_timestral.py --ds "$d" --hist "$h" --ctx 4032 --bs 16 --lr "$lr" --steps 300 \
        --epochs 10 --tag "F2_ft_${d}_${h}_lr$lr"
    done
  done
}
step_long5m() {
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$HOME/.cache/green-routing/s3/network-energy-efficiency-research" \
    OMP_NUM_THREADS=1 "$PYTHON" -W ignore switch_history.py --res 5m
  # Timestral with 16 weeks of 5-minute input (reference; ~50 min on I2, longer on Switch LAN)
  local I2="$GR_ARTIFACTS_DIR/internet2" SW="$GR_ARTIFACTS_DIR/pred_year_sym/merged"
  local FM="--models timestral --tag @5m_16w --fm-context 32256 --fm-step 5min --fm-horizons day,week --batch 32"
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$I2/data/mirror_links" GR_ARTIFACTS_DIR="$I2/cache_links" \
    gr-dense predict --run "$I2/links/merged" --out "$I2/links/merged/dense_fm" $FM
  GR_DATA_SOURCE=local GR_LOCAL_ROOT="$HOME/.cache/green-routing/s3/network-energy-efficiency-research" \
    gr-dense predict --run "$SW" --out "$SW/dense_fm" $FM
  for d in i2 switch; do "$PYTHON" -W ignore refs5m.py "$d"; done
  for g in l5_i2 l5_sw l5b_i2 l5b_sw l5c_i2 l5c_sw l5d_i2; do
    log "grid $g"; "$PYTHON" sweep.py run "grids/$g.txt" long5m.py > "$OUT/$g.log" 2>&1
  done
  for d in i2 switch; do "$PYTHON" sweep.py table5 "$d"; done
}

cmd="${1:-}"; shift || true
case "$cmd" in
  baselines|history|timestral|grids|finetune|table|long5m|why) "step_$cmd" ;;
  all) step_baselines; step_history; step_timestral; step_grids; step_finetune; step_table ;;
  *) sed -n '2,24p' "$0"; exit 1 ;;
esac
