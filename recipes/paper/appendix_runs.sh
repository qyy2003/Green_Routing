#!/usr/bin/env bash
# The appendix experiments of recipes/ladder_diagnosis on the paper's common periods (periods.py): the runs of
# ladder_common.sh (links, test block, validation) through the GR_LD_* overrides of ladder_diagnosis/common.py,
# with the overlay artifacts root ($PAPER_COMMON/root: round 2's Internet2 setup, a fresh ladder_diagnosis
# cache). Needs ladder_common.sh all and timestral_links.py (the reference block logs) first.
#
#   recipes/paper/appendix_runs.sh history   SWITCHlan's history before the panel for the new link set (CPU)
#   recipes/paper/appendix_runs.sh hourly    grids/h_ctx.txt: hourly architectures at 1 / 4 / 24 weeks, resolution
#   recipes/paper/appendix_runs.sh long5m    grids/l5_ctx.txt: 5-minute long-context transformer
#   recipes/paper/appendix_runs.sh data      grids/h_data.txt, h_doors.txt: Internet2 learning curve
#   recipes/paper/appendix_runs.sh finetune  Timestral fine-tuned on Internet2's history since 2021; uses_context.py
#   recipes/paper/appendix_runs.sh fm5m      Timestral on 5-minute input, 4 weeks (gr-dense)
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
eval "$("$PYTHON" "$RECIPES/paper/periods.py" --sh)"
LD="$RECIPES/ladder_diagnosis"
G="$RECIPES/paper/grids"
OUTL="$PAPER_COMMON/logs"
mkdir -p "$OUTL"
CACHE_PANEL() { ls "$PAPER_COMMON/cache_$1"/panel_*.npz | head -1; }
export GR_ARTIFACTS_DIR="$PAPER_COMMON/root"
for n in i2 switch; do                 # defaults; a value set by the caller wins
  N=${n^^}
  eval ": \"\${GR_LD_RUN_$N:=$PAPER_COMMON/links/$n/merged}\" \"\${GR_LD_PANEL_$N:=$(CACHE_PANEL $n)}\" \"\${GR_LD_FM_$N:=$PAPER_COMMON/links/$n/dense_fm}\""
  export GR_LD_RUN_$N GR_LD_PANEL_$N GR_LD_FM_$N
done
S3="$HOME/.cache/green-routing/s3/network-energy-efficiency-research"
cd "$LD"

case "${1:-}" in
  history)
    GR_DATA_SOURCE=local GR_LOCAL_ROOT="$S3" OMP_NUM_THREADS=1 "$PYTHON" -W ignore switch_history.py
    GR_DATA_SOURCE=local GR_LOCAL_ROOT="$S3" OMP_NUM_THREADS=1 "$PYTHON" -W ignore switch_history.py --res 5m ;;
  hourly) "$PYTHON" sweep.py run "$G/${2:-h_ctx.txt}" ;;
  long5m) "$PYTHON" sweep.py run "$G/${2:-l5_ctx.txt}" long5m.py ;;
  data)   "$PYTHON" sweep.py run "$G/h_data.txt"; "$PYTHON" sweep.py run "$G/h_doors.txt" ;;
  finetune)
    for lr in 1e-5 3e-5; do
      "$PYTHON" -W ignore ft_timestral.py --ds i2 --hist ext21 --ctx 4032 --bs 16 --lr "$lr" --steps 300 \
        --epochs 10 --tag "F2_ft_i2_ext21_lr$lr"
    done
    for d in i2 switch; do
      "$PYTHON" -W ignore uses_context.py "$d" timestral L5s_seg_multires_4w_s3 L5s_seg_multires_24w_s3
    done ;;
  fm5m)
    FM="--models timestral --tag @5m_4w --fm-context 8064 --fm-step 5min --fm-horizons day,week --batch 64"
    GR_DATA_SOURCE=local GR_LOCAL_ROOT="$REPO/artifacts/internet2/data/mirror_links" GR_ARTIFACTS_DIR="$PAPER_COMMON/cache_i2" \
      gr-dense predict --run "$PAPER_COMMON/links/i2/merged" --out "$PAPER_COMMON/links/i2/dense_fm" $FM
    GR_DATA_SOURCE=local GR_LOCAL_ROOT="$S3" GR_ARTIFACTS_DIR="$PAPER_COMMON/cache_switch" \
      gr-dense predict --run "$PAPER_COMMON/links/switch/merged" --out "$PAPER_COMMON/links/switch/dense_fm" $FM ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
