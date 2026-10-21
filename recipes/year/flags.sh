# The pred_year configuration: the year-scale benchmark behind the "Per-Link Traffic
# Prediction for Carbon Accounting" report (docs/reports/). The original command was
# never recorded; these flags were reconstructed on 2026-10-01 and the data/test part
# is verified exactly — they reproduce the report's persistence MAE to the Mbps
# (712 / 1541 / 1061 / 1145 at 1 h / 8 h / 1 day / 1 week).
#
#   data   2025-07-01 .. 2026-07-11, links with coverage >= 0.9 and mean >= 1 Mbps
#          (534 links)
#   split  train .. 2026-02-15 | val 2026-02-22 .. 03-15 | test 2026-03-22 .. end
#          (111 days); the train/val dates are the one reconstructed part
#   eval   160 rolling origins, horizons 15 min .. 1 week
#   train  1-week input window, calendar features, asymmetric (peak-aware) loss,
#          up to 100 epochs with early stopping

YEAR_DATA="--start 2025-07-01 --end 2026-07-11 --min-mean-mbps 1"
YEAR_SPLIT="--train-end 2026-02-15 --val-start 2026-02-22 --val-end 2026-03-15 --test-start 2026-03-22"
YEAR_EVAL="--horizons 15min 1h 8h day week --max-origins 160"
YEAR_TRAIN="--tiers 1 3 4 5 --lookback 2016 --epochs 100 --calendar --under-weight 2"
YEAR_FLAGS="$YEAR_DATA $YEAR_SPLIT $YEAR_EVAL $YEAR_TRAIN"

# Parts trained in parallel (each with persistence, which the tables need) and merged.
YEAR_PARTS=(
  "baselines:persistence,seasonal_naive_day,seasonal_naive_week,historical_average"
  "gbdt:persistence,gbdt"
  "gru:persistence,gru"
  "tcn:persistence,tcn"
  "dlinear_patchtst:persistence,dlinear,patchtst"
)
YEAR_MODELS="persistence,seasonal_naive_day,seasonal_naive_week,historical_average,gbdt,gru,tcn,dlinear,patchtst"
