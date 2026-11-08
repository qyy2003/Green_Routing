#!/usr/bin/env bash
# The paper's evaluation (Sec. 6): data, numbers and figures. Run from anywhere, in the `green` env:
#   source ~/anaconda3/etc/profile.d/conda.sh && conda activate green && recipes/paper/run.sh
# Needs the outputs of recipes/internet2 (carbon.py, links/merged/dense_fm with Timestral's block
# logs, timestral_total.csv), recipes/carbon, recipes/carbon_rates (path, tm_capacity.npz) and
# recipes/flat_forecast (run.py cache, grid_candidates.csv.gz). A GPU for I2's grid forecast
# (Timestral, cached after the first run). About 30 minutes on the RTX PRO 4500 box.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
P=recipes/paper
python -W ignore $P/i2_pre_setup.py         # I2: the 28 days before the test block, for the cap
python -W ignore $P/i2_rates.py             # I2: settled / day-ahead capped rates, city pairs, entities, errors
python -W ignore $P/switch_rates.py         # SWITCHlan: capped path rates, city pairs, entities
python -W ignore $P/bill_periods.py         # bill error by accounting period (Fig. eval-forecast, left)
python -W ignore $P/facts.py                # per GB, variance, daily cycle, kWh/GB, power
python -W ignore $P/power_traffic.py all     # Fig. 1: hourly power vs traffic per router / PoP (reads S3 cache)
python -W ignore $P/steering.py all         # one pair alone, everyone (closed loop), centralized
python -W ignore $P/text_numbers.py         # the remaining numbers of the text
for k in 1 3 none; do                        # the cap's k, for the appendix (rates, one pair, everyone)
  python -W ignore $P/steering.py unilateral --k=$k
  python -W ignore $P/steering.py everyone --k=$k
done
python -W ignore $P/total_neural.py all     # DLinear / GRU / TCN / PatchTST on the hourly network totals
python -W ignore $P/appendix.py             # appendix tables and figures -> 6aea077f.../figure_appendix
python -W ignore $P/tradeoff.py all         # context sweep on the totals, GPU timing, Pareto and learning-curve figures (GPU alone)
python -W ignore $P/figures.py "$@"         # PDFs into 6aea077f0929c9a3a1ac5660/figures (or OUTDIR)
