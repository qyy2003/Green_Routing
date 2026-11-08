#!/usr/bin/env bash
# Greenest-path routing on the measured 2021 matrix (Internet2's old network, March 2021).
# Needs recipes/internet2 (fetch tunnels, carbon) and EIA930_BALANCE_2021_Jan_Jun.csv. ~20 min.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-8}
python setup_2021.py
for mb in "meas own" "meas ecmp" "est ecmp"; do
  set -- $mb
  S="--setup tm2021 --matrix $1 --baseline $2"
  python simulate.py unilateral $S --workers "$W"
  python simulate.py static $S --workers "$W"
  python simulate.py everyone $S --workers "$W"
  python simulate.py everyone $S --cap-util 0.8 --workers "$W"
  python simulate.py everyone $S --carry
  python simulate.py everyone $S --carry --cap-util 0.8
done
