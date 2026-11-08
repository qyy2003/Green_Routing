#!/usr/bin/env bash
# Greenest-path routing on Internet2 (needs recipes/internet2/run.sh carbon). ~15 min.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-8}
python simulate.py unilateral --workers "$W"
python simulate.py static --workers "$W"
python simulate.py everyone --workers "$W"
python simulate.py everyone --carry
python simulate.py everyone --cap-util 0.8 --workers "$W"
python simulate.py everyone --carry --cap-util 0.8
