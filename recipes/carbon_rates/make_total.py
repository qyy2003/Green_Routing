"""Write the total network traffic (every door, 5 min, year window) as a one-series mirror.

usage: python make_total.py <mirror root>
"""
import sys

import pandas as pd

from green_routing.carbon import total
from green_routing.paths import ARTIFACTS_DIR

doors = pd.read_csv(ARTIFACTS_DIR / "carbon" / "allocation" / "doors.csv")
tin, tout, cov = total.total_5min(doors, "2025-07-01", "2026-07-11 00:05")
f = total.write_total_mirror(tin, tout, sys.argv[1])
print(f"wrote {f}: {len(tin)} steps, {int(tin.isna().sum())} outage steps, "
      f"mean {((tin + tout) / 2).mean() / 1e3:.1f} Gbit/s one-way")
