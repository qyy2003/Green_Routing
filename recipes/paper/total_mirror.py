"""SWITCHlan's 5-minute network total from periods.TRAIN_START_HOURLY to the end of the test block, as a
one-series mirror for gr-benchmark (the 5-minute ladder on the total). The same series as
recipes/flat_forecast/run.py's load_5min(): the history extract (current doors, 5-minute steps with
door coverage >= 95 %) before 2025-07-01, then the year mirror of recipes/carbon_rates. Octet columns
hold bytes/s, in and out, so total_mbps = in + out (two times the one-way total), as in the year mirror.

writes $PAPER_COMMON/total_mirror/router-interfaces/total/all.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
import periods as P                                                      # noqa: E402

m = pd.read_csv(ARTIFACTS_DIR / "carbon_rates" / "mirror" / "router-interfaces" / "total" / "all.csv")
t_cur = pd.to_datetime(m.timestamp, unit="s", utc=True)
h = pd.read_csv(ARTIFACTS_DIR / "flat_forecast" / "history" / "total_2024.csv.gz", index_col=0, parse_dates=True)
h = h[(h.index >= pd.Timestamp(P.TRAIN_START_HOURLY, tz="UTC")) & (h.index < t_cur.iloc[0])]
ok = h.coverage >= 0.95
old = pd.DataFrame({"timestamp": (h.index.asi8 // 10**9).astype(np.int64),
                    "ifHCInOctets": (h.in_mbps * 1e6 / 8).where(ok).to_numpy(),
                    "ifHCOutOctets": (h.out_mbps * 1e6 / 8).where(ok).to_numpy()})
end = pd.Timestamp(P.end_exclusive(P.TEST), tz="UTC")
cur = m[(t_cur < end).to_numpy()]
out = pd.concat([old, cur], ignore_index=True)
assert out.timestamp.is_monotonic_increasing and (np.diff(out.timestamp) == 300).all(), "gaps in the 5-minute grid"
f = P.ROOT / "total_mirror" / "router-interfaces" / "total" / "all.csv"
f.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(f, index=False)
last_old = old.tail(288 * 7)[["ifHCInOctets", "ifHCOutOctets"]].sum(axis=1).mean()
first_cur = cur.head(288 * 7)[["ifHCInOctets", "ifHCOutOctets"]].sum(axis=1).mean()
print(f"{len(old)} history steps ({(~ok).sum()} below 95 % coverage -> NaN) + {len(cur)} mirror steps -> {f}")
print(f"last week of the history {last_old * 8 / 1e9:.1f} Gbit/s (in + out), first week of the mirror {first_cur * 8 / 1e9:.1f}")
