"""One set of periods for both networks (the paper's round 2, 2026-11-08). Every paper script reads
its periods from here; `python periods.py --sh` prints them as shell variables for the recipes.

Days are inclusive [first, last]. Error periods on the test block: calendar days, 13 consecutive
7-day blocks from TEST[0], calendar months; no overlapping windows in a reported error.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from green_routing.paths import REPO_ROOT

TEST = ("2026-04-01", "2026-06-30")          # 91 days, three calendar months
DEV = ("2025-04-01", "2026-03-31")           # development year (Internet2 measured power from 2024-11)
ANNUAL = ("2025-07-01", "2026-06-30")        # twelve months to the end of the test block
LATER = ("2026-07-01", "2026-09-30")         # flat rate only, both networks
VAL = ("2026-03-04", "2026-03-31")           # trained baselines: the last 4 weeks before the test block
TRAIN_END = VAL[0]                           # training targets end before this day
# Training starts at the later of the two networks' series of each kind:
TRAIN_START_5MIN = "2025-11-15"              # 5-minute link counters: Internet2's start (TSDS keeps 1-minute
#                                              data for a rolling year); SWITCHlan's go back to 2023-12
TRAIN_START_HOURLY = "2025-02-01"            # hourly series: SWITCHlan's current doors (AGENTS pitfall 28);
#                                              Internet2's hourly counters go back to 2021-05
CONTEXT_HOURS = 24 * 7 * 24                  # pre-trained models: 24 weeks before each forecast day
I2_EMISSIONS = ("2024-11-01", "2026-11-01")  # Internet2's hourly emissions series (carbon.py YEAR0, YEAR1 in
#                                              i2_carbon_common.py): measured DC power from 2024-11-01
POWER_TRAFFIC = {"switch": ("2024-02-01", "2024-10-31"), "i2": ("2025-11-15", "2026-10-31")}   # Fig. 2, unchanged
# outputs of round 2 (round 1 stays in paper/). Fixed, so that scripts run with GR_ARTIFACTS_DIR set to the
# overlay root ($PAPER_COMMON/root: round 2's Internet2 setup in place of round 1's) still find it.
ROOT = Path(os.environ.get("GR_PAPER_COMMON", REPO_ROOT / "artifacts" / "paper_common"))
OVERLAY = ROOT / "root"


def days(p: tuple[str, str], tz: str | None = "UTC") -> pd.DatetimeIndex:
    return pd.date_range(p[0], p[1], freq="D", tz=tz)


def hours(p: tuple[str, str], tz: str | None = "UTC") -> pd.DatetimeIndex:
    return pd.date_range(p[0], pd.Timestamp(p[1]) + pd.Timedelta(hours=23), freq="h", tz=tz)


def end_exclusive(p: tuple[str, str]) -> str:
    return str((pd.Timestamp(p[1]) + pd.Timedelta(days=1)).date())


def week_blocks(p: tuple[str, str]) -> pd.Series:
    """Day -> index of its 7-day block from p[0]; days of an incomplete last block get NaN."""
    d = days(p, tz=None)
    k = pd.Series((d - d[0]).days // 7, index=d)
    return k.where(k < len(d) // 7).astype("Int64")


def sh() -> str:
    v = {"TEST0": TEST[0], "TEST1": TEST[1], "TEST_END": end_exclusive(TEST), "DEV0": DEV[0], "DEV1": DEV[1],
         "ANNUAL0": ANNUAL[0], "ANNUAL1": ANNUAL[1], "LATER0": LATER[0], "LATER1": LATER[1],
         "VAL0": VAL[0], "VAL_END": end_exclusive(VAL), "TRAIN_END": TRAIN_END,
         "TRAIN_START_5MIN": TRAIN_START_5MIN, "TRAIN_START_HOURLY": TRAIN_START_HOURLY,
         "PAPER_COMMON": str(ROOT)}
    return "\n".join(f'{k}="{x}"' for k, x in v.items())


if __name__ == "__main__":
    if "--sh" in sys.argv:
        print(sh())
    else:
        print(sh().replace('"', ""))
