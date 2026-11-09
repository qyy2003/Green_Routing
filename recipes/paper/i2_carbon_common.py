"""Internet2's carbon setup on the common periods (recipes/paper/periods.py): recipes/internet2/carbon.py
unchanged except for its windows and output folder.

  YEAR0 = 2024-11-01   the hourly emissions series from the start of the measured DC power (the download
                       starts there; 100-103 plants report in every month, 95-97 % of plant-hours), so the
                       development year 2025-04-01 -> 2026-03-31 and the annual window have measured power
  TEST  = periods.TEST the test block (setup for path rates and steering, hold-out routing check)
  GBDT  trained from periods.TRAIN_START_HOURLY on (the same start as SWITCHlan's), refitted weekly

  python i2_carbon_common.py          -> $PAPER_COMMON/i2_carbon/        (test block)
  python i2_carbon_common.py pre28    -> $PAPER_COMMON/i2_carbon_pre28/  (the 28 days before it, for the cap)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import periods as P                                                     # noqa: E402

spec = importlib.util.spec_from_file_location("i2_carbon", HERE.parent / "internet2" / "carbon.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

c.YEAR0, c.YEAR1 = P.I2_EMISSIONS
pre = len(sys.argv) > 1 and sys.argv[1] == "pre28"
t0 = pd.Timestamp(P.TEST[0])
if pre:
    c.TEST0, c.TEST1 = str((t0 - pd.Timedelta(days=28)).date()), P.TEST[0]
    c.OUT = P.ROOT / "i2_carbon_pre28"
else:
    c.TEST0, c.TEST1 = P.TEST[0], P.end_exclusive(P.TEST)
    c.OUT = P.ROOT / "i2_carbon"
c.OUT.mkdir(parents=True, exist_ok=True)

_gbdt = c.gbdt_dayahead


def gbdt_from_start(s, days, refit_every=7):
    return _gbdt(s[s.index >= pd.Timestamp(P.TRAIN_START_HOURLY, tz="UTC")], days, refit_every)


c.gbdt_dayahead = gbdt_from_start
c.main()

if pre:
    a, b = np.load(c.OUT / "test_setup.npz"), np.load(P.ROOT / "i2_carbon" / "test_setup.npz")
    for k in ("ends", "lks", "res"):
        assert (a[k] == b[k]).all(), k
    print("pre-block setup:", len(a["T"]), "hours; same routers / links / resources as the test block")
