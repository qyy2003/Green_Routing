"""The 28 days before Internet2's test block, built exactly like the test block.

The idle-link cap of the paper is computed from the realized rates of the previous 28 days, so the
first four weeks of the test block need the four weeks before it. This runs
recipes/internet2/carbon.py unchanged with its test window moved to [TEST0 - 28 d, TEST0) and its
outputs sent to $GR_ARTIFACTS_DIR/internet2/carbon_pre28 (the real outputs stay untouched).
Resources, routers and links come out in the same order (they depend on the whole year, not on
the window); check.py-style asserts below make sure.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("i2_carbon", HERE.parent / "internet2" / "carbon.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

test0 = pd.Timestamp(c.TEST0)
c.TEST1 = c.TEST0
c.TEST0 = str((test0 - pd.Timedelta(days=28)).date())
c.OUT = ARTIFACTS_DIR / "internet2" / "carbon_pre28"
c.main()

a = np.load(c.OUT / "test_setup.npz")
b = np.load(ARTIFACTS_DIR / "internet2" / "carbon" / "test_setup.npz")
for k in ("ends", "lks", "res"):
    assert (a[k] == b[k]).all(), k
print("pre-block setup:", len(a["T"]), "hours; same routers / links / resources as the test block")
