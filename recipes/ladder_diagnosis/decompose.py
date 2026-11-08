"""Where does an error (or a gain) sit: the day's volume or the shape within the day?

For each source (as in compare.py: dense:<model> or run:<tag>[+<tag>...]), day-ahead block
forecasts, median over links of:
  hourly   hourly WAPE (the headline metric)
  volume   daily-volume WAPE: sum over days |sum_day F - sum_day A| / sum A
  shape    hourly WAPE after rescaling each forecast day to the true daily volume
           (the error left with a perfect level)
Common hours across all sources; days with all 24 hours complete only.

usage: python decompose.py DS SOURCE [SOURCE ...]
"""
import sys

import numpy as np
import pandas as pd

from common import DS
from compare import load


def main():
    name, srcs = sys.argv[1], sys.argv[2:]
    ds = DS(name)
    A = ds.actual_test()
    n = (A.shape[1] // 24) * 24
    A = A[:, :n]
    P = {s: load(ds, s, "day")[:, :n] for s in srcs}
    ok = np.isfinite(A)
    for p in P.values():
        ok &= np.isfinite(p)
    L, days = A.shape[0], n // 24
    okd = ok.reshape(L, days, 24).all(2)                      # complete days only
    Ad = A.reshape(L, days, 24)
    rows = []
    for s, p in P.items():
        Pd = p.reshape(L, days, 24)
        a_day = np.where(okd, Ad.sum(2), 0)
        p_day = np.where(okd, Pd.sum(2), 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            hourly = 100 * np.where(okd[..., None], np.abs(Pd - Ad), 0).sum((1, 2)) / a_day.sum(1)
            volume = 100 * np.abs(p_day - a_day).sum(1) / a_day.sum(1)
            scale = np.where(p_day > 0, a_day / p_day, 1.0)
            shape = 100 * np.where(okd[..., None], np.abs(Pd * scale[..., None] - Ad), 0).sum((1, 2)) / a_day.sum(1)
        rows.append({"source": s, "hourly": np.nanmedian(hourly), "volume": np.nanmedian(volume),
                     "shape": np.nanmedian(shape), "days_used": int(okd.sum())})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
