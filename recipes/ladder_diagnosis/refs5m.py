"""Reference scores for the 5-minute long-context runs (long5m.py): the ladder's and
Timestral's block logs, and untrained 5-minute weekly profiles over K weeks, on hourly WAPE
and 5-minute WAPE (day and week blocks, median link).

usage: python refs5m.py DS        writes <CACHE>/<ds>/refs5m.csv
Profiles: per-5-minute-of-week median of the last K weeks before each block start (5-minute
history before the panel as in long5m.py --hist ext), optionally rescaled by the ratio of the
last week's mean to the profile's mean ("+level").
"""
import sys

import numpy as np
import pandas as pd

from common import DS, CACHE, score
from long5m import Data5, W, D, wape5, to_hourly


def profile(data, starts, Hb, K, level, n5):
    L = data.L
    P = np.full((L, len(starts) * Hb), np.nan, np.float32)
    for b, s in enumerate(starts):
        a = s - K * W
        if a < 0:
            continue
        X = data.Y[a:s].reshape(K, W, L)                          # raw (NaN gaps)
        with np.errstate(all="ignore"):
            p = np.nanmedian(X, axis=0)[:Hb]                       # [Hb, L]
            if level:
                r = np.nanmean(data.Y[s - W:s], 0) / np.nanmean(np.nanmedian(X, 0), 0)
                p = p * np.where(np.isfinite(r), r, 1.0)
        P[:, b * Hb:(b + 1) * Hb] = p.T
    return P[:, :n5]


def main():
    name = sys.argv[1]
    ds = DS(name)
    data = Data5(ds, "ext")
    A5, A = ds.actual5_test(), ds.actual_test()
    n5, nh = ds.n_test_steps(), ds.n_test_hours()
    rows = []
    for hz, Hb in (("day", D), ("week", W)):
        starts = np.arange(data.test0, data.N, Hb)
        names = ["patchtst", "tcn", "gru", "dlinear", "seasonal_naive_week", "timestral@5m_4w",
                 "timestral@5m_16w", "timestral@1h_24w"]
        names = [m for m in names if (ds.run / "dense_fm" / f"pred_block_{m}.npy").exists()]
        P5 = {m: ds.load_dense_block5(m, hz) for m in names}
        for K in (1, 4, 12, 24):
            for lev in (0, 1):
                P5[f"profile{K}w" + ("+level" * lev)] = profile(data, starts, Hb, K, lev, n5)
        ph = {k: to_hourly(v, nh) for k, v in P5.items()}
        df, _ = score(ph, A, ref="patchtst")
        w5 = wape5(P5, A5)
        df["wape5"] = df.model.map(w5)
        df["horizon"] = hz
        print(df.round(2).to_string(index=False), flush=True)
        rows.append(df)
    out = pd.concat(rows)
    out.to_csv(CACHE / name / "refs5m.csv", index=False)


if __name__ == "__main__":
    main()
