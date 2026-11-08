"""Step 1: calibrate the evaluator against the dense_fm tables, and score simple
multi-week seasonal profiles (no training) next to the ladder and Timestral.

usage: python baselines.py {i2,switch}
"""
import sys

import numpy as np
import pandas as pd

from common import DS, score, CACHE

WEEK = 168


def profile(Hm, starts, H, K, how="mean", level=None):
    """Seasonal profile over the last K weeks at the same hour of the week.

    level: None, or ("week", n) -> rescale by mean(last n hours) / mean(same n hours
    averaged over the K previous weeks) (a recent-level correction)."""
    L = Hm.shape[1]
    out = np.full((L, len(starts), H), np.nan)
    for b, s in enumerate(starts):
        k = np.arange(H)
        lags = [s + k - WEEK * i for i in range(1, K + 1)]
        lags = [x for x in lags if x.min() >= 0]
        if not lags:
            continue
        stack = np.stack([Hm[x] for x in lags])                 # [K, H, L]
        with np.errstate(all="ignore"):
            f = np.nanmean(stack, 0) if how == "mean" else np.nanmedian(stack, 0)
        if level is not None:
            n = level
            recent = np.nanmean(Hm[s - n:s], 0)                 # [L]
            past = np.nanmean(np.stack([np.nanmean(Hm[s - n - WEEK * i:s - WEEK * i], 0)
                                        for i in range(1, K + 1) if s - n - WEEK * i >= 0]), 0)
            with np.errstate(all="ignore"):
                r = np.clip(recent / past, 0.2, 5)
            f = f * np.where(np.isfinite(r), r, 1.0)[None, :]
        out[:, b, :] = f.T
    return out


def to_hours(blocks, n):
    L = blocks.shape[0]
    return blocks.reshape(L, -1)[:, :n]


def main(name):
    ds = DS(name)
    A = ds.actual_test()
    n = ds.n_test_hours()
    res = {}
    for hz, H in (("day", 24), ("week", 168)):
        preds = {}
        for m in ["persistence", "seasonal_naive_week", "dlinear", "gru", "tcn", "patchtst",
                  "timestral@1h_24w"]:
            preds[m] = ds.load_dense_block(m, hz)
        starts = ds.block_starts(H)
        for K in (1, 2, 4, 8, 12, 24):
            preds[f"prof_mean_K{K}"] = to_hours(profile(ds.Hm, starts, H, K), n)
            preds[f"prof_median_K{K}"] = to_hours(profile(ds.Hm, starts, H, K, "median"), n)
        for K in (4, 8, 24):
            for lv in (24, 168):
                preds[f"prof_median_K{K}_lvl{lv}"] = to_hours(
                    profile(ds.Hm, starts, H, K, "median", level=lv), n)
        df, per = score(preds, A, ref="patchtst")
        df.insert(0, "horizon", hz)
        print(df.round(3).to_string(index=False), flush=True)
        res[hz] = df
    out = CACHE / name
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(res.values()).to_csv(out / "baselines.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
