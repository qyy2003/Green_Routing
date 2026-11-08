"""Paired comparisons of hourly block forecasts on common hours (AGENTS.md section 9).

For each pair (a, b): median link hourly WAPE of each, the median of per-link differences
(a - b, points), the share of links where a beats b, and a 95 % interval for the difference
of median link WAPE from a bootstrap over test DAYS (errors of one day are correlated across
links, so days, not links, are the resampling unit).

usage: python compare.py DS HORIZON a:b [a:b ...]     (HORIZON day | week; append "5" -
  "day5", "week5" - to score 5-minute WAPE on 5-minute values: pred5_*.npy of long5m.py
  runs, the raw 5-minute dense logs)
  a source is a dense block log of the run ("dense:patchtst", "dense:timestral@1h_24w") or
  a run tag of this recipe ("run:E_prof_c2016_ext"); several run tags joined by '+' are
  averaged ("run:x+y").
"""
import sys

import numpy as np
import pandas as pd

from common import DS, CACHE


def load(ds, src, hz, five=False):
    kind, name = src.split(":", 1)
    if kind == "dense":
        return ds.load_dense_block5(name, hz) if five else ds.load_dense_block(name, hz)
    f = f"pred5_{hz}.npy" if five else f"pred_{hz}.npy"
    arrs = [np.load(CACHE / ds.name / "runs" / t / f) for t in name.split("+")]
    return np.mean(arrs, 0)


def main():
    name, hz, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
    five = hz.endswith("5")
    hz = hz.rstrip("5")
    ds = DS(name)
    A = ds.actual5_test() if five else ds.actual_test()
    spd = 288 if five else 24                                   # steps per day
    n = (A.shape[1] // spd) * spd
    rng = np.random.default_rng(0)
    rows = []
    for pair in pairs:
        # pair syntax: kindA:nameA:kindB:nameB
        p = pair.split(":")
        sa, sb = f"{p[0]}:{p[1]}", f"{p[2]}:{p[3]}"
        Pa, Pb = load(ds, sa, hz, five)[:, :n], load(ds, sb, hz, five)[:, :n]
        Ad = A[:, :n]
        ok = np.isfinite(Ad) & np.isfinite(Pa) & np.isfinite(Pb)
        days = n // spd
        ea = np.where(ok, np.abs(Pa - Ad), 0).reshape(ds.L, days, spd).sum(2)
        eb = np.where(ok, np.abs(Pb - Ad), 0).reshape(ds.L, days, spd).sum(2)
        aa = np.where(ok, Ad, 0).reshape(ds.L, days, spd).sum(2)

        def med(e, a_, idx=slice(None)):
            s = a_[:, idx].sum(1)
            w = 100 * e[:, idx].sum(1) / np.where(s > 1e-9, s, np.nan)
            return w

        wa, wb = med(ea, aa), med(eb, aa)
        d = wa - wb
        boots = []
        for _ in range(1000):
            idx = rng.integers(0, days, days)
            boots.append(np.nanmedian(med(ea, aa, idx)) - np.nanmedian(med(eb, aa, idx)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"a": sa, "b": sb, "med_a": np.nanmedian(wa), "med_b": np.nanmedian(wb),
                     "diff_of_medians": np.nanmedian(wa) - np.nanmedian(wb),
                     "ci95_lo": lo, "ci95_hi": hi, "median_link_diff": np.nanmedian(d),
                     "a_beats_b_links": np.nanmean(d < 0), "links": int(np.isfinite(d).sum())})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"== {name} {hz}{' (5-minute WAPE)' if five else ''}")
    print(df.round(2).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
