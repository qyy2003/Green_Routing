"""Timestral (Mistral, 2026-10) on Internet2's total, a day ahead: the same series, issue times
and scored hours as carbon.py's day-ahead table (carbon/dayahead_hourly.csv), so the rows add to
carbon/rate_errors.csv's. Writes carbon/timestral_total.csv (forecasts) and
carbon/rate_errors_timestral.csv (every forecast of the table, rescored with Timestral's).
Needs carbon.py's outputs and a GPU (CPU works, slowly).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carbon as C                                                   # noqa: E402
from green_routing.carbon import dayahead                            # noqa: E402

VARIANTS = {
    "Timestral-209M, 12 weeks": dict(model="timestral", context=2016),
    "Timestral-209M, 24 weeks": dict(model="timestral", context=4032),
    "Timestral-209M, 48 weeks": dict(model="timestral", context=8064),
    "Timestral-28M, 12 weeks": dict(model="timestral-28M", context=2016),
    "Chronos-2, 24 weeks": dict(model="chronos-2", context=4032),
}


def total_series():
    """carbon.py's hourly total (each byte once, complete and sane hours, gaps <= 24 h filled)."""
    links, LH, D, DI, DO, W = C.load()
    core = sorted(set(links.u) | set(links.v) | set(D.node))
    H = pd.date_range(C.YEAR0, C.YEAR1, freq="h", inclusive="left", tz="UTC")
    door_router = D.set_index("door").node
    ing = DI.T.groupby(door_router.reindex(DI.columns)).sum().T.reindex(H)
    ends = sorted(set(ing.columns) & set(core))
    n_rep = DI.notna().sum(axis=1)
    complete = (n_rep > 0) & (n_rep >= 0.9 * n_rep.rolling(24 * 14, center=True, min_periods=24).max())
    sane = lambda v: v > 0.2 * v.rolling(24 * 7, center=True, min_periods=24).median()     # noqa: E731
    full_idx = pd.date_range(DI.index.min(), pd.Timestamp(C.YEAR1, tz="UTC"), freq="h", inclusive="left")
    ing_all = DI.T.groupby(door_router.reindex(DI.columns)).sum().T
    Vh = ing_all[ends].sum(axis=1).where(complete)
    Vh = Vh.where(sane(Vh)).reindex(full_idx)
    return Vh.interpolate(limit=24).ffill()


def main():
    s = total_series()
    days = pd.date_range(C.TEST0, pd.Timestamp(C.TEST1) - pd.Timedelta(days=1), freq="D", tz="UTC")
    tab = pd.read_csv(C.OUT / "dayahead_hourly.csv", index_col=0, parse_dates=True)
    hh = tab.index
    # the series must be the one carbon.py forecast from: its actuals on the scored hours
    assert np.allclose(s.reindex(hh).to_numpy(), tab.actual.to_numpy(), rtol=1e-6), "series differs from carbon.py's"
    out = C.OUT / "timestral_total.csv"
    fc = pd.read_csv(out, index_col=0, parse_dates=True) if out.exists() else pd.DataFrame()
    for name, kw in VARIANTS.items():
        if name in fc:
            continue
        print(name, flush=True)
        fc[name] = dayahead.foundation(s.dropna(), days, **kw)
        fc.to_csv(out)
    GB = C.GB
    Vt, Et, E_fc = tab.actual, tab.E_g_per_h, tab.E_dayahead_g_per_h
    r_true = Et / (Vt * GB)
    F = {k: tab[k] for k in tab.columns if k not in ("actual", "E_g_per_h", "E_dayahead_g_per_h")}
    F |= {k: fc[k].reindex(hh) for k in fc}
    rows = []
    for name, p in F.items():
        rr, rf = Et / (p * GB), E_fc / (p * GB)
        rows.append(dict(model=name, traffic_hourly_WAPE=float((p - Vt).abs().sum() / Vt.sum()),
                         traffic_bias=float((p - Vt).sum() / Vt.sum()),
                         rate_WAPE_actual_grid=float((rr - r_true).abs().sum() / r_true.sum()),
                         rate_WAPE_grid_forecast=float((rf - r_true).abs().sum() / r_true.sum())))
    RE = pd.DataFrame(rows).sort_values("traffic_hourly_WAPE")
    RE.to_csv(C.OUT / "rate_errors_timestral.csv", index=False)
    print(RE.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
