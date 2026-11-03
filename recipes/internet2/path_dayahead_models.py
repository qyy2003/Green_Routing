"""Day-ahead bills on Internet2 for every tier of forecast: baselines, trained, pretrained.

For each (link forecast, total forecast) pair, ``path_dayahead.main`` builds the day-ahead
matrix and scores path-aware bills (uncapped, capped at 100x and 10x the hour's flat rate)
per router and hour against the bills from the actual matrix. The flat-rate bills are scored
from the total forecast alone: every router pays its actual bytes x E / V_forecast, so over all
routers the hour's bill is E x V_actual / V_forecast, and
    flat bill WAPE = sum_h |E_h V_h / F_h - E_h| / sum_h E_h      (E: actual or day-ahead grid).
Also the same bills with perfect traffic and the day-ahead grid rule (the grid's part alone).

usage: python path_dayahead_models.py [--workers 6] [--with LINKMODEL:TOTAL ...]
writes artifacts/internet2/carbon/path_dayahead_models.csv
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import path_dayahead as PD

C = PD.C
TOTALS = {"dayahead": C / "dayahead_hourly.csv", "timestral": C / "timestral_total.csv"}
PAIRS = [   # (tier, link model, total forecast)
    ("baseline", "persistence", "persistence (same hour yesterday)"),
    ("baseline", "seasonal_naive_day", "persistence (same hour yesterday)"),
    ("baseline", "seasonal_naive_week", "same hour last week"),
    ("baseline", "historical_average", "mean of the same hour, last 4 weeks"),
    ("trained", "dlinear", "GBDT"),
    ("trained", "gru", "GBDT"),
    ("trained", "tcn", "GBDT"),
    ("trained", "patchtst", "GBDT"),
    ("trained + pretrained total", "patchtst", "Chronos-2, no training"),
    ("trained + pretrained total", "patchtst", "Timestral-209M, 24 weeks"),
]


def totals_csv(total: str) -> Path:
    for p in TOTALS.values():
        if p.exists() and total in pd.read_csv(p, nrows=0).columns:
            return p
    raise KeyError(total)


def one(job):
    tier, model, total, dense = job
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    r = PD.main(model, total, dense=dense, totals_csv=totals_csv(total), out=None)
    return dict(tier=tier, link_model=model, total=total, **{k: v for k, v in r.items() if k not in ("model", "total")})


def perfect_traffic():
    """Actual matrix, day-ahead grid rule: the grid's part of the bill error."""
    S = np.load(C / "test_setup.npz", allow_pickle=True)
    A_res = sp.load_npz(C / "A_res.npz").tocsr()
    n = len(S["ends"])
    E, E_da, Xa = S["E"], S["E_da"], S["X_em"]
    ok = np.isfinite(E_da).all(axis=1)
    E_da = np.nan_to_num(E_da)
    row = dict(tier="perfect traffic", link_model="actual", total="actual", hours=int(ok.sum()))
    for cap in (None, 100, 10):
        k = "uncapped" if cap is None else f"cap{cap}x"
        truth = PD.bills_at(A_res, E, Xa, Xa, n, cap)
        fg = PD.bills_at(A_res, E_da, Xa, Xa, n, cap)
        row[f"path_{k}_bill_wape_actual_grid"] = 0.0
        row[f"path_{k}_bill_wape_grid_rule"] = float(np.abs(fg[ok] - truth[ok]).sum() / np.abs(truth[ok]).sum())
    Et, Ed = E.sum(1), E_da.sum(1)
    row["flat_bill_wape_actual_grid"] = 0.0
    row["flat_bill_wape_grid_rule"] = float(np.abs(Ed[ok] - Et[ok]).sum() / Et[ok].sum())
    return row


def flat_bills(total: str) -> dict:
    """Flat-rate bills from the total forecast alone, on the hours the path scores use."""
    S = np.load(C / "test_setup.npz", allow_pickle=True)
    T = pd.DatetimeIndex(S["T"]).tz_localize("UTC")
    E, E_da, Xa = S["E"].sum(1), S["E_da"].sum(1), S["X_em"].sum(1)
    F = pd.read_csv(totals_csv(total), index_col=0, parse_dates=True)[total].reindex(T).to_numpy()
    ok = np.isfinite(F) & np.isfinite(E_da) & (F > 0)
    w = lambda Eg: float(np.abs(Eg[ok] * Xa[ok] / F[ok] - E[ok]).sum() / E[ok].sum())   # noqa: E731
    return dict(flat_bill_wape_actual_grid=w(E), flat_bill_wape_grid_rule=w(E_da), flat_hours=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--with", dest="extra", nargs="*", default=[],
                    help="more pairs TIER|LINKMODEL|TOTAL[|DENSE_DIR]")
    a = ap.parse_args()
    jobs = [(t, m, tot, PD.DENSE) for t, m, tot in PAIRS]
    for x in a.extra:
        p = x.split("|")
        jobs.append((p[0], p[1], p[2], Path(p[3]) if len(p) > 3 else PD.DENSE))
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        pf = ex.submit(perfect_traffic)
        rows = []
        for f, j in zip(futs, jobs):
            r = f.result()
            r.update(flat_bills(j[2]))
            rows.append(r)
            print(f"{r['tier']:28s} {r['link_model']:24s} {r['total']:38s} path10x {r['path_cap10x_bill_wape_actual_grid']:.4f} "
                  f"flat {r['flat_bill_wape_actual_grid']:.4f}", flush=True)
        rows.append(pf.result())
    out = C / "path_dayahead_models.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
