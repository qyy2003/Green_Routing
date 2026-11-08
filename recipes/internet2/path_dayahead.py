"""Day-ahead path rates on Internet2: how much harder are they to forecast than the flat rate?

For every test hour, a forecast issued at 00:00 UTC for the whole day:
- link loads: the dense *block* log of a ladder model at the day horizon (each panel series is a
  router pair, both directions summed), hourly means, split into the two directions with the
  same hour last week's ratio;
- router door totals: the day-ahead total (Chronos-2, carbon/dayahead_hourly.csv) split over the
  routers with their shares in the same hour last week;
- matrix: EM from these forecasts (as for the actual hours), then each OD pair's usage rate.
The rate is scored as in the paper: the bills it implies *at the actual traffic*, per router and
hour, against the bills from the actual rate; with the actual grid factor and with the day-ahead
grid rule (same hour yesterday). The flat rate is scored the same way.

Needs: carbon.py (test_setup.npz, A_res.npz, A_link.npz, dayahead_hourly.csv) and the ladder's
dense logs (links/merged/dense). Output: carbon/path_dayahead.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from green_routing.carbon import tomography
from green_routing.dataset import internet2 as i2
from green_routing.paths import ARTIFACTS_DIR

ROOT = ARTIFACTS_DIR / "internet2"
C, DATA, DENSE = ROOT / "carbon", ROOT / "data", ROOT / "links" / "merged" / "dense"
TINY = 1e-3


def bills_at(A_res, E, X_rate, X_bill, n, cap=None):
    """Per hour: rates from the matrix ``X_rate`` (usage: emissions over use), charged to the
    bytes of ``X_bill``, half to each end. Resources unused under X_rate are left out. ``cap``:
    no resource charges more than cap x the hour's flat rate (the paper's rule for idle links)."""
    out = np.zeros((len(E), n))
    At = A_res.T.tocsr()
    for h in range(len(E)):
        U = A_res @ X_rate[h]
        per = np.divide(E[h], U, out=np.zeros(len(U)), where=U > 0)
        if cap is not None:
            per = np.minimum(per, cap * E[h].sum() / max(X_rate[h].sum(), 1e-9))
        Cm = (X_bill[h] * (At @ per)).reshape(n, n)
        out[h] = (Cm.sum(1) + Cm.sum(0)) / 2
    return out


def forecast_inputs(model: str | None = None, total: str = "Chronos-2, no training", dense: Path = DENSE,
                    totals_csv: Path | None = None) -> dict:
    """The test block's inputs (test_setup.npz) and the day-ahead forecast of its traffic matrix.
    ``model``: the link forecast (a block log in ``dense``; default the best by hourly WAPE);
    ``total``: the day-ahead total that splits into router door totals (a column of
    ``totals_csv``, default carbon/dayahead_hourly.csv). Returns a dict (T, n, A_res, A_link, In,
    Eg, Y, X_em, E, E_da, Y_fc, X_fc, ok, model)."""
    DENSE = Path(dense)
    S = np.load(C / "test_setup.npz", allow_pickle=True)
    A_res, A_link = sp.load_npz(C / "A_res.npz").tocsr(), sp.load_npz(C / "A_link.npz").tocsr()
    T = pd.DatetimeIndex(S["T"]).tz_localize("UTC")
    ends, lks = list(S["ends"]), list(S["lks"])
    n = len(ends)
    In, Eg, Y, X_em, E, E_da = (S[k] for k in ("In", "Eg", "Y", "X_em", "E", "E_da"))

    # --- link forecasts from the dense block logs ---------------------------------------
    meta = json.loads((DENSE / "meta.json").read_text())
    if model is None:
        agg = pd.read_csv(DENSE / "block" / "aggregate.csv")
        agg = agg[(agg.horizon_name == "day") & ~agg.model.isin(["persistence", "historical_average"])]
        model = agg.sort_values("wape_1h").model.iloc[0]
    hi = [h["name"] for h in meta["horizons"]].index("day")
    P = np.load(DENSE / f"pred_block_{model}.npy", mmap_mode="r")[:, hi, :]          # [L, n_t], Mbit/s
    nt = (meta["n_targets"] // 12) * 12
    t0 = pd.Timestamp(meta["t0"], unit="s", tz="UTC")
    Ht = pd.date_range(t0, periods=nt // 12, freq="h")
    Ph = np.asarray(P[:, :nt], dtype=np.float64).reshape(P.shape[0], -1, 12).mean(axis=2)  # hourly
    pair_fc = {}
    for j, name in enumerate(meta["links"]):                      # "core1.alba/to-core1.bost2"
        u, v = name.split("/to-")
        pair_fc[(u + i2.FQDN, v + i2.FQDN)] = pd.Series(Ph[j], index=Ht)

    # direction shares, same hour last week, from the hourly directed loads
    LH = pd.read_csv(DATA / "links_hourly.csv.gz", index_col=0, parse_dates=True)
    full = lambda s: s + i2.FQDN                                   # noqa: E731
    LH.columns = pd.MultiIndex.from_tuples([(full(a), full(b)) for a, b in (c.split("->") for c in LH.columns)])
    Y_fc = np.full((len(T), len(lks)), np.nan)
    for j, l in enumerate(lks):
        u, v = l.split("->")
        f = pair_fc.get((u, v)) if (u, v) in pair_fc else pair_fc.get((v, u))
        if f is None or (u, v) not in LH or (v, u) not in LH:
            continue
        share = (LH[(u, v)] / (LH[(u, v)] + LH[(v, u)])).shift(168)
        Y_fc[:, j] = (f.reindex(T) * share.reindex(T).clip(0, 1)).to_numpy()

    # router door totals: day-ahead total x the router's share last week
    D = pd.read_csv(DATA / "doors.csv")
    DI = pd.read_csv(DATA / "doors_in_hourly.csv.gz", index_col=0, parse_dates=True)
    DO = pd.read_csv(DATA / "doors_out_hourly.csv.gz", index_col=0, parse_dates=True)
    r_of = D.set_index("door").node.reindex(DI.columns)
    ing = DI.T.groupby(r_of).sum().T.reindex(columns=ends)
    egr = DO.T.groupby(r_of).sum().T.reindex(columns=ends)
    sh_in = ing.div(ing.sum(axis=1), axis=0).shift(168).reindex(T).fillna(0).to_numpy()
    sh_eg = egr.div(egr.sum(axis=1), axis=0).shift(168).reindex(T).fillna(0).to_numpy()
    tot = pd.read_csv(totals_csv or C / "dayahead_hourly.csv", index_col=0, parse_dates=True)[total].reindex(T)
    In_fc, Eg_fc = sh_in * tot.to_numpy()[:, None], sh_eg * tot.to_numpy()[:, None]
    ok = np.isfinite(tot.to_numpy()) & (np.isfinite(Y_fc).mean(axis=1) > 0.8)

    X_fc = np.array([tomography.em(A_link, Y_fc[h], In_fc[h], Eg_fc[h]) if ok[h] else X_em[h]
                     for h in range(len(T))])
    X_fc[X_fc < TINY] = 0.0
    return dict(model=model, T=T, n=n, A_res=A_res, A_link=A_link, In=In, Eg=Eg, Y=Y, X_em=X_em, E=E,
                E_da=E_da, Y_fc=Y_fc, X_fc=X_fc, ok=ok)


def main(model: str | None = None, total: str = "Chronos-2, no training", dense: Path = DENSE,
         totals_csv: Path | None = None, out: Path | None = C / "path_dayahead.json") -> dict:
    """Scores (see the module docstring) of ``forecast_inputs(model, total, dense, totals_csv)``.
    ``out`` None: return the dict only."""
    F = forecast_inputs(model, total, dense, totals_csv)
    model, n, A_res, Y, X_em, E, E_da, Y_fc, X_fc, ok = (
        F[k] for k in ("model", "n", "A_res", "Y", "X_em", "E", "E_da", "Y_fc", "X_fc", "ok"))
    Xa = X_em                                                         # the actual matrix (counters)
    ok &= np.isfinite(E_da).all(axis=1)
    E_da = np.nan_to_num(E_da)
    res = {}
    for cap in (None, 100, 10):
        truth = bills_at(A_res, E, Xa, Xa, n, cap)
        fa = bills_at(A_res, E, X_fc, Xa, n, cap)
        fg = bills_at(A_res, E_da, X_fc, Xa, n, cap)
        k = "uncapped" if cap is None else f"cap{cap}x"
        res[k] = (truth, fa, fg)
    V_a, V_fc = Xa.sum(1), X_fc.sum(1)
    v_r = (Xa.reshape(-1, n, n).sum(2) + Xa.reshape(-1, n, n).sum(1)) / 2   # each router's half-bytes
    flat_truth = v_r * (E.sum(1) / V_a)[:, None]
    flat_fc_actual_grid = v_r * (E.sum(1) / V_fc)[:, None]
    flat_fc_grid_rule = v_r * (E_da.sum(1) / V_fc)[:, None]

    def wape(a, b):
        m = ok
        return float(np.abs(a[m] - b[m]).sum() / np.abs(b[m]).sum())
    def med_router(a, b):
        m = ok
        return float(np.median(np.abs(a[m] - b[m]).sum(0) / np.maximum(np.abs(b[m]).sum(0), 1e-9)))
    res_ = dict(model=model, total=total, hours=int(ok.sum()), routers=n,
               link_forecast_hourly_wape=float(np.nansum(np.abs(Y_fc - Y)[ok]) / np.nansum(np.where(np.isfinite(Y_fc), Y, 0)[ok])),
               flat_rate_bill_wape_actual_grid=wape(flat_fc_actual_grid, flat_truth),
               flat_rate_bill_wape_grid_rule=wape(flat_fc_grid_rule, flat_truth),
               flat_rate_median_router_wape_actual_grid=med_router(flat_fc_actual_grid, flat_truth))
    for k, (truth, fa, fg) in res.items():
        res_[f"path_{k}_bill_wape_actual_grid"] = wape(fa, truth)
        res_[f"path_{k}_bill_wape_grid_rule"] = wape(fg, truth)
        res_[f"path_{k}_median_router_wape_actual_grid"] = med_router(fa, truth)
    if out is not None:
        Path(out).write_text(json.dumps(res_, indent=1))
        print(json.dumps(res_, indent=1))
    return res_


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
