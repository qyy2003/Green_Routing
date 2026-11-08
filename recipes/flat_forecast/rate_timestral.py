"""Forecast the grid, or the rate itself? Timestral zero-shot on both networks.

The flat rate is r(h) = E(h) / T(h): backbone emissions (power x grid factor; power is static, so E
follows the power-weighted grid intensity) over total traffic. Day-ahead options, all with
Timestral-209M, 24 weeks of hourly context, issued 00:00 UTC for the next 24 hours, no training:
  rule / TS     E_f by "same hour yesterday" (the current rule) / r_f = E_f / T_f, T_f by Timestral
  TS / TS       E_f by Timestral on the emissions series        / r_f = E_f / T_f
  direct        Timestral on the rate series r itself
  direct, cap   the same, rate history capped at 3x its median of the previous 28 days
  references    perfect grid / Timestral traffic; Timestral grid / perfect traffic
  "mean: ..."   the same with Timestral's mean (from its 21 quantiles) instead of its median
Billed carbon = sum r_f(h) T(h); error against the emitted carbon sum E(h) per day, every 7-day
window and every 30-day window (WAPE), and over the whole test block (bias). The division: the
spread of T / T_f (a forecast near zero would blow the rate up) and the rate history's spikes.

usage: python rate_timestral.py          writes <ARTIFACTS>/flat_forecast/rate_timestral.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.carbon import dayahead as DA
from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
OUT = ARTIFACTS_DIR / "flat_forecast"
CTX = 4032


def switch():
    """Hourly E (g/h), traffic T (Gbit/s, truth with gaps), input series (filled), test days, GB factor."""
    sys.path.insert(0, str(HERE))
    import run as R
    from green_routing.carbon import assumptions as A, bills, grid
    from green_routing.paths import reference_dir
    C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
    H = pd.date_range(pd.Timestamp("2025-07-01", tz="UTC"), R.TEST[-1] + pd.Timedelta(hours=23), freq="h")
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    tiers = pd.read_csv(NET / "routers.csv").set_index("dev").tier
    _, Pbb, _ = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"),
                                   pd.read_csv(C / "allocation" / "device_pools.csv"),
                                   pd.read_csv(C / "inventory" / "shelves.csv"),
                                   pd.read_csv(reference_dir() / "sites.csv"), doors, tiers, H)
    IA = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in Pbb})
    IF = pd.DataFrame({z: grid.day_ahead(grid.load(C / "grid", z)).reindex(H).interpolate() for z in Pbb})
    E = sum(Pbb[z] / 1000 * IA[z] for z in Pbb)
    Erule = sum(Pbb[z] / 1000 * IF[z] for z in Pbb)
    return dict(E=E, Erule=Erule, T=R.T.reindex(H), Tin=R.S.reindex(H).interpolate().bfill(),
                days=R.TEST, GB=A.GB_PER_MBPS_HOUR * 1e3, unit="Gbit/s")


def i2():
    sys.path.insert(0, str(HERE.parent / "internet2"))
    import carbon as C
    import timestral_total as TT
    tab = pd.read_csv(C.OUT / "dayahead_hourly.csv", index_col=0, parse_dates=True)
    fr = pd.read_csv(C.OUT / "flat_rate_hourly.csv", index_col=0, parse_dates=True)
    s = TT.total_series()
    H = fr.index
    E = fr.E_g_per_h
    assert np.allclose(E.reindex(tab.index), tab.E_g_per_h, rtol=1e-6)
    T = pd.Series(np.nan, index=H)
    T[tab.index] = tab.actual                                       # scored hours only
    days = pd.date_range(C.TEST0, pd.Timestamp(C.TEST1) - pd.Timedelta(days=1), freq="D", tz="UTC")
    Erule = pd.Series(np.nan, index=H)
    Erule[tab.index] = tab.E_dayahead_g_per_h
    return dict(E=E, Erule=Erule, T=T, Tin=s.reindex(H).interpolate().bfill(), days=days, GB=C.GB,
                unit="Mbit/s")


def ts(s: pd.Series, days, point: str = "median") -> pd.Series:
    return DA.foundation(s.dropna(), days, model="timestral", context=CTX, point=point)


def capped(r: pd.Series, k: float = 3.0) -> pd.Series:
    """Rate history capped at k x its median over the previous 28 days (causal)."""
    med = r.rolling(24 * 28, min_periods=24 * 7).median().shift(1)
    return r.clip(upper=k * med.bfill())


def errors(E, T, bill, r_true, r_f) -> dict:
    """Hourly rate WAPE and billed-carbon WAPE per day / 7-day / 30-day window, bias."""
    day = E.index.floor("D")
    d = pd.DataFrame({"b": bill.groupby(day).sum(), "e": E.groupby(day).sum()})
    w = lambda k: float((d.b.rolling(k).sum() - d.e.rolling(k).sum()).abs().dropna().sum()   # noqa: E731
                        / d.e.rolling(k).sum().dropna().sum())
    return {"rate hourly": float((r_f - r_true).abs().sum() / r_true.sum()),
            "bill day": w(1), "bill week": w(7), "bill month": w(30),
            "bias": float((bill.sum() - E.sum()) / E.sum())}


def run(name, D):
    E, T, Tin, GB = D["E"], D["T"], D["Tin"], D["GB"]
    r_in = E / (Tin * GB)                                            # rate history (filled traffic)
    Tf = ts(Tin, D["days"])
    Ets = ts(E, D["days"])
    rts = ts(r_in, D["days"])
    rts_c = ts(capped(r_in), D["days"])
    Tf_m, Ets_m, rts_m = ts(Tin, D["days"], "mean"), ts(E, D["days"], "mean"), ts(r_in, D["days"], "mean")
    idx = Tf.index.intersection(T.dropna().index)
    idx = idx[np.isfinite(E.reindex(idx)) & np.isfinite(D["Erule"].reindex(idx))]
    e, t, tf = E.reindex(idx), T.reindex(idx), Tf.reindex(idx)
    r_true = e / (t * GB)
    out = []
    grid_rows = {"same hour yesterday": D["Erule"].reindex(idx), "Timestral": Ets.reindex(idx),
                 "Timestral, mean": Ets_m.reindex(idx)}
    tfm = Tf_m.reindex(idx)
    for g, ef in grid_rows.items():
        day = idx.floor("D")
        out.append(dict(network=name, what="emissions (grid) forecast", method=g,
                        **{"hourly": float((ef - e).abs().sum() / e.sum()),
                           "daily": float((ef.groupby(day).sum() - e.groupby(day).sum()).abs().sum() / e.sum())}))
    rates = {"rule grid / Timestral traffic": grid_rows["same hour yesterday"] / (tf * GB),
             "Timestral grid / Timestral traffic": grid_rows["Timestral"] / (tf * GB),
             "Timestral on the rate directly": rts.reindex(idx),
             "Timestral on the rate, history capped": rts_c.reindex(idx),
             "mean: rule grid / Timestral traffic": grid_rows["same hour yesterday"] / (tfm * GB),
             "mean: Timestral grid / Timestral traffic": grid_rows["Timestral, mean"] / (tfm * GB),
             "mean: Timestral on the rate directly": rts_m.reindex(idx),
             "ref: perfect grid / Timestral traffic": e / (tf * GB),
             "ref: Timestral grid / perfect traffic": grid_rows["Timestral"] / (t * GB)}
    for m, rf in rates.items():
        out.append(dict(network=name, what="rate and bill", method=m, **errors(e, t, rf * t * GB, r_true, rf)))
    ratio = t / tf
    spikes = (r_in / r_in.rolling(24 * 28, min_periods=24 * 7).median().shift(1)).reindex(
        pd.date_range(idx[0] - pd.Timedelta(weeks=24), idx[-1], freq="h"))
    out.append(dict(network=name, what="division check", method="T / T_f and rate history",
                    **{"T/Tf p0.1": float(ratio.quantile(.001)), "T/Tf p99.9": float(ratio.quantile(.999)),
                       "T/Tf max": float(ratio.max()), "min T_f / median T": float(tf.min() / t.median()),
                       "rate history > 3x 28-day median": float((spikes > 3).mean()),
                       "rate history max / median": float(spikes.max()), "hours": len(idx),
                       "traffic hourly WAPE": float((tf - t).abs().sum() / t.sum())}))
    return out


def main():
    rows = []
    for name, f in (("Switch LAN", switch), ("I2 Network", i2)):
        r = run(name, f())
        rows += r
        df = pd.DataFrame(r)
        pd.set_option("display.width", 250)
        for what, g in df.groupby("what", sort=False):
            print(f"== {name}: {what}")
            print(g.dropna(axis=1, how="all").drop(columns=["network", "what"]).round(4).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rate_timestral.csv", index=False)


if __name__ == "__main__":
    main()
