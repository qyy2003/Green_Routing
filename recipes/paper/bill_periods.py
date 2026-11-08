"""The network's day-ahead carbon bill, by accounting period, on both networks (paper Fig. 3a).

The ISP publishes an hourly rate a day ahead, r_f(h) = E_f(h) / T_f(h), and bills every GB of hour h
at r_f(h); what it emitted is E(h). For a period P the bill error is |sum_P r_f T - sum_P E| / sum_P E,
as WAPE over every period of a length (hour, day, every 7-day window, every 30-day window), as
flat_forecast/bill_horizons.py. Forecasts, the same method on both networks, no training:
  traffic T_f  Timestral-209M, 24 weeks of hourly context (Switch LAN: with calendar covariates);
  grid E_f     Timestral's median of the emissions series + a causal 7-day bias correction
               (x actual / forecast of the previous 7 days; chosen on Switch LAN's development
               year, flat_forecast/grid_forecast.py).
Variants: both forecast; perfect traffic (T_f = T); perfect grid (E_f = E). For reference the grid
rule "same hour yesterday" is scored too.

usage: python bill_periods.py        writes $GR_ARTIFACTS_DIR/paper/bill_periods.csv, hourly_{switch,i2}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
FF = HERE.parent / "flat_forecast"
OUT = ARTIFACTS_DIR / "paper"
SWITCH_TRAFFIC = "timestral: 24 weeks + calendar covariates"
I2_TRAFFIC = "Timestral-209M, 24 weeks"


def switch_hours() -> pd.DataFrame:
    sys.path.insert(0, str(FF))
    import grid_forecast as G
    import run as R
    cand = pd.read_csv(G.CAND, index_col=0, parse_dates=True)
    a = cand["actual"]
    Ef = G.bias_corrected(cand["Timestral: median"], a, 7)
    H = pd.date_range(R.TEST[0], R.TEST[-1] + pd.Timedelta(hours=23), freq="h")
    df = pd.DataFrame({"T": R.T.reindex(H), "Tf": R.cache()[SWITCH_TRAFFIC].reindex(H), "E": a.reindex(H),
                       "Ef": Ef.reindex(H), "Ef_rule": cand["rule: yesterday same hour"].reindex(H)})
    return df


def i2_hours() -> pd.DataFrame:
    sys.path.insert(0, str(FF))
    import grid_forecast as G
    from bias import timestral_quantiles
    C = ARTIFACTS_DIR / "internet2" / "carbon"
    tab = pd.read_csv(C / "dayahead_hourly.csv", index_col=0, parse_dates=True)
    fc = pd.read_csv(C / "timestral_total.csv", index_col=0, parse_dates=True)
    fr = pd.read_csv(C / "flat_rate_hourly.csv", index_col=0, parse_dates=True)
    E = fr.E_g_per_h
    days = pd.date_range(tab.index.min().floor("D"), tab.index.max().floor("D"), freq="D")
    pre = pd.date_range(days[0] - pd.Timedelta(days=7), days[0] - pd.Timedelta(days=1), freq="D")
    cache = OUT / "i2_grid_timestral_median.csv"
    if cache.exists():
        med = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        Q, _ = timestral_quantiles(E, pre.append(days))
        med = Q[0.5]
        med.rename("timestral_median").to_csv(cache)
    Ef = G.bias_corrected(med, E, 7)
    df = pd.DataFrame({"T": tab.actual, "Tf": fc[I2_TRAFFIC].reindex(tab.index), "E": tab.E_g_per_h,
                       "Ef": Ef.reindex(tab.index), "Ef_rule": tab.E_dayahead_g_per_h})
    return df


def errors(df: pd.DataFrame) -> list[dict]:
    variants = {"forecast": ("Tf", "Ef"), "perfect traffic": ("T", "Ef"), "perfect grid": ("Tf", "E"),
                "rule grid": ("Tf", "Ef_rule"), "rule grid, perfect traffic": ("T", "Ef_rule")}
    df = df.dropna()
    day = df.index.floor("D")
    rows = []
    for v, (tf, ef) in variants.items():
        bill = df[ef] * df["T"] / df[tf]
        D = pd.DataFrame({"bill": bill.groupby(day).sum(), "E": df["E"].groupby(day).sum()})
        per = {"hour": float((bill - df["E"]).abs().sum() / df["E"].sum())}
        for name, k in (("day", 1), ("week", 7), ("month", 30)):
            rb, re_ = D.bill.rolling(k).sum().dropna(), D.E.rolling(k).sum().dropna()
            per[name] = float((rb - re_).abs().sum() / re_.sum())
        bias = float(bill.sum() / df["E"].sum() - 1)
        for p, w in per.items():
            rows.append(dict(variant=v, period=p, wape=w, bias=bias, hours=len(df)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for name, f in (("Switch LAN", switch_hours), ("I2 Network", i2_hours)):
        df = f()
        df.to_csv(OUT / f"hourly_{'switch' if name.startswith('Switch') else 'i2'}.csv")
        d = df.dropna()
        print(f"== {name}: {len(d)} hours {d.index[0].date()} -> {d.index[-1].date()}; traffic hourly WAPE "
              f"{100 * (d.Tf - d['T']).abs().sum() / d['T'].sum():.1f} %, grid hourly WAPE "
              f"{100 * (d.Ef - d.E).abs().sum() / d.E.sum():.1f} % (rule {100 * (d.Ef_rule - d.E).abs().sum() / d.E.sum():.1f} %)",
              flush=True)
        R = pd.DataFrame(errors(df))
        R.insert(0, "network", name)
        print(R.pivot(index="variant", columns="period", values="wape")[["hour", "day", "week", "month"]]
              .mul(100).round(1).to_string(), flush=True)
        print(R.groupby("variant").bias.first().mul(100).round(2).to_string(), flush=True)
        out.append(R)
    pd.concat(out).to_csv(OUT / "bill_periods.csv", index=False)


if __name__ == "__main__":
    main()
