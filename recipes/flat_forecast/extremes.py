"""Extreme points in the Swiss grid, and whether a peak-aware or a mean loss gives an unbiased forecast.

  data       Switch LAN backbone emissions (power x grid factor) and the CH grid factor: quantiles,
             hours above 3x / 5x the median, isolated one-hour spikes (> 3x both neighbours: data
             glitches look like this), the top days and how long the high episodes last
  influence  bias of each forecast with and without the top 5 / 10 % of days (by actual)
  losses     the same GBDT (grid_forecast.py features, refitted every 28 days) with three targets:
             log + squared error (current: geometric mean), log + quantile 2/3 (peak-aware: the
             ladder's --under-weight 2 makes under-forecasts cost twice = the 2/3 quantile), linear
             ratio + squared error (the mean); and a distributional one: log-normal, mu from the log
             model, sigma^2 from a second GBDT on out-of-fold squared log residuals, forecast =
             exp(mu + sigma^2/2), the mean of the fitted distribution (maximum likelihood = minimum
             KL divergence within the log-normal family); DEV and TEST scores as grid_forecast.py

usage: python extremes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid_forecast as G              # noqa: E402
from green_routing.carbon import dayahead as DA, grid   # noqa: E402
from green_routing.paths import ARTIFACTS_DIR           # noqa: E402


def data_extremes(E: pd.Series):
    I = grid.load(ARTIFACTS_DIR / "carbon" / "grid", "CH").reindex(E.index).interpolate()
    for name, s in (("CH grid factor (g/kWh)", I), ("backbone emissions (g/h)", E)):
        med = s.median()
        nb = (s.shift(1) + s.shift(-1)) / 2
        iso = (s > 3 * s.shift(1)) & (s > 3 * s.shift(-1))
        hi = s > 3 * med
        runs = hi.astype(int).groupby((~hi).cumsum()).sum()
        runs = runs[runs > 0]
        print(f"{name}, 2023-01 -> 2026-07: median {med:.3g}, p90 {s.quantile(.9) / med:.2f}x, p99 "
              f"{s.quantile(.99) / med:.2f}x, max {s.max() / med:.2f}x median; hours > 3x median "
              f"{100 * hi.mean():.1f} %, > 5x {100 * (s > 5 * med).mean():.2f} %; isolated 1-hour spikes "
              f"(> 3x both neighbours) {int(iso.sum())}; episodes > 3x median: {len(runs)}, median length "
              f"{runs.median():.0f} h, longest {runs.max()} h")
    d = E.groupby(E.index.normalize()).sum()
    top = d.sort_values(ascending=False).head(10)
    print("top days (emissions / median day):", ", ".join(f"{k.date()} {v / d.median():.1f}x" for k, v in top.items()))
    m = d.groupby(d.index.month).median() / d.median()
    print("median day by calendar month / overall median:", " ".join(f"{k}:{v:.2f}" for k, v in m.items()))


def influence(df: pd.DataFrame):
    a = df["actual"]
    day = df.index.normalize()
    ad = a.groupby(day).sum()
    rows = []
    for k in ("rule: yesterday same hour", "Timestral: median", "Timestral: mean", "chronos-2",
              "GBDT (log, squared error)", "GBDT (log, absolute error)"):
        fd = df[k].groupby(day).sum()
        r = {"forecast": k, "bias %": 100 * (fd.sum() / ad.sum() - 1)}
        for q in (0.95, 0.90):
            keep = ad < ad.quantile(q)
            r[f"without top {int(round(100 * (1 - q)))} % days"] = 100 * (fd[keep].sum() / ad[keep].sum() - 1)
        under = (ad - fd).clip(lower=0)
        r["share of under-forecast on top 10 % days"] = 100 * under[ad >= ad.quantile(.9)].sum() / under.sum()
        rows.append(r)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


def gbdt_targets(E: pd.Series) -> dict:
    spec = DA.Spec(hist=168, lags=(1, 2, 7), daily=7, cal="time")
    X, y, base, t, day = DA.design(E, spec, first_day="2023-02-01", last_day=G.DAYS[-1])
    p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)
    variants = {"GBDT log, squared error (current)": ("log", dict(loss="squared_error")),
                "GBDT log, quantile 2/3 (peak-aware)": ("log", dict(loss="quantile", quantile=2 / 3)),
                "GBDT linear, squared error (mean)": ("lin", dict(loss="squared_error")),
                "GBDT log-normal, mean of the fitted distribution": ("lognormal", dict(loss="squared_error"))}
    cuts = list(G.DAYS[::28]) + [G.DAYS[-1] + pd.Timedelta(days=1)]
    out = {}
    for name, (space, kw) in variants.items():
        print(name, flush=True)
        pred = np.full(len(y), np.nan)
        target = np.exp(y) if space == "lin" else y
        for a, b in zip(cuts[:-1], cuts[1:]):
            tr = (day < a) & (t < a)
            te = (day >= a) & (day < b)
            m = HistGradientBoostingRegressor(random_state=0, **p, **kw).fit(X[tr], target[tr])
            q = m.predict(X[te])
            if space == "lognormal":
                # sigma^2: out-of-fold squared residuals (two halves of the training days), then a
                # GBDT on them; the forecast is the log-normal mean exp(mu + sigma^2 / 2)
                idx = np.flatnonzero(tr)
                half = day[idx] < day[idx][len(idx) // 2]
                res = np.empty(len(idx))
                for fit_m, pred_m in ((half, ~half), (~half, half)):
                    mm = HistGradientBoostingRegressor(random_state=0, **p, **kw).fit(X.iloc[idx[fit_m]], y[idx[fit_m]])
                    res[pred_m] = y[idx[pred_m]] - mm.predict(X.iloc[idx[pred_m]])
                vm = HistGradientBoostingRegressor(random_state=0, **p, loss="squared_error").fit(X.iloc[idx], res ** 2)
                s2 = np.maximum(vm.predict(X[te]), 0)
                pred[te] = np.exp(q + s2 / 2 + base[te])
            elif space == "log":
                pred[te] = np.exp(q + base[te])
            else:
                pred[te] = np.maximum(q, 0) * np.exp(base[te])
        keep = (day >= G.DAYS[0]) & (day <= G.DAYS[-1])
        out[name] = pd.Series(pred[keep], index=t[keep])
    return out


def main():
    E = G.switch_emissions()
    data_extremes(E)
    df = pd.read_csv(G.CAND, index_col=0, parse_dates=True)
    influence(df)
    F = gbdt_targets(E)
    rows = []
    for k, f in F.items():
        f = f.reindex(df.index)
        for nm, ff in ((k, f), (k + " + bias corr. 7 d", G.bias_corrected(f, df["actual"], 7))):
            for per, days in (("DEV", G.DEV), ("TEST", G.TEST)):
                rows.append(dict(forecast=nm, period=per, **G.scores(ff, df["actual"], days)))
    S = pd.DataFrame(rows)
    S.to_csv(G.OUT / "grid_losses.csv", index=False)
    cols = ["bias %", "mean |month bias| %", "max |month bias| %", "day %", "30-day %"]
    dev, test = S[S.period == "DEV"].set_index("forecast")[cols], S[S.period == "TEST"].set_index("forecast")[cols]
    pd.set_option("display.width", 250)
    print(dev.join(test, lsuffix=" DEV", rsuffix=" TEST").round(1).to_string())


if __name__ == "__main__":
    main()
