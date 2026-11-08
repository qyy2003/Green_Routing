"""Where the bias of the day-ahead grid forecasts comes from (Switch LAN, DEV + TEST hours of
grid_forecast.py; I2 test block for contrast).

  1. median vs mean: share of hours with actual > forecast, mean and median of actual / forecast,
     and the average size of under- and over-forecasts (% of the mean actual)
  2. log-space training: the bias a log-normal error of spread sigma implies, 1 - exp(-sigma^2/2),
     against the observed bias
  3. "same hour yesterday": the bias of a period telescopes to (E of the day before - E of the
     last day) / E of the period; checked month by month

usage: python bias_sources.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid_forecast as G       # noqa: E402


def shape(f: pd.Series, a: pd.Series) -> dict:
    ok = f.notna() & a.notna() & (f > 0)
    f, a = f[ok], a[ok]
    r = a / f
    lr = np.log(np.maximum(a, 1e-9) / f)
    sig = float(lr.std())
    return {"bias %": 100 * (f.sum() / a.sum() - 1), "actual > forecast %": 100 * float((a > f).mean()),
            "median actual/fc": float(r.median()), "mean actual/fc": float(r.mean()),
            "under-forecasts, % of mean": 100 * float((a - f).clip(lower=0).mean() / a.mean()),
            "over-forecasts, % of mean": 100 * float((f - a).clip(lower=0).mean() / a.mean()),
            "sigma of log error": sig, "log-normal bias %": -100 * (1 - np.exp(-sig ** 2 / 2)),
            "mean log error": float(lr.mean())}


def main():
    df = pd.read_csv(G.CAND, index_col=0, parse_dates=True)
    a = df["actual"]
    rows = []
    for k in ("Timestral: median", "Timestral: mean", "chronos-2", "GBDT (log, absolute error)",
              "GBDT (log, squared error)", "ridge (log)", "rule: yesterday same hour"):
        rows.append(dict(network="Switch LAN", forecast=k, **shape(df[k], a)))
    # I2 for contrast: Timestral median on its test block
    import rate_timestral as RT
    from bias import timestral_quantiles
    D = RT.i2()
    Q, _ = timestral_quantiles(D["E"], D["days"])
    rows.append(dict(network="I2 Network", forecast="Timestral: median", **shape(Q[0.5], D["E"].reindex(Q.index))))
    rows.append(dict(network="I2 Network", forecast="rule: yesterday same hour",
                     **shape(D["Erule"].reindex(Q.index), D["E"].reindex(Q.index))))
    S = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))
    S.to_csv(G.OUT / "bias_sources.csv", index=False)

    # 3. the rule telescopes: monthly bias = (E(day before) - E(last day)) / E(month)
    full = G.switch_emissions()
    d = full.groupby(full.index.normalize()).sum()
    rule_d = df["rule: yesterday same hour"].groupby(df.index.normalize()).sum()
    act_d = a.groupby(a.index.normalize()).sum()
    m = act_d.index.tz_localize(None).to_period("M")
    out = []
    for p in m.unique():
        days = act_d.index[m == p]
        if len(days) < p.days_in_month:
            continue
        obs = (rule_d[days].sum() - act_d[days].sum()) / act_d[days].sum()
        pred = (d[days[0] - pd.Timedelta(days=1)] - d[days[-1]]) / act_d[days].sum()
        out.append(dict(month=str(p), observed=100 * obs, telescoped=100 * pred,
                        **{"E day before / mean day": d[days[0] - pd.Timedelta(days=1)] / act_d[days].mean(),
                           "E last day / mean day": d[days[-1]] / act_d[days].mean()}))
    print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
