"""How biased are the day-ahead forecasts behind the flat-rate bill?

For each network (test blocks as in rate_timestral.py) and each forecast of the two inputs of the
rate, traffic T and emissions E (= power x grid factor):
  bias      (sum forecast - sum actual) / sum actual over the test block, 95 % interval from a
            bootstrap over test days; the same per calendar month
  spikes    bias on the 10 % of days with the highest actual value vs the other days
  calibration (Timestral): share of hours with actual <= the forecast 10 / 50 / 90 / 95 / 99 % quantile
            (ideal: the level), and inside the 10-90 % band (ideal 80 %); plus the 99 % quantile over the
            actual mean (how heavy the forecast upper tail is)
Forecasts: Timestral-209M zero-shot, 24 weeks of hourly context, median and mean (from its 21
quantiles); the grid rule "same hour yesterday"; on Switch LAN also the flat-forecast recipe's
final traffic ensemble.

usage: python bias.py        writes <ARTIFACTS>/flat_forecast/bias.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import rate_timestral as RT
from green_routing.carbon.dayahead import _quantile_mean_weights
from green_routing.paths import ARTIFACTS_DIR

OUT = ARTIFACTS_DIR / "flat_forecast"


def timestral_quantiles(s: pd.Series, days, context: int = 4032) -> tuple[pd.DataFrame, tuple]:
    """[hours x 21] quantile forecasts issued 00:00 UTC of each day for its 24 hours."""
    from timestral import TimestralForecaster
    fm = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda")
    s = s.dropna()
    v = s.to_numpy(np.float32)
    starts = [s.index.get_loc(d) for d in days]
    X = np.full((len(starts), 1, context), np.nan, np.float32)
    for r, i in enumerate(starts):
        c = v[max(0, i - context):i]
        X[r, 0, context - len(c):] = c
    qs = []
    for b in range(0, len(X), 32):
        with torch.inference_mode():
            f = fm.forecast(X[b:b + 32], prediction_length=24)
        qs.append(f.quantiles[:, 0].float().cpu().numpy())               # [B, 24, Q]
    Q = np.concatenate(qs).reshape(-1, len(f.quantile_levels))
    idx = pd.DatetimeIndex(np.concatenate([pd.date_range(d, periods=24, freq="h") for d in days]))
    return pd.DataFrame(Q, index=idx, columns=f.quantile_levels), f.quantile_levels


def bias_stats(f: pd.Series, a: pd.Series, rng) -> dict:
    day = a.index.floor("D")
    fd, ad = f.groupby(day).sum(), a.groupby(day).sum()
    b = float((fd.sum() - ad.sum()) / ad.sum())
    boots = []
    n = len(ad)
    for _ in range(2000):
        k = rng.integers(0, n, n)
        boots.append((fd.iloc[k].sum() - ad.iloc[k].sum()) / ad.iloc[k].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    top = ad >= ad.quantile(0.9)
    month = pd.Series(ad.index.tz_localize(None).to_period("M"), index=ad.index)
    by_m = {str(m): round(100 * float((fd[month == m].sum() - ad[month == m].sum()) / ad[month == m].sum()), 1)
            for m in month.unique()}
    return {"bias %": 100 * b, "95 % low": 100 * lo, "95 % high": 100 * hi,
            "bias on top-10 % days": 100 * float((fd[top].sum() - ad[top].sum()) / ad[top].sum()),
            "bias on other days": 100 * float((fd[~top].sum() - ad[~top].sum()) / ad[~top].sum()),
            "hourly WAPE %": 100 * float((f - a).abs().sum() / a.sum()), "by month": by_m}


def calibration(Q: pd.DataFrame, a: pd.Series) -> dict:
    q = Q.reindex(a.index)
    return {f"actual <= q{int(100 * l)}": 100 * float((a <= q[l]).mean()) for l in (0.1, 0.5, 0.9, 0.95, 0.99)} | \
           {"inside q10-q90": 100 * float(((a >= q[0.1]) & (a <= q[0.9])).mean()),
            "q99 / actual, mean": float((q[0.99] / a).mean()), "q90 / actual, mean": float((q[0.9] / a).mean())}


def main():
    rng = np.random.default_rng(0)
    rows = []
    for name, loader in (("Switch LAN", RT.switch), ("I2 Network", RT.i2)):
        D = loader()
        E, T, Tin, days = D["E"], D["T"], D["Tin"], D["days"]
        QT, lv = timestral_quantiles(Tin, days)
        QE, _ = timestral_quantiles(E, days)
        w = _quantile_mean_weights(lv)
        idx = QT.index.intersection(T.dropna().index)
        idx = idx[np.isfinite(E.reindex(idx)) & np.isfinite(D["Erule"].reindex(idx))]
        t, e = T.reindex(idx), E.reindex(idx)
        fc = {("traffic", "Timestral, median"): (QT[0.5].reindex(idx), t),
              ("traffic", "Timestral, mean"): ((QT * w).sum(1).reindex(idx), t),
              ("emissions (grid)", "same hour yesterday"): (D["Erule"].reindex(idx), e),
              ("emissions (grid)", "Timestral, median"): (QE[0.5].reindex(idx), e),
              ("emissions (grid)", "Timestral, mean"): ((QE * w).sum(1).reindex(idx), e)}
        if name == "Switch LAN":
            import run as R
            fc[("traffic", "recipe's final ensemble")] = (R.cache()[RT_SWITCH_FINAL].reindex(idx), t)
        for (what, m), (f, a) in fc.items():
            r = dict(network=name, series=what, forecast=m, **bias_stats(f, a, rng))
            if m.startswith("Timestral"):
                r |= calibration(QT if what == "traffic" else QE, a)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bias.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 80)
    print(df.drop(columns="by month").round(1).to_string(index=False))
    print(df[["network", "series", "forecast", "by month"]].to_string(index=False))


RT_SWITCH_FINAL = "ensemble: 0.8 x Chronos-2 + 0.2 x GBDT, GBDT alone on holidays"

if __name__ == "__main__":
    main()
