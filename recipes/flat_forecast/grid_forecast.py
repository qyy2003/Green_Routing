"""An unbiased day-ahead forecast of the grid (Switch LAN's backbone emissions = power x grid factor).

Candidates from the codebase, all issued 00:00 UTC for the next 24 hours:
  rules       carbon.grid.RULES (same hour yesterday, mean same hour last 3 / 7 days, last hour
              flat, 7-day profile rescaled to yesterday)
  pretrained  carbon.dayahead.foundation: Timestral-209M (24 weeks; median, mean, other quantiles),
              Chronos-2 (median), TimesFM-2.5 (its point forecast), zero-shot
  trained     carbon.dayahead.backtest: GBDT and ridge on the log ratio to the last week's mean
              (squared error = a geometric-mean target; absolute error = median), refitted every
              28 days on everything before
  + bias      each day's forecast x (actual / forecast) over the previous N days (N = 7, 28), as
              flat_forecast/run.py bias_corrected
  + blends    equal-weight averages of two candidates
Chosen on DEV (2025-03-22 -> 2026-03-21, a full year), reported on TEST (2026-03-22 -> 07-10).
Scores on the emissions: bias over the period, |bias| per calendar month (mean and max), WAPE of
hourly values and of the sums over each day, every 7-day and every 30-day window.

usage: python grid_forecast.py candidates     forecasts -> <ARTIFACTS>/flat_forecast/grid_candidates.csv.gz
       python grid_forecast.py score          scores    -> grid_scores.csv
       python grid_forecast.py bills NAME ... TEST bills with Timestral traffic (Switch LAN)
       python grid_forecast.py i2             the same corrections on I2's test block
       python grid_forecast.py period A B     rule and Timestral (+ correction) on another period A..B
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
DEV = pd.date_range("2025-03-22", "2026-03-21", freq="D", tz="UTC")
TEST = pd.date_range("2026-03-22", "2026-07-10", freq="D", tz="UTC")
DAYS = DEV.append(TEST)
CAND = OUT / "grid_candidates.csv.gz"


def switch_emissions(end=None) -> pd.Series:
    """Hourly backbone emissions (g/h) 2023-01 -> 2026-07-10 (or ``end``) under the actual grid factors."""
    from green_routing.carbon import bills, grid
    from green_routing.paths import reference_dir
    C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
    last = pd.Timestamp(end, tz="UTC") if end is not None else TEST[-1]
    H = pd.date_range(pd.Timestamp("2023-01-01", tz="UTC"), last + pd.Timedelta(hours=23), freq="h")
    # the inventory has power from 2024 on; backbone power is static, so it is taken over the
    # window of rate_timestral.py (2025-07 -> test end) and applied to the whole history
    Hp = pd.date_range(pd.Timestamp("2025-07-01", tz="UTC"), TEST[-1] + pd.Timedelta(hours=23), freq="h")
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    tiers = pd.read_csv(NET / "routers.csv").set_index("dev").tier
    _, Pbb, _ = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"),
                                   pd.read_csv(C / "allocation" / "device_pools.csv"),
                                   pd.read_csv(C / "inventory" / "shelves.csv"),
                                   pd.read_csv(reference_dir() / "sites.csv"), doors, tiers, Hp)
    for z in Pbb:
        v = np.asarray(Pbb[z], float)
        assert np.nanstd(v) <= 1e-6 * max(np.nanmean(v), 1.0), f"backbone power of {z} varies"
    IA = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in Pbb})
    return sum(float(np.nanmean(np.asarray(Pbb[z], float))) / 1000 * IA[z] for z in Pbb)


def rule_forecasts(E: pd.Series) -> dict:
    from green_routing.carbon import grid
    return {f"rule: {k}": f(E.asfreq("h").interpolate()) for k, f in grid.RULES.items()}


def timestral_forecasts(E: pd.Series) -> dict:
    sys.path.insert(0, str(HERE))
    from bias import timestral_quantiles
    Q, lv = timestral_quantiles(E, DAYS)
    w = DA._quantile_mean_weights(lv)
    out = {"Timestral: median": Q[0.5], "Timestral: mean": (Q * w).sum(1)}
    for l in (0.55, 0.6, 0.65, 0.7):
        out[f"Timestral: q{int(round(100 * l))}"] = Q[l]
    return out


def trained_forecasts(E: pd.Series) -> dict:
    base = DA.Spec(hist=168, lags=(1, 2, 7), daily=7, cal="time")
    out = {}
    for name, spec in (("GBDT (log, squared error)", base),
                       ("GBDT (log, absolute error)", base.but(loss="absolute_error")),
                       ("ridge (log)", base.but(model="ridge"))):
        print(name, flush=True)
        out[name] = DA.backtest(E, E, spec, DAYS, refit_every=28, train_start="2023-02-01")
    return out


def bias_corrected(f: pd.Series, a: pd.Series, n: int) -> pd.Series:
    """Each day's forecast x sum(actual) / sum(forecast) over the previous n days."""
    d = f.index.normalize()
    ok = a.reindex(f.index).notna() & f.notna()
    ys = a.reindex(f.index).where(ok).groupby(d).sum()
    fs = f.where(ok).groupby(d).sum()
    r = (ys.rolling(n, min_periods=max(3, n // 2)).sum() / fs.rolling(n, min_periods=max(3, n // 2)).sum()).shift(1)
    return f * r.reindex(d).fillna(1.0).to_numpy()


def candidates():
    E = switch_emissions()
    idx = pd.DatetimeIndex(np.concatenate([pd.date_range(d, periods=24, freq="h") for d in DAYS]))
    F = {k: v.reindex(idx) for k, v in rule_forecasts(E).items()}
    F |= {k: v.reindex(idx) for k, v in timestral_forecasts(E).items()}
    for model, ctx in (("chronos-2", 2048), ("timesfm-2.5", 4032)):
        print(model, flush=True)
        F[f"{model}"] = DA.foundation(E, DAYS, model=model, context=ctx).reindex(idx)
    F |= {k: v.reindex(idx) for k, v in trained_forecasts(E).items()}
    df = pd.DataFrame(F)
    df.insert(0, "actual", E.reindex(idx))
    df.to_csv(CAND)
    print(f"{df.shape[1] - 1} candidates, {len(df)} hours -> {CAND}")


def with_variants(df: pd.DataFrame) -> dict:
    a = df["actual"]
    base = {k: df[k] for k in df.columns if k != "actual"}
    F = dict(base)
    for k, f in base.items():
        for n in (7, 28):
            F[f"{k} + bias corr. {n} d"] = bias_corrected(f, a, n)
    pairs = [("rule: yesterday same hour", "Timestral: median"), ("Timestral: median", "Timestral: mean"),
             ("rule: mean same hour last 7 d", "Timestral: median"), ("GBDT (log, squared error)", "Timestral: median")]
    for x, y in pairs:
        F[f"blend: {x} + {y}"] = (base[x] + base[y]) / 2
    return F


def scores(f: pd.Series, a: pd.Series, days) -> dict:
    hrs = f.index.normalize().isin(days) & f.notna() & a.notna()
    f, a = f[hrs], a[hrs]
    d = pd.DataFrame({"f": f.groupby(f.index.normalize()).sum(), "a": a.groupby(a.index.normalize()).sum()})
    win = lambda k: float((d.f.rolling(k).sum() - d.a.rolling(k).sum()).abs().dropna().sum()   # noqa: E731
                          / d.a.rolling(k).sum().dropna().sum())
    m = d.index.tz_localize(None).to_period("M")
    full = [p for p in m.unique() if (m == p).sum() == p.days_in_month]
    mb = [100 * (d.f[m == p].sum() - d.a[m == p].sum()) / d.a[m == p].sum() for p in full]
    return {"bias %": 100 * float((f.sum() - a.sum()) / a.sum()),
            "mean |month bias| %": float(np.mean(np.abs(mb))) if mb else np.nan,
            "max |month bias| %": float(np.max(np.abs(mb))) if mb else np.nan,
            "hourly %": 100 * float((f - a).abs().sum() / a.sum()), "day %": 100 * win(1),
            "7-day %": 100 * win(7), "30-day %": 100 * win(30), "months": len(mb)}


def score():
    df = pd.read_csv(CAND, index_col=0, parse_dates=True)
    F = with_variants(df)
    rows = []
    for k, f in F.items():
        for per, days in (("DEV", DEV), ("TEST", TEST)):
            rows.append(dict(forecast=k, period=per, **scores(f, df["actual"], days)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "grid_scores.csv", index=False)
    dev = S[S.period == "DEV"].set_index("forecast")
    test = S[S.period == "TEST"].set_index("forecast")
    order = dev.sort_values("30-day %").index
    cols = ["bias %", "mean |month bias| %", "max |month bias| %", "day %", "30-day %"]
    T = dev.loc[order, cols].join(test.loc[order, cols], lsuffix=" DEV", rsuffix=" TEST")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 200)
    print(T.round(1).to_string())


def bills(names: list[str]):
    """TEST bills (every-window WAPE, bias) with Timestral traffic for chosen grid forecasts."""
    sys.path.insert(0, str(HERE))
    import rate_timestral as RT
    df = pd.read_csv(CAND, index_col=0, parse_dates=True)
    F = with_variants(df)
    D = RT.switch()
    Tf = RT.ts(D["Tin"], D["days"])
    idx = Tf.index.intersection(D["T"].dropna().index)
    e, t, tf = D["E"].reindex(idx), D["T"].reindex(idx), Tf.reindex(idx)
    assert np.allclose(e, df["actual"].reindex(idx), rtol=1e-6)
    rows = []
    for k in names:
        ef = F[k].reindex(idx)
        rf = ef / (tf * D["GB"])
        rows.append(dict(forecast=k, **RT.errors(e, t, rf * t * D["GB"], e / (t * D["GB"]), rf)))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def i2check():
    """I2 (test block only: its emissions start 2025-11-15): the same corrections, TEST scores
    and bills with Timestral traffic."""
    sys.path.insert(0, str(HERE))
    import rate_timestral as RT
    from bias import timestral_quantiles
    D = RT.i2()
    E, days = D["E"], D["days"]
    Q, lv = timestral_quantiles(E, days)
    w = DA._quantile_mean_weights(lv)
    # 7 days of forecasts before the test block feed the first corrections
    pre = pd.date_range(days[0] - pd.Timedelta(days=7), days[0] - pd.Timedelta(days=1), freq="D")
    Qp, _ = timestral_quantiles(E, pre)
    med, mean = pd.concat([Qp[0.5], Q[0.5]]), pd.concat([(Qp * w).sum(1), (Q * w).sum(1)])
    F = {"rule: yesterday same hour": D["Erule"], "Timestral: median": med, "Timestral: mean": mean,
         "Timestral: median + bias corr. 7 d": bias_corrected(med, E, 7),
         "Timestral: mean + bias corr. 7 d": bias_corrected(mean, E, 7)}
    Tf = RT.ts(D["Tin"], days)
    idx = Tf.index.intersection(D["T"].dropna().index)
    e, t, tf = E.reindex(idx), D["T"].reindex(idx), Tf.reindex(idx)
    rows = []
    for k, f in F.items():
        ef = f.reindex(idx)
        rf = ef / (tf * D["GB"])
        rows.append(dict(forecast=k, **{f"grid {a}": b for a, b in scores(ef, e, days).items()
                                        if a in ("bias %", "max |month bias| %", "day %", "30-day %")},
                         **{f"bill {a}": b for a, b in RT.errors(e, t, rf * t * D["GB"], e / (t * D["GB"]), rf).items()}))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def period(start: str, end: str):
    """Grid forecasts on another held-out period (no traffic needed): the rule and Timestral
    (median, mean), plain and with the 7-day correction; scores as for DEV / TEST."""
    sys.path.insert(0, str(HERE))
    from bias import timestral_quantiles
    days = pd.date_range(start, end, freq="D", tz="UTC")
    E = switch_emissions(end)
    pre = pd.date_range(days[0] - pd.Timedelta(days=7), days[0] - pd.Timedelta(days=1), freq="D")
    Q, lv = timestral_quantiles(E, pre.append(days))
    w = DA._quantile_mean_weights(lv)
    from green_routing.carbon import grid
    base = {"rule: yesterday same hour": grid.RULES["yesterday same hour"](E.asfreq("h").interpolate()),
            "Timestral: median": Q[0.5], "Timestral: mean": (Q * w).sum(1)}
    F = dict(base)
    for k in ("rule: yesterday same hour", "Timestral: median", "Timestral: mean"):
        F[f"{k} + bias corr. 7 d"] = bias_corrected(base[k].reindex(Q.index), E, 7)
    rows = [dict(forecast=k, **scores(f, E, days)) for k, f in F.items()]
    S = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"{days[0].date()} -> {days[-1].date()} ({len(days)} days)")
    print(S.round(1).to_string(index=False))
    a = E.reindex(Q.index)
    m = a.index.tz_localize(None).to_period("M")
    mb = pd.DataFrame({k: ((F[k].reindex(a.index).groupby(m).sum() / a.groupby(m).sum() - 1) * 100).round(1)
                       for k in ("rule: yesterday same hour", "Timestral: median", "Timestral: median + bias corr. 7 d")})
    print(mb[mb.index >= pd.Period(start[:7], "M")].to_string())


if __name__ == "__main__":
    {"candidates": lambda: candidates(), "score": lambda: score(),
     "bills": lambda: bills(sys.argv[2:]), "i2": lambda: i2check(),
     "period": lambda: period(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
