"""Experiments: how to forecast SWITCHlan's total traffic a day ahead, hour by hour (the
denominator of the flat carbon rate). Every forecast is issued at 00:00 UTC for the 24
hours of the day and scored on the same hours.

usage: python run.py [stage ...]      default: every stage; finished forecasts are cached
writes $GR_ARTIFACTS_DIR/flat_forecast/: forecasts.csv.gz (every forecast, hourly, GBit/s),
       scores_dev.csv, scores_test.csv, paired.csv, summary.json

Periods: DEV 2025-11-01 .. 2026-03-21 (choices are made here), TEST 2026-03-22 .. 07-10 (the
benchmark's test block, 111 days). Hourly models refit weekly on everything known so far
unless stated; the 5-minute ladder keeps the benchmark's fixed training block.

stages
  rules       same hour last week, mean and median of the last 4 weeks
  sweep       direct GBDT / ridge: features, prior knowledge, loss, weights
  protocol    the best and the current GBDT: fixed vs refitted, history length
  foundation  Chronos-2, Chronos-Bolt, TimesFM 2.5 zero-shot (context, covariates, 5-minute input)
  finetune    Chronos-2 fine-tuned on the series before each period
  ladder      the 5-minute ladder (recipes/flat_forecast/run.sh ladder): loss and seeds
  combine     ensembles and bias correction
  score       tables (DEV, TEST, paired differences, errors by kind of day)
  diagnose    level vs shape, error persistence, unforecastable doors, holiday effects
  rate        what the traffic forecast changes for the day-ahead flat rate
  abilene     the same choices on Abilene's total (2004)
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

from green_routing.carbon import dayahead as DA
from green_routing.paths import ARTIFACTS_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ARTIFACTS_DIR / "flat_forecast"
CACHE = OUT / "forecasts.csv.gz"
DEV = pd.date_range("2025-11-01", "2026-03-21", freq="D", tz="UTC")
TEST = pd.date_range("2026-03-22", "2026-07-10", freq="D", tz="UTC")
ALL = DEV.append(TEST)
HISTORY_START = "2025-02-01"          # the export before this has a gap (Dec 2024 - Jan 2025)
LAGS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)


# --- data ------------------------------------------------------------------------------------

def load_5min() -> pd.Series:
    """Total one-way traffic (Gbit/s, 5 min) from the history extract and the year mirror."""
    m = pd.read_csv(ARTIFACTS_DIR / "carbon_rates" / "mirror" / "router-interfaces" / "total" / "all.csv")
    cur = pd.Series((m.ifHCInOctets + m.ifHCOutOctets).to_numpy() * 8 / 1e9 / 2,
                    index=pd.to_datetime(m.timestamp, unit="s", utc=True))
    h = pd.read_csv(OUT / "history" / "total_2024.csv.gz", index_col=0, parse_dates=True)
    old = ((h.in_mbps + h.out_mbps) / 2 / 1e3).where(h.coverage >= 0.95)
    return pd.concat([old[old.index < cur.index[0]], cur])[HISTORY_START:"2026-07-10 23:55"]


T5 = load_5min()
_g = T5.resample("h")
T = _g.mean().where(_g.count() >= 6)                     # the truth: hourly means, >= 6 of 12 steps
S = T.interpolate().bfill()                              # model inputs: gaps filled


def cache() -> pd.DataFrame:
    return pd.read_csv(CACHE, index_col=0, parse_dates=True) if CACHE.exists() else pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))


def save(new: dict):
    """Add forecasts to the cache (locked: stages may run in parallel processes)."""
    import fcntl
    with open(OUT / ".cache.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        C = cache()
        idx = pd.date_range(ALL[0], ALL[-1] + pd.Timedelta(hours=23), freq="h")
        C = C.reindex(idx)
        for k, v in new.items():
            C[k] = v.reindex(idx)
        C.to_csv(CACHE, float_format="%.5f")
    print(f"  saved {len(new)} forecasts ({C.shape[1]} in cache)", flush=True)


def todo(names) -> list:
    have = set(cache().columns)
    return [n for n in names if n not in have]


# --- stages ----------------------------------------------------------------------------------

def stage_rules():
    R = DA.rules(S)
    save({f"rule: {k}": v for k, v in R.items()})


BASE = DA.Spec()                                                    # the current GBDT (total.gbdt_day_ahead)
COMPACT = DA.Spec(hist=24, lags=LAGS, daily=7)
FULL = COMPACT.but(cal="full", holidays="weighted", clean=True)
SPECS = {
    "gbdt: current (168 h, basic calendar)": BASE,
    "gbdt: compact (24 h + same hour 1-28 d + 7 daily means)": COMPACT,
    "gbdt: compact, weighted holidays": COMPACT.but(holidays="weighted"),
    "gbdt: compact, full calendar": COMPACT.but(cal="full", holidays="weighted"),
    "gbdt: compact, full calendar, cleaned history": FULL,
    "gbdt: full, national holidays only": FULL.but(holidays="national"),
    "gbdt: full, old holiday table": FULL.but(holidays="old"),
    "gbdt: full, no calendar": FULL.but(cal="none"),
    "gbdt: full, L1 loss": FULL.but(loss="absolute_error"),
    "gbdt: full, recency weights (half-life 90 d)": FULL.but(halflife=90),
    "gbdt: full, recency weights (half-life 30 d)": FULL.but(halflife=30),
    "gbdt: full, small trees": FULL.but(params=dict(max_leaf_nodes=7, min_samples_leaf=50, max_iter=300)),
    "gbdt: full, larger model": FULL.but(params=dict(max_leaf_nodes=31, learning_rate=0.03, max_iter=800)),
    "gbdt: full + 168 h": FULL.but(hist=168),
    "ridge: full": FULL.but(model="ridge"),
}


def stage_sweep():
    for name in todo(SPECS):
        print(name, flush=True)
        save({name: DA.backtest(S, T, SPECS[name], ALL, refit_every=7)})


def best_gbdt(prefix="gbdt: ") -> str:
    sc = scores(cache(), DEV)
    sc = sc[sc.forecast.str.startswith(prefix) & ~sc.forecast.str.contains("@")]
    return sc.iloc[0].forecast


def stage_protocol():
    best = best_gbdt()
    print("best GBDT on DEV:", best, flush=True)
    for label, name in (("current", "gbdt: current (168 h, basic calendar)"), ("best", best)):
        spec = SPECS[name]
        runs = {
            f"{name} @ fixed, trained to 2026-02-15": dict(refit_every=None, train_end="2026-02-15"),
            f"{name} @ refit every 28 d": dict(refit_every=28),
            f"{name} @ refit daily": dict(refit_every=1),
            f"{name} @ history from 2025-07": dict(refit_every=7, train_start="2025-07-01"),
        }
        for k in todo(runs):
            print(k, flush=True)
            kw = runs[k]
            days = TEST if "fixed" in k else ALL
            s = S["2025-07-01":] if "history from 2025-07" in k else S
            save({k: DA.backtest(s, T, spec, days, **{k2: v for k2, v in kw.items() if k2 != "train_start"})})
        if label == "best":
            k = f"{name} @ 5 seeds"
            if todo([k]):
                save({k: DA.backtest(S, T, spec.but(params={**spec.params, "max_features": 0.8}), ALL, refit_every=7,
                                     seeds=range(5))})


FOUNDATION = {
    "chronos-2: 4 weeks of context": dict(model="chronos-2", context=672),
    "chronos-2: 12 weeks": dict(model="chronos-2", context=2016),
    "chronos-2: 24 weeks": dict(model="chronos-2", context=4032),
    "chronos-2: 48 weeks": dict(model="chronos-2", context=8064),
    "chronos-2: 12 weeks + calendar covariates": dict(model="chronos-2", context=2016, covariates=True),
    "chronos-2: 24 weeks + calendar covariates": dict(model="chronos-2", context=4032, covariates=True),
    "chronos-2: 24 weeks, cleaned history": dict(model="chronos-2", context=4032, clean=True),
    "chronos-2: 24 weeks + calendar covariates, cleaned history": dict(model="chronos-2", context=4032, covariates=True, clean=True),
    "chronos-2: 5-minute input, 4 weeks": dict(model="chronos-2", context=8064, step="5min"),
    "chronos-bolt: 12 weeks": dict(model="chronos-bolt", context=2016),
    "timesfm-2.5: 12 weeks": dict(model="timesfm-2.5", context=2016),
    "timesfm-2.5: 48 weeks": dict(model="timesfm-2.5", context=8064),
    # Mistral's Timestral (2026-10-15; 209M unless named), added 2026-11-03
    "timestral: 12 weeks": dict(model="timestral", context=2016),
    "timestral: 48 weeks": dict(model="timestral", context=8064),
    "timestral: 24 weeks + calendar covariates": dict(model="timestral", context=4032, covariates=True),
    "timestral: 5-minute input, 4 weeks": dict(model="timestral", context=8064, step="5min"),
    "timestral: 5-minute input, 16 weeks": dict(model="timestral", context=32256, step="5min"),
    "timestral-28M: 12 weeks": dict(model="timestral-28M", context=2016),
}                       # (TimesFM's covariates need timesfm[xreg], i.e. JAX with its own CUDA: not tested)


def stage_foundation():
    s5 = T5.interpolate(limit_area="inside").bfill().ffill()
    s5 = s5.reindex(pd.date_range(s5.index[0], pd.Timestamp("2026-07-10 23:55", tz="UTC"), freq="5min")).interpolate().bfill()
    for name in todo(FOUNDATION):
        print(name, flush=True)
        kw = dict(FOUNDATION[name])
        src = DA.cleaned(S, "weighted") if kw.pop("clean", False) else S
        save({name: DA.foundation(s5 if kw.get("step") == "5min" else src, ALL, **kw)})


def stage_finetune():
    """Chronos-2 fine-tuned (all weights) on the series before each period, then forecasting it."""
    for name, steps in (("chronos-2 fine-tuned: 12 weeks, 1000 steps", 1000),):
        if not todo([name]):
            continue
        parts = []
        for days in (DEV, TEST):
            parts.append(DA.foundation(S, days, "chronos-2", 2016,
                                       finetune=dict(cutoff=days[0], steps=steps, out=OUT / "chronos_finetuned")))
        save({name: pd.concat(parts)})


LADDER = OUT / "ladder"
LADDER_MODELS = ("tcn", "patchtst", "gru", "dlinear")


def ladder_hourly(run, model) -> pd.Series | None:
    f = run / "dense" / f"pred_block_{model}.npy"
    if not f.exists():
        return None
    meta = json.load(open(run / "dense" / "meta.json"))
    hi = [h["name"] for h in meta["horizons"]].index("day")
    n = meta["n_targets"] // 288 * 288
    p = np.load(f)[0, hi, :n].astype(float) / 2 / 1e3              # Mbit/s (in + out) -> one-way Gbit/s
    t0 = pd.Timestamp(meta["t0"], unit="s", tz="UTC")
    return pd.Series(p.reshape(-1, 12).mean(1), index=pd.date_range(t0, periods=n // 12, freq="h"))


def stage_ladder():
    new = {}
    orig = ARTIFACTS_DIR / "carbon_rates" / "flat" / "merged"
    for m in LADDER_MODELS:
        f = ladder_hourly(orig, m)
        if f is not None:
            new[f"5-min {m}: original run (peak-aware loss)"] = f
        for uw, lab in ((1, "symmetric loss"), (2, "peak-aware loss")):
            seeds = [ladder_hourly(LADDER / f"{m}_uw{uw}_s{s}", m) for s in range(3)]
            seeds = [x for x in seeds if x is not None]
            for s, x in enumerate(seeds):
                new[f"5-min {m}: {lab} @ seed {s}"] = x
            if len(seeds) == 3:
                new[f"5-min {m}: {lab}, mean of 3 seeds"] = sum(seeds) / 3
    if new:
        save(new)


def bias_corrected(f: pd.Series, days: int) -> pd.Series:
    """Multiply each day's forecast by actual / forecast over the previous ``days`` days
    (only days that are over by the issue time)."""
    d = f.index.normalize()
    ok = T.reindex(f.index).notna() & f.notna()
    ys = T.reindex(f.index).where(ok).groupby(d).sum()
    fs = f.where(ok).groupby(d).sum()
    r = (ys.rolling(days, min_periods=max(3, days // 2)).sum() / fs.rolling(days, min_periods=max(3, days // 2)).sum()).shift(1)
    return f * r.reindex(d).fillna(1.0).to_numpy()


def off_days(idx: pd.DatetimeIndex) -> np.ndarray:
    """Hours on a holiday (weight >= 0.5) or in the Christmas closure, local time."""
    d = DA._local_days(pd.DatetimeIndex(idx))
    return (DA._holiday_weights(d, "weighted") >= 0.5) | (DA.christmas(d) > 0)


def stage_combine():
    """Ensembles and bias correction. Members and weights are chosen on DEV; the 5-minute
    ladder has no DEV forecasts, so ensembles with it use equal weights."""
    C = cache()
    sd = scores(C, DEV)
    plain = sd[~sd.forecast.str.contains("@|ensemble")]
    best_g = best_gbdt()
    fm = plain[plain.forecast.str.startswith("chronos-2")].iloc[0].forecast
    tf = plain[plain.forecast.str.startswith("timesfm")].iloc[0].forecast
    new = {}
    for k in (best_g, fm):
        for n in (7, 14, 28):
            new[f"{k} @ bias-corrected, {n} d"] = bias_corrected(C[k], n)
    ctx = [f"chronos-2: {w} weeks" for w in (12, 24, 48)]
    new["chronos-2: mean of 12, 24 and 48 weeks of context"] = sum(C[k] for k in ctx) / 3
    new["ensemble: GBDT + Chronos-2"] = (C[best_g] + C[fm]) / 2
    new["ensemble: GBDT + Chronos-2 + TimesFM"] = (C[best_g] + C[fm] + C[tf]) / 3
    new["ensemble: median of GBDT, Chronos-2, TimesFM"] = pd.concat([C[best_g], C[fm], C[tf]], axis=1).median(axis=1)
    # the weight of Chronos-2 against the GBDT, chosen on DEV
    idx = DA.common_hours({"a": C[best_g], "b": C[fm]}, T, DEV)
    yy = T.reindex(idx)
    wapes = {w: float((w * C[fm].reindex(idx) + (1 - w) * C[best_g].reindex(idx) - yy).abs().sum() / yy.sum())
             for w in np.round(np.arange(0, 1.01, 0.1), 1)}
    w = max(wapes, key=lambda k: -wapes[k])
    new[f"ensemble: {w:.1f} x Chronos-2 + {1 - w:.1f} x GBDT (weight from DEV)"] = w * C[fm] + (1 - w) * C[best_g]
    cov = "chronos-2: 24 weeks + calendar covariates"
    new["ensemble: GBDT + Chronos-2 with calendar"] = (C[best_g] + C[cov]) / 2
    # prior knowledge in the combination: on holidays and in the Christmas closure, only the
    # model that knows the calendar (Chronos-2 has no holiday table)
    off = off_days(C.index)
    blend = w * C[fm] + (1 - w) * C[best_g]
    new[f"ensemble: {w:.1f} x Chronos-2 + {1 - w:.1f} x GBDT, GBDT alone on holidays"] = blend.where(~off, C[best_g])
    tcn = "5-min tcn: symmetric loss, mean of 3 seeds"
    if tcn in C:
        new["ensemble: GBDT + Chronos-2 + TCN"] = (C[best_g] + C[fm] + C[tcn]) / 3
        new["ensemble: GBDT + Chronos-2 + TimesFM + TCN"] = (C[best_g] + C[fm] + C[tf] + C[tcn]) / 4
        new["ensemble: GBDT + Chronos-2 + TCN, GBDT alone on holidays"] = ((C[best_g] + C[fm] + C[tcn]) / 3).where(~off, C[best_g])
    save(new)
    json.dump({"best_gbdt": best_g, "best_chronos": fm, "best_timesfm": tf, "chronos_weight_dev": w,
               "weight_wapes_dev": wapes}, open(OUT / "choices.json", "w"), indent=1)


def stage_diagnose():
    """Where the error is: daily level vs shape within the day, persistence of errors, the
    doors whose traffic nobody can forecast, the effect of holidays on the traffic."""
    from green_routing.carbon import total
    C = cache()
    st = pd.read_csv(OUT / "scores_test.csv")
    idx = DA.common_hours({k: C[k] for k in st.forecast}, T, TEST)
    y = T.reindex(idx)
    d = idx.normalize()
    out = {"forecasts": {}}
    for k in st.forecast[:12].tolist() + ["gbdt: current (168 h, basic calendar)", "5-min tcn: original run (peak-aware loss)"]:
        if k not in C or k in out["forecasts"]:
            continue
        f = C[k].reindex(idx)
        e = f - y
        yd, fd = y.groupby(d).transform("mean"), f.groupby(d).transform("mean")
        bias_d = e.groupby(d).sum() / y.groupby(d).sum()
        ae = e.abs().groupby(d).sum()
        out["forecasts"][k] = dict(wape=float(e.abs().sum() / y.sum()),
                                   perfect_daily_level=float((f / fd * yd - y).abs().sum() / y.sum()),
                                   level_only=float((fd / yd * y - y).abs().sum() / y.sum()),
                                   worst_tenth_days_share=float(ae.nlargest(len(ae) // 10).sum() / ae.sum()),
                                   daily_bias_autocorr=float(bias_d.autocorr(1)))
    # unforecastable doors: share of the variance of the total's same-hour-last-week error
    tr = dict(np.load(ARTIFACTS_DIR / "network" / "traffic.npz"))
    doors = pd.read_csv(ARTIFACTS_DIR / "carbon" / "allocation" / "doors.csv")
    _, din, dout, _ = total.total_series(tr, doors)
    V = ((din + dout) / 2 / 1e3).reindex(idx)
    err = V.shift(168) - V
    et = err.sum(axis=1)
    share = err.mul(et, axis=0).sum() / (et ** 2).sum()
    party = doors.set_index("link").party.reindex(V.columns)
    g = pd.DataFrame({"party": party.values, "gbps": V.mean().values, "share": share.values}).groupby("party").sum()
    g = g.sort_values("share", ascending=False)
    out["doors"] = {"top4_variance_share": float(g.share.iloc[:4].sum()), "top4_traffic_share": float(g.gbps.iloc[:4].sum() / g.gbps.sum()),
                    "n_parties": int(len(g))}
    # holidays in the traffic: day / same weekday within 3 weeks, pre-test 2025 and test 2026
    lt = T5.tz_convert(DA.TZ)
    daily = lt.resample("D").mean()
    daily.index = daily.index.tz_localize(None)
    hol = []
    for k, (name, w) in DA.HOLIDAYS.items():
        ts = pd.Timestamp(k)
        if ts not in daily.index or ts.dayofweek >= 5 or not np.isfinite(daily[ts]):
            continue
        ref = daily[(daily.index.dayofweek == ts.dayofweek) & (abs((daily.index - ts).days) <= 21) & (daily.index != ts)]
        ref = ref[~ref.index.strftime("%Y-%m-%d").isin(list(DA.HOLIDAYS))]
        hol.append(dict(date=k, name=name, weight=w, ratio=float(daily[ts] / ref.mean())))
    out["holidays"] = hol
    json.dump(out, open(OUT / "diagnose.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:3000])


# --- the flat rate --------------------------------------------------------------------------

def stage_rate():
    """What the traffic forecast changes for the day-ahead flat rate B / T (test block)."""
    from green_routing.carbon import assumptions as A, bills, grid
    from green_routing.paths import reference_dir
    C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
    H = pd.date_range(TEST[0], TEST[-1] + pd.Timedelta(hours=23), freq="h")
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    tiers = pd.read_csv(NET / "routers.csv").set_index("dev").tier
    _, Pbb, _ = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"), pd.read_csv(C / "allocation" / "device_pools.csv"),
                                   pd.read_csv(C / "inventory" / "shelves.csv"), pd.read_csv(reference_dir() / "sites.csv"),
                                   doors, tiers, H)
    IA = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in Pbb})
    IF = pd.DataFrame({z: grid.day_ahead(grid.load(C / "grid", z)).reindex(H).interpolate() for z in Pbb})
    B = lambda I_: sum(Pbb[z] / 1000 * I_[z] for z in Pbb)            # noqa: E731  g per hour
    GB = A.GB_PER_MBPS_HOUR * 1e3                                    # GB per (Gbit/s x hour)
    Cc = cache()
    st = pd.read_csv(OUT / "scores_test.csv")
    keep = list(st.forecast)
    idx = DA.common_hours({k: Cc[k] for k in keep}, T, TEST)
    y = T.reindex(idx)
    rate = B(IA).reindex(idx) / (y * GB)
    rows = []
    for k in ["actual traffic"] + keep:
        f = y if k == "actual traffic" else Cc[k].reindex(idx)
        r = dict(forecast=k)
        for g, I_ in (("actual grid", IA), ("grid forecast", IF)):
            rf = B(I_).reindex(idx) / (f * GB)
            r[f"rate WAPE, {g}"] = float((rf - rate).abs().sum() / rate.sum())
            d = idx.normalize()
            r[f"daily rate WAPE, {g}"] = float((rf.groupby(d).mean() - rate.groupby(d).mean()).abs().sum() / rate.groupby(d).mean().sum())
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rate_errors.csv", index=False)
    print(R.round(4).to_string())


# --- the second network: Abilene (2004) ---------------------------------------------------------

AB_HOLIDAYS = (("2004-05-31", 1.0), ("2004-07-05", 1.0), ("2004-09-06", 1.0))     # Memorial, Independence (obs.), Labor Day


def stage_abilene():
    """The choices made on SWITCHlan, unchanged, on Abilene's total (no DEV period: 19 weeks)."""
    import pickle
    from green_routing.dataset import abilene
    ds = pickle.load(open(ARTIFACTS_DIR / "abilene" / "dataset.pkl", "rb"))
    t5 = abilene.series(ds)["total"] / 1e3
    g = t5.resample("h")
    Ta = g.mean().where(g.count() >= 6)
    Sa = Ta.interpolate().bfill()
    run = ARTIFACTS_DIR / "abilene" / "total" / "merged"
    meta = json.load(open(run / "dense" / "meta.json"))
    test = pd.date_range(pd.Timestamp(meta["t0"], unit="s", tz="UTC"), periods=meta["n_targets"] // 288, freq="D")
    tab = dict(holidays="table", table=AB_HOLIDAYS, tz="America/Chicago")
    cur = DA.Spec(cal="basic", **tab)
    full = DA.Spec(hist=24, lags=(1, 2, 3, 4, 5, 6, 7, 14), daily=7, cal="basic", clean=True, **tab)
    hol = lambda t: {"holiday": DA._holiday_weights(DA._local_days(t, "America/Chicago"), "table", AB_HOLIDAYS)}   # noqa: E731
    F = {f"rule: {k}": v for k, v in DA.rules(Sa).items()}
    F["gbdt: current (168 h, basic calendar) @ fixed, trained to 2004-07-24"] = DA.backtest(Sa, Ta, cur, test, refit_every=None, train_end="2004-07-24")
    F["gbdt: current (168 h, basic calendar)"] = DA.backtest(Sa, Ta, cur, test, refit_every=7)
    F["gbdt: compact, holidays, cleaned history"] = DA.backtest(Sa, Ta, full, test, refit_every=7)
    F["chronos-2: 12 weeks"] = DA.foundation(Sa, test, "chronos-2", 2016)
    F["chronos-2: 12 weeks + holiday covariate"] = DA.foundation(Sa, test, "chronos-2", 2016, covariates=hol)
    F["timesfm-2.5: 12 weeks"] = DA.foundation(Sa, test, "timesfm-2.5", 2016)
    for m in LADDER_MODELS:
        f = ladder_hourly(run, m)
        if f is not None:
            F[f"5-min {m}: original run (peak-aware loss)"] = f * 2       # its mirror holds the total once, not in + out
    F["ensemble: GBDT + Chronos-2"] = (F["gbdt: compact, holidays, cleaned history"] + F["chronos-2: 12 weeks"]) / 2
    F["ensemble: GBDT + Chronos-2 + TimesFM"] = (F["gbdt: compact, holidays, cleaned history"] + F["chronos-2: 12 weeks"]
                                                 + F["timesfm-2.5: 12 weeks"]) / 3
    idx = DA.common_hours(F, Ta, test)
    sc = DA.score(F, Ta, idx).assign(hours=len(idx))
    sc.to_csv(OUT / "scores_abilene.csv", index=False)
    pd.DataFrame(F).to_csv(OUT / "forecasts_abilene.csv.gz", float_format="%.5f")
    ref = "gbdt: current (168 h, basic calendar) @ fixed, trained to 2004-07-24"
    pd.DataFrame([dict(a=ref, b=b, **DA.paired(F[ref], F[b], Ta, idx)) for b in F if b != ref]).to_csv(OUT / "paired_abilene.csv", index=False)
    print(sc.round(4).to_string())


# --- scoring -----------------------------------------------------------------------------------

def scores(C: pd.DataFrame, days, common=None) -> pd.DataFrame:
    F = {k: C[k] for k in C if C[k].reindex(pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h")).notna().mean() > 0.95}
    idx = DA.common_hours(F if common is None else {k: F[k] for k in common if k in F}, T, days)
    return DA.score(F, T, idx).assign(hours=len(idx))


def stage_score():
    C = cache()
    sd, st = scores(C, DEV), scores(C, TEST)
    sd.to_csv(OUT / "scores_dev.csv", index=False)
    st.to_csv(OUT / "scores_test.csv", index=False)
    # the final pipeline is the one with the lowest error on DEV (test-only forecasts can't be chosen)
    json.dump({"name": sd.iloc[0].forecast, "rule": "lowest hourly WAPE on DEV among forecasts with DEV values"},
              open(OUT / "final.json", "w"), indent=1)
    T.reindex(pd.date_range(ALL[0], ALL[-1] + pd.Timedelta(hours=23), freq="h")).rename("T_gbps").to_csv(OUT / "truth.csv.gz")
    idx = DA.common_hours({k: C[k] for k in st.forecast}, T, TEST)
    ref = "gbdt: current (168 h, basic calendar) @ fixed, trained to 2026-02-15"     # the GBDT as used so far
    rows = [dict(a=ref, b=b, **DA.paired(C[ref], C[b], T, idx)) for b in st.forecast if b != ref]
    tcn0 = "5-min tcn: original run (peak-aware loss)"
    if tcn0 in C:
        rows += [dict(a=tcn0, b=b, **DA.paired(C[tcn0], C[b], T, idx)) for b in st.forecast if b != tcn0]
    # one lever at a time: (without, with)
    levers = [(f"5-min {m}: peak-aware loss, mean of 3 seeds", f"5-min {m}: symmetric loss, mean of 3 seeds") for m in LADDER_MODELS]
    levers += [("chronos-2: 24 weeks", "chronos-2: 24 weeks + calendar covariates"), ("chronos-2: 12 weeks", "chronos-2 fine-tuned: 12 weeks, 1000 steps"),
               ("chronos-2: 4 weeks of context", "chronos-2: 5-minute input, 4 weeks"), ("chronos-2: 4 weeks of context", "chronos-2: 24 weeks"),
               ("gbdt: full, old holiday table", "gbdt: compact, full calendar, cleaned history"),
               ("gbdt: full, no calendar", "gbdt: compact, full calendar, cleaned history"),
               ("gbdt: current (168 h, basic calendar) @ fixed, trained to 2026-02-15", "gbdt: current (168 h, basic calendar)"),
               ("gbdt: current (168 h, basic calendar)", "gbdt: current (168 h, basic calendar) @ 5 seeds"),
               ("chronos-2: 24 weeks", "ensemble: 0.8 x Chronos-2 + 0.2 x GBDT (weight from DEV)"),
               ("ensemble: 0.8 x Chronos-2 + 0.2 x GBDT (weight from DEV)", "ensemble: 0.8 x Chronos-2 + 0.2 x GBDT, GBDT alone on holidays"),
               ("ensemble: GBDT + Chronos-2 + TCN", "ensemble: GBDT + Chronos-2 + TCN, GBDT alone on holidays")]
    rows += [dict(a=a, b=b, **DA.paired(C[a], C[b], T, idx)) for a, b in levers if a in st.forecast.values and b in st.forecast.values]
    pd.DataFrame(rows).to_csv(OUT / "paired.csv", index=False)
    # where the errors are: hourly WAPE by kind of day (test block)
    d = DA._local_days(idx)
    hol = DA._holiday_weights(d, "weighted") >= 0.5
    kind = np.where(hol, "holiday", np.where(d.dayofweek >= 5, "weekend", "weekday"))
    after = np.isin(d - pd.Timedelta(days=7), d[hol])                    # a week after a holiday
    kind = np.where(after & ~hol, "week after a holiday", kind)
    yy = T.reindex(idx)
    rows = []
    for k in st.forecast:
        e = (C[k].reindex(idx) - yy).abs()
        rows.append({"forecast": k, **{g: float(e[kind == g].sum() / yy[kind == g].sum()) for g in np.unique(kind)}})
    pd.DataFrame(rows).to_csv(OUT / "by_day_kind.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print("== DEV", sd.hours.iloc[0], "hours"); print(sd.drop(columns="hours").round(4).to_string())
    print("== TEST", st.hours.iloc[0], "hours"); print(st.drop(columns="hours").round(4).to_string())


STAGES = ["rules", "sweep", "protocol", "foundation", "finetune", "ladder", "combine", "score", "diagnose", "rate", "abilene"]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for st in sys.argv[1:] or STAGES:
        print(f"== stage {st}", flush=True)
        globals()[f"stage_{st}"]()
