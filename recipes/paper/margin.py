"""The margin on the posted flat rate (paper Sec. 3.3 "The posted rate is binding", Sec. 5.2 "Bias
decides the ISP's own share").

The posted flat rate of hour h is r^(h) = (1 - m) x E^(h) / T^(h): Timestral's day-ahead median of the
shared emissions, scaled by the 7-day ratio of actual to forecast emissions (as bill_periods.py), over
Timestral's day-ahead traffic, lowered by the margin m. Bills are final: bill(h) = r^(h) x T(h). The ISP's
own share over a period P is

    own(P) = (sum_P E - sum_P bill) / sum_P E        (> 0: the ISP absorbs emissions; < 0: over-billing)

with E the emissions the flat rate splits (SWITCHlan: the backbone pool, dedicated access equipment billed
apart; Internet2: everything). Rule for m (chosen per network on its development period only): the smallest m
on a 0.5 % grid with own >= 0 in at least 90 % of the 30-day windows (every window, daily step) of the
development period.

Periods. SWITCHlan: DEV 2025-03-22 -> 2026-03-21 (grid_forecast.py's development year), TEST 2026-03-22 ->
07-10, LATER 2026-07-11 -> 10-28 (the four months after the test block of Appendix M; traffic rebuilt from
the door counters in the local S3 cache). Internet2: DEV 2026-03-01 -> 07-18 (recipes/internet2/carbon.py
DEV0 .. TEST0; its emissions series starts 2025-11-15 = YEAR0; power and grid data go back to 2024-11), TEST 2026-07-19 -> 10-31 (the hours of
bill_periods.py). TEST forecasts are bill_periods.py's (hourly_{switch,i2}.csv), unchanged.

usage: python margin.py forecasts    the missing Timestral forecasts (GPU) and SWITCHlan's later traffic
       python margin.py score        choice of m, sensitivity table, path-rate own share (Internet2)
       python margin.py check_paths  path choices unchanged under the margin (one pair alone, sample of hours)
writes $GR_ARTIFACTS_DIR/paper/margin/*
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
FF = HERE.parent / "flat_forecast"
P = ARTIFACTS_DIR / "paper"
OUT = P / "margin"
SW_TRAFFIC = "timestral: 24 weeks + calendar covariates"
SW_DEV = ("2025-03-22", "2026-03-21")
SW_TEST = ("2026-03-22", "2026-07-10")
SW_LATER = ("2026-07-11", "2026-10-28")
I2_DEV = ("2026-03-01", "2026-07-18")
I2_TEST = ("2026-07-19", "2026-10-31")
GRID = [round(x, 4) for x in np.arange(0, 0.2001, 0.005)]
TABLE_M = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]


def days(a, b):
    return pd.date_range(a, b, freq="D", tz="UTC")


def log(*a):
    print(*a, flush=True)


# --- forecasts -----------------------------------------------------------------------------------

def switch_later_traffic() -> pd.Series:
    """SWITCHlan's total one-way traffic (Gbit/s, hourly means of >= 6 of 12 steps) from 2026-07-01 to
    2026-10-29, built as recipes/carbon_rates/make_total.py (every door of carbon/allocation/doors.csv),
    read from the frozen local S3 cache."""
    f = OUT / "switch_total_later.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
    from green_routing.carbon import total
    doors = pd.read_csv(ARTIFACTS_DIR / "carbon" / "allocation" / "doors.csv")
    tin, tout, cov = total.total_5min(doors, "2026-07-01", "2026-10-29")
    t5 = (tin + tout) / 2 / 1e3
    g = t5.resample("h")
    T = g.mean().where(g.count() >= 6)
    T.rename("T_gbps").to_csv(f)
    return T


def forecasts():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(FF))
    from green_routing.carbon import dayahead as DA

    # SWITCHlan traffic: Timestral 24 w + calendar covariates, as flat_forecast/run.py's cache (which starts
    # 2025-11-01): the rest of the development year and the later period
    f = OUT / "switch_traffic_forecasts.csv"
    if not f.exists():
        import run as R
        later = switch_later_traffic()
        ov = R.T.reindex(later.index).dropna()
        lv = later.reindex(ov.index)
        log(f"SWITCHlan later traffic vs the year mirror on the overlap ({len(ov)} h): "
            f"mean |rel. diff| {float((lv - ov).abs().mean() / ov.mean()):.2e}")
        T_ext = pd.concat([R.T, later[later.index > R.T.index[-1]]])
        S_ext = T_ext.interpolate().bfill()
        d_dev = days(SW_DEV[0], "2025-10-31")
        d_chk = days("2025-11-01", "2025-11-30")
        d_later = days(*SW_LATER)
        fc = DA.foundation(S_ext, d_dev.append(d_chk).append(d_later), model="timestral", context=4032, covariates=True)
        cache = R.cache()[SW_TRAFFIC]
        chk = fc.reindex(pd.date_range(d_chk[0], d_chk[-1] + pd.Timedelta(hours=23), freq="h"))
        rel = (chk - cache.reindex(chk.index)).abs() / cache.reindex(chk.index)
        log(f"SWITCHlan traffic, reproduction of the cached forecasts (Nov 2025): max rel. diff {rel.max():.2e}, "
            f"median {rel.median():.2e}")
        pd.DataFrame({"Tf": fc, "T": T_ext.reindex(fc.index)}).to_csv(f)

    # Internet2: Timestral 24 w on the total (as recipes/internet2/timestral_total.py) and Timestral's median
    # of the emissions (as bill_periods.py), on the development period (+ 7 days for the correction)
    f = OUT / "i2_dev_forecasts.csv"
    if not f.exists():
        sys.path.insert(0, str(HERE.parent / "internet2"))
        import timestral_total as TT
        from bias import timestral_quantiles
        s = TT.total_series()
        d_dev = days(*I2_DEV)
        d_chk = days(I2_TEST[0], "2026-08-01")
        fc = DA.foundation(s.dropna(), d_dev.append(d_chk), model="timestral", context=4032)
        ref = pd.read_csv(ARTIFACTS_DIR / "internet2" / "carbon" / "timestral_total.csv", index_col=0,
                          parse_dates=True)["Timestral-209M, 24 weeks"]
        chk = fc[fc.index >= pd.Timestamp(I2_TEST[0], tz="UTC")]
        rel = (chk - ref.reindex(chk.index)).abs() / ref.reindex(chk.index)
        log(f"Internet2 traffic, reproduction of timestral_total.csv (first 14 test days): max rel. diff {rel.max():.2e}")
        fr = pd.read_csv(ARTIFACTS_DIR / "internet2" / "carbon" / "flat_rate_hourly.csv", index_col=0, parse_dates=True)
        E = fr.E_g_per_h
        pre = pd.date_range(d_dev[0] - pd.Timedelta(days=7), d_dev[0] - pd.Timedelta(days=1), freq="D")
        Q, _ = timestral_quantiles(E, pre.append(d_dev))
        df = pd.DataFrame({"Ef_median": Q[0.5]})
        df["Tf"] = fc.reindex(df.index)
        df.to_csv(f)


# --- hourly inputs per network and period ----------------------------------------------------------

def switch_hours() -> pd.DataFrame:
    """Hourly T, Tf, E, Ef (corrected median) and period on SWITCHlan."""
    sys.path.insert(0, str(FF))
    G3 = pd.read_csv(P / "appendix" / "grid_hourly_three_periods.csv.gz", index_col=0, parse_dates=True)
    test = pd.read_csv(P / "hourly_switch.csv", index_col=0, parse_dates=True)
    new = pd.read_csv(OUT / "switch_traffic_forecasts.csv", index_col=0, parse_dates=True)
    import run as R
    cache = R.cache()[SW_TRAFFIC]
    H = G3.index
    Tf = cache.reindex(H).combine_first(new.Tf.reindex(H))
    T = R.T.reindex(H).combine_first(new["T"].reindex(H))
    df = pd.DataFrame({"T": T, "Tf": Tf, "E": G3.actual, "Ef": G3.median_corr, "Ef_raw": G3["median"],
                       "period": G3.period.map({"dev": "DEV", "test": "TEST", "later": "LATER"})})
    # TEST: exactly bill_periods.py's hours and values (the appendix's copy differs by CSV rounding, < 2e-6)
    t = test.dropna()
    assert np.allclose(df.loc[t.index, "Ef"], t.Ef, rtol=1e-5) and np.allclose(df.loc[t.index, "Tf"], t.Tf, rtol=1e-6)
    tt = df.period == "TEST"
    df.loc[tt, ["T", "Tf", "E", "Ef"]] = test[["T", "Tf", "E", "Ef"]].reindex(df.index[tt]).to_numpy()
    return df


def i2_hours() -> pd.DataFrame:
    sys.path.insert(0, str(FF))
    import grid_forecast as G
    test = pd.read_csv(P / "hourly_i2.csv", index_col=0, parse_dates=True)
    dev = pd.read_csv(OUT / "i2_dev_forecasts.csv", index_col=0, parse_dates=True)
    fr = pd.read_csv(ARTIFACTS_DIR / "internet2" / "carbon" / "flat_rate_hourly.csv", index_col=0, parse_dates=True)
    ok = fr.flat_g_per_GB.notna()
    Ef = G.bias_corrected(dev.Ef_median, fr.E_g_per_h, 7)
    H = dev.index[dev.index >= pd.Timestamp(I2_DEV[0], tz="UTC")]
    d = pd.DataFrame({"T": fr.V_mbps.where(ok).reindex(H), "Tf": dev.Tf.reindex(H), "E": fr.E_g_per_h.reindex(H),
                      "Ef": Ef.reindex(H), "period": "DEV"})
    t = test[["T", "Tf", "E", "Ef"]].assign(period="TEST")
    # the actuals on the test hours are the same series
    assert np.allclose(fr.E_g_per_h.reindex(t.dropna().index), t.dropna().E, rtol=1e-9)
    return pd.concat([d, t])


# --- scores ----------------------------------------------------------------------------------------

def daily(df: pd.DataFrame, m: float) -> pd.DataFrame:
    d = df.dropna(subset=["T", "Tf", "E", "Ef"])
    bill = (1 - m) * d.Ef * d["T"] / d.Tf
    day = d.index.floor("D")
    return pd.DataFrame({"bill": bill.groupby(day).sum(), "E": d.E.groupby(day).sum(), "hours": bill.groupby(day).size()})


def own(D: pd.DataFrame) -> float:
    return float(1 - D.bill.sum() / D.E.sum())


def windows(D: pd.DataFrame, k: int, step: int = 1) -> pd.DataFrame:
    """Sums over k consecutive calendar days (windows with a missing day are dropped), every ``step`` days."""
    full = D.reindex(pd.date_range(D.index[0], D.index[-1], freq="D"))
    b, e = full.bill.rolling(k).sum(), full.E.rolling(k).sum()
    W = pd.DataFrame({"bill": b, "E": e}).dropna()
    if step > 1:                                   # tiling: from the first day, non-overlapping
        starts = full.index[::step]
        ends = starts + pd.Timedelta(days=k - 1)
        W = W.reindex(ends).dropna()
    return W


def wape(W: pd.DataFrame) -> float:
    return float((W.bill - W.E).abs().sum() / W.E.sum())


def months(D: pd.DataFrame, full_only: bool = True) -> pd.Series:
    m = D.index.tz_localize(None).to_period("M")
    g = D.groupby(m)
    n = g.size()
    o = 1 - g.bill.sum() / g.E.sum()
    if full_only:
        o = o[[p for p in o.index if n[p] == p.days_in_month]]
    return o


def scores(D: pd.DataFrame) -> dict:
    W30 = windows(D, 30)
    W30t = windows(D, 30, 30)
    W7, W7t = windows(D, 7), windows(D, 7, 7)
    mo, mo_all = months(D), months(D, False)
    r = {"days": int(len(D)), "own_share": own(D),
         "own_month_min": float(mo.min()) if len(mo) else np.nan, "own_month_max": float(mo.max()) if len(mo) else np.nan,
         "own_month_min_incl_partial": float(mo_all.min()), "own_month_max_incl_partial": float(mo_all.max()),
         "months_full": int(len(mo)),
         "err_day": wape(D), "err_7d_rolling": wape(W7), "err_7d_tiling": wape(W7t),
         "err_30d_rolling": wape(W30), "err_30d_tiling": wape(W30t),
         "err_month_calendar": float((mo.abs() * D.E.groupby(D.index.tz_localize(None).to_period("M")).sum().reindex(mo.index)).sum()
                                     / D.E.groupby(D.index.tz_localize(None).to_period("M")).sum().reindex(mo.index).sum()) if len(mo) else np.nan,
         "windows_30d": int(len(W30)), "share_30d_negative": float((W30.bill > W30.E).mean()),
         "windows_30d_tiling": int(len(W30t))}
    return r


def choose(D: pd.DataFrame) -> float:
    for m in GRID:
        W = windows(D.assign(bill=D.bill * (1 - m)), 30)
        if (W.bill <= W.E).mean() >= 0.9:
            return m
    return np.nan


def path_own_share(m: float) -> dict:
    """Internet2 path rates (i2_rates.py, capped, k = 2): posted per-resource rates times the actual
    loads, summed: the bills of every byte. Own share over the test block, by calendar month, and over
    30-day windows; the posted rates use the grid of the same hour yesterday (i2_rates.py). Also the
    flat rate posted with the same inputs, for comparison."""
    sys.path.insert(0, str(HERE))
    import i2_rates as IR
    R = IR.load_rates()
    ok = R["ok"]
    T = pd.to_datetime(R["T"], utc=True)[ok]
    U, X = R["U"][ok], R["X"][ok]
    path_bill = (1 - m) * ((R["per_post"][ok] * U).sum(1) + R["add_post"][ok] * X.sum(1))
    flat_bill = (1 - m) * R["E_da"][ok].sum(1) / R["X_fc"][ok].sum(1) * X.sum(1)
    E = R["E"][ok].sum(1)
    out = {}
    for name, b in (("path", path_bill), ("flat_same_inputs", flat_bill)):
        D = pd.DataFrame({"bill": pd.Series(b, index=T).groupby(T.floor("D")).sum(),
                          "E": pd.Series(E, index=T).groupby(T.floor("D")).sum()})
        mo = months(D)
        W30 = windows(D, 30)
        out[name] = {"own_share": own(D), "own_month_min": float(mo.min()), "own_month_max": float(mo.max()),
                     "share_30d_negative": float((W30.bill > W30.E).mean()), "err_day": wape(D),
                     "err_30d_rolling": wape(W30)}
    return out


def score():
    OUT.mkdir(parents=True, exist_ok=True)
    nets = {"SWITCHlan": switch_hours(), "Internet2": i2_hours()}
    rows, chosen = [], {}
    for net, df in nets.items():
        D0 = {p: daily(df[df.period == p], 0.0) for p in df.period.unique()}
        m_star = choose(D0["DEV"])
        chosen[net] = m_star
        log(f"{net}: chosen m = {100 * m_star:.1f} % (DEV {D0['DEV'].index[0].date()} -> {D0['DEV'].index[-1].date()}, "
            f"{len(D0['DEV'])} days)")
        for m in sorted(set(TABLE_M + [m_star])):
            for p, D in D0.items():
                Dm = D.assign(bill=D.bill * (1 - m))
                rows.append(dict(network=net, m=m, chosen=(m == m_star), period=p, **scores(Dm)))
        # the own share by calendar month, every m of the table (for the report)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["network", "period", "m", "own_share", "own_month_min", "own_month_max", "err_day", "err_30d_rolling",
            "err_30d_tiling", "err_month_calendar", "share_30d_negative"]
    T = S[show].copy()
    for c in show[2:]:
        T[c] = (100 * T[c]).round(2)
    log(T.to_string(index=False))
    # monthly own shares (all months, incl. partial) for each network and chosen m
    mon = []
    for net, df in nets.items():
        for m in (0.0, chosen[net]):
            for p in df.period.unique():
                D = daily(df[df.period == p], m)
                mm = D.index.tz_localize(None).to_period("M")
                g = D.groupby(mm)
                for per, o in (1 - g.bill.sum() / g.E.sum()).items():
                    mon.append(dict(network=net, m=m, period=p, month=str(per), days=int(g.size()[per]), own_share=float(o)))
    M = pd.DataFrame(mon)
    M.to_csv(OUT / "months.csv", index=False)
    log(M.assign(own_share=(100 * M.own_share).round(2)).to_string(index=False))
    # Fig. 4 left with the margin: bill error by period as bill_periods.py (every 7- / 30-day window)
    per_rows = []
    for net, df in nets.items():
        t = df[df.period == "TEST"].dropna(subset=["T", "Tf", "E", "Ef"])
        for m in (0.0, chosen[net]):
            for v, (tf, ef) in {"forecast": ("Tf", "Ef"), "perfect grid": ("Tf", "E"), "perfect traffic": ("T", "Ef")}.items():
                bill = (1 - m) * t[ef] * t["T"] / t[tf]
                D = pd.DataFrame({"bill": bill.groupby(t.index.floor("D")).sum(), "E": t.E.groupby(t.index.floor("D")).sum()})
                per = {"hour": float((bill - t.E).abs().sum() / t.E.sum()), "day": wape(D),
                       "week": wape(windows(D, 7)), "month": wape(windows(D, 30))}
                for k, w in per.items():
                    per_rows.append(dict(network=net, m=m, variant=v, period=k, wape=w, bias=float(bill.sum() / t.E.sum() - 1)))
    BP = pd.DataFrame(per_rows)
    BP.to_csv(OUT / "bill_periods_margin.csv", index=False)
    log(BP.pivot_table(index=["network", "m", "variant"], columns="period", values="wape")[["hour", "day", "week", "month"]]
        .mul(100).round(2).to_string())
    paths = {f"m={m}": path_own_share(m) for m in (0.0, chosen["Internet2"])}
    log(json.dumps(paths, indent=1))
    json.dump({"chosen": chosen, "rule": "smallest m on a 0.5 % grid with own share >= 0 in >= 90 % of the 30-day "
               "windows (daily step) of the development period", "dev": {"SWITCHlan": SW_DEV, "Internet2": I2_DEV},
               "test": {"SWITCHlan": SW_TEST, "Internet2": I2_TEST}, "later": {"SWITCHlan": SW_LATER},
               "path_rates_i2": paths}, open(OUT / "margin.json", "w"), indent=1)


def check_paths(m: float = 0.05, n_hours: int = 200):
    """One pair alone (steering.py uni_hour) with the posted rates x (1 - m): same moves, same paths, same
    realized costs on a sample of test hours."""
    sys.path.insert(0, str(HERE))
    import steering as ST
    ST._init()
    hrs = [h for h in range(len(ST.R["T"])) if ST.R["ok"][h]]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(hrs, min(n_hours, len(hrs)), replace=False))
    base = {h: ST.uni_hour(h)[1] for h in pick}
    ST.R["per_post"] = ST.R["per_post"] * (1 - m)
    ST.R["add_post"] = ST.R["add_post"] * (1 - m)
    diff = 0
    for h in pick:
        o = ST.uni_hour(h)[1]
        b = base[h]
        same = (np.array_equal(o["moved_fc"], b["moved_fc"]) and np.allclose(o["c_fc"], b["c_fc"], rtol=1e-12)
                and np.array_equal(o["same"], b["same"]))
        diff += not same
    res = {"m": m, "hours": len(pick), "hours_with_any_difference": diff}
    log(json.dumps(res))
    json.dump(res, open(OUT / "check_paths.json", "w"))


def check_everyone(m: float = 0.06):
    """Everyone follows the posted rates (steering.py everyone, closed loop) with every posted resource
    rate x (1 - m): the posted emissions E_da scaled (capped_rates is linear in E). Writes
    margin/everyone_hourly_margin.csv; compare with paper/i2/everyone_hourly.csv."""
    sys.path.insert(0, str(HERE))
    import i2_rates as IR
    import steering as ST
    orig = IR.load_rates_k

    def scaled(k):
        R = dict(orig(k))
        R["E_da"] = R["E_da"] * (1 - m)
        R["per_post"], R["add_post"] = R["per_post"] * (1 - m), R["add_post"] * (1 - m)
        return R
    IR.load_rates_k = scaled
    ST.OUT = OUT
    ST._tag = lambda: "_margin"
    ST.everyone()
    a = pd.read_csv(P / "i2" / "everyone_hourly.csv")
    b = pd.read_csv(OUT / "everyone_hourly_margin.csv")
    cols = ["spans_dark", "dark_E_share", "util_max", "links_over_80", "hops_per_byte", "bytes_paying_more"]
    res = {"m": m, "hours": len(a), "identical_hours": int((a[cols].round(12) == b[cols].round(12)).all(axis=1).sum()),
           "mean_base": a[cols].mean().to_dict(), "mean_margin": b[cols].mean().to_dict()}
    log(json.dumps(res, indent=1))
    json.dump(res, open(OUT / "check_everyone.json", "w"), indent=1)


if __name__ == "__main__":
    {"forecasts": forecasts, "score": score, "check_paths": check_paths, "check_everyone": check_everyone}[sys.argv[1]]()
