"""The flat rate on the common periods (periods.py), both networks.

  forecasts   Timestral (24 weeks, zero-shot) on every day from DEV[0] - 7 d to LATER[1]:
              Internet2's total traffic and the quantiles of its emissions (median, mean), from the hourly
              emissions series of i2_carbon_common.py (measured DC power from 2024-11); Chronos-2 on
              Internet2's total (test block); SWITCHlan's dedicated (access) emissions A and their median;
              Timestral's traffic quantiles on the test block (calibration), both networks.
              SWITCHlan's traffic and backbone (shared) emissions forecasts already exist for these days
              (flat_forecast cache, margin.py, appendix grid_hourly_three_periods); see switch_hours().
  score       margin chosen on DEV (margin.py's rule), own share (shared pool; all emissions with A_k posted
              from the forecast grid), bill error per calendar day / 7-day block / calendar month on TEST and
              LATER (and DEV), with and without the margin; perfect-traffic / perfect-grid variants; bias table
              with day-bootstrap intervals; SWITCHlan's grid-forecast table (DEV / TEST, calendar months).

writes $PAPER_COMMON/flat/*
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FF = HERE.parent / "flat_forecast"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FF))
import periods as P                                                      # noqa: E402
from green_routing.paths import ARTIFACTS_DIR                            # noqa: E402

OUT = P.ROOT / "flat"
SW_TRAFFIC = "timestral: 24 weeks + calendar covariates"
CTX = 4032
GRID = [round(x, 4) for x in np.arange(0, 0.2001, 0.005)]
TABLE_M = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]


def log(*a):
    print(*a, flush=True)


def all_days() -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(P.DEV[0]) - pd.Timedelta(days=7), P.LATER[1], freq="D", tz="UTC")


# --- inputs ---------------------------------------------------------------------------------------

def i2_series():
    """Internet2: hourly emissions E (g/h), scored traffic T (Mbit/s, complete and sane hours) and the filled
    input series s (as recipes/internet2/timestral_total.py)."""
    sys.path.insert(0, str(HERE.parent / "internet2"))
    import timestral_total as TT
    fr = pd.read_csv(P.ROOT / "i2_carbon" / "flat_rate_hourly.csv", index_col=0, parse_dates=True)
    T = fr.V_mbps.where(fr.flat_g_per_GB.notna())
    return fr.E_g_per_h, T, TT.total_series()


def switch_access(start="2024-06-01", end=None) -> pd.Series:
    """SWITCHlan's dedicated (access) equipment emissions, g/h: static site power of the access pool per grid
    zone (carbon.bills.window_power, as factcheck.switch_window) x the zone's hourly factor."""
    from green_routing.carbon import bills, grid
    from green_routing.paths import reference_dir
    C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
    end = end or P.LATER[1]
    H = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23), freq="h")
    Hp = pd.date_range(pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2026-07-10 23:00", tz="UTC"), freq="h")
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    tiers = pd.read_csv(NET / "routers.csv").set_index("dev").tier
    ACC, _, _ = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"),
                                   pd.read_csv(C / "allocation" / "device_pools.csv"),
                                   pd.read_csv(C / "inventory" / "shelves.csv"),
                                   pd.read_csv(reference_dir() / "sites.csv"), doors, tiers, Hp)
    zones = sorted({z for _, z in ACC.index})
    acc_z = {z: float(sum(v for (_, zz), v in ACC.items() if zz == z)) for z in zones}
    I = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in zones})
    return sum(acc_z[z] / 1000 * I[z] for z in zones)


def forecasts():
    OUT.mkdir(parents=True, exist_ok=True)
    from green_routing.carbon import dayahead as DA
    from bias import timestral_quantiles
    days = all_days()
    test = P.days(P.TEST)
    w = None
    f = OUT / "i2_forecasts.csv"
    if not f.exists():
        E, T, s = i2_series()
        log(f"Internet2: E from {E.dropna().index[0]}, {len(days)} forecast days")
        Tf = DA.foundation(s.dropna(), days, model="timestral", context=CTX)
        Q, lv = timestral_quantiles(E, days)
        w = DA._quantile_mean_weights(lv)
        df = pd.DataFrame({"Tf": Tf.reindex(Q.index), "Ef_median": Q[0.5], "Ef_mean": (Q * w).sum(1)})
        df["Tf_chronos2"] = DA.foundation(s.dropna(), test, model="chronos-2", context=CTX).reindex(df.index)
        # reproduction: round 1's Internet2 traffic forecasts (same series, same model) on their overlap
        r1 = pd.read_csv(ARTIFACTS_DIR / "paper" / "margin" / "i2_dev_forecasts.csv", index_col=0, parse_dates=True).Tf
        ov = r1.index.intersection(df.index)
        log(f"Internet2 traffic vs round 1 ({len(ov)} h): max rel. diff "
            f"{float(((df.Tf.reindex(ov) - r1[ov]).abs() / r1[ov]).max()):.2e}")
        df.to_csv(f)
    f = OUT / "switch_access.csv"
    if not f.exists():
        A = switch_access()
        Q, lv = timestral_quantiles(A, days)
        pd.DataFrame({"A": A.reindex(Q.index), "Af_median": Q[0.5]}).to_csv(f)
        log(f"SWITCHlan access emissions: mean {A.mean() / 1e3:.2f} kg/h")
    f = OUT / "traffic_quantiles_test.csv"
    if not f.exists():
        import run as R
        _, _, s = i2_series()
        out = []
        for net, x in (("switch", R.S), ("i2", s)):
            Q, lv = timestral_quantiles(x.dropna(), test)
            w = DA._quantile_mean_weights(lv)
            q = Q[[0.1, 0.5, 0.9, 0.95, 0.99]].copy()
            q.columns = [f"q{int(round(100 * c))}" for c in q.columns]
            q["mean"] = (Q * w).sum(1)
            out.append(q.assign(network=net))
        pd.concat(out).to_csv(f)


def gridq():
    """Timestral's quantiles of each network's emissions on the test block (q99 / actual)."""
    from bias import timestral_quantiles
    import grid_forecast as G
    f = OUT / "grid_quantiles_test.csv"
    if f.exists():
        return
    test = P.days(P.TEST)
    E_sw = G.switch_emissions(P.TEST[1])
    E_i2, _, _ = i2_series()
    out = []
    for net, E in (("switch", E_sw), ("i2", E_i2)):
        Q, _ = timestral_quantiles(E, test)
        q = Q[[0.1, 0.5, 0.9, 0.99]].copy()
        q.columns = ["q10", "q50", "q90", "q99"]
        out.append(q.assign(network=net, actual=E.reindex(q.index)))
    pd.concat(out).to_csv(f)


# --- hourly frames -----------------------------------------------------------------------------------

def label(idx: pd.DatetimeIndex) -> pd.Series:
    d = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    lab = pd.Series("PRE", index=idx)
    for name, (a, b) in (("DEV", P.DEV), ("TEST", P.TEST), ("LATER", P.LATER)):
        lab[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))] = name
    return lab


def switch_hours() -> pd.DataFrame:
    """SWITCHlan, hourly: T, Tf (Timestral 24 w + calendar), E (backbone = shared emissions), Ef (Timestral
    median + 7-day correction), Ef_raw, Ef_mean, Ef_rule, A (dedicated access equipment), Af (its median +
    correction). Sources: flat_forecast cache (Nov 2025 -> Jul 2026), margin.py (the rest), appendix.py's
    grid series (Timestral on the emissions 2025-03-22 -> 2026-10-28)."""
    import run as R
    import grid_forecast as G
    PA = ARTIFACTS_DIR / "paper"
    G3 = pd.read_csv(PA / "appendix" / "grid_hourly_three_periods.csv.gz", index_col=0, parse_dates=True)
    new = pd.read_csv(PA / "margin" / "switch_traffic_forecasts.csv", index_col=0, parse_dates=True)
    later = pd.read_csv(PA / "margin" / "switch_total_later.csv", index_col=0, parse_dates=True).iloc[:, 0]
    acc = pd.read_csv(OUT / "switch_access.csv", index_col=0, parse_dates=True)
    H = pd.date_range(all_days()[0], pd.Timestamp(P.LATER[1], tz="UTC") + pd.Timedelta(hours=23), freq="h")
    Tf = R.cache()[SW_TRAFFIC].reindex(H).combine_first(new.Tf.reindex(H))
    T = R.T.reindex(H).combine_first(new["T"].reindex(H)).combine_first(later.reindex(H))
    Ef = G.bias_corrected(G3["median"], G3.actual, 7)
    assert np.allclose(Ef, G3.median_corr, rtol=1e-9), "corrected median differs from appendix.py's"
    df = pd.DataFrame({"T": T, "Tf": Tf, "E": G3.actual.reindex(H), "Ef": Ef.reindex(H), "Ef_raw": G3["median"].reindex(H),
                       "Ef_mean": G3["mean"].reindex(H), "Ef_rule": G3.rule.reindex(H), "Ef_blend": G3.blend.reindex(H),
                       "A": acc.A.reindex(H), "Af": G.bias_corrected(acc.Af_median, acc.A, 7).reindex(H)})
    df["period"] = label(H)
    return df


def i2_hours() -> pd.DataFrame:
    import grid_forecast as G
    E, T, _ = i2_series()
    fc = pd.read_csv(OUT / "i2_forecasts.csv", index_col=0, parse_dates=True)
    H = fc.index
    Ef = G.bias_corrected(fc.Ef_median, E, 7)
    rule = E.shift(24)
    df = pd.DataFrame({"T": T.reindex(H), "Tf": fc.Tf, "E": E.reindex(H), "Ef": Ef.reindex(H), "Ef_raw": fc.Ef_median,
                       "Ef_mean": fc.Ef_mean, "Ef_rule": rule.reindex(H),
                       "Ef_blend": ((rule.reindex(H) + Ef.reindex(H)) / 2), "A": 0.0, "Af": 0.0,
                       "Tf_chronos2": fc.Tf_chronos2})
    df["period"] = label(H)
    return df


# --- scores --------------------------------------------------------------------------------------------

def daily(df: pd.DataFrame, m: float = 0.0, tf="Tf", ef="Ef") -> pd.DataFrame:
    """Day sums over the hours with every input: posted bill (1 - m) Ef T / Tf, shared emissions, and the
    dedicated equipment's actual (A) and posted (Af) emissions."""
    d = df.dropna(subset=["T", tf, "E", ef])
    bill = (1 - m) * d[ef] * d["T"] / d[tf]
    g = d.index.floor("D").tz_localize(None)
    return pd.DataFrame({"bill": bill.groupby(g).sum(), "E": d.E.groupby(g).sum(), "A": d.A.groupby(g).sum(),
                         "Af": d.Af.groupby(g).sum(), "hours": bill.groupby(g).size()})


def own(D: pd.DataFrame) -> float:
    return float(1 - D.bill.sum() / D.E.sum())


def own_all(D: pd.DataFrame) -> float:
    return float(1 - (D.bill.sum() + D.Af.sum()) / (D.E.sum() + D.A.sum()))


def period_sums(D: pd.DataFrame, p: tuple[str, str], kind: str) -> pd.DataFrame:
    """Sums of bill and E per calendar day, per 7-day block from p[0] (complete blocks only), or per calendar
    month (complete months only) inside period p."""
    D = D[(D.index >= pd.Timestamp(p[0])) & (D.index <= pd.Timestamp(p[1]))]
    if kind == "day":
        return D[["bill", "E", "A", "Af"]]
    if kind == "week":
        k = P.week_blocks(p).reindex(D.index)
        g = D[["bill", "E", "A", "Af"]].groupby(k)
        S = g.sum()[g.size() == 7]
        return S
    m = D.index.to_period("M")
    g = D[["bill", "E", "A", "Af"]].groupby(m)
    n = g.size()
    return g.sum()[[n[q] == q.days_in_month for q in n.index]]


def wape(S: pd.DataFrame) -> float:
    return float((S.bill - S.E).abs().sum() / S.E.sum())


def rolling30(D: pd.DataFrame, p) -> pd.DataFrame:
    D = D[(D.index >= pd.Timestamp(p[0])) & (D.index <= pd.Timestamp(p[1]))]
    full = D.reindex(pd.date_range(D.index[0], D.index[-1], freq="D"))
    return pd.DataFrame({"bill": full.bill.rolling(30).sum(), "E": full.E.rolling(30).sum()}).dropna()


def choose(D: pd.DataFrame, share: float = 0.9) -> float:
    for m in GRID:
        W = rolling30(D.assign(bill=D.bill * (1 - m)), P.DEV)
        if (W.bill <= W.E).mean() >= share:
            return m
    return np.nan


def scores(D: pd.DataFrame, p) -> dict:
    Dp = D[(D.index >= pd.Timestamp(p[0])) & (D.index <= pd.Timestamp(p[1]))]
    mo = period_sums(D, p, "month")
    om = 1 - mo.bill / mo.E
    oma = 1 - (mo.bill + mo.Af) / (mo.E + mo.A)
    W30 = rolling30(D, p)
    return {"days": int(len(Dp)), "hours": int(Dp.hours.sum()), "own_share": own(Dp), "own_share_all": own_all(Dp),
            "bias": -own(Dp),
            "own_month": {str(k): float(v) for k, v in om.items()},
            "own_month_all": {str(k): float(v) for k, v in oma.items()},
            "own_month_min": float(om.min()), "own_month_max": float(om.max()),
            "err_day": wape(period_sums(D, p, "day")), "err_week": wape(period_sums(D, p, "week")),
            "err_month": wape(mo), "weeks": int(len(period_sums(D, p, "week"))), "months": int(len(mo)),
            "err_day_abs_kg": float((Dp.bill - Dp.E).abs().mean() / 1e3),
            "share_30d_negative": float((W30.bill > W30.E).mean()), "windows_30d": int(len(W30)),
            "err_30d_rolling_overlapping": wape(W30)}


def score():
    from bias import bias_stats
    nets = {"switch": switch_hours(), "i2": i2_hours()}
    res, sens = {}, []
    for net, df in nets.items():
        df.to_csv(OUT / f"hourly_{net}.csv.gz")
        D0 = daily(df)
        m_star = choose(D0)
        r = {"m": m_star, "rule_robustness": {str(s): choose(D0, s) for s in (0.8, 0.9, 0.95, 1.0)}}
        # m that makes the development own share exactly 0 (the bill is linear in 1 - m)
        Dd = D0[(D0.index >= pd.Timestamp(P.DEV[0])) & (D0.index <= pd.Timestamp(P.DEV[1]))]
        r["m_dev_zero"] = float(1 - Dd.E.sum() / Dd.bill.sum())
        for m in sorted(set(TABLE_M + [m_star])):
            D = daily(df, m)
            for per, p in (("DEV", P.DEV), ("TEST", P.TEST), ("LATER", P.LATER)):
                sens.append(dict(network=net, m=m, period=per, **{k: v for k, v in scores(D, p).items()
                                                                  if not isinstance(v, dict)}))
        for m, tag in ((0.0, "no_margin"), (m_star, "margin")):
            for v, (tf, ef) in {"forecast": ("Tf", "Ef"), "perfect_grid": ("Tf", "E"), "perfect_traffic": ("T", "Ef"),
                                "timestral_uncorrected": ("Tf", "Ef_raw"), "rule_grid": ("Tf", "Ef_rule")}.items():
                D = daily(df, m, tf, ef)
                for per, p in (("DEV", P.DEV), ("TEST", P.TEST), ("LATER", P.LATER)):
                    r[f"{tag}/{v}/{per}"] = scores(D, p)
        # biases per period (sums over the scored hours)
        for per, p in (("DEV", P.DEV), ("TEST", P.TEST), ("LATER", P.LATER)):
            x = df[(df.period == per)].dropna(subset=["T", "Tf", "E", "Ef"])
            r[f"bias/{per}"] = {"traffic": float(x.Tf.sum() / x["T"].sum() - 1),
                                "grid_corrected": float(x.Ef.sum() / x.E.sum() - 1),
                                "grid_raw_median": float(x.Ef_raw.sum() / x.E.sum() - 1),
                                "grid_mean": float(x.Ef_mean.sum() / x.E.sum() - 1),
                                "grid_rule": float(x.Ef_rule.sum() / x.E.sum() - 1),
                                "grid_blend": float(x.Ef_blend.sum() / x.E.sum() - 1),
                                "bill": float((x.Ef * x["T"] / x.Tf).sum() / x.E.sum() - 1),
                                "hours": int(len(x))}
        # Table app-bias: test block, 95 % day-bootstrap intervals (bias.py's statistics)
        x = df[df.period == "TEST"].dropna(subset=["T", "Tf", "E", "Ef"])
        rng = np.random.default_rng(0)
        r["bias_table"] = {k: bias_stats(x[f], x[a], rng) for k, (f, a) in {
            "traffic_timestral": ("Tf", "T"), "grid_rule": ("Ef_rule", "E"), "grid_median": ("Ef_raw", "E"),
            "grid_mean": ("Ef_mean", "E"), "grid_median_corrected": ("Ef", "E")}.items()}
        bill = x.Ef * x["T"] / x.Tf
        r["bias_table"]["bill"] = bias_stats(bill, x.E, rng)
        res[net] = r
        log(f"{net}: m = {100 * m_star:.1f} %, test own share {r['margin/forecast/TEST']['own_share']:+.4f} "
            f"(no margin {r['no_margin/forecast/TEST']['own_share']:+.4f}), month error "
            f"{r['margin/forecast/TEST']['err_month']:.4f}")
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)
    json.dump(res, open(OUT / "flat.json", "w"), indent=1, default=float)


def grid_scores(f: pd.Series, a: pd.Series, p) -> dict:
    """A grid (emissions) forecast against the actual over period p: bias, worst calendar month, WAPE of
    calendar days, 7-day blocks and calendar months (non-overlapping), and of every 30-day window (round 1's)."""
    ok = f.notna() & a.notna()
    g = f.index.floor("D").tz_localize(None)
    D = pd.DataFrame({"bill": f[ok].groupby(g[ok]).sum(), "E": a[ok].groupby(g[ok]).sum(), "A": 0.0, "Af": 0.0})
    mo = period_sums(D, p, "month")
    mb = mo.bill / mo.E - 1
    Dp = D[(D.index >= pd.Timestamp(p[0])) & (D.index <= pd.Timestamp(p[1]))]
    return {"bias %": 100 * float(Dp.bill.sum() / Dp.E.sum() - 1), "worst month %": 100 * float(mb.abs().max()),
            "months within 4 %": int((mb.abs() <= 0.04).sum()), "months": int(len(mb)),
            "months low": int((mb < 0).sum()),
            "day %": 100 * wape(period_sums(D, p, "day")), "week %": 100 * wape(period_sums(D, p, "week")),
            "month %": 100 * wape(mo), "30-day rolling %": 100 * wape(rolling30(D, p)),
            "by month %": {str(k): round(100 * float(v), 2) for k, v in mb.items()}}


def grid():
    """SWITCHlan's grid-forecast table on DEV and TEST (grid_forecast.py's candidates, causal: refitted or
    zero-shot day by day, so any period inside 2025-03-22 -> 2026-07-10 is out of sample), the GBDT loss
    variants (extremes.py, refitted every 28 days), and the monthly bias over DEV + TEST + LATER."""
    import grid_forecast as G
    df = pd.read_csv(G.CAND, index_col=0, parse_dates=True)
    F = G.with_variants(df)
    f = OUT / "grid_losses_forecasts.csv.gz"
    if not f.exists():
        import extremes as X
        L = X.gbdt_targets(G.switch_emissions())
        pd.DataFrame({k: v.reindex(df.index) for k, v in L.items()}).to_csv(f)
    L = pd.read_csv(f, index_col=0, parse_dates=True)
    for k in L.columns:
        F[k] = L[k]
        F[k + " + bias corr. 7 d"] = G.bias_corrected(L[k], df["actual"], 7)
    rows = []
    for k, x in F.items():
        for per, p in (("DEV", P.DEV), ("TEST", P.TEST)):
            s = grid_scores(x, df["actual"], p)
            rows.append(dict(forecast=k, period=per, **{a: b for a, b in s.items() if a != "by month %"},
                             by_month=json.dumps(s["by month %"])))
    pd.DataFrame(rows).to_csv(OUT / "grid_scores.csv", index=False)
    # three periods (Fig. app-bias): appendix.py's hourly series from 2025-03-22 to 2026-10-28
    G3 = pd.read_csv(ARTIFACTS_DIR / "paper" / "appendix" / "grid_hourly_three_periods.csv.gz", index_col=0, parse_dates=True)
    rows = []
    for k in ("rule", "median", "mean", "median_corr", "blend"):
        for per, p in (("DEV", P.DEV), ("TEST", P.TEST), ("LATER", P.LATER)):
            s = grid_scores(G3[k], G3.actual, p)
            rows.append(dict(forecast=k, period=per, **{a: b for a, b in s.items() if a != "by month %"},
                             by_month=json.dumps(s["by month %"])))
    pd.DataFrame(rows).to_csv(OUT / "grid_three_periods.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="by_month").round(2).to_string(index=False))


def dedicated():
    """SWITCHlan's dedicated equipment as design §3.3 now specifies it: A_k posted from the intensity of each grid
    zone in the same hour of the previous day (static power, so A_rule(h) = A(h - 24 h)), and lowered by the same
    margin as every other posted charge. Own share over all emissions = 1 - (bills_B + bills_A) / (E + A). The
    margin chosen two ways on the development year (the rule: own share >= 0 in >= 90 % of the 30-day windows):
    on the shared pool (as flat.json) and on all emissions. Also the Timestral variant of flat.json (Af).
    Writes dedicated.json."""
    df = pd.read_csv(OUT / "hourly_switch.csv.gz", index_col=0, parse_dates=True)
    df["Af_rule"] = df.A.shift(24)
    res = {"m_shared": float(json.load(open(OUT / "flat.json"))["switch"]["m"])}

    def dd(m_b, m_a, a_col):
        d = df.dropna(subset=["T", "Tf", "E", "Ef", "A", a_col])
        g = d.index.floor("D").tz_localize(None)
        bill = (1 - m_b) * d.Ef * d["T"] / d.Tf + (1 - m_a) * d[a_col]
        return pd.DataFrame({"bill": bill.groupby(g).sum(), "E": (d.E + d.A).groupby(g).sum(), "A": 0.0, "Af": 0.0,
                             "hours": bill.groupby(g).size()})

    def choose_all(a_col):
        for m in GRID:
            W = rolling30(dd(m, m, a_col), P.DEV)
            if (W.bill <= W.E).mean() >= 0.9:
                return m
        return np.nan

    res["m_all_rule"] = choose_all("Af_rule")
    res["m_all_timestral"] = choose_all("Af")
    for per, p in (("DEV", P.DEV), ("TEST", P.TEST), ("LATER", P.LATER)):
        x = df[df.period == per].dropna(subset=["A", "Af", "Af_rule"])
        res[f"A_posted_bias/{per}"] = {"rule": float(x.Af_rule.sum() / x.A.sum() - 1),
                                       "timestral_corrected": float(x.Af.sum() / x.A.sum() - 1),
                                       "A_share_of_all": float(x.A.sum() / (x.A.sum() + x.E.sum()))}
        for name, (mb, ma, col) in {
                "rule, margin on B and A (m_shared)": (res["m_shared"], res["m_shared"], "Af_rule"),
                "rule, margin on B only (m_shared)": (res["m_shared"], 0.0, "Af_rule"),
                "rule, margin on B and A (m_all_rule)": (res["m_all_rule"], res["m_all_rule"], "Af_rule"),
                "rule, no margin": (0.0, 0.0, "Af_rule"),
                "timestral, margin on B and A (m_shared)": (res["m_shared"], res["m_shared"], "Af"),
                "timestral, margin on B only (m_shared)": (res["m_shared"], 0.0, "Af")}.items():
            D = dd(mb, ma, col)
            s = scores(D, p)
            res[f"{name}/{per}"] = {k: s[k] for k in ("own_share", "own_month", "own_month_min", "err_day", "err_week",
                                                     "err_month", "share_30d_negative")}
    json.dump(res, open(OUT / "dedicated.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith(("DEV", "LATER"))}, indent=1, default=float))


def bill_periods():
    """Fig. 4 (left): the bill's error on the test block by billing period (hour, calendar day, 7-day block,
    calendar month): posted with the margin (forecast), without it, with the actual grid (perfect grid, no margin)
    and with the actual traffic (perfect traffic, no margin). Writes bill_periods.csv."""
    res = json.load(open(OUT / "flat.json"))
    rows = []
    for net in ("switch", "i2"):
        df = pd.read_csv(OUT / f"hourly_{net}.csv.gz", index_col=0, parse_dates=True)
        m = float(res[net]["m"])
        for v, (mm, tf, ef) in {"forecast": (m, "Tf", "Ef"), "forecast, no margin": (0.0, "Tf", "Ef"),
                                "perfect grid": (0.0, "Tf", "E"), "perfect traffic": (0.0, "T", "Ef")}.items():
            d = df[df.period == "TEST"].dropna(subset=["T", tf, "E", ef])
            bill = (1 - mm) * d[ef] * d["T"] / d[tf]
            rows.append(dict(network=net, variant=v, m=mm, period="hour",
                             wape=float((bill - d.E).abs().sum() / d.E.sum()), bias=float(bill.sum() / d.E.sum() - 1)))
            D = daily(df, mm, tf, ef)
            for k in ("day", "week", "month"):
                S = period_sums(D, P.TEST, k)
                rows.append(dict(network=net, variant=v, m=mm, period=k, wape=wape(S),
                                 bias=float(S.bill.sum() / S.E.sum() - 1), n=int(len(S))))
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "bill_periods.csv", index=False)
    print(B.pivot_table(index=["network", "variant"], columns="period", values="wape")[["hour", "day", "week", "month"]]
          .mul(100).round(2).to_string())


if __name__ == "__main__":
    {"forecasts": forecasts, "gridq": gridq, "score": score, "grid": grid,
     "periods": bill_periods, "dedicated": dedicated}[sys.argv[1]]()
