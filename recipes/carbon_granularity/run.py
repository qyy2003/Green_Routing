"""At what grain to publish the carbon rate, at what grain to forecast it, and whether to
publish a curve or a distribution: experiments on SWITCHlan (2026) and Abilene (2004).

usage: python run.py
writes $GR_ARTIFACTS_DIR/carbon_granularity/<dataset>/*.csv and summary.json

E1 settlement   publish the true rate at grain G, bill real 5-minute traffic: what is lost?
E2 forecast     the day-ahead rate at grain G (issued 00:00 UTC): error against the 5-minute
                truth, against the block truth, on party bills
E3 model grain  5-minute models summed up vs a model trained directly at the grain
E4 curves       Fourier and piecewise-linear curves fitted to the forecast path
E5 distribution quantiles from the last W days' joint traffic and grid errors: coverage,
                CRPS, and what they change for decisions
E6 cadence      the same rate forecast 15 min, 1 h, 8 h and a day ahead (intraday updates)
E7 scheduling   flexible traffic moved by a rate published at grain G
E8 resolution   the same traffic forecast scored at each grain against that grain's truth,
                and how much of the within-hour variation its 5-minute shape explains
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from green_routing.carbon import bills as billmod, grid, granularity as G, total
from green_routing.dataset import abilene
from green_routing.paths import ARTIFACTS_DIR, reference_dir
from green_routing.prediction.calendar_features import HOLIDAYS

OUT = ARTIFACTS_DIR / "carbon_granularity"
GR = G.GRAINS
US_HOLIDAYS_2004 = {"2004-05-31", "2004-07-05", "2004-09-06"}


def dense(run, mode, model, horizon, n, scale):
    meta = json.load(open(run / "dense" / "meta.json"))
    hi = {h["name"]: i for i, h in enumerate(meta["horizons"])}[horizon]
    return np.load(run / "dense" / f"pred_{mode}_{model}.npy")[0, hi, :n].astype(float) * scale


def load_switch():
    C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
    run = ARTIFACTS_DIR / "carbon_rates" / "flat" / "merged"
    meta = json.load(open(run / "dense" / "meta.json"))
    n = meta["n_targets"] // 288 * 288
    H5 = pd.date_range(pd.Timestamp(meta["t0"], unit="s", tz="UTC"), periods=n, freq="5min")
    m = pd.read_csv(ARTIFACTS_DIR / "carbon_rates" / "mirror" / "router-interfaces" / "total" / "all.csv")
    full = pd.Series((m.ifHCInOctets + m.ifHCOutOctets).to_numpy() * 8 / 1e6 / 2,
                     index=pd.to_datetime(m.timestamp, unit="s", utc=True))       # one-way Mbit/s
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    din, dout, _ = total.door_5min(doors, str(H5[0]), str(H5[-1] + pd.Timedelta("5min")))
    V = (din + dout)
    V5 = V.T.groupby(doors.set_index("link").party.reindex(V.columns).values).sum().T.to_numpy(float)
    tiers = pd.read_csv(NET / "routers.csv").set_index("dev").tier
    Hh = pd.date_range(H5[0], periods=n // 12, freq="h")
    _, Pbb, _ = billmod.window_power(pd.read_csv(C / "inventory" / "routers.csv"), pd.read_csv(C / "allocation" / "device_pools.csv"),
                                     pd.read_csv(C / "inventory" / "shelves.csv"), pd.read_csv(reference_dir() / "sites.csv"),
                                     doors, tiers, Hh)
    zones = list(Pbb)
    load = {z: grid.load(C / "grid", z) for z in zones}
    IA = pd.DataFrame({z: load[z].reindex(Hh).interpolate() for z in zones})
    lead = {"day": lambda s: grid.day_ahead(s)}
    for L, sh in (("15min", 2), ("1h", 2), ("8h", 9)):      # last complete hour before the issue time
        lead[L] = (lambda k: (lambda s: s.asfreq("h").interpolate().shift(k)))(sh)
    IF = {L: pd.DataFrame({z: f(load[z]).reindex(Hh).interpolate() for z in zones}) for L, f in lead.items()}
    Bh = lambda I: sum(Pbb[z] * I[z].to_numpy() / 1000 for z in zones)            # noqa: E731  g per hour
    return dict(name="switchlan", H5=H5, T5=full.reindex(H5).to_numpy(float), full=full, V5=V5,
                B5=np.repeat(Bh(IA), 12) / 12, B5_fc={L: np.repeat(Bh(I), 12) / 12 for L, I in IF.items()},
                Pbb=Pbb, IA=IA, IF=IF["day"], run=run, scale=0.5, train_end="2026-02-15",
                holidays=HOLIDAYS, tz="Europe/Zurich", grid_varies=True)


def load_abilene():
    ds = pickle.load(open(ARTIFACTS_DIR / "abilene" / "dataset.pkl", "rb"))
    S = abilene.series(ds)
    run = ARTIFACTS_DIR / "abilene" / "total" / "merged"
    meta = json.load(open(run / "dense" / "meta.json"))
    n = meta["n_targets"] // 288 * 288
    H5 = pd.date_range(pd.Timestamp(meta["t0"], unit="s", tz="UTC"), periods=n, freq="5min")
    inv = pd.read_csv(ARTIFACTS_DIR / "abilene" / "carbon" / "inventory.csv")
    Bh = float((inv.W * 2.0 / 1000 * inv.g_per_kWh).sum())                        # g per hour (PUE 2.0)
    V5 = (S["ingress"] + S["egress"]).reindex(H5).to_numpy(float)
    B5 = np.full(n, Bh / 12)
    return dict(name="abilene", H5=H5, T5=S["total"].reindex(H5).to_numpy(float), full=S["total"], V5=V5,
                B5=B5, B5_fc={L: B5 for L in ("15min", "1h", "8h", "day")}, run=run, scale=1.0, train_end="2004-07-24",
                holidays=US_HOLIDAYS_2004, tz="America/Chicago", grid_varies=False)


def experiments(D):
    out = OUT / D["name"]
    out.mkdir(parents=True, exist_ok=True)
    T5, B5, V5, n = D["T5"], D["B5"], D["V5"], len(D["H5"])
    meta = json.load(open(D["run"] / "dense" / "meta.json"))
    models = [m for m in meta["modes"]["block"] if "@" not in m and "+" not in m]
    fc = {f"5-min {m}": dense(D["run"], "block", m, "day", n, D["scale"]) for m in models}
    for g, k in GR.items():
        if k >= 3:
            fc[f"direct GBDT at {g}"] = G.gbdt_direct(D["full"], k, D["train_end"], D["holidays"], D["tz"]).reindex(D["H5"]).to_numpy(float)
    # score every source on the same days: real data and every forecast available (the
    # 1-week input windows have no forecast for a week after a gap in the data)
    days_ok = np.isfinite(T5).reshape(-1, 288).all(axis=1) & np.isfinite(V5).reshape(-1, 288, V5.shape[1]).all(axis=(1, 2))
    for p in fc.values():
        days_ok &= np.isfinite(p).reshape(-1, 288).all(axis=1)
    mask = np.repeat(days_ok, 288)
    T5m, V5m = np.where(mask, T5, np.nan), np.where(mask[:, None], V5, 0.0)
    r5 = B5 / (T5m * G.GB5)                                                    # the 5-minute truth
    r5z = np.nan_to_num(r5)
    S = {"dataset": D["name"], "days": int(days_ok.sum()), "parties": int(V5.shape[1])}

    # E1: settlement at each grain, true rates
    rows = []
    var = np.nansum((r5 - np.nansum(r5 * T5m) / np.nansum(T5m)) ** 2 * T5m)
    for g, k in GR.items():
        rg = np.nan_to_num(G.block_rate(np.where(mask, B5, 0), np.nan_to_num(T5m), k))
        be, bias = G.bill_error(rg, r5z, V5m)
        bg, b5 = G.bills(rg, V5m), G.bills(r5z, V5m)
        rel = np.abs(bg - b5) / np.maximum(b5, 1e-9)
        big = b5 > np.quantile(b5, 0.5)                          # the larger half of the parties
        rows.append(dict(grain=g, bill_error=be, daily_bill_error=G.daily_bill_error(rg, r5z, V5m), bias=bias,
                         emission_error=G.emission_error(rg, r5z, np.nan_to_num(T5m)),
                         variance_kept=float(1 - np.nansum((r5 - rg) ** 2 * T5m) / var),
                         max_party_error=float(rel.max()), max_party_bill_share=float(b5[np.argmax(rel)] / b5.sum()),
                         max_error_larger_half=float(rel[big].max()),
                         bill_share_over_1pct=float(b5[rel > 0.01].sum() / b5.sum())))
    E1 = pd.DataFrame(rows)
    E1.to_csv(out / "e1_settlement.csv", index=False)

    # E2/E3: day-ahead forecasts at each grain, 5-minute models summed vs direct GBDT
    rows = []
    for src, T_hat in fc.items():
        k_src = GR[src.split(" at ")[1]] if src.startswith("direct") else 1
        for g, k in GR.items():
            if k < k_src:
                continue
            for gname, Bf in (("grid forecast", D["B5_fc"]["day"]), ("actual grid", B5)):
                if gname == "actual grid" and not D["grid_varies"]:
                    continue
                rf = np.nan_to_num(G.block_rate(np.where(mask, Bf, 0), np.where(mask, T_hat, 0), k))
                rt = np.nan_to_num(G.block_rate(np.where(mask, B5, 0), np.nan_to_num(T5m), k))
                be, bias = G.bill_error(rf, r5z, V5m)
                rows.append(dict(source=src, grain=g, grid=gname, error_vs_5min=G.emission_error(rf, r5z, np.nan_to_num(T5m)),
                                 error_vs_block=G.emission_error(rf, rt, np.nan_to_num(T5m)), bill_error=be,
                                 daily_bill_error=G.daily_bill_error(rf, r5z, V5m), bias=bias))
    E2 = pd.DataFrame(rows)
    E2.to_csv(out / "e2_forecast_by_grain.csv", index=False)
    g1 = E2[(E2.grain == "1h") & (E2.grid == "grid forecast")].sort_values("error_vs_5min")
    best5 = g1[g1.source.str.startswith("5-min")].iloc[0].source
    S["best_5min_model"] = best5

    # E8: traffic accuracy of the same forecast at each resolution; the within-hour part
    rows = []
    for src, T_hat in fc.items():
        k_src = GR[src.split(" at ")[1]] if src.startswith("direct") else 1
        Th = np.where(mask, T_hat, np.nan)
        skill = G.within_block_skill(Th, T5m) if k_src == 1 else np.nan
        for g, k in GR.items():
            if k >= k_src:
                rows.append(dict(source=src, grain=g, wape=G.wape(Th, T5m, k),
                                 wape_vs_5min=G.wape(G.expand(G.blocks(Th, k, "mean"), k), T5m), within_hour_skill=skill))
    E8 = pd.DataFrame(rows)
    E8.to_csv(out / "e8_resolution.csv", index=False)
    ok = np.isfinite(T5m)
    S["within_hour_share"] = float((G.within_block(T5m)[ok] ** 2).sum() / ((T5m[ok] - T5m[ok].mean()) ** 2).sum())

    # E4: curves fitted to the best 5-minute forecast path, and to the rate
    Th = fc[best5]
    rows = []
    paths = np.where(mask, Th, np.nan).reshape(-1, 288)
    ok = np.isfinite(paths).all(axis=1)
    Bf = D["B5_fc"]["day"]

    def score(T_fit, label, params):
        rf = np.nan_to_num(Bf / (np.where(mask, T_fit, np.nan) * G.GB5))
        be, _ = G.bill_error(rf, r5z, V5m)
        rows.append(dict(curve=label, parameters_per_day=params, error_vs_5min=G.emission_error(rf, r5z, np.nan_to_num(T5m)),
                         bill_error=be, daily_bill_error=G.daily_bill_error(rf, r5z, V5m)))
    score(Th, "raw 5-minute values", 288)
    for k in (3, 12, 36):
        stair = G.expand(G.blocks(np.nan_to_num(Th), k, "mean"), k)
        score(stair, f"staircase, {[g for g, v in GR.items() if v == k][0]} values", 288 // k)
    for K in (1, 2, 3, 4, 6, 8, 12, 24):
        fit = np.full_like(paths, np.nan)
        fit[ok] = G.fourier_fit(paths[ok], K)
        score(fit.ravel(), f"Fourier, {K} harmonics", 2 * K + 1)
    for k in (12, 36):
        fit = np.full_like(paths, np.nan)
        fit[ok] = G.knot_fit(paths[ok], k)
        score(fit.ravel(), f"curve through {[g for g, v in GR.items() if v == k][0]} knots", 288 // k)
    # a curve fitted to the rate itself (with the grid's hourly steps)
    rate_path = np.where(mask, Bf / (Th * G.GB5), np.nan).reshape(-1, 288)
    for K in (6, 12, 24):
        fit = np.full_like(rate_path, np.nan)
        fit[ok] = G.fourier_fit(rate_path[ok], K)
        rf = np.nan_to_num(fit.ravel())
        be, _ = G.bill_error(rf, r5z, V5m)
        rows.append(dict(curve=f"Fourier on the rate, {K} harmonics", parameters_per_day=2 * K + 1,
                         error_vs_5min=G.emission_error(rf, r5z, np.nan_to_num(T5m)), bill_error=be,
                         daily_bill_error=G.daily_bill_error(rf, r5z, V5m)))
    E4 = pd.DataFrame(rows)
    E4.to_csv(out / "e4_curves.csv", index=False)

    # E5: distributions at 1 hour, from the last W days' joint errors (traffic and grid)
    nd = n // 288
    Tt = G.blocks(np.nan_to_num(T5m), 12, "sum").reshape(nd, 24)                      # sums of 12 steps
    Tt[~np.repeat(days_ok, 1)] = np.nan
    Tp = G.blocks(np.nan_to_num(Th), 12, "sum").reshape(nd, 24)
    Bt = G.blocks(B5, 12).reshape(nd, 24)
    Bp = G.blocks(Bf, 12).reshape(nd, 24)
    r_true = Bt / (Tt * G.GB5)
    r_point = Bp / (Tp * G.GB5)
    eT = np.log(Tt / Tp)                      # traffic error (NaN on gap days)
    eB = np.log(Bt / Bp)                      # grid error (0 when the grid factor is constant)
    rows, dec = [], []
    variants = ([(7, 1), (14, 1), (28, 1), (28, 3), (56, 1)] if D["grid_varies"] else [(7, 1), (7, 3), (14, 1)])
    main = (28, 3) if D["grid_varies"] else (7, 3)
    for W, spread in variants:
        comps = ("traffic + grid", "traffic only", "grid only") if (D["grid_varies"] and (W, spread) == (28, 1)) else \
                (("traffic + grid",) if D["grid_varies"] else ("traffic only",))
        for comp in comps:
            Q = np.full((nd, 24, len(G.LEVELS)), np.nan)
            for d in range(W, nd):
                if not days_ok[d]:
                    continue
                past = range(d - W, d)
                for h in range(24):
                    hs = [(h + j) % 24 for j in range(-spread, spread + 1)]
                    et = eT[list(past)][:, hs].ravel() if comp != "grid only" else np.zeros(1)
                    eb = eB[list(past)][:, hs].ravel() if comp != "traffic only" else np.zeros(1)
                    if comp == "traffic + grid":       # joint: same past day and hour
                        sc = Bp[d, h] * np.exp(eb) / (Tp[d, h] * np.exp(et) * G.GB5)
                    else:
                        sc = Bp[d, h] * np.exp(eb[:, None]).ravel() / (Tp[d, h] * np.exp(et) * G.GB5)
                    sc = sc[np.isfinite(sc)]
                    if len(sc):
                        Q[d, h] = np.quantile(sc, G.LEVELS)
            ev = np.isfinite(Q[..., 0]) & np.isfinite(r_true)
            i10, i90 = list(G.LEVELS).index(0.1), list(G.LEVELS).index(0.9)
            y = np.where(ev, r_true, np.nan)
            pt = np.where(ev, r_point, np.nan)
            med = Q[..., list(G.LEVELS).index(0.5)]
            rows.append(dict(window_days=W, hours_pooled=2 * spread + 1, uncertainty=comp, hours=int(ev.sum()),
                             coverage_80=G.coverage(Q[ev], y[ev], 0.1, 0.9), coverage_90=G.coverage(Q[ev], y[ev], 0.05, 0.95),
                             width_80_rel=float(np.nanmean((Q[..., i90] - Q[..., i10])[ev] / med[ev])),
                             crps=G.pinball(Q[ev], y[ev]), mae_point=float(np.nanmean(np.abs(pt - y))),
                             mae_median=float(np.nanmean(np.abs(med - y))), mean_rate=float(np.nanmean(y))))
            if comp == ("traffic + grid" if D["grid_varies"] else "traffic only") and (W, spread) == main:
                pd.DataFrame({"level": G.LEVELS, "observed": [float((y[ev] <= Q[..., i][ev]).mean()) for i in range(len(G.LEVELS))]}
                             ).to_csv(out / "e5_reliability.csv", index=False)
                S["main_distribution"] = dict(window_days=W, hours_pooled=2 * spread + 1)
                # decisions: pick tomorrow's cheapest hour by the point forecast, the median, or the P90
                for label, crit in (("point forecast", pt), ("median", np.where(ev, med, np.nan)),
                                    ("P90 (risk-averse)", np.where(ev, Q[..., i90], np.nan))):
                    pick = [int(np.nanargmin(crit[d])) for d in range(nd) if np.isfinite(crit[d]).any()]
                    dd = [d for d in range(nd) if np.isfinite(crit[d]).any()]
                    real = np.array([r_true[d, p] for d, p in zip(dd, pick)])
                    dec.append(dict(choice=label, realised=float(np.nanmean(real)), day_average=float(np.nanmean(r_true[dd])),
                                    hindsight=float(np.nanmean(np.nanmin(r_true[dd], axis=1)))))
                exceed = (y > Q[..., i90])[ev].mean()
                S["p90_exceedance"] = float(exceed)
                S["p90_premium"] = float(np.nanmean(Q[..., i90][ev]) / np.nanmean(y[ev]) - 1)
    E5 = pd.DataFrame(rows)
    E5.to_csv(out / "e5_distributions.csv", index=False)
    pd.DataFrame(dec).to_csv(out / "e5_decisions.csv", index=False)

    # E6: cadence, the best model's rolling forecasts at each lead with the matching grid forecast
    rows = []
    base = best5.split("5-min ")[1]
    roll = {L: dense(D["run"], "rolling", base, L, n, D["scale"]) for L in ("15min", "1h", "8h", "day")}
    m6 = days_ok.copy()
    for p in roll.values():
        m6 &= np.isfinite(p).reshape(-1, 288).all(axis=1)
    m6 = np.repeat(m6, 288)
    T6, V6 = np.where(m6, T5, np.nan), np.where(m6[:, None], V5, 0.0)
    r6 = np.nan_to_num(B5 / (T6 * G.GB5))
    rules = {"last known hour": D["B5_fc"], "same hour yesterday": {L: D["B5_fc"]["day"] for L in roll}}
    for L, Tl in roll.items():
        for rule, Bl in rules.items():
            if rule == "last known hour" and L == "day":
                continue
            if not D["grid_varies"] and rule != "same hour yesterday":
                continue
            for g in ("15min", "1h", "3h"):
                k = GR[g]
                rf = np.nan_to_num(G.block_rate(np.where(m6, Bl[L], 0), np.where(m6, Tl, 0), k))
                rows.append(dict(lead=L, grid_rule=rule if D["grid_varies"] else "constant", grain=g,
                                 error_vs_5min=G.emission_error(rf, r6, np.nan_to_num(T6)), days=int(m6.sum() // 288)))
        rf = np.nan_to_num(G.block_rate(np.where(m6, B5, 0), np.where(m6, Tl, 0), 12))
        rows.append(dict(lead=L, grid_rule="actual grid", grain="1h", error_vs_5min=G.emission_error(rf, r6, np.nan_to_num(T6)),
                         days=int(m6.sum() // 288)))
    E6 = pd.DataFrame(rows)
    E6.to_csv(out / "e6_cadence.csv", index=False)

    # E7: scheduling flexible traffic with the rate published at each grain
    rows = []
    rf5 = np.nan_to_num(Bf / (np.where(mask, Th, np.nan) * G.GB5), nan=np.nan)
    for g, k in GR.items():
        if k == 288:
            continue
        rfk = G.block_rate(np.where(mask, Bf, 0), np.where(mask, Th, np.nan), k)
        rtk = G.block_rate(np.where(mask, B5, 0), T5m, k)
        a = G.schedule(rfk, r5, k)
        b = G.schedule(rtk, r5, k)
        rows.append(dict(grain=g, forecast_realised=a["realised"], forecast_captured=a["captured"],
                         perfect_realised=b["realised"], perfect_captured=b["captured"], day_average=a["day_average"],
                         hindsight_5min=a["hindsight"]))
    E7 = pd.DataFrame(rows)
    E7.to_csv(out / "e7_scheduling.csv", index=False)
    json.dump(S, open(out / "summary.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    for name, t in (("E1", E1), ("E2 (1h, 5-min truth)", E2[E2.grain == "1h"]), ("E8", E8.pivot(index="source", columns="grain", values="wape")[list(GR)]), ("E4", E4), ("E5", E5), ("E6", E6), ("E7", E7)):
        print(f"== {D['name']} {name}")
        print(t.round(4).to_string(index=False))
    print(S)


if __name__ == "__main__":
    for loader in (load_abilene, load_switch):
        experiments(loader())
