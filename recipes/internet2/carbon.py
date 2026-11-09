"""Carbon rates and bills on Internet2 (2025-26): measured PoP power, hourly US grid factors,
counters only (no flow data), the same method as Switch LAN.

Energy: every DC power plant at the PoPs of the 73 core routers, hourly amps x 48 V (A and B
feeds summed), x PUE. Nearly all plants feed a whole PoP, so the PoP's power is split by an
assumed share (OPTICAL_SHARE, Switch LAN's measured 45 %) between its routers and its spans. PoPs
with no plant data get the median power per core router. Power at sites without a core router
(amplifier / regeneration huts) is spread over all inter-PoP spans by capacity. Grid: each PoP's balancing authority, hourly production-based g/kWh from EIA-930
(``carbon.us_grid``). Internet2 has no per-member equipment data, so all power is shared.

Resources for path rates: each core router (its PoP's router-side power, split over the PoP's
core routers) and each inter-PoP span (half of each end PoP's optical power, split over the
PoP's spans by capacity, plus its share of the huts). Routing: ECMP shortest paths, weights
inverse to capacity (checked by hiding links). Matrix: non-negative EM from the directed link
loads and each router's door totals, then IPF (``carbon.tomography``).

Outputs ($GR_ARTIFACTS_DIR/internet2/carbon): energy_pops.csv, dependence.csv, router_bills.csv,
od_rates.csv, rate_errors.csv, flat_rate_hourly.csv, summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from green_routing.carbon import path_rate, tomography, us_grid
from green_routing.dataset import internet2 as i2
from green_routing.paths import ARTIFACTS_DIR, reference_dir

HERE = Path(__file__).resolve().parent
ROOT = ARTIFACTS_DIR / "internet2"
DATA, OUT = ROOT / "data", ROOT / "carbon"
GB = 3600 / 8 / 1000                       # GB per Mbit/s-hour
PUE = {"central": 1.5, "low": 1.3, "high": 1.8}
OPTICAL_SHARE = 0.45                       # of a PoP's power, to its spans (Switch LAN's measured share)
YEAR0, YEAR1 = "2025-11-15", "2026-11-01"  # the year with 5-minute data (forecasting window)
TEST0, TEST1 = "2026-07-19", "2026-11-01"  # test block (forecasting and day-ahead rates)
DEV0 = "2026-03-01"                        # development period for choices: DEV0 .. TEST0
US_HOLIDAYS = ["2024-11-11", "2024-11-28", "2024-11-29", "2024-12-24", "2024-12-25", "2024-12-31", "2025-01-01",
               "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13",
               "2025-11-11", "2025-11-27", "2025-11-28", "2025-12-24", "2025-12-25", "2025-12-31", "2026-01-01",
               "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-10-12"]


def log(*a):
    print(*a, flush=True)


def wq(vals, wts, p):
    o = np.argsort(vals)
    c = np.cumsum(np.asarray(wts, float)[o])
    return float(np.interp(p * c[-1], c, np.asarray(vals, float)[o]))


def load():
    links = pd.read_csv(DATA / "links.csv")
    LH = pd.read_csv(DATA / "links_hourly.csv.gz", index_col=0, parse_dates=True)
    D = pd.read_csv(DATA / "doors.csv")
    DI = pd.read_csv(DATA / "doors_in_hourly.csv.gz", index_col=0, parse_dates=True)
    DO = pd.read_csv(DATA / "doors_out_hourly.csv.gz", index_col=0, parse_dates=True)
    W = pd.read_csv(DATA / "power_hourly.csv.gz", parse_dates=["time"])
    return links, LH, D, DI, DO, W


def grid_factors(pops: pd.DataFrame) -> pd.DataFrame:
    """Hourly g/kWh per PoP (columns), from its balancing authority."""
    eia = reference_dir() / "eia930"
    G = us_grid.load_balance(sorted(eia.glob("EIA930_BALANCE_*.csv")))
    F = pd.read_csv(HERE / "fuel_factors.csv").set_index("source").g_per_kWh
    I = us_grid.intensity(G, F).unstack("ba")
    out = {p: I[b] for p, b in zip(pops["pop"], pops["ba"]) if isinstance(b, str) and b in I}
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    links, LH, D, DI, DO, W = load()
    pops = pd.read_csv(HERE / "pops.csv")
    H = pd.date_range(YEAR0, YEAR1, freq="h", inclusive="left", tz="UTC")
    core = sorted(set(links.u) | set(links.v) | set(D.node))
    pop_of = {r: i2.pop_of(r) for r in core}
    core_pops = sorted(set(pop_of.values()))

    # --- energy: hourly W per PoP, router side and optical --------------------------------
    W = W[W.dc & W["pop"].notna()]
    # sum of a system's circuits; an hour in which every circuit is missing stays missing (pivot_table's sum
    # would make it 0 W: export outages then read as idle PoPs, AGENTS pitfall 49) and is filled below
    per_sys = W.groupby(["time", "node"]).watts.sum(min_count=1).unstack().reindex(H)
    med = per_sys.median()
    live = med[med > 0].index                                   # systems that report a load
    per_sys = per_sys[live].fillna(per_sys[live].median())      # power is near constant: fill gaps
    popn = pd.Series({n: (i2.pop_of(n) or ("manlan" if "manlan" in n else None)) for n in live})
    at_core = popn.isin(core_pops)
    # Almost every plant is a generic "dcp" feeding the whole PoP (4 of 106 are optical-only), so
    # the router / optical split is an assumption: Switch LAN's measured optical share.
    P_pop = per_sys.T.groupby([popn[at_core]]).sum().T                         # W per core PoP
    n_rt = pd.Series({p: sum(1 for r in core if pop_of[r] == p) for p in core_pops})
    per_router = (P_pop.mean() / n_rt.reindex(P_pop.columns)).median()
    missing = [p for p in core_pops if p not in P_pop.columns]                 # PoPs with no plant data
    for p in missing:
        P_pop[p] = per_router * n_rt[p]
    P_router, P_opt = P_pop * (1 - OPTICAL_SHARE), P_pop * OPTICAL_SHARE
    P_huts = per_sys.loc[:, ~at_core].sum(axis=1)                              # sites without a core router
    other_sites = sorted(set(popn[~at_core].dropna()))
    EP = pd.DataFrame({"W": P_pop.mean(), "router_W": P_router.mean(), "optical_W": P_opt.mean(),
                       "imputed": [p in missing for p in P_pop.columns], "core_routers": n_rt.reindex(P_pop.columns)})
    EP["ba"] = pops.set_index("pop").ba.reindex(EP.index)
    EP.to_csv(OUT / "energy_pops.csv")

    # --- grid --------------------------------------------------------------------------------
    I = grid_factors(pops).reindex(H)
    I = I.interpolate(limit=6, limit_direction="both")
    pops_no_grid = [p for p in EP.index if p not in I.columns]
    hut_I = I[[p for p in core_pops if p in I.columns]].mean(axis=1)            # huts: network-mean factor

    # --- resources: routers and inter-PoP spans --------------------------------------------
    spans = sorted({tuple(sorted((u, v))) for u, v in zip(links.u, links.v) if pop_of[u] != pop_of[v]})
    cap_span = {s: 0.0 for s in spans}
    for u, v, c in zip(links.u, links.v, links.capacity_bps.fillna(links.capacity_bps.median())):
        s = tuple(sorted((u, v)))
        if s in cap_span:
            cap_span[s] += c / 2                                   # both directions listed
    routers_at = {p: [r for r in core if pop_of[r] == p] for p in core_pops}
    spans_at = {p: [s for s in spans if p in (pop_of[s[0]], pop_of[s[1]])] for p in core_pops}
    tot_cap = sum(cap_span.values())

    def emissions(I):
        """g CO2 per hour of every resource (routers, then spans) under grid factors ``I``."""
        hut = I[[p for p in core_pops if p in I.columns]].mean(axis=1)            # huts: network-mean factor
        g = lambda p: I[p] if p in I.columns else hut                              # noqa: E731
        E_router = {}
        for p in core_pops:
            Pw = P_router[p] if p in P_router else pd.Series(0.0, index=H)
            for r in routers_at[p]:
                E_router[r] = Pw / len(routers_at[p]) * PUE["central"] / 1000 * g(p)
        E_span = {s: pd.Series(0.0, index=H) for s in spans}
        for p in core_pops:
            Po = P_opt[p] if p in P_opt else pd.Series(0.0, index=H)
            tot = sum(cap_span[s] for s in spans_at[p]) or 1.0
            for s in spans_at[p]:
                E_span[s] = E_span[s] + Po * cap_span[s] / tot * PUE["central"] / 1000 * g(p)
        for s in spans:
            E_span[s] = E_span[s] + P_huts * cap_span[s] / tot_cap * PUE["central"] / 1000 * hut
        return pd.DataFrame({**E_router, **{f"{s[0]}|{s[1]}": E_span[s] for s in spans}})

    E = emissions(I)
    E_dayahead = emissions(I.shift(24))                    # grid: same hour yesterday (the Switch LAN rule)

    # --- traffic: router door totals, directed link loads -----------------------------------
    door_router = D.set_index("door").node
    ing = DI.T.groupby(door_router.reindex(DI.columns)).sum().T.reindex(H)        # Mbit/s into the network
    egr = DO.T.groupby(door_router.reindex(DO.columns)).sum().T.reindex(H)
    ends = sorted(set(ing.columns) & set(core))
    LHn = LH.copy()
    LHn.columns = pd.MultiIndex.from_tuples([tuple(c.split("->")) for c in LHn.columns])
    full = lambda s: s + i2.FQDN                                                  # noqa: E731
    lk = {(full(a), full(b)): c for (a, b), c in zip(LHn.columns, LHn.columns)}
    V = ing[ends].sum(axis=1, min_count=1)                                        # each byte once
    # complete hours: nearly as many doors report as in the surrounding two weeks (doors come and go)
    n_rep = DI.notna().sum(axis=1)
    complete = (n_rep > 0) & (n_rep >= 0.9 * n_rep.rolling(24 * 14, center=True, min_periods=24).max())
    sane = lambda v: v > 0.2 * v.rolling(24 * 7, center=True, min_periods=24).median()     # noqa: E731  outages report 0
    ok = V.notna() & sane(V) & E.notna().all(axis=1) & complete.reindex(H).fillna(False)

    # flat rate
    flat_rate = (E.sum(axis=1) / (V * GB)).where(ok)                               # g per GB
    pd.DataFrame({"E_g_per_h": E.sum(axis=1), "V_mbps": V, "flat_g_per_GB": flat_rate}).to_csv(OUT / "flat_rate_hourly.csv")

    # routing (ECMP, inverse-capacity weights) over core routers
    def routing(weight):
        edges = {}
        for u, v, c in zip(links.u, links.v, links.capacity_bps.fillna(links.capacity_bps.median())):
            edges[(u, v)] = (1e11 / c) if weight == "capacity" else 1.0
        return tomography.ecmp_routing(core, edges, ends)
    lks, A_link, A_node, od = routing("capacity")
    nE = len(ends)
    Y_all = np.column_stack([LHn[lk[l]].reindex(H).to_numpy() if l in lk else np.full(len(H), np.nan) for l in lks])

    T = H[ok.to_numpy() & (H >= TEST0) & (H < TEST1)]
    tix = H.get_indexer(T)
    In, Eg = np.nan_to_num(ing[ends].to_numpy()[tix]), np.nan_to_num(egr[ends].to_numpy()[tix])
    Eg = Eg * (In.sum(1, keepdims=True) / np.maximum(Eg.sum(1, keepdims=True), 1e-9))   # close the cut
    Y = Y_all[tix]
    log(f"[carbon] {len(T)} test hours, {nE} routers with doors, {len(lks)} directed links, {len(spans)} spans")
    X_em = np.array([tomography.em(A_link, Y[h], In[h], Eg[h]) for h in range(len(T))])
    X_gr = np.array([tomography.gravity(In[h], Eg[h]) for h in range(len(T))])
    # EM leaves tiny positive flows everywhere; on a nearly idle router they make per-byte rates
    # explode (1e18 g/GB). Flows under 1 kbit/s are set to zero (their resources' emissions then
    # count as idle and are shared over all bytes, as node_bills does for unused resources).
    TINY = 1e-3                                                                   # Mbit/s
    X_em[X_em < TINY] = 0.0
    X_gr[X_gr < TINY] = 0.0

    # hold-out: hide a fifth of the links, predict them (capacity vs hop-count weights)
    ho = {}
    for wname in ("capacity", "hops"):
        lw, Aw, _, _ = routing(wname)
        Yw = np.column_stack([LHn[lk[l]].reindex(T).to_numpy() if l in lk else np.full(len(T), np.nan) for l in lw])
        errs = []
        sub = range(0, len(T), 24)
        for seed in range(5):
            obs = np.where(np.isfinite(Yw).all(axis=0))[0]
            drop = np.random.default_rng(seed).choice(obs, size=len(obs) // 5, replace=False)
            Yd = Yw.copy()
            Yd[:, drop] = np.nan
            Xd = np.array([tomography.em(Aw, Yd[h], In[h], Eg[h]) for h in sub])
            P_ = (Aw[drop] @ Xd.T).T
            errs.append(float(np.abs(P_ - Yw[list(sub)][:, drop]).sum() / Yw[list(sub)][:, drop].sum()))
        ho[wname] = float(np.mean(errs))
    log(f"[carbon] hold-out link WAPE {ho}")

    # resources x OD: routers (A_node over ends, all core routers), spans (links mapped to spans)
    rix = {r: i for i, r in enumerate(core)}
    res_names = list(E.columns)
    S = sp.lil_matrix((len(res_names), len(lks)))
    for j, (u, v) in enumerate(lks):
        s = tuple(sorted((u, v)))
        if pop_of[u] != pop_of[v]:
            S[res_names.index(f"{s[0]}|{s[1]}"), j] = 1.0
    Rn = sp.lil_matrix((len(res_names), len(core)))
    for r in core:
        Rn[res_names.index(r), rix[r]] = 1.0
    A_res = (Rn.tocsr() @ A_node + S.tocsr() @ A_link).tocsr()
    Eo = E.loc[T].to_numpy()
    flat = (In + Eg) / (In + Eg).sum(1, keepdims=True) * Eo.sum(1, keepdims=True)
    path_em = path_rate.node_bills(A_res, Eo, X_em, nE)
    path_gr = path_rate.node_bills(A_res, Eo, X_gr, nE)

    def wape(b, ref):
        return float(np.abs(b.sum(0) - ref.sum(0)).sum() / ref.sum())

    # hop-count routing: path bills under another routing model
    lh_, Ah_, Anh_, _ = routing("hops")
    Sh = sp.lil_matrix((len(res_names), len(lh_)))
    for j, (u, v) in enumerate(lh_):
        s = tuple(sorted((u, v)))
        if pop_of[u] != pop_of[v]:
            Sh[res_names.index(f"{s[0]}|{s[1]}"), j] = 1.0
    Yh = np.column_stack([LHn[lk[l]].reindex(T).to_numpy() if l in lk else np.full(len(T), np.nan) for l in lh_])
    X_h = np.array([tomography.em(Ah_, Yh[h], In[h], Eg[h]) for h in range(len(T))])
    X_h[X_h < TINY] = 0.0
    path_hops = path_rate.node_bills((Rn.tocsr() @ Anh_ + Sh.tocsr() @ Ah_).tocsr(), Eo, X_h, nE)

    # the test-period setup, for the day-ahead path-rate forecast (path_dayahead.py)
    np.savez_compressed(OUT / "test_setup.npz", T=T.asi8, In=In, Eg=Eg, Y=Y, X_em=X_em, E=Eo,
                        E_da=E_dayahead.loc[T].to_numpy(), ends=np.array(ends),
                        lks=np.array([f"{u}->{v}" for u, v in lks]), res=np.array(res_names))
    sp.save_npz(OUT / "A_res.npz", A_res)
    sp.save_npz(OUT / "A_link.npz", A_link.tocsr())
    RB = pd.DataFrame({"router": [r.replace(i2.FQDN, "") for r in ends], "GB_in_out": (In + Eg).sum(0) * GB,
                       "flat_g": flat.sum(0), "path_g": path_em.sum(0)})
    RB["path_over_flat"] = RB.path_g / RB.flat_g
    RB.to_csv(OUT / "router_bills.csv", index=False)

    # per-OD usage rates (g per GB), bytes-weighted, and resource rates
    used = (A_res @ X_em.T).T
    per_u = np.divide(Eo, used, out=np.zeros_like(Eo), where=used > 0)
    rate_od = (A_res.T @ per_u.T).T / GB
    wts = X_em.sum(0)
    odr = pd.DataFrame({"s": [a for a, _ in od], "t": [b for _, b in od], "GB": wts * GB,
                        "g_per_GB": (rate_od * X_em).sum(0) / np.maximum(wts, 1e-12)})
    odr = odr[odr.GB > 0]
    odr.to_csv(OUT / "od_rates.csv", index=False)
    carried = used.mean(0) >= 1.0                                                  # >= 1 Mbit/s on average
    res_rate = pd.Series(Eo.sum(0) / np.maximum(used.sum(0) * GB, 1e-12), index=res_names)[carried]
    span_util = []
    for j, (u, v) in enumerate(lks):
        c = links.set_index(["u", "v"]).capacity_bps.get((u, v), np.nan)
        span_util.append(np.nanmean(Y[:, j]) * 1e6 / c if c and np.isfinite(c) else np.nan)
    util = pd.Series(span_util, index=[f"{a}->{b}" for a, b in lks])
    # emission-weighted utilisation of the spans: each span's mean utilisation over its two directions
    su, sw_ = [], []
    for s in spans:
        us = [util.get(f"{s[0]}->{s[1]}"), util.get(f"{s[1]}->{s[0]}")]
        us = [x for x in us if x is not None and np.isfinite(x)]
        if us:
            su.append(np.mean(us))
            sw_.append(E[f"{s[0]}|{s[1]}"].loc[T].sum())
    util_w = float(np.average(su, weights=sw_)) if su else None

    # --- power vs traffic: does a PoP's router-side power follow the bits it switches? ----------
    port_tr = (pd.concat([DI, DO]).groupby(level=0).sum()).T.groupby(door_router.reindex(DI.columns).map(pop_of)).sum().T
    bb_tr = LH.T.groupby([pop_of.get(full(c.split("->")[0])) for c in LH.columns]).sum().T \
        .add(LH.T.groupby([pop_of.get(full(c.split("->")[1])) for c in LH.columns]).sum().T, fill_value=0)
    sw = port_tr.add(bb_tr, fill_value=0).reindex(H)                               # Mbit/s through each PoP
    dep = []
    Wraw = pd.read_csv(DATA / "power_hourly.csv.gz", parse_dates=["time"])
    Wraw = Wraw[Wraw.dc & Wraw["pop"].isin(core_pops) & ~Wraw.type.str.startswith("dwdm")]
    PR = Wraw.pivot_table(index="time", columns="pop", values="watts", aggfunc="sum").reindex(H)
    for p in core_pops:
        if p not in PR or p not in sw:
            continue
        d = pd.DataFrame({"W": PR[p], "gbps": sw[p] / 1e3}).dropna()
        d = d[d.W > 0]
        if len(d) < 24 * 30:
            continue
        x, y = d.gbps.to_numpy(), d.W.to_numpy()
        b, a = np.polyfit(x, y, 1)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if x.std() > 0 and y.std() > 0 else 0.0
        dep.append(dict(pop=p, hours=len(d), mean_W=float(y.mean()), mean_gbps=float(x.mean()),
                        r2=r2, slope_W_per_gbps=float(b), traffic_share_of_power=float(b * x.mean() / y.mean())))
    DEP = pd.DataFrame(dep).sort_values("r2", ascending=False)
    DEP.to_csv(OUT / "dependence.csv", index=False)

    # --- day-ahead forecast of the total, the flat rate a day ahead -----------------------------
    # the total over the whole hourly record (from 2024-11), so the models see history before YEAR0
    full_idx = pd.date_range(DI.index.min(), pd.Timestamp(YEAR1, tz="UTC"), freq="h", inclusive="left")
    ing_all = DI.T.groupby(door_router.reindex(DI.columns)).sum().T
    Vh = ing_all[ends].sum(axis=1).where(complete)
    Vh = Vh.where(sane(Vh)).reindex(full_idx)
    rows, fc = [], {}
    days = pd.date_range(TEST0, pd.Timestamp(TEST1) - pd.Timedelta(days=1), freq="D", tz="UTC")
    s = Vh.interpolate(limit=24).ffill()
    fc["persistence (same hour yesterday)"] = s.shift(24)
    fc["same hour last week"] = s.shift(168)
    fc["mean of the same hour, last 4 weeks"] = sum(s.shift(168 * k) for k in range(1, 5)) / 4
    fc["GBDT"] = gbdt_dayahead(s, days)
    try:
        from green_routing.carbon import dayahead
        fc["Chronos-2, no training"] = dayahead.foundation(s.dropna(), days, model="chronos-2", context=2048)
    except Exception as e:                       # noqa: BLE001  (no GPU / package: skip, say so)
        log(f"[carbon] Chronos-2 skipped: {e}")
    idx = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h")
    good = Vh.reindex(idx).notna() & E.sum(axis=1).reindex(idx).notna()
    for f in fc.values():
        good &= f.reindex(idx).notna()
    hh = idx[good.to_numpy()]
    Et, Vt = E.sum(axis=1).reindex(hh), Vh.reindex(hh)
    E_fc = E_dayahead.sum(axis=1).reindex(hh)
    E_fc = E_fc if E_fc.notna().all() else None
    r_true = Et / (Vt * GB)
    for name, f in fc.items():
        p = f.reindex(hh)
        rr = Et / (p * GB)
        row = dict(model=name, traffic_hourly_WAPE=float((p - Vt).abs().sum() / Vt.sum()),
                   traffic_bias=float((p - Vt).sum() / Vt.sum()),
                   rate_WAPE_actual_grid=float((rr - r_true).abs().sum() / r_true.sum()))
        if E_fc is not None:
            rf = E_fc / (p * GB)
            row["rate_WAPE_grid_forecast"] = float((rf - r_true).abs().sum() / r_true.sum())
        rows.append(row)
    if E_fc is not None:
        rows.append(dict(model="actual traffic", traffic_hourly_WAPE=0.0, traffic_bias=0.0, rate_WAPE_actual_grid=0.0,
                         rate_WAPE_grid_forecast=float((E_fc / (Vt * GB) - r_true).abs().sum() / r_true.sum())))
    pd.DataFrame({k: v.reindex(hh) for k, v in fc.items()} | {"actual": Vt, "E_g_per_h": Et,
                  "E_dayahead_g_per_h": E_dayahead.sum(axis=1).reindex(hh)}).to_csv(OUT / "dayahead_hourly.csv")
    RE = pd.DataFrame(rows).sort_values("traffic_hourly_WAPE")
    RE.to_csv(OUT / "rate_errors.csv", index=False)

    # --- summary -------------------------------------------------------------------------------
    yr = H[ok.to_numpy()]
    site_kW = (P_router.reindex(yr).sum(axis=1) + P_opt.reindex(yr).sum(axis=1) + P_huts.reindex(yr)) * PUE["central"] / 1e3
    summary = {
        "year": [YEAR0, YEAR1], "test": [TEST0, TEST1], "test_hours": int(len(T)), "year_hours": int(len(yr)),
        "core_routers": len(core), "core_pops": len(core_pops), "routers_with_doors": nE, "doors": int(len(D)),
        "parties": int(D.party.nunique()), "directed_links": len(lks), "spans": len(spans),
        "power_systems_used": int(len(live)), "huts_and_other_sites": other_sites,
        "pops_without_grid_factor": pops_no_grid,
        "it_kW_core_pops": float(P_pop.reindex(yr).sum(axis=1).mean() / 1e3),
        "it_kW_imputed_pops": float(P_pop[missing].reindex(yr).sum(axis=1).mean() / 1e3) if missing else 0.0,
        "pops_imputed": missing, "W_per_core_router_median": float(per_router), "optical_share_assumed": OPTICAL_SHARE,
        "it_kW_router_side": float(P_router.reindex(yr).sum(axis=1).mean() / 1e3),
        "it_kW_optical_at_pops": float(P_opt.reindex(yr).sum(axis=1).mean() / 1e3),
        "it_kW_other_sites": float(P_huts.reindex(yr).mean() / 1e3),
        "site_kW": float(site_kW.mean()), "site_kW_pue_low_high": [float(site_kW.mean() * PUE["low"] / PUE["central"]),
                                                                   float(site_kW.mean() * PUE["high"] / PUE["central"])],
        "g_per_kWh_power_weighted": float(E.reindex(yr).sum().sum() / (site_kW.sum())),     # g/h over kW = g/kWh
        "emissions_t_year": float(E.reindex(yr).sum().sum() / 1e6),
        "PB_year": float(V.reindex(yr).sum() * GB / 1e6), "mean_total_gbps": float(V.reindex(yr).mean() / 1e3),
        "flat_rate_g_per_GB": float(E.reindex(yr).sum().sum() / (V.reindex(yr).sum() * GB)),
        "flat_rate_hour_p10_p90": [float(flat_rate.reindex(yr).quantile(.1)), float(flat_rate.reindex(yr).quantile(.9))],
        "test_emissions_t": float(Eo.sum() / 1e6), "test_PB": float(In.sum() * GB / 1e6),
        "test_flat_rate_g_per_GB": float(Eo.sum() / (In.sum() * GB)),
        "path_usage_g_per_GB_p10_p50_p90": [wq(odr.g_per_GB, odr.GB, p) for p in (.1, .5, .9)],
        "resource_rate_max_g_per_GB": float(res_rate.max()), "resource_rate_max_name": str(res_rate.idxmax()),
        "utilisation_link_median": float(util.median()),
        "utilisation_emission_weighted": util_w,
        "flat_vs_path_WAPE": wape(flat, path_em), "moved_share_M": wape(flat, path_em) / 2,
        "gravity_vs_em_path_WAPE": wape(path_gr, path_em), "hops_vs_capacity_path_WAPE": wape(path_hops, path_em),
        "holdout_link_WAPE": ho,
        "dependence_median_r2": float(DEP.r2.median()) if len(DEP) else None,
        "dependence_pops_r2_below_0.01": int((DEP.r2 < 0.01).sum()) if len(DEP) else None,
        "dependence_pops": int(len(DEP)),
        "dependence_max_r2": float(DEP.r2.max()) if len(DEP) else None,
        "dependence_median_traffic_share": float(DEP.traffic_share_of_power.median()) if len(DEP) else None,
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    log(json.dumps(summary, indent=1))
    log(RE.round(4).to_string(index=False))
    log(DEP.head(10).round(4).to_string(index=False))


def gbdt_dayahead(s: pd.Series, days: pd.DatetimeIndex, refit_every: int = 7) -> pd.Series:
    """A plain direct GBDT: per target hour, the same hour 1..7 and 14, 21, 28 days back, the last
    week's mean, hour, weekday and a US federal-holiday flag; refitted weekly on all known days."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    hol = set(pd.to_datetime(US_HOLIDAYS).date)
    y = np.log(s.clip(lower=1))
    t = s.index
    feats = {f"lag{k}d": y.shift(24 * k) for k in (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)}
    # a forecast issued at 00:00 UTC knows nothing of the target day: shift day-1 lags to the issue time
    issue = t.floor("D")
    prev = y.shift(1)                                                            # last hour before the issue time
    known = prev.groupby(issue).transform("first")
    X = pd.DataFrame(feats)
    X["week_mean"] = prev.rolling(168, min_periods=100).mean().reindex(issue).to_numpy()
    X["issue_level"] = known.to_numpy()
    X["hour"] = t.hour
    X["dow"] = t.dayofweek
    X["holiday"] = [d in hol for d in t.date]
    base = X["week_mean"]
    yt = y - base
    out = pd.Series(np.nan, index=t)
    for k in range(0, len(days), refit_every):
        d0 = days[k]
        tr = (t < d0 - pd.Timedelta(hours=24)) & yt.notna() & X.notna().all(axis=1)
        te = (t >= d0) & (t < days[min(k + refit_every, len(days) - 1)] + pd.Timedelta(days=1)) & X.notna().all(axis=1)
        if tr.sum() < 24 * 60 or te.sum() == 0:
            continue
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)
        m.fit(X[tr], yt[tr])
        out[te] = np.exp(m.predict(X[te]) + base[te])
    return out


if __name__ == "__main__":
    main()
