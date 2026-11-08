"""Numbers for the fact check of the paper's text (REPORT.md, section 5): values the other scripts do not
write, each on a stated window.

- energy and carbon per byte on common windows (both networks), the annual window of Table 1 / Fig. 1;
- SWITCHlan's site power by window and pool (dedicated access vs shared backbone);
- the within-day rate (hour of day of the median rate vs of the traffic);
- the ten largest customers' share, by party (all parties of the allocation);
- the path bills summed over routers (Internet2) against the emissions.

usage: python factcheck.py        writes $GR_ARTIFACTS_DIR/paper/factcheck.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR, reference_dir

HERE = Path(__file__).resolve().parent
P = ARTIFACTS_DIR / "paper"
C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
I2D, I2C = ARTIFACTS_DIR / "internet2" / "data", ARTIFACTS_DIR / "internet2" / "carbon"
GB_PER_MBPS_H = 3600 / 8 / 1000

WINDOWS = {"switch_year_to_block_end": ("2025-07-11", "2026-07-10"), "switch_test": ("2026-03-22", "2026-07-10"),
           "calendar_2025": ("2025-01-01", "2025-12-31"), "overlap": ("2025-11-15", "2026-07-10"),
           "i2_year_to_block_end": ("2025-11-15", "2026-10-31"), "i2_test": ("2026-07-19", "2026-10-31")}


def hours(a, b):
    return pd.date_range(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC") + pd.Timedelta(hours=23), freq="h")


def switch_window(a, b) -> dict:
    """Site power (access / backbone, with PUE), energy, emissions (actual CH/FR grid), traffic and per-GB
    figures of SWITCHlan over [a, b]."""
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    import run as R
    from green_routing.carbon import bills, grid
    H = hours(a, b)
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    tiers = pd.read_csv(NET / "routers.csv").set_index("dev").tier
    ACC, Pbb, info = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"),
                                        pd.read_csv(C / "allocation" / "device_pools.csv"),
                                        pd.read_csv(C / "inventory" / "shelves.csv"),
                                        pd.read_csv(reference_dir() / "sites.csv"), doors, tiers, H)
    zones = sorted(set(Pbb) | {z for _, z in ACC.index})
    I = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in zones})
    acc_z = {z: float(sum(v for (_, zz), v in ACC.items() if zz == z)) for z in zones}
    bb_z = {z: float(np.nanmean(np.asarray(Pbb[z], float))) for z in zones}
    E_acc = sum(acc_z[z] / 1e3 * I[z].sum() for z in zones)            # g
    E_bb = sum(bb_z[z] / 1e3 * I[z].sum() for z in zones)
    kw_acc, kw_bb = sum(acc_z.values()) / 1e3, sum(bb_z.values()) / 1e3
    T = R.T.reindex(H)
    gb = float(T.mean() * 1e3 * GB_PER_MBPS_H * len(H))                 # mean one-way Gbit/s over the window x hours
    return {"window": [a, b], "hours": len(H), "traffic_hours_present": int(T.notna().sum()),
            "site_kW_access": kw_acc, "site_kW_backbone": kw_bb, "site_kW_total": kw_acc + kw_bb,
            "info_routers_kW": info["routers_kW"], "info_optical_kW": info["optical_kW"],
            "emissions_t_total": (E_acc + E_bb) / 1e6, "emissions_t_backbone": E_bb / 1e6,
            "mean_factor_g_per_kWh_total": (E_acc + E_bb) / ((kw_acc + kw_bb) * len(H)),
            "mean_factor_g_per_kWh_backbone": E_bb / (kw_bb * len(H)),
            "CH_mean_g_per_kWh": float(I["CH"].mean()), "FR_mean_g_per_kWh": float(I["FR"].mean()),
            "mean_traffic_gbps": float(T.mean()), "PB": gb / 1e6,
            "Wh_per_GB_backbone": kw_bb * len(H) / gb * 1e3, "Wh_per_GB_total": (kw_acc + kw_bb) * len(H) / gb * 1e3,
            "g_per_GB_backbone": E_bb / gb, "g_per_GB_total": (E_acc + E_bb) / gb}


def i2_site_power() -> pd.Series:
    """Internet2's hourly site power (kW, PUE 1.5) as recipes/internet2/carbon.py computes it."""
    sys.path.insert(0, str(HERE.parent / "internet2"))
    import carbon as IC
    from green_routing.dataset import internet2 as i2
    links, LH, D, DI, DO, W = IC.load()
    H = pd.date_range(IC.YEAR0, IC.YEAR1, freq="h", inclusive="left", tz="UTC")
    core = sorted(set(links.u) | set(links.v) | set(D.node))
    pop_of = {r: i2.pop_of(r) for r in core}
    core_pops = sorted(set(pop_of.values()))
    W = W[W.dc & W["pop"].notna()]
    per_sys = W.pivot_table(index="time", columns="node", values="watts", aggfunc="sum").reindex(H)
    med = per_sys.median()
    live = med[med > 0].index
    per_sys = per_sys[live].fillna(per_sys[live].median())
    popn = pd.Series({n: (i2.pop_of(n) or ("manlan" if "manlan" in n else None)) for n in live})
    at_core = popn.isin(core_pops)
    P_pop = per_sys.T.groupby([popn[at_core]]).sum().T
    n_rt = pd.Series({p: sum(1 for r in core if pop_of[r] == p) for p in core_pops})
    per_router = (P_pop.mean() / n_rt.reindex(P_pop.columns)).median()
    missing = [p for p in core_pops if p not in P_pop.columns]
    imputed = per_router * n_rt[missing].sum()
    huts = per_sys.loc[:, ~at_core.to_numpy()].sum(axis=1)
    return (P_pop.sum(axis=1) + imputed + huts) * IC.PUE["central"] / 1e3


def i2_window(a, b, site_kw: pd.Series) -> dict:
    fr = pd.read_csv(I2C / "flat_rate_hourly.csv", index_col=0, parse_dates=True)
    H = hours(a, b)
    d = fr.reindex(H)
    ok = d.flat_g_per_GB.notna()
    kw = site_kw.reindex(H)
    gb = float(d.V_mbps[ok].mean() * GB_PER_MBPS_H * len(H))          # mean over complete hours x hours
    E = float(d.E_g_per_h.sum())
    return {"window": [a, b], "hours": len(H), "hours_complete": int(ok.sum()), "site_kW_mean": float(kw.mean()),
            "emissions_t": E / 1e6, "mean_factor_g_per_kWh": E / float(kw.sum()),
            "mean_traffic_gbps": float(d.V_mbps[ok].mean() / 1e3), "PB": gb / 1e6,
            "Wh_per_GB": float(kw.mean()) * len(H) / gb * 1e3, "g_per_GB": E / gb,
            "flat_rate_hours_complete_g_per_GB": float(d.E_g_per_h[ok].sum() / (d.V_mbps[ok].sum() * GB_PER_MBPS_H))}


def within_day() -> dict:
    out = {}
    for net, tz in (("switch", "Europe/Zurich"), ("i2", "America/Chicago")):
        h = pd.read_csv(P / f"hourly_year_{net}.csv", index_col=0, parse_dates=True)
        loc = h.index.tz_convert(tz).hour
        r, t = h.r.groupby(loc).median(), h.T_gbps.groupby(loc).median()
        peak = int(t.idxmax())
        out[net] = {"rate_dearest_hour": int(r.idxmax()), "rate_cheapest_hour": int(r.idxmin()),
                    "traffic_peak_hour": peak, "traffic_trough_hour": int(t.idxmin()),
                    "rate_max_over_min": float(r.max() / r.min()),
                    "rate_max_over_rate_at_traffic_peak": float(r.max() / r[peak]),
                    "rate_at_traffic_trough_over_rate_at_traffic_peak": float(r[int(t.idxmin())] / r[peak]),
                    "night_00_06_median_over_peak_hour": float(r[[0, 1, 2, 3, 4, 5]].median() / r[peak]),
                    "traffic_peak_over_trough": float(t.max() / t.min())}
    return out


def top_customers() -> dict:
    """Ten largest parties' share of the bills: as the paper's Fig. 1 (entity_bills.csv, flat rate, parties
    with traffic) and over every party of the allocation (SWITCHlan: party_bills.csv over the test block,
    settled, access + backbone; Internet2: every door party)."""
    sw = pd.read_csv(P / "switch" / "entity_bills.csv", index_col=0)
    i2 = pd.read_csv(P / "i2" / "entity_bills.csv", index_col=0)
    pb = pd.read_csv(C / "bills" / "party_bills.csv", index_col=0)
    i2d = pd.read_csv(I2D / "doors.csv")
    top = lambda s: float(s.nlargest(10).sum() / s.sum())                      # noqa: E731
    i2_nr = i2[~i2.index.str.startswith("router ")]
    return {"switch_entities_fig1": int(len(sw)), "switch_top10_fig1_flat": top(sw.flat_kg_per_year),
            "switch_top10_fig1_path": top(sw.path_kg_per_year),
            "switch_parties_all": int(len(pb)), "switch_parties_with_bill": int((pb.settled_g > 0).sum()),
            "switch_top10_all_parties_settled": top(pb.settled_g),
            "i2_entities_fig1": int(len(i2)), "i2_parties_all": int(i2d.party.nunique()),
            "i2_top10_fig1_flat": top(i2.flat_kg_per_year), "i2_top10_fig1_path": top(i2.path_kg_per_year),
            "i2_aggregation_router_parties": int(i2.index.str.startswith("router ").sum()),
            "i2_top10_without_aggregation_routers_flat": float(i2_nr.flat_kg_per_year.nlargest(10).sum() / i2.flat_kg_per_year.sum()),
            "i2_top10_names": i2.flat_kg_per_year.nlargest(10).index.tolist()}


def path_bill_sum() -> dict:
    """Internet2: posted path rates x actual traffic, summed over router bills (i2_rates.router_bills), vs E."""
    sys.path.insert(0, str(HERE))
    import scipy.sparse as sp
    import i2_rates as IR
    R = IR.load_rates()
    A_res = sp.load_npz(IR.OUT / "A_res.npz").tocsr()
    ok, n = R["ok"], len(R["ends"])
    post = IR.router_bills(R["X"][ok], IR.od_costs(R["per_post"][ok], R["add_post"][ok], A_res), n)
    sett = IR.router_bills(R["X"][ok], IR.od_costs(R["per_set"][ok], R["add_set"][ok], A_res), n)
    E = R["E"][ok].sum()
    return {"posted_path_bills_over_E": float(post.sum() / E), "settled_path_bills_over_E": float(sett.sum() / E),
            "hours": int(ok.sum())}


def window_definitions() -> dict:
    """B1: the bill's 30-day and 7-day errors under three window definitions (SWITCHlan test block, no margin),
    and why overlapping windows can fall below the block's bias; the same for the corrected grid forecast."""
    sys.path.insert(0, str(HERE))
    import margin as M
    sw = M.switch_hours()
    D = M.daily(sw[sw.period == "TEST"], 0.0)
    n, k = len(D), 30
    w = np.array([min(i + 1, k, n - i, n - k + 1) for i in range(n)])
    mm = D.index.tz_localize(None).to_period("M")
    g = D.groupby(mm)
    W = M.windows(D, 30)
    out = {"days": n, "block_bias": float(D.bill.sum() / D.E.sum() - 1),
           "rolling30_windows": int(len(W)), "rolling30_windows_overbilled": int((W.bill > W.E).sum()),
           "rolling30_wape": M.wape(W), "rolling30_day_weighted_bias": float(((D.bill - D.E) * w).sum() / (D.E * w).sum()),
           "tiling30_from_day1": [float(b / e - 1) for b, e in zip(M.windows(D, 30, 30).bill, M.windows(D, 30, 30).E)],
           "tiling30_from_day1_wape": M.wape(M.windows(D, 30, 30)),
           "tiling30_last90_wape": M.wape(M.windows(D.iloc[n - 90:], 30, 30)),
           "months_bias": {str(p): float(v) for p, v in (g.bill.sum() / g.E.sum() - 1).items()},
           "months_E_share": {str(p): float(v) for p, v in (g.E.sum() / D.E.sum()).items()},
           "months_full_wape": M.scores(D)["err_month_calendar"],
           "months_all_incl_partial_wape": float((g.bill.sum() - g.E.sum()).abs().sum() / g.E.sum().sum()),
           "rolling7_wape": M.wape(M.windows(D, 7)), "tiling7_wape": M.wape(M.windows(D, 7, 7)),
           "tiling7_windows": int(len(M.windows(D, 7, 7)))}
    G3 = pd.read_csv(P / "appendix" / "grid_hourly_three_periods.csv.gz", index_col=0, parse_dates=True)
    gt = G3[G3.period == "test"]
    Dg = pd.DataFrame({"bill": gt.median_corr.groupby(gt.index.floor("D")).sum(), "E": gt.actual.groupby(gt.index.floor("D")).sum()})
    out["grid_corrected_median_test"] = {"bias": float(Dg.bill.sum() / Dg.E.sum() - 1), "rolling30_wape": M.wape(M.windows(Dg, 30)),
                                         "tiling30_wape": M.wape(M.windows(Dg, 30, 30)),
                                         "months_full_wape": M.scores(Dg)["err_month_calendar"]}
    return out


def steering_stats() -> dict:
    """B2 / B3: the controller's asleep and dark spans; dark-span statistics of the closed loop for every k."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "green_paths"))
    import i2_rates as IR
    import simulate as SIM
    c = json.load(open(P / "i2" / "central.json"))
    R, S = IR.load_rates(), SIM.Net()
    ok = R["ok"]
    E = R["E"][ok]
    spans = [r for r, s in zip(S.res, S.is_span) if s]
    Uspan = R["U"][ok][:, [S.res.index(s) for s in spans]]
    dark_not_asleep = [s for s in spans if s not in c["asleep_spans"] and Uspan[:, spans.index(s)].mean() < 1e-3]
    out = {"central": {"asleep": c["asleep"], "dark": c["dark"], "dark_E_share": c["dark_E_share"],
                       "asleep_E_share": float(E[:, [S.res.index(s) for s in c["asleep_spans"]]].sum() / E.sum()),
                       "dark_not_asleep": dark_not_asleep, "hops_per_byte": c["hops_per_byte"]}}
    for k, tag in (("1", "_k1.0"), ("2", ""), ("3", "_k3.0"), ("none", "_kNone")):
        D = pd.read_csv(P / "i2" / f"everyone_hourly{tag}.csv", parse_dates=["hour"]).set_index("hour")
        s = pd.read_csv(P / "i2" / f"everyone_spans{tag}.csv", index_col=0).share_hours_dark
        es = E[:, [S.res.index(r) for r in s.index]].sum(0) / E.sum()
        t0 = D.index[0].floor("D")
        a = D[D.index >= t0 + pd.Timedelta(days=14)]
        out[f"everyone_k{k}"] = {"block_mean": float(D.spans_dark.mean()), "block_median": float(D.spans_dark.median()),
                                 "mean_after_14_days": float(a.spans_dark.mean()), "first_day_mean": float(D.spans_dark.iloc[:24].mean()),
                                 "dark_ge90": int((s >= 0.9).sum()), "dark_ge90_E_share": float(es[(s >= 0.9).to_numpy()].sum()),
                                 "dark_all_hours": int((s == 1).sum()), "hourly_dark_E_share_mean": float(D.dark_E_share.mean()),
                                 "util_max": float(D.util_max.max()), "links_over_80_max": int(D.links_over_80.max()),
                                 "today_spans_dark_mean": float(D.today_spans_dark.mean()),
                                 "hops_today": float(D.today_hops_per_byte.mean()), "hops": float(D.hops_per_byte.mean()),
                                 "bytes_paying_more": float(D.bytes_paying_more.mean())}
    u = json.load(open(P / "i2" / "unilateral.json"))
    out["unilateral"] = {"reduction_forecast_bytes": u["bytes_weighted_reduction_forecast"],
                         "reduction_oracle_bytes": u["bytes_weighted_reduction_oracle"],
                         "captured": u["bytes_weighted_reduction_forecast"] / u["bytes_weighted_reduction_oracle"],
                         "same_choice_bytes": u["bytes_same_choice"], "pairs_ge1": u["pairs_reduction_forecast_ge_1pct"],
                         "pairs_ge20": u["pairs_reduction_forecast_ge_20pct"]}
    return out


def cap_and_rates() -> dict:
    """The non-robust cap (mean + 2 sd of the pooled log rates, previous 28 days) for comparison, the skew,
    rate multiples of nearly idle resources, rate errors with the margin, and the split of the posted path bill."""
    sys.path.insert(0, str(HERE))
    import scipy.sparse as sp
    from scipy.stats import skew
    import i2_rates as IR
    R = IR.load_rates()
    Pp = np.load(IR.PRE / "test_setup.npz")
    A_res = sp.load_npz(IR.OUT / "A_res.npz").tocsr()
    E = np.r_[Pp["E"], R["E"]]
    U = np.r_[(A_res @ Pp["X_em"].T).T, R["U"]]
    T = pd.to_datetime(np.r_[Pp["T"], R["T"]], utc=True)
    days = T.floor("D")
    robust, classic, sk = [], [], []
    for d in days[len(Pp["T"]):].unique():
        w = (days >= d - pd.Timedelta(days=28)) & (days < d)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.log10(E[w] / U[w] / (E[w].sum(1) / U[w].sum(1))[:, None])
        v = rel[(U[w] >= 1e-3) & (E[w] > 0) & np.isfinite(rel)]
        robust.append(10 ** (np.median(v) + 2 * 1.4826 * np.median(np.abs(v - np.median(v)))))
        classic.append(10 ** (v.mean() + 2 * v.std()))
        sk.append(skew(v))
    fr = IR.flat_rate(R)
    conv = float(np.median((R["E"].sum(1) / R["U"].sum(1)) / fr))
    ok = R["ok"]
    Eo, Uo, Xo = R["E"][ok], R["U"][ok], R["X"][ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (Eo / Uo) / (Eo.sum(1) / Xo.sum(1))[:, None]
    util = Uo / IR.res_capacity()[None, :]
    idle = (Uo > 1e-3) & (Eo > 0) & (util < 1e-4)
    out = {"robust_cap_over_flat_median": float(np.median(robust)) * conv, "classic_cap_over_flat_median": float(np.median(classic)) * conv,
           "log_rate_skew_median": float(np.median(sk)),
           "rate_over_flat_load_below_0.01pct_p50_p90": np.quantile(ratio[idle], [.5, .9]).tolist()}
    for m in (0.0, 0.005):
        R2 = dict(R)
        R2["per_post"], R2["add_post"], R2["E_da"] = R["per_post"] * (1 - m), R["add_post"] * (1 - m), R["E_da"] * (1 - m)
        pe = IR.path_rate_error(R2, A_res)
        out[f"rate_errors_m{m}"] = {"path_bytes_wape": pe["path_rate_wape_bytes_capped"], "flat_hourly_wape": pe["flat_rate_hourly_wape"]}
    out["posted_path_bill_split_over_E"] = {"resource_part": float((R["per_post"][ok] * Uo).sum() / Eo.sum()),
                                            "flat_addon_part": float((R["add_post"][ok] * Xo.sum(1)).sum() / Eo.sum()),
                                            "resource_loads_forecast_bias": float(R["U_fc"][ok].sum() / Uo.sum() - 1),
                                            "posted_emissions_bias": float(R["E_da"][ok].sum() / Eo.sum() - 1)}
    return out


def timing() -> dict:
    """Wall time on this machine (CPU, numpy) for one day of Internet2's path rates after the link forecasts:
    the 24 hourly EM matrices and the capped rates (with the 28-day cap)."""
    import time
    import scipy.sparse as sp
    sys.path.insert(0, str(HERE))
    import i2_rates as IR
    from green_routing.carbon import path_rate, tomography
    S = np.load(I2C / "test_setup.npz", allow_pickle=True)
    A_link = sp.load_npz(I2C / "A_link.npz").tocsr()
    A_res = sp.load_npz(I2C / "A_res.npz").tocsr()
    R = IR.load_rates()
    hs = range(240, 264)
    t_em, t_rate = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        [tomography.em(A_link, np.nan_to_num(S["Y"][h]), S["In"][h], S["Eg"][h]) for h in hs]
        t_em.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        mult = path_rate.cap_multiple(R["E"][:672], R["U"][:672])
        for h in hs:
            per, add = path_rate.capped_rates(R["E_da"][h], A_res @ R["X_fc"][h], R["X_fc"][h].sum(), mult)
            A_res.T @ per + add
        t_rate.append(time.perf_counter() - t0)
    return {"em_24h_s_min": min(t_em), "rates_and_cap_24h_s_min": min(t_rate), "device": "CPU (numpy/scipy)"}


def main():
    out = {}
    for k in ("switch_year_to_block_end", "switch_test", "calendar_2025", "overlap"):
        out[f"switch_{k}"] = switch_window(*WINDOWS[k])
    kw = i2_site_power()
    for k in ("i2_year_to_block_end", "i2_test", "overlap"):
        out[f"i2_{k}"] = i2_window(*WINDOWS[k], kw)
    out["within_day"] = within_day()
    out["top_customers"] = top_customers()
    out["path_bill_sum_i2"] = path_bill_sum()
    out["window_definitions"] = window_definitions()
    out["steering"] = steering_stats()
    out["cap_and_rates"] = cap_and_rates()
    out["timing"] = timing()
    json.dump(out, open(P / "factcheck.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
