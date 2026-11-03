"""Path-based carbon rate on SWITCHlan, from the counters, over the bill window.

usage: python path_rate.py
writes $GR_ARTIFACTS_DIR/carbon_rates/path/: summary.json, routing_check.csv,
party_rates.csv, by_type.csv, resources.csv, od_rates.csv
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from green_routing.carbon import assumptions as A, bills, grid, path_rate, tomography
from green_routing.network.roles import site_of
from green_routing.paths import ARTIFACTS_DIR, reference_dir

C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
OUT = ARTIFACTS_DIR / "carbon_rates" / "path"
OUT.mkdir(parents=True, exist_ok=True)
RUN = ARTIFACTS_DIR / "pred_year" / "merged"
GB = A.GB_PER_MBPS_HOUR

I = pd.read_csv(NET / "interfaces.csv")
doors = pd.read_csv(C / "allocation" / "doors.csv")
traffic = dict(np.load(NET / "traffic.npz"))
sites = pd.read_csv(reference_dir() / "sites.csv")
H = bills.window(RUN)
t0 = time.time()
net = path_rate.Network(I, doors, traffic, H)
print(f"graph: {len(net.nodes)} routers, {net.loads.shape[1]} directed links, {len(net.ends)} edge nodes "
      f"({len(net.hidden)} missing from the export); {time.time() - t0:.0f}s", flush=True)

# --- which routing explains the counters? fit, and prediction of links hidden from the estimator
sample = list(range(0, len(H), 37))                                  # 72 hours spread over the window
rows = []
for weight in (() if (OUT / "routing_check.csv").exists() else ("hops", "distance", "capacity")):
    net.route(weight, sites)
    Xs = net.estimate(hours=sample)
    fit = np.mean([np.abs(net.A_link @ Xs[h] - net.y[h]).sum() / net.y[h].sum() for h in sample])
    busy = np.where(net.y[sample].mean(0) > 100)[0]
    for seed in range(3):
        drop = np.random.default_rng(seed).choice(busy, size=len(busy) // 5, replace=False)
        Xd = net.estimate(hours=sample, drop_links=drop)
        ho = np.mean([np.abs(net.A_link[drop] @ Xd[h] - net.y[h][drop]).sum() / net.y[h][drop].sum() for h in sample])
        gr = np.mean([np.abs(net.A_link[drop] @ tomography.ipf(tomography.gravity(net.ingress[h], net.egress[h]) * net.allowed,
                                                               net.ingress[h], net.egress[h]) - net.y[h][drop]).sum()
                      / net.y[h][drop].sum() for h in sample])
        rows.append(dict(routing=weight, seed=seed, fit_WAPE=fit, holdout_WAPE=ho, holdout_WAPE_gravity_only=gr))
RC = pd.DataFrame(rows) if rows else pd.read_csv(OUT / "routing_check.csv")
RC.to_csv(OUT / "routing_check.csv", index=False)
print(RC.groupby("routing")[["fit_WAPE", "holdout_WAPE", "holdout_WAPE_gravity_only"]].mean().round(3).to_string(), flush=True)
best = RC.groupby("routing").holdout_WAPE.mean().idxmin()

# --- the traffic matrix for every hour, and the bills
net.route(best, sites)
t0 = time.time()
cache = OUT / f"tm_{best}.npz"
if cache.exists():
    X, Xg = np.load(cache)["X"], np.load(cache)["Xg"]
else:
    X = net.estimate()
    Xg = np.array([tomography.ipf(tomography.gravity(net.ingress[h], net.egress[h]) * net.allowed, net.ingress[h], net.egress[h])
                   for h in range(len(H))])
    np.savez_compressed(cache, X=X, Xg=Xg)
print(f"traffic matrices ({best} routing): {time.time() - t0:.0f}s", flush=True)
R, E = bills.window_units(pd.read_csv(C / "inventory" / "routers.csv"), pd.read_csv(C / "allocation" / "device_pools.csv"),
                          pd.read_csv(C / "inventory" / "shelves.csv"), sites,
                          pd.read_csv(NET / "routers.csv").set_index("dev").tier, H)
router_W, span_W, unplaced = path_rate.resources(net, R, E, doors)
IA = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in ("CH", "FR")})
al = path_rate.Allocation(net, router_W, span_W, unplaced, IA)
res = al.bills(X)
resg = al.bills(Xg)
resc = al.bills(X, basis="capacity")
rescg = al.bills(Xg, basis="capacity")

types = doors.groupby("party").ptype.first()
V = al.V.sum() * GB                                             # GB in + out per party over the window
P = pd.DataFrame({"type": types.reindex(al.V.columns).fillna("customer site"), "GB_in_out": V,
                  "flat_g": res["flat_g"].sum(), "path_g": res["path_g"].sum(), "path_g_gravity_matrix": resg["path_g"].sum(),
                  "pathcap_g": resc["path_g"].sum(), "pathcap_g_gravity_matrix": rescg["path_g"].sum()})
P = P[P.GB_in_out > 0]
P["flat_rate_g_per_GB"] = 2 * P.flat_g / P.GB_in_out             # g per GB moved (half paid at each end)
P["path_rate_g_per_GB"] = 2 * P.path_g / P.GB_in_out
P["path_over_flat"] = P.path_g / P.flat_g
P["pathcap_rate_g_per_GB"] = 2 * P.pathcap_g / P.GB_in_out
P["pathcap_over_flat"] = P.pathcap_g / P.flat_g
P.sort_values("path_g", ascending=False).rename_axis("party").to_csv(OUT / "party_rates.csv")
T = P.groupby("type")[["GB_in_out", "flat_g", "path_g", "pathcap_g"]].sum()
T["parties"] = P.groupby("type").size()
T["path_over_flat"] = T.path_g / T.flat_g
T["pathcap_over_flat"] = T.pathcap_g / T.flat_g
T.to_csv(OUT / "by_type.csv")

# --- resources: power, measured throughput, own rate (counters only)
rate = al.rate_counters() / GB                                   # g per GB crossing the resource
Etot = al.E.sum(0)
RS = pd.DataFrame({"kind": [k for k, _ in al.names], "name": [n for _, n in al.names],
                   "site": [site_of(n.split("-")[0]) for _, n in al.names],
                   "W": np.r_[router_W.to_numpy(), span_W.sum(axis=1).to_numpy()],
                   "throughput_gbps": np.nanmean(al.X_meas, axis=0) / 1e3,
                   "emissions_g": Etot, "rate_g_per_GB_median": np.nanmedian(rate, axis=0),
                   "capacity_gbps": al.capacity / 1e3,
                   "utilisation": np.nanmean(al.X_meas, axis=0) / al.capacity,
                   "capacity_rate_g_per_GB_median": np.nanmedian(al.E / al.capacity[None, :], axis=0) / GB})
RS.to_csv(OUT / "resources.csv", index=False)

# --- path rates between edge nodes (g per GB moved), bytes-weighted over the window
odr = pd.DataFrame({"s": [s for s, _ in net.od], "t": [t for _, t in net.od],
                    "g_per_GB": res["od_g_per_GB"].to_numpy(), "cap_g_per_GB": resc["od_g_per_GB"].to_numpy(),
                    "GB": res["od_GB"].to_numpy()})
odr = odr[odr.GB > 0]
odr.to_csv(OUT / "od_rates.csv", index=False)

flat_rate = res["total_g"].sum() / (al.V.sum().sum() * GB / 2)
def wq(values, weights, p):
    """Weighted quantile (weights: GB moved)."""
    o = np.argsort(values)
    c = np.cumsum(np.asarray(weights, float)[o])
    return float(np.interp(p * c[-1], c, np.asarray(values, float)[o]))
summary = {
    "window": [str(H[0]), str(H[-1])], "routers": len(net.nodes), "directed_links": int(net.loads.shape[1]),
    "edge_nodes": len(net.ends), "hidden_edge_nodes": len(net.hidden), "routing": best,
    "routing_check": RC.groupby("routing")[["fit_WAPE", "holdout_WAPE", "holdout_WAPE_gravity_only"]].mean().to_dict("index"),
    "fit_WAPE_all_hours": float(np.mean([np.abs(net.A_link @ X[h] - net.y[h]).sum() / net.y[h].sum() for h in range(len(H))])),
    "local_share": float(np.mean([np.trace(X[h].reshape(len(net.ends), -1)) / X[h].sum() for h in range(len(H))])),
    "unseen_share_of_ingress": float(net.ed_in[[c for c in net.ed_in if c.startswith("unseen:")]].sum().sum() / net.ed_in.sum().sum()),
    "isolated_door_traffic_share": float(doors[doors.link.isin(net.isolated_doors)].gbps.sum() / doors.gbps.sum()),
    "backbone_t": float(res["total_g"].sum() / 1e6), "path_t": float(res["path_g"].sum().sum() / 1e6),
    "flat_t": float(res["flat_g"].sum().sum() / 1e6),
    "spread_flat_share": float(res["unallocated_g"].sum() / res["total_g"].sum()),
    "unseen_cost_share": float(res["unseen_g"].sum() / res["total_g"].sum()),
    "router_share_of_backbone": float(router_W.sum() / (router_W.sum() + span_W.to_numpy().sum() + sum(unplaced.values()))),
    "unplaced_W": unplaced,
    "flat_rate_g_per_GB": float(flat_rate),
    "path_rate_g_per_GB_bytes_weighted_p10_p50_p90": [wq(odr.g_per_GB, odr.GB, p) for p in (0.1, 0.5, 0.9)],
    "party_path_over_flat_p10_p50_p90": [float(P.path_over_flat.quantile(p)) for p in (0.1, 0.5, 0.9)],
    "party_bill_change_gravity_vs_em_WAPE": float((P.path_g_gravity_matrix - P.path_g).abs().sum() / P.path_g.sum()),
    "flat_vs_path_WAPE": float((P.flat_g - P.path_g).abs().sum() / P.path_g.sum()),
    "capacity_basis": {
        "path_share_of_emissions": float(1 - resc["unallocated_g"].sum() / resc["total_g"].sum()),
        "path_rate_g_per_GB_bytes_weighted_p10_p50_p90": [wq(odr.cap_g_per_GB, odr.GB, p) for p in (0.1, 0.5, 0.9)],
        "party_over_flat_p10_p50_p90": [float(P.pathcap_over_flat.quantile(p)) for p in (0.1, 0.5, 0.9)],
        "party_bill_change_gravity_vs_em_WAPE": float((P.pathcap_g_gravity_matrix - P.pathcap_g).abs().sum() / P.pathcap_g.sum()),
        "flat_vs_path_WAPE": float((P.flat_g - P.pathcap_g).abs().sum() / P.pathcap_g.sum()),
        "share_paid_by_parties_over_10x_flat": float(P.loc[P.pathcap_over_flat > 10, "pathcap_g"].sum() / P.pathcap_g.sum())},
    "usage_share_paid_by_parties_over_10x_flat": float(P.loc[P.path_over_flat > 10, "path_g"].sum() / P.path_g.sum()),
    "usage_traffic_share_of_parties_over_10x_flat": float(P.loc[P.path_over_flat > 10, "GB_in_out"].sum() / P.GB_in_out.sum()),
}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print(T.round(3).to_string())
