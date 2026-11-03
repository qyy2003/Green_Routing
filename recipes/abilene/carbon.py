"""Carbon rates and bills on Abilene, with measured traffic matrices (NetFlow), real
equipment power and 2004 grid factors.

Energy: Juniper T640 per PoP from the hardware guide's component currents and an assumed
line-card configuration, the Atlanta Juniper M5, and for every backbone link one leased
OC-192 wavelength on Qwest's DWDM (two transponders plus a share of the amplifier sites
along the route), all x PUE (energy.csv). Grid: EPA eGRID2006, year-2004 CO2 output rate
of each city's subregion (egrid2004.csv), constant over the year (no hourly data exists
for 2004). Abilene has no per-customer equipment data, so the whole network is one pool.

Outputs ($GR_ARTIFACTS_DIR/abilene/carbon): inventory.csv, node_bills.csv, od_rates.csv,
rate_errors.csv (flat-rate forecasts), horizons_links.csv / horizons_total.csv, summary.json
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from green_routing.carbon import path_rate, tomography
from green_routing.dataset import abilene
from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
ROOT = ARTIFACTS_DIR / "abilene"
OUT = ROOT / "carbon"
OUT.mkdir(parents=True, exist_ok=True)
GB = 3600 / 8 / 1000                      # GB per Mbit/s-hour
EN = pd.read_csv(HERE / "energy.csv").set_index("item")
EG = pd.read_csv(HERE / "egrid2004.csv").set_index("node")
v = lambda k, col="value": float(EN.loc[k, col])          # noqa: E731

ds = pickle.load(open(ROOT / "dataset.pkl", "rb"))
nodes, n, od = ds["nodes"], len(ds["nodes"]), ds["od"]
Xh5 = ds["X"]
Xh = Xh5.resample("1h").mean().where(Xh5.resample("1h").count().min(axis=1) >= 6, axis=0)
ok = Xh.notna().all(axis=1).to_numpy()
H = Xh.index
X = np.nan_to_num(Xh.to_numpy())                               # Mbit/s, hourly mean, [hours, OD]


# --- energy inventory ------------------------------------------------------------------
def km(a, b):
    (la1, lo1), (la2, lo2) = np.radians(ds["coord"][a]), np.radians(ds["coord"][b])
    return 6371 * 2 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))


spans = sorted({tuple(sorted(xy)) for xy in ds["link_xy"]})
deg = {x: sum(x in s for s in spans) for x in nodes}
gc = np.array([km(a, b) for a, b in spans])
route = gc * v("route_miles") * 1.609344 / gc.sum()             # scaled to the published route length


def router_W(x, case):
    if x == "ATLA-M5":
        return {"central": v("m5_W"), "low": v("m5_W", "low"), "high": v("m5_W", "high")}[case]
    col = {"central": "value", "low": "low", "high": "high"}[case]
    fpc2, pics = v("t640_customer_fpc2", col), v("t640_customer_pics", col)
    amps = v("t640_base_A") + v("t640_fpc3_A") * math.ceil(deg[x] / 4) + v("t640_fpc2_A") * fpc2 + v("t640_pic_A") * (deg[x] + pics)
    return 48 * amps


def span_W(km_, case):
    if km_ < 1:                      # a cross-connect inside one building (Atlanta): no DWDM wavelength
        return 0.0, 0
    ch = {"central": v("channels"), "low": v("channels", "low"), "high": v("channels", "high")}[case]
    sk = {"central": v("span_km"), "low": v("span_km", "high"), "high": v("span_km", "low")}[case]
    txp = v("txp_10g_W") if case != "high" else v("txp_10g_W", "high")
    ila = max(0, math.ceil(km_ / sk) - 1)
    return 2 * txp + 2 * v("terminal_shared_W") / ch + ila * v("ila_site_W") / ch, ila


pue = {"central": v("pue"), "low": v("pue", "low"), "high": v("pue", "high")}
g_kwh = EG.co2_lb_per_mwh * v("lb_to_g") / 1000                 # g per kWh
rows = []
for x in nodes:
    rows.append(dict(kind="router", name=x, model="Juniper M5" if x == "ATLA-M5" else "Juniper T640",
                     W=router_W(x, "central"), W_low=router_W(x, "low"), W_high=router_W(x, "high"),
                     g_per_kWh=g_kwh[x], subregion=EG.egrid_subregion[x]))
for (a, b), r_km in zip(spans, route):
    w, ila = span_W(r_km, "central")
    rows.append(dict(kind="span", name=f"{a}-{b}", model=f"OC-192 wavelength, {r_km:.0f} km, {ila} amplifier sites",
                     W=w, W_low=span_W(r_km, "low")[0], W_high=span_W(r_km, "high")[0],
                     g_per_kWh=(g_kwh[a] + g_kwh[b]) / 2, subregion=f"{EG.egrid_subregion[a]}/{EG.egrid_subregion[b]}"))
INV = pd.DataFrame(rows)
INV.to_csv(OUT / "inventory.csv", index=False)
E_res = (INV.W * pue["central"] / 1000 * INV.g_per_kWh).to_numpy()           # g per hour, constant
Eh = np.tile(E_res, (len(H), 1))

# --- routing onto resources, capacities
S = sp.lil_matrix((len(spans), len(ds["link_xy"])))
for j, xy in enumerate(ds["link_xy"]):
    S[spans.index(tuple(sorted(xy))), j] = 1.0
A_res = sp.vstack([ds["A_node"], S.tocsr() @ ds["A_link"]]).tocsr()
cap_out = {x: sum(c for (a, _), c in ds["cap"].items() if a == x) for x in nodes}
CAP = np.r_[[2 * cap_out[x] for x in nodes],                        # backbone ports + as much toward connectors
            [ds["cap"][(a, b)] + ds["cap"][(b, a)] for a, b in spans]]

# --- bills per router (each router = one party, the connectors behind it), hours with data
Xo, Eo = X[ok], Eh[ok]
ing, egr = Xo.reshape(-1, n, n).sum(2), Xo.reshape(-1, n, n).sum(1)
V = ing + egr
flat = V / V.sum(1, keepdims=True) * Eo.sum(1, keepdims=True)
path_u = path_rate.node_bills(A_res, Eo, Xo, n)
path_c = path_rate.node_bills(A_res, Eo, Xo, n, capacity=CAP)
Y = (ds["A_link"] @ Xo.T).T
X_em = np.array([tomography.em(ds["A_link"], Y[h], ing[h], egr[h]) for h in range(len(Xo))])
X_gr = np.array([tomography.gravity(ing[h], egr[h]) for h in range(len(Xo))])
em_u = path_rate.node_bills(A_res, Eo, X_em, n)
em_c = path_rate.node_bills(A_res, Eo, X_em, n, capacity=CAP)
gr_u = path_rate.node_bills(A_res, Eo, X_gr, n)
gr_c = path_rate.node_bills(A_res, Eo, X_gr, n, capacity=CAP)


# hold-out: hide a fifth of the links from the estimator and predict them (as on SWITCHlan)
ho = []
for seed in range(10):
    drop = np.random.default_rng(seed).choice(Y.shape[1], size=Y.shape[1] // 5, replace=False)
    Yd = Y.copy()
    Yd[:, drop] = np.nan
    sub = range(0, len(Xo), 12)
    Xd = np.array([tomography.em(ds["A_link"], Yd[h], ing[h], egr[h]) for h in sub])
    ho.append(float(np.abs((ds["A_link"][drop] @ Xd.T).T - Y[list(sub)][:, drop]).sum() / Y[list(sub)][:, drop].sum()))


def wape_total(b, ref):
    return float(np.abs(b.sum(0) - ref.sum(0)).sum() / ref.sum())


NB = pd.DataFrame({"node": nodes, "GB_in_out": V.sum(0) * GB, "flat_g": flat.sum(0), "path_usage_g": path_u.sum(0),
                   "path_capacity_g": path_c.sum(0), "counters_path_usage_g": em_u.sum(0)})
NB["flat_g_per_GB"] = 2 * NB.flat_g / NB.GB_in_out
NB["path_usage_g_per_GB"] = 2 * NB.path_usage_g / NB.GB_in_out
NB.to_csv(OUT / "node_bills.csv", index=False)

# per-path rates (g per GB moved), bytes-weighted over the period
used = (A_res @ Xo.T).T
per_u = np.divide(Eo, used, out=np.zeros_like(Eo), where=used > 0)
per_c = Eo / np.maximum(CAP[None, :], used)
rate_u = (A_res.T @ per_u.T).T / GB
rate_c = (A_res.T @ per_c.T).T / GB
w = Xo.sum(0)
odr = pd.DataFrame({"s": [s for s, _ in od], "t": [t for _, t in od], "GB": w * GB,
                    "usage_g_per_GB": (rate_u * Xo).sum(0) / np.maximum(w, 1e-12),
                    "capacity_part_g_per_GB": (rate_c * Xo).sum(0) / np.maximum(w, 1e-12)})
odr.to_csv(OUT / "od_rates.csv", index=False)


def wq(vals, wts, p):
    o = np.argsort(vals)
    c = np.cumsum(np.asarray(wts, float)[o])
    return float(np.interp(p * c[-1], c, np.asarray(vals, float)[o]))


T_all = pd.Series(np.where(ok, X.sum(1), np.nan), index=H)       # Mbit/s carried (each byte once)
flat_rate_h = E_res.sum() / (T_all * GB)
util = float((used / CAP[None, :] * Eo).sum() / Eo.sum())

# --- forecasting the flat rate: the total-traffic models (block logs, day horizon)
rate_rows, horizons = [], {}
for panel in ("total", "links"):
    run = ROOT / panel / "merged"
    dn = run / "dense"
    for mode in ("rolling", "block"):
        f = dn / mode / "aggregate.csv"
        if f.exists():
            horizons[(panel, mode)] = pd.read_csv(f)
    if panel == "total" and (dn / "meta.json").exists():
        meta = json.load(open(dn / "meta.json"))
        hi = {h["name"]: i for i, h in enumerate(meta["horizons"])}["day"]
        t0 = pd.Timestamp(meta["t0"], unit="s", tz="UTC")
        nt = (meta["n_targets"] // 12) * 12
        Ht = pd.date_range(t0, periods=nt // 12, freq="h")
        y = T_all.reindex(Ht)
        good = y.notna()
        for f in sorted(dn.glob("pred_block_*.npy")):
            name = f.stem.split("pred_block_")[1]
            p = pd.Series(np.load(f)[0, hi, :nt].astype(float).reshape(-1, 12).mean(1), index=Ht)
            e = (p - y)[good]
            rate_f, rate_a = E_res.sum() / (p * GB), E_res.sum() / (y * GB)
            rate_rows.append(dict(model=name, traffic_hourly_WAPE=float(e.abs().sum() / y[good].sum()),
                                  traffic_daily_WAPE=float((p[good].resample("D").mean() - y[good].resample("D").mean()).abs().sum()
                                                           / y[good].resample("D").mean().sum()),
                                  traffic_bias=float(e.sum() / y[good].sum()),
                                  rate_hourly_WAPE=float((rate_f - rate_a)[good].abs().sum() / rate_a[good].sum())))
RE = pd.DataFrame(rate_rows).sort_values("traffic_hourly_WAPE") if rate_rows else pd.DataFrame()
RE.to_csv(OUT / "rate_errors.csv", index=False)
for (panel, mode), t in horizons.items():
    t.to_csv(OUT / f"horizons_{panel}_{mode}.csv", index=False)

hours = int(ok.sum())
summary = {
    "period": [str(H[0]), str(H[-1])], "hours_with_data": hours, "nodes": n, "spans": len(spans),
    "route_km": float(route.sum()), "amplifier_sites": int(sum(span_W(r, "central")[1] for r in route)),
    "router_kW_it": float(INV[INV.kind == "router"].W.sum() / 1e3), "optical_kW_it": float(INV[INV.kind == "span"].W.sum() / 1e3),
    "it_kW_low_high": [float(INV.W_low.sum() / 1e3), float(INV.W_high.sum() / 1e3)],
    "site_kW": float(INV.W.sum() * pue["central"] / 1e3),
    "site_kW_low_high": [float(INV.W_low.sum() * pue["low"] / 1e3), float(INV.W_high.sum() * pue["high"] / 1e3)],
    "g_per_kWh_power_weighted": float((INV.W * INV.g_per_kWh).sum() / INV.W.sum()),
    "emissions_t_per_year": float(E_res.sum() * 8760 / 1e6),
    "emissions_t_period": float(Eo.sum() / 1e6), "PB_carried_period": float(Xo.sum() * GB / 1e6),
    "mean_total_gbps": float(T_all.mean() / 1e3),
    "flat_rate_g_per_GB": float(Eo.sum() / (Xo.sum() * GB)),
    "flat_rate_hour_p10_p90": [float(flat_rate_h.quantile(.1)), float(flat_rate_h.quantile(.9))],
    "path_usage_g_per_GB_p10_p50_p90": [wq(odr.usage_g_per_GB, odr.GB, p) for p in (.1, .5, .9)],
    "utilisation_emission_weighted": util,
    "capacity_path_share": float((Eo * np.minimum(used / CAP[None, :], 1)).sum() / Eo.sum()),
    "flat_vs_path_usage_WAPE": wape_total(flat, path_u), "flat_vs_path_capacity_WAPE": wape_total(flat, path_c),
    "counters_vs_flows_usage_WAPE": wape_total(em_u, path_u), "counters_vs_flows_capacity_WAPE": wape_total(em_c, path_c),
    "gravity_vs_flows_usage_WAPE": wape_total(gr_u, path_u),
    "gravity_vs_flows_capacity_WAPE": wape_total(gr_c, path_c),
    "period_weeks": round(hours / 168),
    "counters_vs_flows_usage_WAPE_hourly": float(np.abs(em_u - path_u).sum() / path_u.sum()),
    "counters_worst_node_error": float((np.abs(em_u.sum(0) - path_u.sum(0)) / path_u.sum(0)).max()),
    "holdout_link_WAPE": float(np.mean(ho)),
    "tm_rel_l2_median_em": float(np.median(np.linalg.norm(X_em - Xo, axis=1) / np.linalg.norm(Xo, axis=1))),
    "tm_rel_l2_median_gravity": float(np.median(np.linalg.norm(X_gr - Xo, axis=1) / np.linalg.norm(Xo, axis=1))),
}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(json.dumps(summary, indent=1))
print(INV.round(1).to_string(index=False))
print(NB.round(3).to_string(index=False))
print(RE.round(4).to_string(index=False))
