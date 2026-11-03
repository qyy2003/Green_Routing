"""Build the carbon-rates report from the outputs of flat_rate.py, path_rate.py and abilene.py.

usage: python build_report.py [out.html]
       (default $GR_ARTIFACTS_DIR/carbon_rates/report/carbon-rates.html)

Every number in the text comes from those outputs; the page lives in report_template.html.
Only aggregates are embedded (the hourly grid series is licensed).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.carbon import assumptions as A
from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
ROOT = ARTIFACTS_DIR / "carbon_rates"
FL, PA = ROOT / "flat", ROOT / "path"
AB = ARTIFACTS_DIR / "abilene" / "carbon"
ABR = HERE.parent / "abilene"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "carbon-rates.html"
OUT.parent.mkdir(parents=True, exist_ok=True)
GB = A.GB_PER_MBPS_HOUR

FS = json.load(open(FL / "summary.json"))
RE = pd.read_csv(FL / "rate_errors.csv")
TB = pd.read_csv(FL / "traffic_by_horizon.csv")
HR = pd.read_csv(FL / "hourly.csv", index_col=0, parse_dates=True)
PS = json.load(open(PA / "summary.json"))
BT = pd.read_csv(PA / "by_type.csv")
PR = pd.read_csv(PA / "party_rates.csv")
RC = pd.read_csv(PA / "routing_check.csv")
RS = pd.read_csv(PA / "resources.csv")
AS = json.load(open(AB / "summary.json"))
AINV = pd.read_csv(AB / "inventory.csv")
ANB = pd.read_csv(AB / "node_bills.csv")
ARE = pd.read_csv(AB / "rate_errors.csv") if (AB / "rate_errors.csv").stat().st_size > 1 else pd.DataFrame()
AEG = pd.read_csv(ABR / "egrid2004.csv")
BS = json.load(open(ARTIFACTS_DIR / "carbon" / "bills" / "summary.json"))
HN = pd.read_csv(ARTIFACTS_DIR / "carbon" / "bills" / "hourly_network.csv")
TOT = json.load(open(ARTIFACTS_DIR / "carbon" / "total" / "summary.json"))
LAB = {"persistence": "Persistence", "seasonal_naive_day": "Same time yesterday", "seasonal_naive_week": "Same time last week",
       "historical_average": "Historical average", "holt_winters": "Holt-Winters", "sarima": "SARIMA", "gbdt": "GBDT",
       "gru": "GRU", "tcn": "TCN", "dlinear": "DLinear", "patchtst": "PatchTST"}
TYPES = {"customer port": "Customer ports on PoP routers", "customer site": "Customer sites (own access router)",
         "external, behind hidden border router": "External networks behind ix3, ce1, ba3",
         "behind hidden router (customers/transit)": "Behind other hidden routers", "cdn": "Facebook direct link"}


def pct(x, d=0):
    return f"{100 * x:.{d}f} %"


def round1(x):
    """One significant digit: 203317 -> 200000."""
    return float(f"{x:.1g}")


def g(x):
    return f"{x:.4f}" if x >= 0.001 else f"{x:.5f}"


# --- plan A: models
blk = RE[RE.protocol == "block"].sort_values("traffic_hourly_WAPE")
ref = RE[RE.protocol == "reference"].set_index("model")
rows = [dict(m=LAB.get(r.model, r.model), t=r.traffic_hourly_WAPE, td=r.traffic_daily_WAPE, b=r.traffic_bias,
             ra=r.rate_WAPE_actual_grid, rf=r.rate_WAPE_grid_forecast) for r in blk.itertuples()]
for k, lab in (("GBDT on the total (top-down)", "GBDT on the hourly total"),
               ("sum of tcn door forecasts (bottom-up)", "Sum of 414 door forecasts (TCN)")):
    r = ref.loc[k]
    rows.append(dict(m=lab, t=r.traffic_hourly_WAPE, td=r.traffic_daily_WAPE, b=r.traffic_bias,
                     ra=r.rate_WAPE_actual_grid, rf=r.rate_WAPE_grid_forecast))
rows.sort(key=lambda r: r["t"])
grid_only = float(ref.loc["actual traffic", "rate_WAPE_grid_forecast"])
grid_only_d = float(ref.loc["actual traffic", "rate_daily_WAPE_grid_forecast"])
dr = TB[TB.protocol == "dense rolling"].pivot_table(index="model", columns="horizon_name", values="wape")
horizon = [dict(m=LAB.get(m, m), **{h: float(dr.loc[m, h]) for h in ["15min", "1h", "8h", "day", "week"]})
           for m in dr.sort_values("day").index]
sm = TB[TB.protocol == "sampled 160 origins"].pivot_table(index="model", columns="horizon_name", values="wape")
best = blk.iloc[0]
pers = blk.set_index("model").loc["persistence"]
ha = blk.set_index("model").loc["historical_average"]
hist = {m: pd.read_csv(FL / "merged" / "history" / f"{m}.csv") for m in ("tcn", "gru", "patchtst", "dlinear")}
ep = {m: int(h[h.phase == "fit"].epoch.max()) for m, h in hist.items()}
second = blk.iloc[1]
nb_ = blk.set_index("model").loc[["tcn", "gru", "patchtst", "dlinear"], "traffic_bias"] * 100
nn_bias = f"{nb_.min():+.1f} to {nb_.max():+.1f} %"
feedback = (f"{LAB[best.model]} is the best single model day-ahead ({pct(best.traffic_hourly_WAPE, 1)} hourly, "
            f"{pct(best.traffic_daily_WAPE, 1)} on daily totals), ahead of {LAB[second.model]} "
            f"({pct(second.traffic_hourly_WAPE, 1)}); the learned models all beat persistence ({pct(pers.traffic_hourly_WAPE, 1)}) "
            f"and the historical average ({pct(ha.traffic_hourly_WAPE, 1)}). With one series to learn from they stop "
            f"early (TCN after {ep['tcn']} epochs, GRU after {ep['gru']}, PatchTST after {ep['patchtst']}), so there is "
            f"little to gain from bigger networks. A small GBDT on the hourly total does as well "
            f"({pct(ref.loc['GBDT on the total (top-down)', 'traffic_hourly_WAPE'], 1)}) and trains in seconds; that is "
            f"the one I would run. The neural models over-forecast slightly ({nn_bias}), the hourly GBDT "
            f"under-forecasts ({ref.loc['GBDT on the total (top-down)', 'traffic_bias'] * 100:+.1f} %).")
hw_note = (f"at 8 hours they miss by {sm.loc['holt_winters', '8h']:.0f} % and {sm.loc['sarima', '8h']:.0f} %, "
           f"against {sm.loc['persistence', '8h']:.0f} % for persistence and {sm.loc['tcn', '8h']:.0f} % for TCN on the same origins")

# --- plan B
types = [dict(t=TYPES.get(r.type, r.type), n=int(r.parties), f=r.flat_g / 1e6, u=r.path_g / 1e6, c=r.pathcap_g / 1e6)
         for r in BT.sort_values("flat_g", ascending=False).itertuples()]
edges = [0, 0.25, 0.5, 0.8, 1.25, 2, 4, 10, 100, np.inf]
labs = ["< ¼", "¼–½", "½–0.8", "0.8–1.25", "1.25–2", "2–4", "4–10", "10–100", "> 100"]
ratio = {"bins": labs, "usage": np.histogram(PR.path_over_flat, edges)[0].tolist(),
         "capacity": np.histogram(PR.pathcap_over_flat, edges)[0].tolist()}
rc = RC.groupby("routing")[["fit_WAPE", "holdout_WAPE", "holdout_WAPE_gravity_only"]].mean()
bestr = PS["routing"]
names = {"capacity": "By link capacity (OSPF default)", "distance": "By fibre distance", "hops": "Every hop equal"}
routing = [dict(r=names[k], fit=float(v.fit_WAPE), ho=float(v.holdout_WAPE), gr=float(v.holdout_WAPE_gravity_only), best=k == bestr)
           for k, v in rc.sort_values("holdout_WAPE").iterrows()]
spans = RS[(RS.kind == "span") & RS.rate_g_per_GB_median.notna()].sort_values("rate_g_per_GB_median", ascending=False)
ws = spans.iloc[0]
util = float((RS.utilisation.fillna(0) * RS.emissions_g).sum() / RS.emissions_g.sum())
tiny = PR.path_over_flat > 10
cap = PS["capacity_basis"]

# --- abilene: flow data vs counters (real equipment and grid)
ab_rows = [dict(k="Counters, non-negative estimate (used for SWITCHlan)", tm=AS["tm_rel_l2_median_em"],
                b=AS["counters_vs_flows_usage_WAPE"], bmax=AS["counters_worst_node_error"], bc=AS["counters_vs_flows_capacity_WAPE"]),
           dict(k="Counters, gravity guess only", tm=AS["tm_rel_l2_median_gravity"], b=AS["gravity_vs_flows_usage_WAPE"],
                bmax=None, bc=AS["gravity_vs_flows_capacity_WAPE"]),
           dict(k="Flat rate instead of path", tm=None, b=AS["flat_vs_path_usage_WAPE"], bmax=None, bc=AS["flat_vs_path_capacity_WAPE"])]
rt = AINV[(AINV.kind == "router") & (AINV.model == "Juniper T640")]
ab_inputs = [
    dict(k="Juniper T640 routers (11 PoPs)", v=f"{rt.W.min():,.0f}–{rt.W.max():,.0f} W each (range {rt.W_low.min():,.0f}–{rt.W_high.max():,.0f}): base 16.9 A + one FPC3 for the OC-192 links + 2 FPC2 and (links + 8) interface cards for the connectors, at −48 V",
         s="Juniper T640 Core Router Hardware Guide, Table 78 (component currents); line-card count assumed"),
    dict(k="Juniper M5 (Atlanta)", v="300 W (200–434)", s="Juniper: M5/M10 power supplies rated 434 W each, redundant pair"),
    dict(k="Each backbone link", v="one leased OC-192 wavelength: two 10G transponders (32.5 W) plus 1/40 of the amplifier sites every 80 km (148 W each) and of the two terminals",
         s="Cisco ONS 15454 DWDM Reference Manual, Table A-1 (card power)"),
    dict(k="Fibre route", v=f"{AS['route_km']:,.0f} km ({AS['amplifier_sites']} amplifier sites): great-circle spans scaled to the published 13,000 route-miles",
         s="Lightwave, 4 Feb 2004 (Abilene upgrade to 10 Gbit/s)"),
    dict(k="Facility overhead", v="PUE 2.0 (1.6–2.5)", s="assumption for 2004 carrier PoPs; EPA Report to Congress on data centres (2007)"),
    dict(k="Grid factor", v="CO₂ output rate of each city's eGRID subregion, year 2004, constant", s="EPA eGRID2006 v2.1, Year 2004 summary tables"),
    dict(k="Traffic", v="NetFlow, 1 in 100 packets, scaled; real OSPF routing", s="Zhang et al., Abilene traffic matrix archive"),
]
ab_egrid = [dict(n=r.node, c=r.city, r=f"{r.egrid_subregion} ({r.subregion_name})", g=r.co2_lb_per_mwh * 0.453592) for r in AEG.itertuples()]
ab_models = [dict(m=LAB.get(r.model, r.model), t=r.traffic_hourly_WAPE, td=r.traffic_daily_WAPE, b=r.traffic_bias, r=r.rate_hourly_WAPE)
             for r in ARE.itertuples()] if len(ARE) else []
ab_horizon = []
for panel in ("total", "links"):
    f = AB / f"horizons_{panel}_rolling.csv"
    if f.exists():
        t = pd.read_csv(f).pivot_table(index="model", columns="horizon_name", values="wape")
        for m in t.sort_values("day").index:
            ab_horizon.append(dict(p=panel, m=LAB.get(m, m), **{h: (float(t.loc[m, h]) if h in t and pd.notna(t.loc[m, h]) else None)
                                                                 for h in ["15min", "1h", "8h", "day", "week"]}))
ab_nodes = [dict(n=r.node, tb=r.GB_in_out / 1e3, f=r.flat_g_per_GB, u=r.path_usage_g_per_GB,
                 c=2 * r.path_capacity_g / r.GB_in_out, e=2 * r.counters_path_usage_g / r.GB_in_out) for r in ANB.itertuples()]
ab_best = ARE.iloc[0] if len(ARE) else None
sw_best = blk.iloc[0]
compare = [
    dict(k="Period", s=f"{pd.Timestamp(FS['window'][0]).day} Mar – {pd.Timestamp(FS['window'][1]).day} Jul 2026 (111 days)",
         a=f"{pd.Timestamp(AS['period'][0]).day} May – {pd.Timestamp(AS['period'][1]).day} Sep 2004 ({AS['period_weeks']} weeks)"),
    dict(k="Network", s=f"{PS['routers']} routers, {PS['directed_links'] // 2} links", a=f"{AS['nodes']} routers, {AS['spans']} links"),
    dict(k="Traffic carried (mean)", s=f"{FS['mean_total_gbps']:.0f} Gbit/s", a=f"{AS['mean_total_gbps']:.1f} Gbit/s"),
    dict(k="Power at the sites", s=f"{BS['backbone_kW']:.0f} kW shared (+ {BS['access_kW']:.0f} kW access)", a=f"{AS['site_kW']:.0f} kW (one pool)"),
    dict(k="Grid factor", s=f"{HN.I_CH.mean():.0f} g/kWh (hourly, flow-traced)", a=f"{AS['g_per_kWh_power_weighted']:.0f} g/kWh (annual, by region)"),
    dict(k="Flat rate", s=f"{PS['flat_rate_g_per_GB']:.4f} g/GB", a=f"{AS['flat_rate_g_per_GB']:.1f} g/GB"),
    dict(k="Utilisation (emission-weighted)", s=pct(util, 0), a=pct(AS["utilisation_emission_weighted"], 0)),
    dict(k="Emissions moved between parties, path by usage", s=pct(PS["flat_vs_path_WAPE"] / 2, 0), a=pct(AS["flat_vs_path_usage_WAPE"] / 2, 0)),
    dict(k="Who sends to whom", s="estimated from counters", a="measured (NetFlow)"),
    dict(k="Counters instead of flows, path bills", s="no flows to compare", a=pct(AS["counters_vs_flows_usage_WAPE"], 1)),
    dict(k="Best one-model forecast of the total (day-ahead, hourly)", s=f"{LAB[sw_best.model]} {pct(sw_best.traffic_hourly_WAPE, 1)}",
         a=(f"{LAB.get(ab_best.model, ab_best.model)} {pct(ab_best.traffic_hourly_WAPE, 1)}" if ab_best is not None else "—")),
]

H0, H1 = pd.Timestamp(FS["window"][0]), pd.Timestamp(FS["window"][1])
bytes_pb = float((HR.T_gbps * 1e3 * GB).sum() / 1e6)
N = {
    "days": f"{(H1 - H0).days + 1}", "window_text": f"{H0.day} {H0:%B} to {H1.day} {H1:%B %Y}",
    "built": f"{pd.Timestamp.now().day} {pd.Timestamp.now():%B %Y}",
    "bb_t": f"{PS['backbone_t']:.2f}", "tb_bytes": f"{bytes_pb:.0f}", "flat_rate": g(PS["flat_rate_g_per_GB"]),
    "rate_p10": g(FS["rate_hour_p10_p90"][0]), "rate_p90": g(FS["rate_hour_p10_p90"][1]),
    "best_model": LAB[best.model], "best_wape": pct(best.traffic_hourly_WAPE, 1), "pers_wape": pct(pers.traffic_hourly_WAPE, 1),
    "bu_wape": pct(ref.loc["sum of tcn door forecasts (bottom-up)", "traffic_hourly_WAPE"], 1),
    "rate_err_a": pct(best.rate_WAPE_actual_grid, 0), "rate_err_f": pct(best.rate_WAPE_grid_forecast, 0),
    "rate_err_fd": pct(best.rate_daily_WAPE_grid_forecast, 0), "rate_err_grid": pct(grid_only, 0),
    "usage_moved": pct(PS["flat_vs_path_WAPE"] / 2, 0),
    "gbdt_wape": pct(ref.loc["GBDT on the total (top-down)", "traffic_hourly_WAPE"], 1),
    "usage_tiny_traffic": pct(PS["usage_traffic_share_of_parties_over_10x_flat"], 1),
    "usage_tiny_share": pct(PS["usage_share_paid_by_parties_over_10x_flat"], 0),
    "util": pct(util, 0), "cap_share": pct(cap["path_share_of_emissions"], 0), "cap_idle": pct(1 - cap["path_share_of_emissions"], 0),
    "ab_em_total": pct(AS["counters_vs_flows_usage_WAPE"], 0), "ab_flat": pct(AS["flat_vs_path_usage_WAPE"], 0),
    "sw_sens": pct(PS["party_bill_change_gravity_vs_em_WAPE"], 0), "sw_sens_cap": pct(cap["party_bill_change_gravity_vs_em_WAPE"], 0),
    "cons": pct(abs(TOT["conservation"]["in_over_out"] - 1), 1),
    "steps": f"{int(round((pd.Timestamp('2026-07-11') - pd.Timestamp('2025-07-01')).total_seconds() / 300)) + 1:,}",
    "hw_note": hw_note, "model_feedback": feedback,
    "routers": f"{PS['routers']}", "links": f"{PS['directed_links']}", "hidden": f"{PS['hidden_edge_nodes']}",
    "router_share": pct(PS["router_share_of_backbone"], 0), "unseen": pct(PS["unseen_share_of_ingress"], 1),
    "unseen_cost": pct(PS["unseen_cost_share"], 0),
    "worst_span": ws["name"].replace("-", "–"), "worst_span_rate": f"{ws.rate_g_per_GB_median:,.0f}",
    "worst_span_x": f"about {round1(ws.rate_g_per_GB_median / PS['flat_rate_g_per_GB']):,.0f} times",
    "n_parties": f"{len(PR)}",
    "p_usage_p10": g(PS["path_rate_g_per_GB_bytes_weighted_p10_p50_p90"][0]),
    "p_usage_p50": g(PS["path_rate_g_per_GB_bytes_weighted_p10_p50_p90"][1]),
    "p_usage_p90": g(PS["path_rate_g_per_GB_bytes_weighted_p10_p50_p90"][2]),
    "p_cap_p50": g(cap["path_rate_g_per_GB_bytes_weighted_p10_p50_p90"][1]),
    "ab_weeks": f"{AS['period_weeks']}", "ab_hours": f"{AS['hours_with_data']:,}", "ab_nodes": f"{AS['nodes']}",
    "ab_cap": pct(AS["counters_vs_flows_capacity_WAPE"], 1), "ab_util": pct(AS["utilisation_emission_weighted"], 0),
    "sw_holdout": pct(rc.loc[bestr, "holdout_WAPE"], 0), "ab_holdout": pct(AS["holdout_link_WAPE"], 0),
    "ab_em_max": pct(AS["counters_worst_node_error"], 0),
    "ab_flat_rate": f"{AS['flat_rate_g_per_GB']:.1f}", "ab_vs_sw": f"{round1(AS['flat_rate_g_per_GB'] / PS['flat_rate_g_per_GB']):,.0f}",
    "ab_gkwh": f"{AS['g_per_kWh_power_weighted']:.0f}",
    "ab_best_wape": pct(ab_best.traffic_hourly_WAPE, 1) if ab_best is not None else "—",
    "ab_best_model": LAB.get(ab_best.model, ab_best.model) if ab_best is not None else "—",
    "ab_best_rate": pct(ab_best.rate_hourly_WAPE, 1) if ab_best is not None else "—",
    "ab_period": f"{pd.Timestamp(AS['period'][0]).day} May to {pd.Timestamp(AS['period'][1]).day} September 2004",
    "ab_gbps": f"{AS['mean_total_gbps']:.1f}", "ab_it": f"{AS['router_kW_it'] + AS['optical_kW_it']:.1f}",
    "ab_rt": f"{AS['router_kW_it']:.1f}", "ab_opt": f"{AS['optical_kW_it']:.1f}", "ab_site": f"{AS['site_kW']:.0f}",
    "ab_site_lo": f"{AS['site_kW_low_high'][0]:.0f}", "ab_site_hi": f"{AS['site_kW_low_high'][1]:.0f}",
    "ab_tyr": f"{AS['emissions_t_per_year']:.0f}", "ab_cap_vs_flat": pct(AS["flat_vs_path_capacity_WAPE"], 1),
}
html = (HERE / "report_template.html").read_text()
for k, v in N.items():
    html = html.replace("{{" + k + "}}", v)
data = {"models": rows, "grid_only": grid_only, "horizon": horizon, "routing": routing, "types": types, "ratio": ratio,
        "abilene": ab_rows, "ab_inputs": ab_inputs, "ab_egrid": ab_egrid, "ab_models": ab_models, "ab_horizon": ab_horizon,
        "ab_nodes": ab_nodes, "compare": compare}
html = html.replace("/*DATA*/", "const DATA = " + json.dumps(data, separators=(",", ":"), default=float) + ";")
OUT.write_text(html)
left = sorted(set(re.findall(r"\{\{(\w+)\}\}", html)))
print("wrote", OUT, len(html), "bytes; unreplaced:", left or "none", "; NaN" if "NaN" in html else "")
