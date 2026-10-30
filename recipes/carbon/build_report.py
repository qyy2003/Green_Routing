"""Build the forecast-carbon-bills report (one HTML page) from the pipeline outputs.

usage: python build_report.py [out.html]
       (default $GR_ARTIFACTS_DIR/carbon/report/carbon-bills.html)

Reads $GR_ARTIFACTS_DIR/network and $GR_ARTIFACTS_DIR/carbon (gr-network, gr-carbon);
every number in the text comes from those outputs. The page text and charts live in
report_template.html next to this script. Only aggregates are embedded: the hourly
Electricity Maps series is licensed and never leaves the grid/ directory.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

from green_routing.carbon import assumptions as A
from green_routing.carbon import grid, inventory
from green_routing.network import dns
from green_routing.paths import ARTIFACTS_DIR, reference_dir

HERE = Path(__file__).resolve().parent
C, NET, REF = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network", reference_dir()
B, INV, G = C / "bills", C / "inventory", C / "grid"
RUN = ARTIFACTS_DIR / "pred_year" / "merged"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else C / "report" / "carbon-bills.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

S = json.load(open(B / "summary.json"))
YEAR = int(sorted(INV.glob("routers_*.csv"))[-1].stem.split("_")[1])
E = pd.read_csv(B / "bill_errors.csv").set_index("scenario")
T = pd.read_csv(B / "errors_by_type.csv").merge(pd.read_csv(B / "settled_by_type.csv"), on="type")
HOD = pd.read_csv(B / "hour_of_day_median.csv", index_col=0)
DX = pd.read_csv(B / "daily_examples.csv", header=[0, 1], index_col=0)
TOP = pd.read_csv(B / "top_parties.csv", index_col=0)
SCH = pd.read_csv(B / "scheduling_test.csv")
HN = pd.read_csv(B / "hourly_network.csv", index_col=0, parse_dates=True)

MODEL_LAB = {"persistence": "Persistence", "seasonal_naive_week": "Same hour last week",
             "seasonal_naive_day": "Same hour yesterday", "dlinear": "DLinear", "gru": "GRU", "tcn": "TCN",
             "patchtst": "PatchTST"}
TYPES = {"customer port": "Customer ports on PoP routers", "customer site": "Customer sites (own access router)",
         "external, behind hidden border router": "External networks behind ix3, ce1, ba3",
         "behind hidden router (customers/transit)": "Behind other hidden routers", "cdn": "Facebook direct link"}


def pct(x):
    return f"{x:.0%}"


def sci(x):
    e = math.floor(math.log10(abs(x)))
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{x / 10 ** e:.1f} × 10{sup}"


def err(name):
    return float(E.WAPE_daily[name])


def day(ts):
    return f"{ts.day} {ts:%B %Y}"


# --- charts (as before) -----------------------------------------------------------------
models = [k for k in MODEL_LAB if f"{k} + grid actual" in E.index]
dumb = [{"m": MODEL_LAB[k], "a": err(f"{k} + grid actual"), "f": err(f"{k} + grid fc")} for k in models]
stack = [{"t": TYPES.get(r.type, r.type), "a": r.access_t, "b": r.backbone_t, "w": r.WAPE, "n": int(r.parties)}
         for r in T.sort_values("settled_t", ascending=False).itertuples()]
hod = {"h": list(range(24)), "I": HOD.I.round(2).tolist(), "V": HOD.V.round(1).tolist(), "g": (HOD.g * 1000).round(3).tolist()}
site_col = next(c for c in DX.columns.get_level_values(0).unique() if c.startswith("site:"))
ix, lg = DX["hidden:ix3"], DX[site_col]
daily = {"d": [d[:10] for d in DX.index], "ix_s": (ix.settled_g / 1000).round(2).tolist(),
         "ix_f": (ix.forecast_g / 1000).round(2).tolist(), "lg_s": (lg.settled_g / 1000).round(3).tolist(),
         "lg_f": (lg.forecast_g / 1000).round(3).tolist()}
med = SCH[["avg", "oracle", "forecast_pick", "worst"]].median()
captured = (SCH.avg - SCH.forecast_pick).sum() / (SCH.avg - SCH.oracle).sum()
sched = [{"k": "Hour with the highest g/GB", "v": med.worst}, {"k": "Any hour (day average)", "v": med.avg},
         {"k": "Hour picked by the day-ahead forecast", "v": med.forecast_pick}, {"k": "Best hour in hindsight", "v": med.oracle}]
top = [{"p": p.replace("hidden:", "behind ").replace("port:", "").replace("site:", "site "), "t": TYPES.get(r.type, r.type),
        "s": r.settled_t * 1000, "f": r.forecast_t * 1000, "acc": r.access_share} for p, r in TOP.iterrows()]

# --- numbers for the text ------------------------------------------------------------------
best_key = S["best_scenario"].split(" + ")[0]
H0, H1 = pd.Timestamp(S["window"][0]), pd.Timestamp(S["window"][1])
sites = pd.read_csv(REF / "sites.csv").drop_duplicates("code").set_index("code")
doors = pd.read_csv(C / "allocation" / "doors.csv")
I_net = pd.read_csv(NET / "interfaces.csv")
L = pd.read_csv(NET / "dns_links.csv")
live = dns.live_routers(L)
dev = pd.read_csv(INV / "devices.csv")
last = pd.to_datetime(dev.assign(d=dev.dev.str.replace("^swi", "", regex=True)).drop_duplicates("d").set_index("d")["last"])
missing = [d for d in live if d not in last.index or pd.isna(last[d]) or last[d] < pd.Timestamp("2026-07-01")]
act = I_net[I_net.role != "idle/absent"]
both = act[act.peer.notna() & act.dns_peer.notna()]
psu = pd.read_csv(INV / "psu_survey.csv")
psu_prod = psu[~psu.dev.str.contains("lab|test|mgmt") & psu.W_med.notna()]
zoo = pd.read_csv(INV / "zoo_match.csv")
sh = pd.read_csv(INV / "shelves.csv")
sh_live = sh[pd.to_datetime(sh["last"]) >= H0.tz_localize(None)]
ext = pd.read_csv(REF / "external_links.csv")
dep = pd.read_csv(INV / "power_vs_traffic.csv")
dbig = dep.sort_values("W_mean", ascending=False).iloc[0]
R = pd.read_csv(INV / f"routers_{YEAR}.csv")
sweep = (REF / "dns" / "sweep_date.txt").read_text().strip()
nodns = inventory.router_energy(inventory.router_power(dev, set(), sweep), YEAR)
ch = grid.load(G, "CH")
chy = ch.loc[str(YEAR)]
hod_y = chy.groupby(chy.index.tz_convert("Europe/Zurich").hour).mean()
mon_y = chy.resample("ME").mean()
est = pd.read_csv(G / "em_CH_hourly.csv", index_col=0, parse_dates=True).estimated.loc[str(YEAR)].mean()
rules = pd.read_csv(G / "rules.csv")
# grid-factor change on the worst-forecast days of the two example parties
DX.index = pd.to_datetime(DX.index, utc=True)
Id = ch.resample("D").mean()
dI = (Id - Id.shift(1)).abs().reindex(DX.index)
worst = []
for k in ("hidden:ix3", site_col):
    e = (DX[k].forecast_g - DX[k].settled_g).abs() / DX[k].settled_g
    worst.append(float(dI.loc[e.nlargest(10).index].mean()))
by_type = pd.read_csv(B / "settled_by_type.csv").set_index("type")
tot_t = (by_type.access_t + by_type.backbone_t)
hidden_t = tot_t[[t for t in tot_t.index if "hidden" in t]].sum()
border_t = tot_t.get("external, behind hidden border router", 0.0)
oom = round(math.log10(med.avg / S["marginal_g_per_GB"][1]))
def rule_sentence():
    used, best = S["grid_rule"], rules.iloc[0]
    u = rules.set_index("rule").loc[used]
    if used == best.rule:
        return (f"The grid factor is “{used}”, the best of {len(rules)} simple rules tested "
                f"(error {u.MAE:.1f} g/kWh on a mean of {u.mean_actual:.0f}).")
    return (f"The grid factor is “{used}” (error {u.MAE:.1f} g/kWh on a mean of {u.mean_actual:.0f}); the best of "
            f"{len(rules)} simple rules tested was “{best.rule}” ({best.MAE:.1f} g/kWh).")


words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
host = re.split(r"[,;(]", str(sites.host_or_facility.get(site_col.split(":", 1)[1], "") or ""))[0].strip()
rank0 = TOP.index[0] == "hidden:ix3"

N = {
    "t_total": f"{S['settled_t_total']:.2f}", "kw_total": f"{S['access_kW'] + S['backbone_kW']:.0f}",
    "kw_acc": f"{S['access_kW']:.1f}", "kw_bb": f"{S['backbone_kW']:.1f}", "days": f"{S['hours'] // 24}",
    "window_text": f"{H0.day} {H0:%B} to {H1.day} {H1:%B %Y}", "window_short": f"{H0.day} {H0:%b} – {H1.day} {H1:%b %Y}",
    "built": day(pd.Timestamp.now()), "year": str(YEAR),
    "best_model": MODEL_LAB.get(best_key, best_key), "grid_rule": S["grid_rule"],
    "e_best_a": pct(err(f"{best_key} + grid actual")), "e_pers_a": pct(err("persistence + grid actual")),
    "e_best_f": pct(err(f"{best_key} + grid fc")), "e_grid": pct(err("traffic actual + grid fc")),
    "bias": f"{float(E.total_bill_error[S['best_scenario']]):+.0%}",
    "g_avg": f"{med.avg:.4f}", "g_pick": f"{med.forecast_pick:.4f}", "captured": pct(captured),
    "n_sched": f"{SCH.party.nunique()}", "model_doors": f"{S['model_doors']}", "fb_doors": f"{S['fallback_doors']}",
    "model_share": pct(S["model_traffic_share"]), "I_mean": f"{HN.I_CH.mean():.0f}",
    "marg_lo": sci(S["marginal_g_per_GB"][0]), "marg_hi": sci(S["marginal_g_per_GB"][1]), "oom": words.get(oom, str(oom)),
    "pj_lo": f"{A.MARGINAL_J_PER_BIT[0] * 1e12:.0f}", "pj_hi": f"{A.MARGINAL_J_PER_BIT[1] * 1e12:.0f}",
    "n_path": f"{A.ROUTERS_PER_PATH}",
    "n_if": f"{len(I_net):,}", "n_links": f"{len(json.load(open(RUN / 'config.json'))['links'])}",
    "n_models": f"{len(models)}", "n_doors": f"{len(doors)}", "n_parties": f"{S['parties']}",
    "n_missing": f"{len(missing)}", "n_dns_routers": f"{len(live)}",
    "n_psu": f"{psu_prod.dev.nunique()}",
    "psu_span": f"{pd.Timestamp(psu_prod['first'].min()).day} {pd.Timestamp(psu_prod['first'].min()):%b %Y} → "
                f"{pd.Timestamp(psu_prod['last'].max()).day} {pd.Timestamp(psu_prod['last'].max()):%b %Y}",
    "psu_var": f"{((dep.W_p95 - dep.W_p5) / 2 / dep.W_mean).median():.1%}",
    "zoo_matched": f"{zoo.dev.dropna().nunique()}", "zoo_total": f"{len(zoo)}",
    "n_shelves": f"{len(sh_live)}", "n_shelf_sites": f"{sh_live.site.nunique()}",
    "n_sites_high": f"{sites[sites.confidence == 'high'].index.nunique()}",
    "n_ptr4": f"{len(dns.read_ptr(REF / 'dns' / 'ptr_130.59.0.0_16.txt')):,}",
    "n_ptr6": f"{len(dns.read_ptr(REF / 'dns' / 'ptr_v6_2001-620-0.txt')):,}",
    "n_dns_pairs": f"{L.peer_link.notna().sum():,}",
    "dns_agree": f"{(both.peer.str.split('/').str[0] == both.dns_peer.str.split('/').str[0]).sum()}",
    "dns_both": f"{len(both)}",
    "n_ext": f"{len(ext)}", "n_ext_sites": f"{ext.switch_site_code.str.rstrip('?').nunique()}",
    "em_span": f"{ch.index.min():%Y} → {ch.index.max():%b %Y}", "em_requests": f"{len(list((G / 'raw').glob('*.json'))):,}",
    "em_est": pct(est),
    "dep_n": f"{len(dep)}", "dep_under1": f"{(dep.dyn_share.abs() < 0.01).sum()}", "dep_max": f"{dep.dyn_share.max():.1%}",
    "dep_big": dbig.dev, "dep_p5": f"{dbig.W_p5:,.0f}", "dep_p95": f"{dbig.W_p95:,.0f}", "dep_gbps": f"{dbig.Gbps_p95:.0f}",
    "opt_share": pct(S["optical_kW"] / S["backbone_kW"]),
    "mwh_nodns": f"{nodns.MWh_W.sum():.0f}", "mwh_dns": f"{R.MWh_W.sum():.0f}",
    "hod_ratio": f"{hod_y.max() / hod_y.min():.0f}", "hod_min": f"{hod_y.min():.0f}", "hod_max": f"{hod_y.max():.0f}",
    "mon_ratio": f"{mon_y.max() / mon_y.min():.0f}", "mon_min": f"{mon_y.min():.0f}", "mon_max": f"{mon_y.max():.0f}",
    "rule_sentence": rule_sentence(),
    "ix_note": " (largest party)" if rank0 else "", "site_ex": site_col.split(":", 1)[1],
    "site_ex_host": host if host and host != "nan" else "customer site",
    "dI_lo": f"{min(worst):.1f}", "dI_hi": f"{max(worst):.1f}", "dI_all": f"{dI.mean():.1f}",
    "hidden_share": pct(hidden_t / tot_t.sum()), "border_share": pct(border_t / tot_t.sum()),
    "modelled_share": pct(1 - R.loc[R.W_src == "measured 2024", "MWh_W"].sum() / R.MWh_W.sum()),
}
html = (HERE / "report_template.html").read_text()
for k, v in N.items():
    html = html.replace("{{" + k + "}}", v)
data = {"dumb": dumb, "grid_only": err("traffic actual + grid fc"), "stack": stack, "hod": hod, "daily": daily,
        "sched": sched, "top": top}
html = html.replace("/*DATA*/", "const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
OUT.write_text(html)
unknown = sorted(set(re.findall(r"\{\{(\w+)\}\}", html)))
print("wrote", OUT, len(html), "bytes; unreplaced:", unknown or "none")
