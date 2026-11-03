"""Build the carbon-rate granularity report from run.py's outputs.

usage: python build_report.py [out.html]
       (default $GR_ARTIFACTS_DIR/carbon_granularity/report/carbon-granularity.html)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
ROOT = ARTIFACTS_DIR / "carbon_granularity"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "carbon-granularity.html"
OUT.parent.mkdir(parents=True, exist_ok=True)
GRAINS = ["5min", "15min", "1h", "3h", "6h", "1d"]
LAB = {"5min": "5 min", "15min": "15 min", "1h": "1 h", "3h": "3 h", "6h": "6 h", "1d": "1 day"}
MOD = {"persistence": "Persistence", "seasonal_naive_day": "Same time yesterday", "seasonal_naive_week": "Same time last week",
       "historical_average": "Historical average", "gru": "GRU", "tcn": "TCN", "dlinear": "DLinear", "patchtst": "PatchTST"}


def load(ds):
    d = ROOT / ds
    return {k: pd.read_csv(d / f"{k}.csv") for k in ("e1_settlement", "e2_forecast_by_grain", "e4_curves", "e5_distributions",
                                                    "e5_reliability", "e5_decisions", "e6_cadence", "e7_scheduling", "e8_resolution")} | \
        {"summary": json.load(open(d / "summary.json"))}


SW, AB = load("switchlan"), load("abilene")


def pct(x, d=0):
    return f"{100 * x:.{d}f} %"


def by_grain(t, col):
    t = t.set_index("grain")
    return [float(t.loc[g, col]) if g in t.index else None for g in GRAINS]


def e2(D, source, gridname, col="error_vs_5min"):
    t = D["e2_forecast_by_grain"]
    t = t[(t.source == source) & (t.grid == gridname)]
    return by_grain(t, col)


sw_model, ab_model = SW["summary"]["best_5min_model"], AB["summary"]["best_5min_model"]
settle = {"sw_var": by_grain(SW["e1_settlement"], "variance_kept"), "sw_dbe": by_grain(SW["e1_settlement"], "daily_bill_error"),
          "ab_var": by_grain(AB["e1_settlement"], "variance_kept"), "ab_dbe": by_grain(AB["e1_settlement"], "daily_bill_error")}
fc = {"sw_fc": e2(SW, sw_model, "grid forecast"), "sw_act": e2(SW, sw_model, "actual grid"), "ab": e2(AB, ab_model, "grid forecast")}

# model grain table: every 5-minute model summed, and the direct GBDTs (real grid on SWITCHlan)
rows = []
srcs = sorted(set(SW["e2_forecast_by_grain"].source), key=lambda s: (s.startswith("direct"), s))
for s in srcs:
    def get(D, g, gridname):
        t = D["e2_forecast_by_grain"]
        t = t[(t.source == s) & (t.grain == g) & (t.grid == gridname)]
        return float(t.error_vs_5min.iloc[0]) if len(t) else None
    name = ("5-minute " + MOD.get(s.split("5-min ")[1], s.split("5-min ")[1]) + ", summed") if s.startswith("5-min") else \
        s.replace("direct GBDT at ", "GBDT trained on ") .replace("15min", "15-minute data").replace(" 1h", " hourly data") \
         .replace(" 3h", " 3-hour data").replace(" 6h", " 6-hour data").replace(" 1d", " daily data")
    rows.append(dict(s=name, sw1=get(SW, "1h", "actual grid"), sw3=get(SW, "3h", "actual grid"),
                     ab1=get(AB, "1h", "grid forecast"), ab3=get(AB, "3h", "grid forecast")))
grain_rows = [r for r in rows if any(v is not None for v in (r["sw1"], r["sw3"], r["ab1"], r["ab3"]))]

# the same traffic forecast at each resolution (best 5-minute model and the hourly GBDT)
def res_row(D, src, label):
    t = D["e8_resolution"]
    t = t[t.source == src].set_index("grain")
    return dict(s=label, v=[float(t.loc[g, "wape"]) if g in t.index else None for g in GRAINS])


res = [res_row(SW, sw_model, "SWITCHlan, 5-minute " + MOD[sw_model.split("5-min ")[1]]),
       res_row(SW, "direct GBDT at 1h", "SWITCHlan, GBDT trained on hourly data"),
       res_row(AB, ab_model, "Abilene, 5-minute " + MOD[ab_model.split("5-min ")[1]]),
       res_row(AB, "direct GBDT at 1h", "Abilene, GBDT trained on hourly data")]
skill = {D["summary"]["dataset"]: D["e8_resolution"].dropna(subset=["within_hour_skill"]).groupby("source").within_hour_skill.first()
         for D in (SW, AB)}
learned = [f"5-min {m}" for m in ("tcn", "patchtst", "gru", "dlinear")]
sk_sw = skill["switchlan"].reindex(learned).dropna()

c_sw, c_ab = SW["e4_curves"].set_index("curve"), AB["e4_curves"].set_index("curve")
CURVE = {"raw 5-minute values": "Values every 5 minutes", "staircase, 15min values": "Values every 15 minutes",
         "staircase, 1h values": "Values every hour", "staircase, 3h values": "Values every 3 hours",
         "curve through 1h knots": "Line through hourly points", "curve through 3h knots": "Line through 3-hourly points"}


def curve_name(c):
    if c in CURVE:
        return CURVE[c]
    k = re.search(r"(\d+) harmonics", c).group(1)
    return (f"Fourier curve of the rate, {k} harmonics" if "rate" in c else f"Fourier curve, {k} harmonics")


curves = [dict(c=curve_name(c), p=int(c_sw.loc[c, "parameters_per_day"]), sw=float(c_sw.loc[c, "error_vs_5min"]),
               ab=float(c_ab.loc[c, "error_vs_5min"])) for c in c_sw.index]
min_k = min(int(re.search(r"(\d+) harmonics", r["c"]).group(1)) for r in curves
            if r["c"].startswith("Fourier curve,") and r["sw"] <= c_sw.loc["staircase, 1h values", "error_vs_5min"] + 0.0005)

cad_sw = SW["e6_cadence"][SW["e6_cadence"].grain == "1h"]
cad_ab = AB["e6_cadence"][(AB["e6_cadence"].grain == "1h") & (AB["e6_cadence"].grid_rule == "constant")].set_index("lead")
cadence = []
for L, lab in (("15min", "15 minutes ahead"), ("1h", "1 hour ahead"), ("8h", "8 hours ahead"), ("day", "day ahead (00:00 UTC)")):
    t = cad_sw[cad_sw.lead == L].set_index("grid_rule").error_vs_5min
    cadence.append(dict(l=lab, last=float(t["last known hour"]) if "last known hour" in t else None, yest=float(t["same hour yesterday"]),
                        act=float(t["actual grid"]), ab=float(cad_ab.loc[L, "error_vs_5min"])))
best_sw = {r["l"]: min(v for v in (r["last"], r["yest"]) if v is not None) for r in cadence}

cov_sw, cov_ab = SW["e5_distributions"], AB["e5_distributions"]
msw, mab = SW["summary"]["main_distribution"], AB["summary"]["main_distribution"]
main_sw = cov_sw[(cov_sw.window_days == msw["window_days"]) & (cov_sw.hours_pooled == msw["hours_pooled"]) & (cov_sw.uncertainty == "traffic + grid")].iloc[0]
traffic_only = cov_sw[(cov_sw.uncertainty == "traffic only")].iloc[0]
cov = []
for D, net in ((cov_sw, "SWITCHlan"), (cov_ab, "Abilene")):
    for r in D.itertuples():
        cov.append(dict(k=f"{net}, {r.uncertainty}", w=int(r.window_days), h=int(r.hours_pooled), c80=r.coverage_80, c90=r.coverage_90,
                        wd=r.width_80_rel, sk=1 - r.crps / r.mae_point))
rel = {"sw": {"level": SW["e5_reliability"].level.tolist(), "obs": SW["e5_reliability"].observed.tolist()},
       "ab": {"level": AB["e5_reliability"].level.tolist(), "obs": AB["e5_reliability"].observed.tolist()}}
dsw, dab = SW["e5_decisions"].set_index("choice"), AB["e5_decisions"].set_index("choice")
dec = [dict(c=c, sw=float(dsw.loc[c, "realised"]), ab=float(dab.loc[c, "realised"])) for c in dsw.index] + \
      [dict(c="day average (no shifting)", sw=float(dsw.day_average.iloc[0]), ab=float(dab.day_average.iloc[0])),
       dict(c="best hour in hindsight", sw=float(dsw.hindsight.iloc[0]), ab=float(dab.hindsight.iloc[0]))]
s7, a7 = SW["e7_scheduling"].set_index("grain"), AB["e7_scheduling"].set_index("grain")
sg = [g for g in GRAINS if g in s7.index]
sched = {"grains": [LAB[g] for g in sg], "sw_p": [float(s7.loc[g, "perfect_captured"]) for g in sg],
         "sw_f": [float(s7.loc[g, "forecast_captured"]) for g in sg], "ab_p": [float(a7.loc[g, "perfect_captured"]) for g in sg],
         "ab_f": [float(a7.loc[g, "forecast_captured"]) for g in sg]}
e1s = SW["e1_settlement"].set_index("grain")


def src_err(D, s, g, gridname):
    t = D["e2_forecast_by_grain"]
    return float(t[(t.source == s) & (t.grain == g) & (t.grid == gridname)].error_vs_5min.iloc[0])


N = {
    "built": f"{pd.Timestamp.now().day} {pd.Timestamp.now():%B %Y}", "sw_days": str(SW["summary"]["days"]), "ab_days": str(AB["summary"]["days"]),
    "sw_parties": str(SW["summary"]["parties"]), "ab_parties": str(AB["summary"]["parties"]),
    "sw_var_1h": pct(settle["sw_var"][2], 1), "ab_var_1h": pct(settle["ab_var"][2], 1), "sw_var_3h": pct(settle["sw_var"][3]),
    "sw_var_1d": pct(settle["sw_var"][5]), "sw_dbe_1h": pct(settle["sw_dbe"][2], 1), "ab_dbe_1h": pct(settle["ab_dbe"][2], 1),
    "sw_share_1pct": pct(e1s.loc["1h", "bill_share_over_1pct"], 1), "sw_share_1pct_3h": pct(e1s.loc["3h", "bill_share_over_1pct"], 0),
    "sw_sum_1h": pct(src_err(SW, sw_model, "1h", "actual grid"), 1), "sw_dir_1h": pct(src_err(SW, "direct GBDT at 1h", "1h", "actual grid"), 1),
    "ab_sum_1h": pct(src_err(AB, ab_model, "1h", "grid forecast"), 1), "ab_dir_1h": pct(src_err(AB, "direct GBDT at 1h", "1h", "grid forecast"), 1),
    "sw_res_5": pct(res[0]["v"][0], 1), "sw_res_1h": pct(res[0]["v"][2], 1), "sw_res_1d": pct(res[0]["v"][5], 1),
    "ab_res_5": pct(res[2]["v"][0], 1), "ab_res_1h": pct(res[2]["v"][2], 1),
    "sw_wh": pct(SW["summary"]["within_hour_share"], 1), "ab_wh": pct(AB["summary"]["within_hour_share"], 1),
    "sw_sk_lo": pct(sk_sw.min(), 0), "sw_sk_hi": pct(sk_sw.max(), 0),
    "sw_snw_sk": pct(skill["switchlan"]["5-min seasonal_naive_week"], 0),
    "sw_day": pct(best_sw["day ahead (00:00 UTC)"], 0), "sw_1h": pct(best_sw["1 hour ahead"], 0),
    "ab_day": pct(cadence[3]["ab"], 1), "ab_1h": pct(cadence[1]["ab"], 1),
    "sw_trafday": pct(cadence[3]["act"], 0), "sw_traf1h": pct(cadence[1]["act"], 0),
    "fourier_k": str(min_k), "fourier_p": str(2 * min_k + 1),
    "q_window": str(msw["window_days"]), "q_window_ab": str(mab["window_days"]),
    "sw_cov80": pct(main_sw.coverage_80, 1), "sw_cov90": pct(main_sw.coverage_90, 1), "sw_crps": pct(1 - main_sw.crps / main_sw.mae_point, 0),
    "sw_cov80_traffic": pct(traffic_only.coverage_80, 0),
    "dec_sw": pct(1 - dsw.loc["P90 (risk-averse)", "realised"] / dsw.loc["point forecast", "realised"], 0),
    "dec_ab": pct(1 - dab.loc["P90 (risk-averse)", "realised"] / dab.loc["point forecast", "realised"], 0),
    "p90_exc": pct(SW["summary"]["p90_exceedance"], 0), "p90_prem": pct(SW["summary"]["p90_premium"], 0),
    "sw_model": MOD.get(sw_model.split("5-min ")[1]), "ab_model": MOD.get(ab_model.split("5-min ")[1]),
    "sw_dbe_fc_1h": pct(e2(SW, sw_model, "grid forecast", "daily_bill_error")[2], 0),
    "sw_dbe_fc_1d": pct(e2(SW, sw_model, "grid forecast", "daily_bill_error")[5], 0),
    "ab_dbe_fc_1h": pct(e2(AB, ab_model, "grid forecast", "daily_bill_error")[2], 0),
    "ab_dbe_fc_1d": pct(e2(AB, ab_model, "grid forecast", "daily_bill_error")[5], 0),
    "sw_sched_1h": pct(s7.loc["1h", "perfect_captured"], 0), "ab_sched_1h": pct(a7.loc["1h", "perfect_captured"], 0),
    "sw_sched_fc": pct(s7.loc[["5min", "15min", "1h"], "forecast_captured"].mean(), 0),
}
html = (HERE / "report_template.html").read_text()
style = re.search(r"<style>.*?</style>", (HERE.parent / "carbon_rates" / "report_template.html").read_text(), re.S).group(0)
html = html.replace("/*STYLE*/", style.replace("</style>", ".kref { fill: var(--ref); }\n.multi { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }\n.multi > div { min-width: 0; }\n</style>"))
for k, v in N.items():
    html = html.replace("{{" + k + "}}", v)
data = {"grains": [LAB[g] for g in GRAINS], "settle": settle, "fc": fc, "grain": grain_rows, "res": res, "curves": curves, "cadence": cadence,
        "rel": rel, "cov": cov, "dec": dec, "sched": sched}
html = html.replace("/*DATA*/", "const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
OUT.write_text(html)
print("wrote", OUT, len(html), "bytes; unreplaced:", sorted(set(re.findall(r"\{\{(\w+)\}\}", html))) or "none",
      "; NaN!" if "NaN" in html else "")
