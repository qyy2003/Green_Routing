"""Build the day-ahead traffic forecast report from run.py's outputs.

usage: python build_report.py [out.html]
       (default $GR_ARTIFACTS_DIR/flat_forecast/report/flat-forecast.html)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
ROOT = ARTIFACTS_DIR / "flat_forecast"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "flat-forecast.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

SD = pd.read_csv(ROOT / "scores_dev.csv").set_index("forecast")
ST = pd.read_csv(ROOT / "scores_test.csv").set_index("forecast")
PA = pd.read_csv(ROOT / "paired.csv")
DK = pd.read_csv(ROOT / "by_day_kind.csv").set_index("forecast")
DG = json.load(open(ROOT / "diagnose.json"))
RE = pd.read_csv(ROOT / "rate_errors.csv").set_index("forecast")
AB = pd.read_csv(ROOT / "scores_abilene.csv").set_index("forecast")
CH = json.load(open(ROOT / "choices.json"))
FINAL = json.load(open(ROOT / "final.json"))
GRAN = ARTIFACTS_DIR / "carbon_granularity" / "switchlan" / "e8_resolution.csv"

CUR = "gbdt: current (168 h, basic calendar) @ fixed, trained to 2026-02-15"
CUR_W = "gbdt: current (168 h, basic calendar)"
TCN0 = "5-min tcn: original run (peak-aware loss)"


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f} %"


def pp(x, d=1):
    """A difference in percentage points, signed."""
    return f"{100 * x:+.{d}f} pp".replace("-", "−")


def label(k: str) -> str:
    k = k.replace("gbdt: ", "GBDT, ").replace("ridge: ", "Ridge, ").replace("rule: ", "")
    k = k.replace("chronos-2", "Chronos-2").replace("chronos-bolt", "Chronos-Bolt").replace("timesfm-2.5", "TimesFM 2.5")
    k = k.replace("5-min tcn", "5-minute TCN").replace("5-min patchtst", "5-minute PatchTST").replace("5-min gru", "5-minute GRU")
    k = k.replace("5-min dlinear", "5-minute DLinear").replace(" @ ", ", ").replace("ensemble: ", "Ensemble: ")
    return k[0].upper() + k[1:]


def get(T, k, col="hourly_WAPE"):
    return float(T.loc[k, col]) if k in T.index else None


def paired(a, b):
    r = PA[(PA.a == a) & (PA.b == b)]
    return r.iloc[0] if len(r) else None


def row(k, note=""):
    p = paired(CUR, k) if k != CUR else None
    return dict(k=label(k), dev=get(SD, k), test=get(ST, k), daily=get(ST, k, "daily_WAPE"), bias=get(ST, k, "bias"),
                diff=None if p is None else float(p["diff"]), lo=None if p is None else float(p["lo"]),
                hi=None if p is None else float(p["hi"]), note=note)


# --- the model families ----------------------------------------------------------------------
best_c, best_g, best_t = CH["best_chronos"], CH["best_gbdt"], CH["best_timesfm"]
final = FINAL["name"]
tcn_sym = "5-min tcn: symmetric loss, mean of 3 seeds"
families = [k for k in [final, best_c, "chronos-2: 24 weeks + calendar covariates", "chronos-bolt: 12 weeks", best_t, best_g, CUR_W, CUR,
                        tcn_sym, TCN0, "5-min patchtst: symmetric loss, mean of 3 seeds", "rule: mean of the same hour, last 4 weeks",
                        "rule: same hour last week"] if k in ST.index]
families = list(dict.fromkeys(families))
models = [row(k) for k in families]

# every forecast, for the full table
everything = [row(k) for k in ST.index]

# --- 5 minutes or hours --------------------------------------------------------------------
res = []
for k, note in (("chronos-2: 4 weeks of context", "hourly values, 4 weeks (672 steps)"),
                ("chronos-2: 5-minute input, 4 weeks", "5-minute values, 4 weeks (8,064 steps), averaged into hours"),
                (best_c, "hourly values, its best context"),
                (tcn_sym, "trained on 5-minute data, forecasts summed into hours"),
                (best_g, "trained on hourly data")):
    if k in ST.index:
        res.append(row(k, note))
gran = None
if GRAN.exists():
    g = pd.read_csv(GRAN)
    g = g[g.grain == "1h"].set_index("source").wape
    gran = {s: float(g[s]) for s in ("direct GBDT at 15min", "direct GBDT at 1h", "5-min tcn") if s in g.index}

# --- prior knowledge ----------------------------------------------------------------------------
cal_keys = ["gbdt: full, no calendar", "gbdt: full, old holiday table", "gbdt: full, national holidays only",
            "gbdt: compact (24 h + same hour 1-28 d + 7 daily means)", "gbdt: compact, weighted holidays",
            "gbdt: compact, full calendar", "gbdt: compact, full calendar, cleaned history",
            "chronos-2: 24 weeks", "chronos-2: 24 weeks + calendar covariates", "chronos-2: 12 weeks",
            "chronos-2: 12 weeks + calendar covariates"]
calendar = [row(k) for k in cal_keys if k in ST.index]
kinds = [c for c in ("weekday", "weekend", "holiday", "week after a holiday") if c in DK.columns]
dk_keys = [k for k in dict.fromkeys([final, best_c, "chronos-2: 24 weeks + calendar covariates", "gbdt: full, old holiday table",
                                     "gbdt: compact, full calendar, cleaned history", CUR, TCN0]) if k in DK.index]
daykind = [dict(k=label(k), v=[float(DK.loc[k, c]) for c in kinds]) for k in dk_keys]
hol = [h for h in DG["holidays"]]

# --- training process -----------------------------------------------------------------------------
train_keys = [(TCN0, "the benchmark's run: under-forecasts cost twice"),
              ("5-min tcn: peak-aware loss, mean of 3 seeds", "same loss, three seeds averaged"),
              (tcn_sym, "symmetric loss (the metric is symmetric), three seeds"),
              ("5-min patchtst: original run (peak-aware loss)", ""), ("5-min patchtst: symmetric loss, mean of 3 seeds", ""),
              ("5-min dlinear: original run (peak-aware loss)", ""), ("5-min dlinear: symmetric loss, mean of 3 seeds", ""),
              ("5-min gru: original run (peak-aware loss)", ""), ("5-min gru: symmetric loss, mean of 3 seeds", ""),
              (CUR, "trained once, to 15 Feb"), (CUR_W, "refitted every week on everything known"),
              (f"{CUR_W} @ refit every 28 d", ""), (f"{CUR_W} @ refit daily", ""), (f"{CUR_W} @ history from 2025-07", "without Feb–Jun 2025"),
              (f"{best_g} @ fixed, trained to 2026-02-15", ""), (best_g, "refitted weekly"), (f"{best_g} @ refit daily", ""),
              (f"{best_g} @ history from 2025-07", ""), (f"{best_g} @ 5 seeds", "five seeds, 80 % of features each"),
              ("gbdt: full, L1 loss", "absolute instead of squared error"), ("gbdt: full, recency weights (half-life 90 d)", ""),
              ("gbdt: full, recency weights (half-life 30 d)", ""), ("gbdt: full, small trees", ""), ("gbdt: full, larger model", ""),
              ("gbdt: full + 168 h", ""), ("ridge: full", "linear model, same features"),
              ("chronos-2: 12 weeks", "zero-shot"), ("chronos-2 fine-tuned: 12 weeks, 1000 steps", "all weights fine-tuned on the series before each period")]
training = [row(k, n) for k, n in train_keys if k in ST.index]

# one lever at a time (paired, test)
lev = PA[(PA.a != CUR) & (PA.a != TCN0)]
levers = [dict(a=label(r.a), b=label(r.b), ea=get(ST, r.a), eb=get(ST, r.b), diff=float(r["diff"]), lo=float(r.lo), hi=float(r.hi),
               days=float(r.days_better)) for _, r in lev.iterrows()]

# --- combining --------------------------------------------------------------------------------------
comb_keys = [k for k in ST.index if k.startswith("ensemble") or "bias-corrected" in k or k.startswith("chronos-2: mean of")]
combining = [row(k) for k in sorted(comb_keys, key=lambda k: get(ST, k))]

# --- diagnosis ------------------------------------------------------------------------------------
dfc = DG["forecasts"]
diag = [dict(k=label(k), w=v["wape"], lvl=v["perfect_daily_level"], lo=v["level_only"], worst=v["worst_tenth_days_share"],
             ac=v["daily_bias_autocorr"]) for k, v in dfc.items() if k in (final, best_c, CUR, TCN0, best_g)]

# --- Abilene ----------------------------------------------------------------------------------------
ab = [dict(k=label(k), test=float(AB.loc[k, "hourly_WAPE"]), daily=float(AB.loc[k, "daily_WAPE"]), bias=float(AB.loc[k, "bias"]))
      for k in AB.index]

# --- the rate -----------------------------------------------------------------------------------------
rate_keys = [k for k in dict.fromkeys(["actual traffic", final, best_c, best_g, CUR, TCN0]) if k in RE.index]
rate = [dict(k=label(k) if k != "actual traffic" else "Actual traffic (perfect forecast)",
             a=float(RE.loc[k, "rate WAPE, actual grid"]) if k != "actual traffic" else 0.0,
             f=float(RE.loc[k, "rate WAPE, grid forecast"]), fd=float(RE.loc[k, "daily rate WAPE, grid forecast"])) for k in rate_keys]

# --- an example fortnight ---------------------------------------------------------------------------
C = pd.read_csv(ROOT / "forecasts.csv.gz", index_col=0, parse_dates=True)
win = slice("2026-05-04", "2026-05-17 23:00")
truth = pd.read_csv(ROOT / "truth.csv.gz", index_col=0, parse_dates=True).iloc[:, 0]
ex = {"t": [x.isoformat() for x in C.loc[win].index], "y": truth.reindex(C.loc[win].index).round(2).tolist(),
      "final": C.loc[win, final].round(2).tolist(), "cur": C.loc[win, CUR].round(2).tolist()}

pf = paired(CUR, final)
pt = paired(TCN0, final) if TCN0 in ST.index else None
N = {
    "built": f"{pd.Timestamp.now().day} {pd.Timestamp.now():%B %Y}",
    "dev_days": str(int(SD.hours.iloc[0] // 24)), "test_days": str(int(ST.hours.iloc[0] // 24)),
    "final": label(final), "final_test": pct(get(ST, final)), "final_dev": pct(get(SD, final)),
    "final_daily": pct(get(ST, final, "daily_WAPE")), "cur_test": pct(get(ST, CUR)), "cur_daily": pct(get(ST, CUR, "daily_WAPE")),
    "tcn0_test": pct(get(ST, TCN0)), "gain_cur": f"{-100 * pf['diff']:.1f}", "gain_lo": f"{-100 * pf['hi']:.1f}", "gain_hi": f"{-100 * pf['lo']:.1f}",
    "gain_rel": pct(-pf["diff"] / get(ST, CUR), 0), "days_better": pct(pf["days_better"], 0),
    "gain_tcn": pp(-pt["diff"]) if pt is not None else "—",
    "c_best": label(best_c), "c_dev": pct(get(SD, best_c)), "c_test": pct(get(ST, best_c)),
    "c4w": pct(get(ST, "chronos-2: 4 weeks of context")), "c5": pct(get(ST, "chronos-2: 5-minute input, 4 weeks")),
    "c_cov_test": pct(get(ST, "chronos-2: 24 weeks + calendar covariates")), "c24_test": pct(get(ST, "chronos-2: 24 weeks")),
    "c_cov_dev": pct(get(SD, "chronos-2: 24 weeks + calendar covariates")), "c24_dev": pct(get(SD, "chronos-2: 24 weeks")),
    "tf_test": pct(get(ST, best_t)), "bolt_test": pct(get(ST, "chronos-bolt: 12 weeks")),
    "g_best": label(best_g), "g_test": pct(get(ST, best_g)), "g_dev": pct(get(SD, best_g)),
    "g_old": pct(get(ST, "gbdt: full, old holiday table")), "g_new": pct(get(ST, "gbdt: compact, full calendar, cleaned history")),
    "g_nocal": pct(get(ST, "gbdt: full, no calendar")),
    "tcn_sym": pct(get(ST, tcn_sym)), "tcn_uw2": pct(get(ST, "5-min tcn: peak-aware loss, mean of 3 seeds")),
    "tcn0_bias": pct(get(ST, TCN0, "bias")), "tcn_sym_bias": pct(get(ST, tcn_sym, "bias")),
    "ft_test": pct(get(ST, "chronos-2 fine-tuned: 12 weeks, 1000 steps")), "ft_dev": pct(get(SD, "chronos-2 fine-tuned: 12 weeks, 1000 steps")),
    "c12_test": pct(get(ST, "chronos-2: 12 weeks")), "c12_dev": pct(get(SD, "chronos-2: 12 weeks")),
    "lvl": pct(dfc[final]["perfect_daily_level"]) if final in dfc else "—", "lvlonly": pct(dfc[final]["level_only"]) if final in dfc else "—",
    "worst": pct(dfc[final]["worst_tenth_days_share"], 0) if final in dfc else "—",
    "top4_var": pct(DG["doors"]["top4_variance_share"], 0), "top4_gbps": pct(DG["doors"]["top4_traffic_share"], 0),
    "ab_c": pct(float(AB.loc["chronos-2: 12 weeks + holiday covariate", "hourly_WAPE"])),
    "ab_g": pct(float(AB.loc["gbdt: current (168 h, basic calendar) @ fixed, trained to 2004-07-24", "hourly_WAPE"])),
    "ab_p": pct(float(AB.loc["5-min patchtst: original run (peak-aware loss)", "hourly_WAPE"])),
    "rate_a_cur": pct(float(RE.loc[CUR, "rate WAPE, actual grid"])), "rate_a_new": pct(float(RE.loc[final, "rate WAPE, actual grid"])),
    "rate_f_cur": pct(float(RE.loc[CUR, "rate WAPE, grid forecast"])), "rate_f_new": pct(float(RE.loc[final, "rate WAPE, grid forecast"])),
    "rate_f_perfect": pct(float(RE.loc["actual traffic", "rate WAPE, grid forecast"])),
    "w_dev": f"{CH['chronos_weight_dev']:.1f}",
}
# the combining sentence, from the numbers (members and weights chosen on DEV)
ens_dev = [k for k in comb_keys if k.startswith("ensemble") and k in SD.index]
best_ens = min(ens_dev, key=lambda k: get(SD, k)) if ens_dev else None
bc = [k for k in comb_keys if "bias-corrected" in k and k.startswith(best_c) and k in SD.index]
best_bc = min(bc, key=lambda k: get(SD, k)) if bc else None
parts = []
if best_ens:
    d = get(ST, best_ens) - get(ST, best_c)
    parts.append(f"The best ensemble on the development period, {label(best_ens).replace('Ensemble: ', '')}, scores {pct(get(ST, best_ens))} "
                 f"on the test, {'better' if d < 0 else 'worse'} than Chronos-2 alone ({pct(get(ST, best_c))}) by {abs(100 * d):.1f} points: "
                 + ("the GBDT's errors are partly independent of Chronos-2's." if d < 0 else "the GBDT adds noise rather than information."))
if best_bc:
    d = get(ST, best_bc) - get(ST, best_c)
    parts.append(f"Correcting Chronos-2 with its recent bias ({label(best_bc).split(', ')[-1]}) {'helps' if d < -0.001 else 'does not help'} "
                 f"({pct(get(ST, best_bc))}): its errors do not persist from day to day.")
N["comb_text"] = " ".join(parts)
# combining: the holiday rule vs the plain blend, and the test-only ensemble with the TCN
blend = [k for k in comb_keys if "weight from DEV" in k]
sw_tcn = "ensemble: GBDT + Chronos-2 + TCN, GBDT alone on holidays"
if blend:
    N["comb_text"] = (f"Averaging Chronos-2 with the GBDT ({N['w_dev']} to {1 - CH['chronos_weight_dev']:.1f}, the weight chosen on the development period) "
                      f"gives {pct(get(ST, blend[0]))}; adding the holiday rule gives {pct(get(ST, final))}. "
                      + (f"Adding the symmetric TCN as a third member (equal weights; it has no development-period forecasts, so it could not be chosen) "
                         f"gives {pct(get(ST, sw_tcn))}." if sw_tcn in ST.index else ""))
ratio = {h["date"]: h["ratio"] for h in DG["holidays"]}
cut = lambda ds: f"{100 * (1 - max(ratio[d] for d in ds)):.0f}–{100 * (1 - min(ratio[d] for d in ds)):.0f} %"      # noqa: E731
N["hol25"] = cut(["2025-04-18", "2025-04-21", "2025-05-29", "2025-06-09"])
N["hol26"] = cut(["2026-04-03", "2026-04-06", "2026-05-14", "2026-05-25"])
N["xmas"] = cut(["2025-12-25", "2025-12-26", "2026-01-01", "2026-01-02"])
N["cc"] = cut(["2025-06-19", "2026-06-04"])
N["c_hol"] = pct(float(DK.loc[best_c, "holiday"]), 0)
N["c_cov_hol"] = pct(float(DK.loc["chronos-2: 24 weeks + calendar covariates", "holiday"]), 0)
N["g_hol"] = pct(float(DK.loc[CUR_W, "holiday"]), 0)
g5 = f"{CUR_W} @ 5 seeds"
N["g5_dev"], N["g5_test"] = pct(get(SD, g5)), pct(get(ST, g5))
pt2 = paired("5-min tcn: peak-aware loss, mean of 3 seeds", tcn_sym)
if pt2 is not None:
    N["tcn_gain"] = f"{100 * pt2['diff']:+.1f}".replace("-", "−")
    N["tcn_lo"], N["tcn_hi"] = f"{100 * pt2['lo']:.1f}".replace("-", "−"), f"{100 * pt2['hi']:.1f}".replace("-", "−")
pr = paired(CUR, CUR_W)
N["refit_gain"], N["refit_lo"], N["refit_hi"] = (f"{100 * pr[c]:+.1f}".replace("-", "−") for c in ("diff", "lo", "hi"))
fr = [float(RE.loc[k, "rate WAPE, grid forecast"]) for k in rate_keys if k != "actual traffic"]
N["rate_f_range"] = f"{100 * min(fr):.0f}–{100 * max(fr):.0f} %"
if gran:
    N.update(gr_15=pct(gran.get("direct GBDT at 15min")), gr_1h=pct(gran.get("direct GBDT at 1h")), gr_tcn=pct(gran.get("5-min tcn")))
html = (HERE / "report_template.html").read_text()
style = re.search(r"<style>.*?</style>", (HERE.parent / "carbon_rates" / "report_template.html").read_text(), re.S).group(0)
html = html.replace("/*STYLE*/", style.replace("</style>", "tr.final td { font-weight: 600; }\n.multi { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }\n</style>"))
for k, v in N.items():
    html = html.replace("{{" + k + "}}", str(v))
data = dict(models=models, everything=everything, res=res, gran=gran, calendar=calendar, kinds=kinds, daykind=daykind, hol=hol,
            training=training, levers=levers, combining=combining, diag=diag, ab=ab, rate=rate, ex=ex, final=label(final))
html = html.replace("/*DATA*/", "const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
OUT.write_text(html)
print("wrote", OUT, len(html), "bytes; unreplaced:", sorted(set(re.findall(r"\{\{(\w+)\}\}", html))) or "none",
      "; NaN!" if "NaN" in html else "")
