"""The appendix's forecasting and steering details (paper appendix E-L), on the common periods of periods.py
(round 2, 2026-11-08): the same test block, development year and training window for both networks.

- models:   every forecaster on the network total and per link, both networks: hourly traffic error, daily bill
            error with the paper's grid forecast, training cost (GPU minutes from the training logs); per link
            from the block logs of ladder_common.sh and timestral_links.py (day ahead, median link hourly WAPE).
- time:     context length (Timestral 1-24 weeks, the hourly architectures and the 5-minute transformer of
            appendix_runs.sh), input resolution, forecast horizon (1 h ... week).
- levels:   Level 1 (one rate) vs Level 2 (per path): error of traffic, rate, bills.
- grid:     the grid-forecast candidates on SWITCHlan (development year, test block; flat_common.py grid).
- decision: does the day-ahead choice equal the choice made on the settled rates (steering.py).
- cap:      the cap's k (1, 2, 3, none): level, share capped, rate error, steering (steering.py --k).
- bias:     where forecast bias sits (traffic / grid, network / link, bill), SWITCHlan's grid forecasts by month
            over the development year, the test block and the later period.
- params:   parameters of every trained network (checkpoints).

Errors over periods are on calendar days, 7-day blocks from the period start and calendar months (no overlapping
windows). Run with the overlay root, after flat_common.py, i2_rates.py, steering.py (k = 2, 1, 3, none),
total_neural.py, ladder_common.sh (links, total) and appendix_runs.sh:

usage: GR_ARTIFACTS_DIR=$PAPER_COMMON/root python appendix.py
writes $GR_ARTIFACTS_DIR/paper/appendix/*.csv, numbers.json and the PDFs into
<repo>/6aea077f0929c9a3a1ac5660/figure_appendix (or the directory given as the first argument).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

from green_routing.paths import ARTIFACTS_DIR, REPO_ROOT   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import periods as PER                      # noqa: E402

A = ARTIFACTS_DIR                          # the overlay root
P = A / "paper"
OUT = P / "appendix"
PC = PER.ROOT
plt.style.use(HERE / "paper.mplstyle")     # matplotlib's own fonts, also where LaTeX is installed
SW, I2 = "#348ABD", "#A60628"
PCT = "%"
PNG = None
DENSE = {n: PC / "links" / n / "merged" for n in ("switch", "i2")}
FM = {n: PC / "links" / n / "dense_fm" for n in ("switch", "i2")}
LDC = A / "ladder_diagnosis"               # appendix_runs.sh's cache (overlay)
# recipes/ladder_diagnosis/common.py's DS on the round-2 runs (as appendix_runs.sh sets them)
import os                                  # noqa: E402
for _n in ("i2", "switch"):
    _p = sorted((PC / f"cache_{_n}").glob("panel_*.npz"))
    os.environ.setdefault(f"GR_LD_RUN_{_n.upper()}", str(DENSE[_n]))
    if _p:
        os.environ.setdefault(f"GR_LD_PANEL_{_n.upper()}", str(_p[0]))
    os.environ.setdefault(f"GR_LD_FM_{_n.upper()}", str(FM[_n]))
sys.path.insert(0, str(HERE.parent / "ladder_diagnosis"))


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    if PNG:
        fig.savefig(PNG / f"{name}.png", dpi=200)
    plt.close(fig)


# --- training cost ------------------------------------------------------------------------------
def train_minutes(log: Path) -> dict:
    """GPU minutes and epochs per model from a gr-benchmark log ("[tcn] epoch 3/100 ... (28.0s)")."""
    out = {}
    if not log.exists():
        return out
    for line in log.read_text(errors="ignore").splitlines():
        m = re.match(r"\[(\w+)\] epoch (\d+)/\d+ .*\(([\d.]+)s\)", line)
        if m:
            d = out.setdefault(m.group(1), {"minutes": 0.0, "epochs": 0})
            d["minutes"] += float(m.group(3)) / 60
            d["epochs"] = max(d["epochs"], int(m.group(2)))
    return out


def costs() -> pd.DataFrame:
    rows = []
    for net in ("switch", "i2"):
        for part in ("part_dlinear_patchtst", "part_gru", "part_tcn"):
            for model, d in train_minutes(PC / "links" / net / part / "run.log").items():
                rows.append(dict(network=net, target="links", model=model, **d))
    for m in ("dlinear", "gru", "tcn", "patchtst"):
        for s in (0, 1, 2):
            for model, d in train_minutes(PC / "total" / f"{m}_uw1_s{s}" / "run.log").items():
                rows.append(dict(network="switch", target=f"total (seed {s})", model=model, **d))
    return pd.DataFrame(rows)


# --- the network total: every forecaster, traffic and bill ---------------------------------------
SWITCH_MODELS = {"Persistence (same hour yesterday)": None, "Same hour last week": "rule: same hour last week",
                 "GBDT": "gbdt: current (168 h, basic calendar)",
                 "Chronos-2": "chronos-2: 24 weeks", "TimesFM-2.5": "timesfm-2.5: 48 weeks"}
I2_MODELS = {"Persistence (same hour yesterday)": ("d", "persistence (same hour yesterday)"),
             "Same hour last week": ("d", "same hour last week"), "GBDT": ("d", "GBDT"),
             "Chronos-2": ("t", "Chronos-2, 24 weeks")}


def bill_scores(T, Tf, E, Ef) -> dict:
    """Traffic hourly WAPE and bias; bill error per calendar day, 7-day block and calendar month (test block)."""
    ok = T.notna() & Tf.notna() & E.notna() & Ef.notna()
    T, Tf, E, Ef = T[ok], Tf[ok], E[ok], Ef[ok]
    bill = Ef * T / Tf
    g = bill.index.floor("D").tz_localize(None)
    D = pd.DataFrame({"bill": bill.groupby(g).sum(), "E": E.groupby(g).sum(), "A": 0.0, "Af": 0.0})
    import flat_common as FC
    return {"traffic_hourly_wape": float((Tf - T).abs().sum() / T.sum()),
            "traffic_bias": float(Tf.sum() / T.sum() - 1),
            "bill_day_wape": FC.wape(FC.period_sums(D, PER.TEST, "day")),
            "bill_week_wape": FC.wape(FC.period_sums(D, PER.TEST, "week")),
            "bill_month_wape": FC.wape(FC.period_sums(D, PER.TEST, "month")), "hours": int(ok.sum())}


def ladder_total(m: str, uw: int = 1) -> pd.Series | None:
    """The 5-minute ladder on SWITCHlan's total (ladder_common.sh total): mean of three seeds, hourly, one-way."""
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    import run as R
    seeds = [R.ladder_hourly(PC / "total" / f"{m}_uw{uw}_s{s}", m) for s in range(3)]
    seeds = [x for x in seeds if x is not None]
    return sum(seeds) / len(seeds) if len(seeds) == 3 else None


def flat_hours(net: str) -> pd.DataFrame:
    h = pd.read_csv(PC / "flat" / f"hourly_{net}.csv.gz", index_col=0, parse_dates=True)
    return h[h.period == "TEST"]


def total_models() -> pd.DataFrame:
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    import run as R
    hs = flat_hours("switch")
    cache = R.cache()
    rows = []
    fc = {name: (R.S.shift(24) if key is None else cache[key]) for name, key in SWITCH_MODELS.items()}
    for m, lab in (("dlinear", "DLinear"), ("gru", "GRU"), ("tcn", "TCN"), ("patchtst", "PatchTST")):
        f = ladder_total(m)
        if f is not None:
            fc[lab] = f
    fc["Timestral"] = hs.Tf
    for name, Tf in fc.items():
        rows.append(dict(network="switch", model=name, **bill_scores(hs["T"], Tf.reindex(hs.index), hs.E, hs.Ef)))
    hi = flat_hours("i2")
    d = pd.read_csv(A / "internet2" / "carbon" / "dayahead_hourly.csv", index_col=0, parse_dates=True)
    t = pd.read_csv(A / "internet2" / "carbon" / "timestral_total.csv", index_col=0, parse_dates=True)
    for name, (src, col) in I2_MODELS.items():
        Tf = (d if src == "d" else t)[col].reindex(hi.index)
        rows.append(dict(network="i2", model=name, **bill_scores(hi["T"], Tf, hi.E, hi.Ef)))
    rows.append(dict(network="i2", model="Timestral", **bill_scores(hi["T"], hi.Tf, hi.E, hi.Ef)))
    for net, h in (("i2", hi), ("switch", hs)):
        f = OUT / f"total_neural_{net}.csv"
        if not f.exists():
            continue
        nn_ = pd.read_csv(f, index_col=0, parse_dates=True)
        for arch, (lab, *_) in ARCH.items():
            name = lab if net == "i2" else f"{lab} (hourly, 4 weeks)"
            rows.append(dict(network=net, model=name, **bill_scores(h["T"], nn_[arch].reindex(h.index), h.E, h.Ef)))
    return pd.DataFrame(rows)


def link_models() -> pd.DataFrame:
    """Day-ahead median link hourly WAPE per model (common links and targets with Timestral)."""
    rows = []
    for net in DENSE:
        p = pd.read_csv(FM[net] / "tables_both" / "block" / "per_link.csv")
        t = p[p.horizon_name == "day"].groupby("model").wape_1h.median()
        for m, v in t.items():
            rows.append(dict(network=net, model=m, link_day_wape=v / 100, links=int(p.link.nunique())))
    return pd.DataFrame(rows)


def horizons() -> pd.DataFrame:
    """Median link hourly WAPE by horizon: the ladder's full block logs (all horizons) and Timestral's
    hourly-input logs (1 h on Internet2 only, 8 h, day, week)."""
    rows = []
    for net in DENSE:
        p = pd.read_csv(DENSE[net] / "dense" / "block" / "per_link.csv")
        q = pd.read_csv(FM[net] / "tables_1h_24w" / "block" / "per_link.csv")
        q = q[q.model == "timestral@1h_24w"]
        for src in (p, q):
            g = src.groupby(["model", "horizon_name"]).wape_1h.median()
            for (m, h), v in g.items():
                rows.append(dict(network=net, model=m, horizon=h, link_wape=v / 100))
    return pd.DataFrame(rows)


def _ds(net):
    sys.path.insert(0, str(HERE.parent / "ladder_diagnosis"))
    from common import DS
    return DS(net)


def timestral_context() -> pd.DataFrame:
    """Timestral by context (timestral_links.py ctx): median link hourly WAPE, day and week, on the points every
    reference forecasts (the ladder's PatchTST, Timestral 24 w), as timestral_ctx.py scores them."""
    from common import score
    rows = []
    for net in DENSE:
        ds = _ds(net)
        A_ = ds.actual_test()
        for hz in ("day", "week"):
            refs = {m: ds.load_dense_block(m, hz) for m in ("patchtst", "timestral@1h_24w")}
            for f in sorted((PC / "links" / net / "ctx").glob(f"pred_{hz}_c*.npy")):
                ctx = int(f.stem.split("_c")[1])
                df, _ = score({**refs, "x": np.load(f)}, A_, ref="patchtst")
                r = df.set_index("model").loc["x"]
                rows.append(dict(network=net, horizon=hz, ctx=ctx, weeks=ctx / 168, median_wape=float(r.median_wape),
                                 median_link_bias=float(r.median_link_bias), beats_patchtst=float(r.beats_ref)))
    return pd.DataFrame(rows)


def run_summary(net: str, tag: str) -> dict | None:
    f = LDC / net / "runs" / tag / "summary.json"
    return json.load(open(f)) if f.exists() else None


def context() -> pd.DataFrame:
    """Day-ahead median link hourly WAPE by context (weeks): Timestral (hourly input), the 5-minute
    multi-resolution transformer (mean of 3 seeds), the hourly architectures, the ladder (PatchTST, 1 week)."""
    rows = []
    tc = timestral_context()
    tc.to_csv(OUT / "timestral_context.csv", index=False)
    for _, r in tc[tc.horizon == "day"].iterrows():
        rows.append(dict(network=r.network, model="Timestral (hourly)", weeks=r.weeks, day=r.median_wape / 100))
    for net in DENSE:
        for w, tags in ((1, ["L5s_seg_full_1w", "L5s_seg_full_1w_s1", "L5s_seg_full_1w_s2"]),
                        (4, ["L5s_seg_multires_4w", "L5s_seg_multires_4w_s1", "L5s_seg_multires_4w_s2"]),
                        (12, ["L5s_seg_multires_12w"]),
                        (24, ["L5s_seg_multires_24w", "L5s_seg_multires_24w_s1", "L5s_seg_multires_24w_s2"])):
            v = [run_summary(net, t) for t in tags]
            v = [x["test"]["day"]["median_wape"] for x in v if x]
            if v:
                rows.append(dict(network=net, model="trained (5-min)", weeks=float(w), day=float(np.mean(v)) / 100,
                                 seeds=len(v)))
        for arch in ("dlinear", "gru", "tcn", "patchtst"):
            for ctx in (168, 672, 4032):
                d = run_summary(net, f"H_{arch}_c{ctx}")
                if d:
                    rows.append(dict(network=net, model=f"{arch} (hourly)", weeks=ctx / 168,
                                     day=d["test"]["day"]["median_wape"] / 100))
        lk = link_models()
        v = lk[(lk.network == net) & (lk.model == "patchtst")].link_day_wape
        if len(v):
            rows.append(dict(network=net, model="ladder PatchTST (5-min)", weeks=1.0, day=float(v.iloc[0])))
        u = lk[(lk.network == net) & (lk.model == "timestral@5m_4w")].link_day_wape
        if len(u):
            rows.append(dict(network=net, model="Timestral (5-min)", weeks=4.0, day=float(u.iloc[0])))
        for tag in ("D_pt_5m_log_huber", "A_pt_c168_log_huber"):
            d = run_summary(net, tag)
            if d:
                rows.append(dict(network=net, model=f"resolution check {tag}", weeks=1.0,
                                 day=d["test"]["day"]["median_wape"] / 100))
    return pd.DataFrame(rows)


def grid_table() -> pd.DataFrame:
    g = pd.read_csv(PC / "flat" / "grid_scores.csv")
    keep = ["rule: yesterday same hour", "Timestral: median", "Timestral: mean", "chronos-2", "timesfm-2.5",
            "GBDT (log, squared error)"]
    keep = keep + [k + " + bias corr. 7 d" for k in keep]
    g = g[g.forecast.isin(keep)]
    w = g.pivot(index="forecast", columns="period", values=["bias %", "worst month %", "day %", "week %", "month %"])
    w.columns = [f"{a} {b}" for a, b in w.columns]
    return w.reindex(keep)


def levels() -> dict:
    s = json.load(open(P / "i2" / "summary.json"))
    pe = json.load(open(P / "i2" / "path_rate_error.json"))
    lk = link_models()
    tm = total_models()
    pooled = {}
    for net in DENSE:
        w = pd.read_csv(FM[net] / "tables_both" / "block" / "wape.csv")
        r = w[(w.model == "timestral@1h_24w") & (w.horizon_name == "day")]
        pooled[net] = float(r.pooled_wape.iloc[0]) / 100 if len(r) else None
    return {"traffic_total": {n: float(tm[(tm.network == n) & (tm.model == "Timestral")].traffic_hourly_wape.iloc[0]) for n in ("switch", "i2")},
            "traffic_link_median": {n: float(lk[(lk.network == n) & (lk.model == "timestral@1h_24w")].link_day_wape.iloc[0]) for n in ("switch", "i2")},
            "traffic_link_pooled_note": "pooled = sum of absolute errors over all links / sum of traffic (5-minute WAPE)",
            "traffic_link_pooled": pooled,
            "rate_flat_hourly": pe["flat_rate_hourly_wape"], "rate_path_bytes": pe["path_rate_wape_bytes_capped"],
            "router_bill_flat": {k: s[f"flat_router_{k}_wape"] for k in ("hour", "day", "week", "month", "30day")},
            "router_bill_path": {k: s[f"path_router_{k}_wape"] for k in ("hour", "day", "week", "month", "30day")}}


def cap_table() -> pd.DataFrame:
    import i2_rates as IR
    import scipy.sparse as sp
    A_res = sp.load_npz(IR.OUT / "A_res.npz").tocsr()
    rows = []
    for k in (1.0, 2.0, 3.0, None):
        R = IR.load_rates_k(k)
        E, U, X = R["E"], R["U"], R["X"]
        fr = IR.flat_rate(R)
        live = (U > 0) & (E > 0)
        capped = live & (R["per_set"] < E / np.where(U > 0, U, 1) * (1 - 1e-12))
        pe = IR.path_rate_error(R, A_res)
        be = IR.bill_errors(R, A_res)
        tag = "" if k == 2.0 else f"_k{k}"
        uni = json.load(open(IR.OUT / f"unilateral{tag}.json")) if (IR.OUT / f"unilateral{tag}.json").exists() else {}
        ev = json.load(open(IR.OUT / f"everyone{tag}.json")) if (IR.OUT / f"everyone{tag}.json").exists() else {}
        rows.append(dict(k="none" if k is None else k,
                         cap_over_flat=float(np.median(R["mult"] * (E.sum(1) / U.sum(1)) / fr)) if k is not None else np.inf,
                         resource_hours_capped=float(capped[live].mean()),
                         emissions_spread_flat=float((R["add_set"] * X.sum(1)).sum() / E.sum()),
                         path_rate_wape=pe["path_rate_wape_bytes_capped"],
                         router_bill_day=be["path_router_day_wape"], router_bill_month=be["path_router_month_wape"],
                         pairs_ge20=uni.get("pairs_reduction_forecast_ge_20pct"),
                         bytes_reduction=uni.get("bytes_weighted_reduction_forecast"),
                         same_choice=uni.get("bytes_same_choice"),
                         spans_dark_block_mean=ev.get("mean", {}).get("spans_dark"),
                         spans_dark_from_day15=ev.get("from_day15_mean", {}).get("spans_dark"),
                         spans_dark_ge90_from_day15=ev.get("from_day15_spans_dark_ge90"),
                         E_share_dark_ge90_from_day15=ev.get("from_day15_E_share_spans_dark_ge90"),
                         dark_E=ev.get("mean", {}).get("dark_E_share"),
                         spans_dark_all_block=ev.get("spans_dark_all_block")))
    return pd.DataFrame(rows)


# --- bias ---------------------------------------------------------------------------------------
def grid_monthly() -> pd.DataFrame:
    """SWITCHlan's day-ahead grid (backbone emissions) forecasts, hourly, from appendix.py round 1's series
    (2025-03-22 -> 2026-10-28: grid_forecast.py's candidates, then Timestral re-run on Jul-Oct 2026), labelled by
    the common periods and cut to DEV + TEST + LATER."""
    G3 = pd.read_csv(REPO_ROOT / "artifacts" / "paper" / "appendix" / "grid_hourly_three_periods.csv.gz",
                     index_col=0, parse_dates=True)
    import flat_common as FC
    lab = FC.label(G3.index)
    df = G3.drop(columns="period")
    df["period"] = lab.str.lower().to_numpy()
    return df[df.period.isin(["dev", "test", "later"])]


def grid_bias_tables(df: pd.DataFrame):
    """Bias per calendar month and per period; errors per calendar day / 7-day block / calendar month."""
    import flat_common as FC
    cols = ["rule", "median", "mean", "median_corr", "blend"]
    m = df.index.tz_localize(None).to_period("M")
    g = df[cols + ["actual"]].groupby(m).sum()
    months = (g[cols].div(g.actual, axis=0) - 1) * 100
    months["period"] = df.groupby(m).period.agg(lambda x: x.mode()[0])
    rows = []
    for p_, per in (("dev", PER.DEV), ("test", PER.TEST), ("later", PER.LATER)):
        for c in cols:
            s = FC.grid_scores(df[c], df.actual, per)
            rows.append(dict(period=p_, forecast=c, bias=s["bias %"], day=s["day %"], week=s["week %"],
                             month=s["month %"], worst_month=s["worst month %"], months_low=s["months low"],
                             months_within_4=s["months within 4 %"], months=s["months"]))
    periods = pd.DataFrame(rows).pivot(index="forecast", columns="period", values="bias")
    return months, periods, pd.DataFrame(rows)


def bias_summary() -> dict:
    """Where the bias sits: traffic vs grid, network vs link, and the bill (test block, flat_common.py)."""
    fj = json.load(open(PC / "flat" / "flat.json"))
    out = {}
    q = pd.read_csv(PC / "flat" / "traffic_quantiles_test.csv", index_col=0, parse_dates=True)
    gq = pd.read_csv(PC / "flat" / "grid_quantiles_test.csv", index_col=0, parse_dates=True)
    for net in ("switch", "i2"):
        bt = fj[net]["bias_table"]
        for k, v in bt.items():
            out[f"{net}_{k}"] = [v["bias %"], v["95 % low"], v["95 % high"]]
            out[f"{net}_{k}_top10_days"] = v["bias on top-10 % days"]
            out[f"{net}_{k}_by_month"] = v["by month"]
        h = flat_hours(net).dropna(subset=["T"])
        qq = q[q.network == net].reindex(h.index)
        out[f"{net}_traffic_calibration_q10_q50_q90_q99"] = [float(100 * (h["T"] <= qq[c]).mean())
                                                             for c in ("q10", "q50", "q90", "q99")]
        out[f"{net}_traffic_timestral_median_bias_no_covariates"] = float(100 * (qq.q50.sum() / h["T"].sum() - 1))
        g = gq[gq.network == net]
        out[f"{net}_grid_q99_over_actual"] = float((g.q99 / g.actual).mean())
        out[f"{net}_grid_calibration_q50_q99"] = [float(100 * (g.actual <= g.q50).mean()), float(100 * (g.actual <= g.q99).mean())]
        tc = pd.read_csv(OUT / "timestral_context.csv")
        r = tc[(tc.network == net) & (tc.horizon == "day") & (tc.ctx == 4032)]
        out[f"{net}_link_median_bias_timestral"] = float(r.median_link_bias.iloc[0]) if len(r) else None
    # the Swiss grid: how often Timestral's median is right, and how large its misses are (bias_sources.py)
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    import bias_sources as BS
    hs = flat_hours("switch")
    out["switch_grid_median_shape"] = BS.shape(hs.Ef_raw, hs.E)
    hi = flat_hours("i2")
    out["i2_grid_median_shape"] = BS.shape(hi.Ef_raw, hi.E)
    gs = pd.read_csv(PC / "flat" / "grid_scores.csv")
    out["grid_gbdt_losses_dev_test"] = {f: gs[gs.forecast == f].set_index("period")["bias %"].to_dict() for f in
                                        ("GBDT log, squared error (current)", "GBDT log, quantile 2/3 (peak-aware)",
                                         "GBDT linear, squared error (mean)", "GBDT log-normal, mean of the fitted distribution")}
    # the loss and the bias: round 1's 5-minute ladder on SWITCHlan's total, both losses, three seeds each
    # (recipes/flat_forecast/run.sh ladder: trained to 2026-02-15, before the test block), on the common test block
    tot = {}
    hs_ = flat_hours("switch").dropna(subset=["T"])
    import run as R
    for m in ("dlinear", "gru", "tcn", "patchtst"):
        tot[m] = {}
        for uw, lab in ((2, "peak-aware"), (1, "symmetric")):
            seeds = [R.ladder_hourly(R.LADDER / f"{m}_uw{uw}_s{s}", m) for s in range(3)]
            if all(x is not None for x in seeds):
                f = (sum(seeds) / 3).reindex(hs_.index)
                tot[m][lab] = float(100 * (f.sum() / hs_["T"][f.notna()].sum() - 1))
    out["switch_total_ladder_bias_round1_runs"] = tot
    return out


def model_params() -> pd.DataFrame:
    """Parameters of every trained per-link network (from its checkpoint)."""
    import torch
    rows = []
    for net in DENSE:
        for f in sorted((PC / "links" / net).glob("part_*/checkpoints/*.pt")):
            if ".train." in f.name:
                continue
            sd = torch.load(f, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            rows.append(dict(network=net, model=f.stem, params=sum(v.numel() for v in sd.values() if hasattr(v, "numel"))))
    return pd.DataFrame(rows)


ARCH = {"dlinear": ("DLinear", "#188487", "-.", "D"), "gru": ("GRU", "#467821", ":", "^"),
        "tcn": ("TCN", "#E24A33", "--", "v"), "patchtst": ("PatchTST", "#7A68A6", "--", "s")}


def fig_context(C: pd.DataFrame, out: Path):
    """Per network: median link error a day ahead by weeks of history, Timestral and every trained
    architecture (hourly input, ladder diagnosis), plus the 5-minute long-context transformer."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.75), gridspec_kw={"wspace": 0.18})
    for ax, net, title in ((axes[0], "switch", "SWITCHlan"), (axes[1], "i2", "Internet2")):
        d = C[C.network == net]
        t = d[d.model == "Timestral (hourly)"].sort_values("weeks")
        ax.plot(t.weeks, t.day * 100, color="k", marker="o", ms=3, lw=1.3, label="Timestral")
        for arch, (lab, c, ls, mk) in ARCH.items():
            m = d[d.model == f"{arch} (hourly)"].sort_values("weeks")
            ax.plot(m.weeks, m.day * 100, color=c, ls=ls, marker=mk, ms=3, mfc="white", label=lab)
        m5 = d[d.model == "trained (5-min)"].sort_values("weeks")
        ax.plot(m5.weeks, m5.day * 100, color="0.5", ls="-", marker="x", ms=3.5, label="5-min transformer")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 12, 24], ["1", "2", "4", "12", "24"])
        ax.minorticks_off()
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_xlabel("Weeks of history", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major")
    axes[0].set_ylabel(f"Link error ({PCT})", fontsize=9)
    for ax, net in ((axes[0], "switch"), (axes[1], "i2")):     # every point visible, headroom for the legend
        v = C[(C.network == net) & C.model.isin(["Timestral (hourly)", "trained (5-min)"] +
                                               [f"{a} (hourly)" for a in ARCH])].day * 100
        ax.set_ylim(np.floor(v.min() - 1), np.ceil(v.max() + 0.35 * (v.max() - v.min()) + 1))
    axes[1].legend(loc="upper right", frameon=False, ncol=2, fontsize=7, handlelength=2.0, columnspacing=0.8,
                   borderaxespad=0.1)
    save(fig, out, "app_context")


def fig_horizon(H: pd.DataFrame, out: Path):
    """Per network: median link error by horizon for the two rules, every trained architecture
    (the ladder's 5-minute block logs) and Timestral (hourly input; 1 h not computed on SWITCHlan)."""
    hours = {"1h": 1, "8h": 8, "day": 24, "week": 168}
    show = {"persistence": ("persistence", "0.55", ":", "^"),
            "seasonal_naive_week": ("same hour last week", "0.55", "-.", "v")}
    show |= {k: v for k, v in ARCH.items()}
    show["timestral@1h_24w"] = ("Timestral", "k", "-", "o")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.75), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, net, title in ((axes[0], "switch", "SWITCHlan"), (axes[1], "i2", "Internet2")):
        for m, (lab, c, ls, mk) in show.items():
            d = H[(H.network == net) & (H.model == m) & H.horizon.isin(hours)].copy()
            d["h"] = d.horizon.map(hours)
            d = d.sort_values("h")
            ax.plot(d.h, d.link_wape * 100, color=c, ls=ls, marker=mk, ms=3, lw=1.3 if m.startswith("timestral") else 1,
                    mfc=c if m.startswith("timestral") else "white", label=lab)
        ax.set_xscale("log")
        ax.set_xticks([1, 8, 24, 168], ["1 h", "8 h", "Day", "Week"])
        ax.minorticks_off()
        ax.set_title(title, fontsize=9, pad=2)
        ax.set_xlabel("Forecast horizon", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_ylim(0, 90)                     # room for the legend above the seasonal rule (47.6 % on SWITCHlan)
        ax.grid(True, which="major")
    axes[0].set_ylabel(f"Link error ({PCT})", fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, ncol=2, fontsize=7, handlelength=2.0, columnspacing=0.8,
                   borderaxespad=0.1)
    save(fig, out, "app_horizon")


def fig_decision(out: Path):
    U = pd.read_csv(P / "i2" / "unilateral_pairs.csv")
    fig, ax = plt.subplots()
    ax.scatter(U.reduction_oracle * 100, U.reduction_forecast * 100, s=2, color=I2, alpha=0.4, lw=0, rasterized=True)
    ax.plot([-40, 100], [-40, 100], color="0.4", ls="--", lw=0.8)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(-5, 95)
    ax.set_ylim(-40, 95)
    ax.set_xlabel(f"Reduction, chosen on settled rates ({PCT})")
    ax.set_ylabel(f"Day-ahead choice ({PCT})")
    ax.grid(True, which="major")
    save(fig, out, "app_decision")


def fig_bias(months: pd.DataFrame, out: Path):
    """Monthly bias of SWITCHlan's grid forecasts over three held-out periods."""
    fig, ax = plt.subplots()
    x = np.arange(len(months))
    show = {"rule": ("same hour yesterday", "0.45", ":", "^"), "median": ("Timestral, median", "#E24A33", "--", "v"),
            "mean": ("Timestral, mean", "#7A68A6", "-.", "s"), "median_corr": ("median, corrected", "k", "-", "o")}
    for p_, c in (("dev", "0.93"), ("test", "0.83"), ("later", "0.93")):
        idx = np.where(months.period.to_numpy() == p_)[0]
        if len(idx):
            ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=c, lw=0, zorder=0)
            ax.text((idx[0] + idx[-1]) / 2, 33, {"dev": "development", "test": "test", "later": "later"}[p_],
                    ha="center", va="top", fontsize=7, color="0.3")
    for col, (lab, c, ls, mk) in show.items():
        ax.plot(x, months[col], color=c, ls=ls, marker=mk, ms=2.5, label=lab, mfc="white" if ls != "-" else c)
    ax.axhline(0, color="0.3", lw=0.5)
    lab = [p.strftime("%b") if p.month in (1, 4, 7, 10) else "" for p in months.index]
    lab = [f"{l}\n{p.year}" if (p.month == 1 or i == 0) else l for i, (l, p) in enumerate(zip(lab, months.index))]
    ax.set_xticks(x, lab, fontsize=7)
    ax.set_ylim(-35, 35)
    ax.set_xlim(-0.5, len(months) - 0.5)
    ax.set_ylabel(f"Monthly bias ({PCT})")
    ax.grid(True, which="major", axis="y")
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=6.5, handlelength=1.8, columnspacing=0.8,
              borderaxespad=0.1)
    save(fig, out, "app_bias")




def step(name, f, *a):
    """Run one part; a missing input skips it (and says so) instead of stopping the rest."""
    try:
        return f(*a)
    except FileNotFoundError as e:
        print(f"[skip] {name}: {e}", flush=True)
        return None


def main():
    global PNG
    args = [a for a in sys.argv[1:] if not a.startswith("--png=")]
    png = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--png=")]
    PNG = Path(png[0]) if png else None
    if PNG:
        PNG.mkdir(parents=True, exist_ok=True)
    out = Path(args[0]) if args else REPO_ROOT / "6aea077f0929c9a3a1ac5660" / "figure_appendix"
    out.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    CO = costs()
    CO.to_csv(OUT / "train_cost.csv", index=False)
    print(CO.round(1).to_string(index=False), flush=True)
    TM = step("total models", total_models)
    if TM is not None:
        TM.to_csv(OUT / "total_models.csv", index=False)
        print(TM.round(4).to_string(index=False), flush=True)
    LM = step("link models", link_models)
    if LM is not None:
        LM.to_csv(OUT / "link_models.csv", index=False)
        print(LM.round(4).to_string(index=False), flush=True)
    HZ = step("horizons", horizons)
    if HZ is not None:
        HZ.to_csv(OUT / "horizons.csv", index=False)
        fig_horizon(HZ, out)
    CX = step("context", context)
    if CX is not None:
        CX.to_csv(OUT / "context.csv", index=False)
        print(CX.round(4).to_string(index=False), flush=True)
        fig_context(CX, out)
    GT = grid_table()
    GT.to_csv(OUT / "grid_candidates.csv")
    print(GT.round(1).to_string(), flush=True)
    LV = step("levels", levels)
    CT = step("cap table", cap_table)
    if CT is not None:
        CT.to_csv(OUT / "cap_sensitivity.csv", index=False)
        print(CT.round(4).to_string(index=False), flush=True)
    dec = None
    if (P / "i2" / "unilateral.json").exists():
        uni = json.load(open(P / "i2" / "unilateral.json"))
        U = pd.read_csv(P / "i2" / "unilateral_pairs.csv")
        worse = U[U.reduction_forecast < -0.001]
        dec = {"bytes_same_choice": uni.get("bytes_same_choice"),
               "captured_share_of_oracle_bytes": uni["bytes_weighted_reduction_forecast"] / uni["bytes_weighted_reduction_oracle"],
               "pairs_forecast_within_5pts_of_oracle": float(((U.reduction_oracle - U.reduction_forecast) <= 0.05).mean()),
               "pairs_corr_forecast_oracle": float(np.corrcoef(U.reduction_oracle, U.reduction_forecast)[0, 1]),
               "pairs_worse": float(len(worse) / len(U)), "bytes_worse": float(worse.GB.sum() / U.GB.sum()),
               "worse_median_loss": float(-worse.reduction_forecast.median()) if len(worse) else None,
               "worse_p90_loss": float(-worse.reduction_forecast.quantile(.1)) if len(worse) else None}
        fig_decision(out)
    json.dump({"levels": LV, "decision": dec}, open(OUT / "numbers.json", "w"), indent=1, default=float)
    print(json.dumps({"levels": LV, "decision": dec}, indent=1, default=float), flush=True)
    GM = grid_monthly()
    months, periods, gper = grid_bias_tables(GM)
    months.to_csv(OUT / "grid_bias_months.csv")
    gper.to_csv(OUT / "grid_bias_periods.csv", index=False)
    print(months.round(1).to_string(), flush=True)
    print(gper.round(2).to_string(index=False), flush=True)
    fig_bias(months, out)
    BS = step("bias summary", bias_summary)
    if BS is not None:
        json.dump(BS, open(OUT / "bias.json", "w"), indent=1, default=float)
        print(json.dumps(BS, indent=1, default=float), flush=True)
    MP = model_params()
    MP.to_csv(OUT / "model_params.csv", index=False)
    print(MP.to_string(index=False), flush=True)
    print("figures ->", out)


if __name__ == "__main__":
    main()
