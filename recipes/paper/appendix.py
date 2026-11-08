"""The appendix's forecasting and steering details (paper appendix E-I), from results of this project:

- models:   every forecaster on the network total and per link, both networks: hourly traffic error,
            daily bill error with the paper's grid forecast, training cost (GPU minutes from the
            training logs); per link from the dense block logs (day ahead, median link hourly WAPE).
- time:     context length (Timestral 1-24 weeks vs trained 5-minute multi-resolution models,
            recipes/ladder_diagnosis), input resolution, forecast horizon (1 h ... week).
- levels:   Level 1 (one rate) vs Level 2 (per path): error of traffic, rate, bills.
- grid:     the grid-forecast candidates on SWITCHlan (development year, test block).
- decision: does the day-ahead choice equal the choice made on the settled rates (steering.py).
- cap:      the cap's k (1, 2, 3, none): level, share capped, rate error, steering (steering.py --k).
- bias:     where forecast bias sits (traffic / grid, network / link, bill), SWITCHlan's grid forecasts by
            month over the development year, the test block and Jul-Oct 2026 (Timestral re-run, GPU).
- params:   parameters of every trained network (checkpoints).

usage: python appendix.py            (after the rest of recipes/paper and steering.py --k=1/3/none)
writes $GR_ARTIFACTS_DIR/paper/appendix/*.csv, numbers.json and the PDFs into
<repo>/6aea077f0929c9a3a1ac5660/figure_appendix (or the directory given as the first argument).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402

from green_routing.paths import ARTIFACTS_DIR, REPO_ROOT   # noqa: E402

HERE = Path(__file__).resolve().parent
A = ARTIFACTS_DIR
P = A / "paper"
OUT = P / "appendix"
plt.style.use(HERE / "paper.mplstyle")
if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
SW, I2 = "#348ABD", "#A60628"
PCT = "\\%" if plt.rcParams["text.usetex"] else "%"
PNG = None
DENSE = {"switch": A / "pred_year_sym" / "merged", "i2": A / "internet2" / "links" / "merged"}


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
    for net, base in (("switch", A / "pred_year_sym"), ("i2", A / "internet2" / "links")):
        for part in ("part_dlinear_patchtst", "part_gru", "part_tcn"):
            for model, d in train_minutes(base / part / "run.log").items():
                rows.append(dict(network=net, target="links", model=model, **d))
    for part in ("part_dlinear_patchtst", "part_gru", "part_tcn"):
        for model, d in train_minutes(A / "carbon_rates" / "flat" / "logs" / f"{part}.out").items():
            rows.append(dict(network="switch", target="total", model=model, **d))
    return pd.DataFrame(rows)


# --- the network total: every forecaster, traffic and bill ---------------------------------------
SWITCH_MODELS = {"Persistence (same hour yesterday)": None, "Same hour last week": "rule: same hour last week",
                 "GBDT": "gbdt: current (168 h, basic calendar)",
                 "DLinear": "5-min dlinear: symmetric loss, mean of 3 seeds", "GRU": "5-min gru: symmetric loss, mean of 3 seeds",
                 "TCN": "5-min tcn: symmetric loss, mean of 3 seeds", "PatchTST": "5-min patchtst: symmetric loss, mean of 3 seeds",
                 "Chronos-2": "chronos-2: 24 weeks", "TimesFM-2.5": "timesfm-2.5: 48 weeks",
                 "Timestral": "timestral: 24 weeks + calendar covariates"}
I2_MODELS = {"Persistence (same hour yesterday)": ("d", "persistence (same hour yesterday)"),
             "Same hour last week": ("d", "same hour last week"), "GBDT": ("d", "GBDT"),
             "Chronos-2": ("t", "Chronos-2, 24 weeks"), "Timestral": ("t", "Timestral-209M, 24 weeks")}


def bill_scores(T, Tf, E, Ef) -> dict:
    ok = T.notna() & Tf.notna() & E.notna() & Ef.notna()
    T, Tf, E, Ef = T[ok], Tf[ok], E[ok], Ef[ok]
    bill = Ef * T / Tf
    day = T.index.floor("D")
    bd, ed = bill.groupby(day).sum(), E.groupby(day).sum()
    r30 = lambda s: s.rolling(30).sum().dropna()                                   # noqa: E731
    return {"traffic_hourly_wape": float((Tf - T).abs().sum() / T.sum()),
            "traffic_bias": float(Tf.sum() / T.sum() - 1),
            "bill_day_wape": float((bd - ed).abs().sum() / ed.sum()),
            "bill_30day_wape": float((r30(bd) - r30(ed)).abs().sum() / r30(ed).sum()), "hours": int(ok.sum())}


def total_models() -> pd.DataFrame:
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    import run as R
    hs = pd.read_csv(P / "hourly_switch.csv", index_col=0, parse_dates=True)
    cache = R.cache()
    rows = []
    for name, key in SWITCH_MODELS.items():
        Tf = R.S.shift(24) if key is None else cache[key]
        rows.append(dict(network="switch", model=name, **bill_scores(hs["T"], Tf.reindex(hs.index), hs.E, hs.Ef)))
    hi = pd.read_csv(P / "hourly_i2.csv", index_col=0, parse_dates=True)
    C = A / "internet2" / "carbon"
    d = pd.read_csv(C / "dayahead_hourly.csv", index_col=0, parse_dates=True)
    t = pd.read_csv(C / "timestral_total.csv", index_col=0, parse_dates=True)
    for name, (src, col) in I2_MODELS.items():
        Tf = (d if src == "d" else t)[col].reindex(hi.index)
        rows.append(dict(network="i2", model=name, **bill_scores(hi["T"], Tf, hi.E, hi.Ef)))
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
    for net, base in DENSE.items():
        p = pd.read_csv(base / "dense_fm" / "tables_both" / "block" / "per_link.csv")
        t = p[p.horizon_name == "day"].groupby("model").wape_1h.median()
        for m, v in t.items():
            rows.append(dict(network=net, model=m, link_day_wape=v / 100, links=int(p.link.nunique())))
    return pd.DataFrame(rows)


def horizons() -> pd.DataFrame:
    """Median link hourly WAPE by horizon: the ladder's full block logs (all horizons) and
    Timestral's hourly-input logs (1 h on I2 only, 8 h, day, week)."""
    rows = []
    for net, base in DENSE.items():
        p = pd.read_csv(base / "dense" / "block" / "per_link.csv")
        q = pd.read_csv(base / "dense_fm" / "tables_1h_24w" / "block" / "per_link.csv")
        q = q[q.model == "timestral@1h_24w"]
        for src in (p, q):
            g = src.groupby(["model", "horizon_name"]).wape_1h.median()
            for (m, h), v in g.items():
                rows.append(dict(network=net, model=m, horizon=h, link_wape=v / 100))
    return pd.DataFrame(rows)


def context() -> pd.DataFrame:
    """Day-ahead median link hourly WAPE by context (weeks): Timestral (hourly input), the trained
    5-minute multi-resolution transformer (mean of 3 seeds), and the ladder (PatchTST, 1 week)."""
    a = pd.read_csv(A / "ladder_diagnosis" / "all_runs.csv")
    l5 = pd.read_csv(A / "ladder_diagnosis" / "long5m_runs.csv")
    rows = []
    for _, r in a[a.model == "timestral-209M"].iterrows():
        rows.append(dict(network="i2" if r.ds == "i2" else "switch", model="Timestral (hourly)", weeks=r.ctx / 168, day=r.day / 100))
    # bounded inputs (squash 10), no dropout of old weeks, no profile queries; at 1 week the
    # multi-resolution model is the full-resolution one (one week has a single resolution)
    m = l5[(l5.model == "seg") & (l5.squash == 10) & (l5["drop"] == 0) & (l5.qprof == 0)
           & ((l5.seg == "multires") | ((l5.seg == "full") & (l5.weeks == 1)))]
    for (ds, w), g in m.groupby(["ds", "weeks"]):
        rows.append(dict(network="i2" if ds == "i2" else "switch", model="trained (5-min)", weeks=float(w),
                         day=float(g.day.mean()) / 100, seeds=int(len(g))))
    for net in ("i2", "switch"):
        for arch in ("dlinear", "gru", "tcn", "patchtst"):
            for ctx in (168, 672, 4032):
                f = A / "ladder_diagnosis" / net / "runs" / f"H_{arch}_c{ctx}" / "summary.json"
                if f.exists():
                    d = json.load(open(f))
                    rows.append(dict(network=net, model=f"{arch} (hourly)", weeks=ctx / 168,
                                     day=d["test"]["day"]["median_wape"] / 100))
    for net in DENSE:
        lk = link_models()
        v = lk[(lk.network == net) & (lk.model == "patchtst")].link_day_wape.iloc[0]
        rows.append(dict(network=net, model="ladder PatchTST (5-min)", weeks=1.0, day=v))
        for tag, w in (("timestral@5m_4w", 4.0),):
            u = lk[(lk.network == net) & (lk.model == tag)].link_day_wape
            if len(u):
                rows.append(dict(network=net, model="Timestral (5-min)", weeks=w, day=float(u.iloc[0])))
    return pd.DataFrame(rows)


def grid_table() -> pd.DataFrame:
    g = pd.read_csv(A / "flat_forecast" / "grid_scores.csv")
    keep = ["rule: yesterday same hour", "Timestral: median", "Timestral: mean", "chronos-2", "timesfm-2.5",
            "GBDT (log, squared error)"]
    keep = keep + [k + " + bias corr. 7 d" for k in keep]
    g = g[g.forecast.isin(keep)]
    w = g.pivot(index="forecast", columns="period", values=["bias %", "max |month bias| %", "day %", "30-day %"])
    w.columns = [f"{a} {b}" for a, b in w.columns]
    return w.reindex(keep)


def levels() -> dict:
    s = json.load(open(P / "i2" / "summary.json"))
    pe = json.load(open(P / "i2" / "path_rate_error.json"))
    lk = link_models()
    tm = total_models()
    pooled = {}
    for net, base in DENSE.items():
        w = pd.read_csv(base / "dense_fm" / "tables_both" / "block" / "wape.csv")
        r = w[(w.model == "timestral@1h_24w") & (w.horizon_name == "day")]
        pooled[net] = float(r.pooled_wape.iloc[0]) / 100 if len(r) else None
    return {"traffic_total": {n: float(tm[(tm.network == n) & (tm.model == "Timestral")].traffic_hourly_wape.iloc[0]) for n in ("switch", "i2")},
            "traffic_link_median": {n: float(lk[(lk.network == n) & (lk.model == "timestral@1h_24w")].link_day_wape.iloc[0]) for n in ("switch", "i2")},
            "traffic_link_pooled_note": "pooled = sum of absolute errors over all links / sum of traffic (5-minute WAPE)",
            "traffic_link_pooled": pooled,
            "rate_flat_hourly": pe["flat_rate_hourly_wape"], "rate_path_bytes": pe["path_rate_wape_bytes_capped"],
            "router_bill_flat": {k: s[f"flat_router_{k}_wape"] for k in ("hour", "day", "30day")},
            "router_bill_path": {k: s[f"path_router_{k}_wape"] for k in ("hour", "day", "30day")}}


def cap_table() -> pd.DataFrame:
    sys.path.insert(0, str(HERE))
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
                         router_bill_day=be["path_router_day_wape"], router_bill_30day=be["path_router_30day_wape"],
                         pairs_ge20=uni.get("pairs_reduction_forecast_ge_20pct"),
                         bytes_reduction=uni.get("bytes_weighted_reduction_forecast"),
                         same_choice=uni.get("bytes_same_choice"),
                         spans_dark=ev.get("mean", {}).get("spans_dark"), dark_E=ev.get("mean", {}).get("dark_E_share"),
                         spans_dark_all_block=ev.get("spans_dark_all_block")))
    return pd.DataFrame(rows)


# --- bias ---------------------------------------------------------------------------------------
LATER = ("2026-07-11", "2026-10-28")          # the held-out period after the test block (grid only)


def grid_monthly() -> pd.DataFrame:
    """SWITCHlan's day-ahead grid (backbone emissions) forecasts on the development year, the test
    block and the later period, hourly: actual, the rule, Timestral's median and mean, the median
    with the 7-day correction, and the 50/50 blend of the rule and the corrected median."""
    cache = OUT / "grid_hourly_three_periods.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    import grid_forecast as G
    from bias import timestral_quantiles
    from green_routing.carbon import dayahead as DA, grid
    c = pd.read_csv(G.CAND, index_col=0, parse_dates=True)
    a = c["actual"]
    parts = [pd.DataFrame({"actual": a, "rule": c["rule: yesterday same hour"], "median": c["Timestral: median"],
                           "mean": c["Timestral: mean"]})]
    days = pd.date_range(*LATER, freq="D", tz="UTC")
    E = G.switch_emissions(LATER[1])
    pre = pd.date_range(days[0] - pd.Timedelta(days=7), days[0] - pd.Timedelta(days=1), freq="D")
    Q, lv = timestral_quantiles(E, pre.append(days))
    w = DA._quantile_mean_weights(lv)
    rule = grid.RULES["yesterday same hour"](E.asfreq("h").interpolate())
    parts.append(pd.DataFrame({"actual": E.reindex(Q.index), "rule": rule.reindex(Q.index), "median": Q[0.5],
                               "mean": (Q * w).sum(1)}))
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df["median_corr"] = G.bias_corrected(df["median"], df["actual"], 7)
    df["blend"] = (df["rule"] + df["median_corr"]) / 2
    period = pd.Series("dev", index=df.index)
    period[df.index >= pd.Timestamp(G.TEST[0])] = "test"
    period[df.index >= pd.Timestamp(LATER[0], tz="UTC")] = "later"
    df["period"] = period
    keep = (df.index.normalize().isin(G.DEV) | df.index.normalize().isin(G.TEST) | df.index.normalize().isin(days))
    df = df[keep]
    df.to_csv(cache)
    return df


def grid_bias_tables(df: pd.DataFrame):
    """Bias per calendar month (months with >= 20 days) and per period."""
    cols = ["rule", "median", "mean", "median_corr", "blend"]
    m = df.index.tz_localize(None).to_period("M")
    g = df[cols + ["actual"]].groupby(m).sum()
    n = df.groupby(m).size() / 24
    months = ((g[cols].div(g.actual, axis=0) - 1) * 100)[n >= 20]
    months["period"] = df.groupby(m).period.agg(lambda x: x.mode()[0])[n >= 20]
    per = df.groupby("period")[cols + ["actual"]].sum()
    periods = (per[cols].div(per.actual, axis=0) - 1) * 100
    day = df.index.normalize()
    rows = []
    for p_, d in df.groupby("period"):
        dd = d[cols + ["actual"]].groupby(d.index.normalize()).sum()
        for c in cols:
            r30 = lambda x: x.rolling(30).sum().dropna()                         # noqa: E731
            rows.append(dict(period=p_, forecast=c, bias=float(d[c].sum() / d.actual.sum() - 1) * 100,
                             day=float((dd[c] - dd.actual).abs().sum() / dd.actual.sum()) * 100,
                             d30=float((r30(dd[c]) - r30(dd.actual)).abs().sum() / r30(dd.actual).sum()) * 100,
                             worst_month=float(months[months.period == p_][c].abs().max())))
    del day
    return months, periods, pd.DataFrame(rows)


def bias_summary() -> dict:
    """Where the bias sits: traffic vs grid, network vs link, and the bill (test blocks)."""
    b = pd.read_csv(A / "flat_forecast" / "bias.csv")
    pick = lambda net, ser, fc: b[(b.network == net) & (b.series == ser) & (b.forecast == fc)].iloc[0]   # noqa: E731
    out = {}
    for net, key in (("Switch LAN", "switch"), ("I2 Network", "i2")):
        for ser, fc, name in (("traffic", "Timestral, median", "traffic_total"),
                              ("emissions (grid)", "same hour yesterday", "grid_rule"),
                              ("emissions (grid)", "Timestral, median", "grid_median"),
                              ("emissions (grid)", "Timestral, mean", "grid_mean")):
            r = pick(net, ser, fc)
            out[f"{key}_{name}"] = [float(r["bias %"]), float(r["95 % low"]), float(r["95 % high"])]
            if name == "traffic_total":
                out[f"{key}_traffic_top10_days"] = float(r["bias on top-10 % days"])
                out[f"{key}_traffic_calibration_q10_q50_q90_q99"] = [float(r[f"actual <= q{q}"]) for q in (10, 50, 90, 99)]
            if name == "grid_median":
                out[f"{key}_grid_calibration_q50_q99"] = [float(r["actual <= q50"]), float(r["actual <= q99"])]
                out[f"{key}_grid_q99_over_actual"] = float(r["q99 / actual, mean"])
    a = pd.read_csv(A / "ladder_diagnosis" / "all_runs.csv")
    for net in ("switch", "i2"):
        r = a[(a.ds == net) & (a.tag == "timestral_c4032")].iloc[0]
        out[f"{net}_link_median_bias_timestral"] = float(r.day_bias)
        h = pd.read_csv(P / f"hourly_{net}.csv", index_col=0, parse_dates=True).dropna()
        out[f"{net}_grid_corrected_bias"] = float(h.Ef.sum() / h.E.sum() - 1) * 100
        out[f"{net}_bill_bias"] = float((h.Ef * h["T"] / h.Tf).sum() / h.E.sum() - 1) * 100
    src = pd.read_csv(A / "flat_forecast" / "bias_sources.csv")
    r = src[(src.network == "Switch LAN") & (src.forecast == "Timestral: median")].iloc[0]
    out["switch_grid_median_under_over"] = [float(r["actual > forecast %"]), float(r["under-forecasts, % of mean"]),
                                            float(r["over-forecasts, % of mean"])]
    out["log_error_sigma"] = {n: float(src[(src.network == n) & (src.forecast == "Timestral: median")]["sigma of log error"].iloc[0])
                              for n in ("Switch LAN", "I2 Network")}
    s = pd.read_csv(A / "flat_forecast" / "scores_test.csv").set_index("forecast")
    out["switch_total_ladder_bias"] = {m: [float(s.loc[f"5-min {m}: peak-aware loss, mean of 3 seeds", "bias"]) * 100,
                                           float(s.loc[f"5-min {m}: symmetric loss, mean of 3 seeds", "bias"]) * 100]
                                       for m in ("dlinear", "gru", "tcn", "patchtst")}
    gl = pd.read_csv(A / "flat_forecast" / "grid_losses.csv")
    out["grid_gbdt_losses_dev_test"] = {r["forecast"]: [float(r["bias %"])] for _, r in gl[gl.period == "DEV"].iterrows()}
    for _, r in gl[gl.period == "TEST"].iterrows():
        out["grid_gbdt_losses_dev_test"][r["forecast"]].append(float(r["bias %"]))
    return out


def model_params() -> pd.DataFrame:
    """Parameters of every trained per-link network (from its checkpoint) and of the pretrained models."""
    import torch
    rows = []
    for net, base in (("switch", A / "pred_year_sym"), ("i2", A / "internet2" / "links")):
        for f in sorted(base.glob("part_*/checkpoints/*.pt")):
            if ".train." in f.name:
                continue
            sd = torch.load(f, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            rows.append(dict(network=net, model=f.stem, params=sum(v.numel() for v in sd.values() if hasattr(v, "numel"))))
    return pd.DataFrame(rows)


# --- figures ----------------------------------------------------------------------------------
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
    axes[0].set_ylim(26, 40)
    axes[1].set_ylim(15, 24)
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
        ax.set_ylim(0, 80)
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
    TM = total_models()
    TM.to_csv(OUT / "total_models.csv", index=False)
    print(TM.round(4).to_string(index=False), flush=True)
    LM = link_models()
    LM.to_csv(OUT / "link_models.csv", index=False)
    HZ = horizons()
    HZ.to_csv(OUT / "horizons.csv", index=False)
    CX = context()
    CX.to_csv(OUT / "context.csv", index=False)
    print(CX.round(4).to_string(index=False), flush=True)
    GT = grid_table()
    GT.to_csv(OUT / "grid_candidates.csv")
    print(GT.round(1).to_string(), flush=True)
    LV = levels()
    CT = cap_table()
    CT.to_csv(OUT / "cap_sensitivity.csv", index=False)
    print(CT.round(4).to_string(index=False), flush=True)
    uni = json.load(open(P / "i2" / "unilateral.json"))
    U = pd.read_csv(P / "i2" / "unilateral_pairs.csv")
    dec = {"bytes_same_choice": uni.get("bytes_same_choice"),
           "captured_share_of_oracle_bytes": uni["bytes_weighted_reduction_forecast"] / uni["bytes_weighted_reduction_oracle"],
           "pairs_forecast_within_5pts_of_oracle": float(((U.reduction_oracle - U.reduction_forecast) <= 0.05).mean()),
           "pairs_corr_forecast_oracle": float(np.corrcoef(U.reduction_oracle, U.reduction_forecast)[0, 1]),
           "pairs_same_choice_median": float(U.same_choice.median()) if "same_choice" in U else None}
    json.dump({"levels": LV, "decision": dec}, open(OUT / "numbers.json", "w"), indent=1)
    print(json.dumps({"levels": LV, "decision": dec}, indent=1), flush=True)
    fig_context(CX, out)
    fig_horizon(HZ, out)
    fig_decision(out)
    GM = grid_monthly()
    months, periods, gper = grid_bias_tables(GM)
    months.to_csv(OUT / "grid_bias_months.csv")
    gper.to_csv(OUT / "grid_bias_periods.csv", index=False)
    print(months.round(1).to_string(), flush=True)
    print(gper.round(2).to_string(index=False), flush=True)
    BS = bias_summary()
    json.dump(BS, open(OUT / "bias.json", "w"), indent=1)
    print(json.dumps(BS, indent=1), flush=True)
    MP = model_params()
    MP.to_csv(OUT / "model_params.csv", index=False)
    print(MP.to_string(index=False), flush=True)
    fig_bias(months, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
