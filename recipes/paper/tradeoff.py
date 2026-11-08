"""Accuracy against inference time for every forecaster configuration, and what more training data buys
(paper appendix: "Accuracy and Cost", "Why an ISP Need Not Train").

sweep    Timestral-209M, Chronos-2 and TimesFM-2.5 zero-shot on each network's hourly total with 1, 2, 4, 12,
         24 and 48 weeks of context (``carbon.dayahead.foundation``, issued 00:00 UTC for 24 h), hourly WAPE on
         the test blocks.
timing   inference time of one day of day-ahead forecasts on one GPU (RTX PRO 4500), per configuration: the
         network total (one series) or all its links (113 / 534 series), with the batching of production
         (pretrained models in batches of 32; trained networks at once; GBDT on the CPU). Warm-up, then the
         median of five runs. Trained networks are timed with random weights (their cost does not depend on
         the weights); three-seed ensembles cost three times one model.
figures  app_pareto.pdf (error against inference time, Pareto frontier), app_learning.pdf (Internet2: per-link
         error against the history the trained models learn from, recipes/ladder_diagnosis data-size runs).

usage: python tradeoff.py [sweep|timing|figures|all] [--png=DIR]
writes $GR_ARTIFACTS_DIR/paper/appendix/{sweep_totals.csv, timing.csv, tradeoff_points.csv}
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR, REPO_ROOT

HERE = Path(__file__).resolve().parent
A = ARTIFACTS_DIR
OUT = A / "paper" / "appendix"
FIG = REPO_ROOT / "6aea077f0929c9a3a1ac5660" / "figure_appendix"
CTXS = (168, 336, 672, 2016, 4032, 8064)
PRETRAINED = ("timestral", "chronos-2", "timesfm-2.5")
N_LINKS = {"switch": 534, "i2": 113}
PNG = None


def log(*a):
    print(*a, flush=True)


# --- the network totals: series, test days, truth -------------------------------------------------
def totals():
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    sys.path.insert(0, str(HERE.parent / "internet2"))
    import run as R
    import timestral_total as TT
    C = A / "internet2" / "carbon"
    tab = pd.read_csv(C / "dayahead_hourly.csv", index_col=0, parse_dates=True)
    i2_days = pd.date_range(tab.index.min().floor("D"), tab.index.max().floor("D"), freq="D")
    return {"switch": (R.S, R.TEST, R.T), "i2": (TT.total_series(), i2_days, tab.actual)}


def sweep():
    from green_routing.carbon import dayahead as DA
    rows = []
    for net, (s, days, truth) in totals().items():
        for model in PRETRAINED:
            for ctx in CTXS:
                f = DA.foundation(s.dropna(), days, model=model, context=ctx)
                h = truth.dropna().index.intersection(f.dropna().index)
                w = float((f[h] - truth[h]).abs().sum() / truth[h].sum())
                rows.append(dict(network=net, model=model, context_h=ctx, hourly_wape=w, hours=len(h)))
                log(f"[sweep] {net} {model} {ctx // 168} w: {100 * w:.2f} %")
        pd.DataFrame(rows).to_csv(OUT / "sweep_totals.csv", index=False)
    return pd.DataFrame(rows)


# --- timing ---------------------------------------------------------------------------------------
def _t(fn, reps=5, cuda=True):
    import torch
    fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def timing():
    import torch
    rng = np.random.default_rng(0)
    rows = []

    def add(**kw):
        rows.append(kw)
        log(f"[timing] {kw}")

    # pretrained
    from timestral import TimestralForecaster
    for size in ("209M", "28M"):
        fm = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device="cuda")
        for target, n in (("total", 1), ("links_switch", N_LINKS["switch"]), ("links_i2", N_LINKS["i2"])):
            ctxs = [(c, 24, "1h") for c in CTXS] + ([(8064, 288, "5min"), (32256, 288, "5min")] if size == "209M" else [])
            for ctx, H, step in ctxs:
                X = rng.lognormal(6, 1, (n, 1, ctx)).astype(np.float32)

                def f():
                    with torch.inference_mode():
                        for b in range(0, n, 32):
                            fm.forecast(X[b:b + 32], prediction_length=H)
                add(model=f"timestral-{size}", target=target, context=ctx, step=step, seconds=_t(f))
        del fm
        torch.cuda.empty_cache()
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda", torch_dtype=torch.float32)
    for ctx in CTXS:
        ser = [torch.tensor(rng.lognormal(6, 1, ctx).astype(np.float32))]
        add(model="chronos-2", target="total", context=ctx, step="1h",
            seconds=_t(lambda: pipe.predict_quantiles(ser, prediction_length=24, quantile_levels=[0.5])))
    del pipe
    torch.cuda.empty_cache()
    import timesfm
    torch.set_float32_matmul_precision("high")
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    for ctx in CTXS:
        m.compile(timesfm.ForecastConfig(max_context=ctx, max_horizon=256, normalize_inputs=True,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=True, fix_quantile_crossing=True, per_core_batch_size=32))
        ser = [rng.lognormal(6, 1, ctx)]
        add(model="timesfm-2.5", target="total", context=ctx, step="1h",
            seconds=_t(lambda: m.forecast(horizon=24, inputs=ser)))
    del m
    torch.cuda.empty_cache()

    # trained networks: the ladder (5-minute, one week in, a week of 5-minute steps out) and the hourly versions
    from green_routing.prediction import models_deep as MD
    mods = MD._modules()
    sys.path.insert(0, str(HERE.parent / "ladder_diagnosis"))
    import hourly_train as HT
    for target, n in (("links_switch", N_LINKS["switch"]), ("links_i2", N_LINKS["i2"]), ("total", 1)):
        x = torch.randn(n, 2016, device="cuda")
        for name, cls in (("dlinear", "DLinear"), ("gru", "GRUNet"), ("tcn", "TCNNet"), ("patchtst", "PatchTST")):
            net = mods[cls](2016, 2016).cuda().eval()

            def f():
                with torch.no_grad():
                    net(x)
            add(model=f"{name} (5-min, 1 w)", target=target, context=2016, step="5min", seconds=_t(f))
        for ctx, H in ((168, 168), (672, 168), (4032, 168), (672, 24)):
            if target != "total" and H == 24 or target == "total" and H == 168:
                continue
            xh = torch.randn(n, ctx, device="cuda")
            mh = torch.ones_like(xh)
            for name in ("dlinear", "gru", "tcn", "patchtst"):
                net = HT.MODELS[name](ctx, H, dropout=0.1, patch=24, n_links=0).cuda().eval()

                def f():
                    with torch.no_grad():
                        net(xh, mh)
                add(model=f"{name} (hourly)", target=target, context=ctx, step="1h", seconds=_t(f))
    # GBDT (CPU): predict one day after a fit on a year of hourly rows
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xg = rng.normal(size=(8760, 180))
    yg = rng.normal(size=8760)
    t0 = time.perf_counter()
    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xg, yg)
    fit_s = time.perf_counter() - t0
    Xd = rng.normal(size=(24, 180))
    add(model="gbdt", target="total", context=168, step="1h", seconds=_t(lambda: g.predict(Xd), cuda=False), fit_seconds=fit_s)
    # rules: a shifted copy
    v = rng.normal(size=24 * 400)
    add(model="rules", target="total", context=168, step="1h", seconds=_t(lambda: v[-168:-144].copy(), cuda=False))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "timing.csv", index=False)
    return T


# --- points: (configuration, error, inference time) ----------------------------------------------
def points() -> pd.DataFrame:
    T = pd.read_csv(OUT / "timing.csv")
    S = pd.read_csv(OUT / "sweep_totals.csv")
    tm = lambda **k: float(T.loc[np.logical_and.reduce([T[a] == b for a, b in k.items()]), "seconds"].iloc[0])  # noqa: E731
    rows = []
    a = pd.read_csv(A / "ladder_diagnosis" / "all_runs.csv")
    for net in ("switch", "i2"):
        tgt = f"links_{net}"
        # per link: Timestral by context (hourly), Timestral 5-minute 4 w, the ladder, the hourly architectures
        for _, r in a[(a.ds == net) & (a.model == "timestral-209M")].iterrows():
            rows.append(dict(network=net, level="links", family="Timestral", label=f"{int(r.ctx) // 168} w",
                             weeks=r.ctx / 168, error=r.day, seconds=tm(model="timestral-209M", target=tgt, context=int(r.ctx))))
        base = {"switch": A / "pred_year_sym" / "merged", "i2": A / "internet2" / "links" / "merged"}[net]
        p = pd.read_csv(base / "dense_fm" / "tables_both" / "block" / "per_link.csv")
        med = p[p.horizon_name == "day"].groupby("model").wape_1h.median()
        rows.append(dict(network=net, level="links", family="Timestral (5-min)", label="4 w", weeks=4,
                         error=med["timestral@5m_4w"], seconds=tm(model="timestral-209M", target=tgt, context=8064)))
        for m in ("dlinear", "gru", "tcn", "patchtst"):
            rows.append(dict(network=net, level="links", family=f"{m} (5-min)", label="1 w", weeks=1, error=med[m],
                             seconds=tm(model=f"{m} (5-min, 1 w)", target=tgt)))
            for ctx in (168, 672, 4032):
                f = A / "ladder_diagnosis" / net / "runs" / f"H_{m}_c{ctx}" / "summary.json"
                if f.exists():
                    rows.append(dict(network=net, level="links", family=f"{m} (hourly)", label=f"{ctx // 168} w", weeks=ctx / 168,
                                     error=json.load(open(f))["test"]["day"]["median_wape"],
                                     seconds=tm(model=f"{m} (hourly)", target=tgt, context=ctx)))
        for m, lab in (("persistence", "persistence"), ("seasonal_naive_week", "same hour last week")):
            rows.append(dict(network=net, level="links", family="rule", label=lab, weeks=1, error=med[m],
                             seconds=tm(model="rules", target="total") * N_LINKS[net]))
        # network total: the pretrained sweep, the trained models, GBDT, the rules
        for _, r in S[S.network == net].iterrows():
            fam = {"timestral": "Timestral", "chronos-2": "Chronos-2", "timesfm-2.5": "TimesFM-2.5"}[r.model]
            mname = "timestral-209M" if r.model == "timestral" else r.model
            rows.append(dict(network=net, level="total", family=fam, label=f"{int(r.context_h) // 168} w",
                             weeks=r.context_h / 168, error=100 * r.hourly_wape,
                             seconds=tm(model=mname, target="total", context=int(r.context_h))))
        tmods = pd.read_csv(OUT / "total_models.csv")
        tmods = tmods[tmods.network == net].set_index("model")
        for m, lab in (("dlinear", "DLinear"), ("gru", "GRU"), ("tcn", "TCN"), ("patchtst", "PatchTST")):
            if net == "switch":        # the 5-minute ladder, three seeds
                rows.append(dict(network=net, level="total", family=f"{m} (5-min)", label="1 w", weeks=1,
                                 error=100 * tmods.loc[lab, "traffic_hourly_wape"],
                                 seconds=3 * tm(model=f"{m} (5-min, 1 w)", target="total")))
            else:                      # hourly, 4 weeks, three seeds
                rows.append(dict(network=net, level="total", family=f"{m} (hourly)", label="4 w", weeks=4,
                                 error=100 * tmods.loc[lab, "traffic_hourly_wape"],
                                 seconds=3 * tm(model=f"{m} (hourly)", target="total", context=672)))
        rows.append(dict(network=net, level="total", family="GBDT", label="GBDT", weeks=1,
                         error=100 * tmods.loc["GBDT", "traffic_hourly_wape"], seconds=tm(model="gbdt", target="total")))
        for lab, key in (("persistence", "Persistence (same hour yesterday)"), ("same hour last week", "Same hour last week")):
            rows.append(dict(network=net, level="total", family="rule", label=lab, weeks=1,
                             error=100 * tmods.loc[key, "traffic_hourly_wape"], seconds=tm(model="rules", target="total")))
    P = pd.DataFrame(rows)
    P["pareto"] = False
    for (net, lev), g in P.groupby(["network", "level"]):
        g = g.sort_values(["seconds", "error"])
        best = np.inf
        for i, r in g.iterrows():
            if r.error < best - 1e-9:
                P.loc[i, "pareto"] = True
                best = r.error
    P.to_csv(OUT / "tradeoff_points.csv", index=False)
    return P


# --- figures --------------------------------------------------------------------------------------
STYLE = {"Timestral": ("k", "o", True), "Timestral (5-min)": ("k", "s", False), "Chronos-2": ("#E24A33", "o", True),
         "TimesFM-2.5": ("#7A68A6", "o", True), "GBDT": ("#467821", "D", False), "rule": ("0.55", "^", False)}
TRAINED = {"dlinear": ("#188487", "D"), "gru": ("#467821", "^"), "tcn": ("#E24A33", "v"), "patchtst": ("#7A68A6", "s")}


def _save(fig, name):
    import matplotlib.pyplot as plt
    fig.savefig(FIG / f"{name}.pdf")
    if PNG:
        fig.savefig(PNG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_pareto(P: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.6), gridspec_kw={"wspace": 0.16, "hspace": 0.42})
    for i, lev in enumerate(("links", "total")):
        for j, (net, title) in enumerate((("switch", "SWITCHlan"), ("i2", "Internet2"))):
            ax = axes[i, j]
            g = P[(P.network == net) & (P.level == lev) & (P.family != "rule")]   # rules: ~1e-6 s, far worse (caption)
            fr = g[g.pareto].sort_values("seconds")
            xs = np.r_[fr.seconds.to_numpy(), 30]
            ys = np.r_[fr.error.to_numpy(), fr.error.min()]
            ax.step(xs, ys, where="post", color="#9ECAE1", lw=4, alpha=0.8, zorder=0)
            for fam, d in g.groupby("family"):
                d = d.sort_values("weeks")
                if fam in STYLE:
                    c, mk, line = STYLE[fam]
                    ax.plot(d.seconds, d.error, color=c, marker=mk, ms=3.5, ls="-" if line else "", lw=0.8,
                            mfc=c if fam != "Timestral (5-min)" else "white", zorder=3)
                    if fam == "Timestral":
                        for _, r in d[d.weeks.isin([1, 4, 24])].iterrows():
                            ax.annotate(r.label, (r.seconds, r.error), textcoords="offset points", xytext=(4, 1), fontsize=6)
                else:
                    arch, kind = fam.split(" ")
                    c, mk = TRAINED[arch]
                    ax.plot(d.seconds, d.error, color=c, marker=mk, ms=3.5, ls="", mfc="white" if "hourly" in kind else c,
                            zorder=2)
            ax.set_xscale("log")
            ax.set_xlim(2e-6, 30)
            ax.set_title(f"{title}, {'per link' if lev == 'links' else 'network total'}", fontsize=8, pad=2)
            ax.tick_params(labelsize=7)
            ax.grid(True, which="major")
            if i == 1:
                ax.set_xlabel("Inference time for one day (s)", fontsize=8)
            if j == 0:
                ax.set_ylabel("Error (%)", fontsize=8)
            hi = np.percentile(g.error, 85) * 1.15
            ax.set_ylim(min(g.error) * 0.9, hi)
    h = [Line2D([], [], color=STYLE[k][0], marker=STYLE[k][1], ls="-" if STYLE[k][2] else "", ms=3.5,
                mfc=STYLE[k][0] if k != "Timestral (5-min)" else "white", label=k) for k in STYLE if k != "rule"] + \
        [Line2D([], [], color=c, marker=mk, ls="", ms=3.5, label=a.replace("dlinear", "DLinear").replace("gru", "GRU")
                .replace("tcn", "TCN").replace("patchtst", "PatchTST")) for a, (c, mk) in TRAINED.items()] + \
        [Line2D([], [], color="#9ECAE1", lw=4, label="Pareto frontier")]
    fig.legend(handles=h, loc="upper center", ncol=6, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.05),
               handlelength=1.6, columnspacing=0.9)
    _save(fig, "app_pareto")


def fig_learning():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = pd.read_csv(A / "ladder_diagnosis" / "i2" / "data_size_runs.csv")
    d["cond"] = np.where(d.extra.fillna("") == "i2doors", "61+", d.months.astype(float).astype(str))
    order = ["6.0", "12.0", "24.0", "61.0", "61+"]
    labels = ["6 months", "1 year", "2 years", "5 years", "5 years\n+ doors"]
    fig, ax = plt.subplots()
    for (m, ctx), c, ls, mk in ((("patchtst", 672), "#7A68A6", "--", "s"), (("patchtst", 4032), "#7A68A6", "-", "s"),
                                (("profmlp", 672), "#188487", "--", "D"), (("profmlp", 4032), "#188487", "-", "D")):
        g = d[(d.model == m) & (d.ctx == ctx)].groupby("cond").day.mean().reindex(order)
        ax.plot(range(len(order)), g.to_numpy(), color=c, ls=ls, marker=mk, ms=3, mfc="white" if ctx == 672 else c)
    ax.axhline(16.22, color="k", lw=1.2)
    ax.axhline(16.07, color="k", lw=0.8, ls=":")
    ax.text(0, 16.35, "Timestral, no training", fontsize=7, va="bottom")
    ax.set_xticks(range(len(order)), labels, fontsize=7)
    ax.set_ylabel("Link error (%)")
    ax.set_xlabel("History the trained models learn from")
    ax.set_ylim(15.5, 22.5)
    ax.grid(True, which="major", axis="y")
    h = [Line2D([], [], color="#7A68A6", marker="s", ms=3, label="PatchTST"),
         Line2D([], [], color="#188487", marker="D", ms=3, label="profile MLP"),
         Line2D([], [], color="0.3", ls="--", label="4 weeks of context"), Line2D([], [], color="0.3", ls="-", label="24 weeks"),
         Line2D([], [], color="k", ls=":", label="Timestral, fine-tuned")]
    ax.legend(handles=h, loc="upper right", frameon=False, ncol=2, fontsize=6.5, handlelength=1.8, columnspacing=0.8)
    _save(fig, "app_learning")


def main():
    global PNG
    import matplotlib.pyplot as plt
    plt.style.use(HERE / "paper.mplstyle")
    if shutil.which("latex"):
        plt.rcParams["text.usetex"] = True
    args = [a for a in sys.argv[1:] if not a.startswith("--png=")]
    png = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--png=")]
    PNG = Path(png[0]) if png else None
    if PNG:
        PNG.mkdir(parents=True, exist_ok=True)
    what = args[0] if args else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if what in ("sweep", "all"):
        sweep()
    if what in ("timing", "all"):
        timing()
    if what in ("figures", "all"):
        P = points()
        pd.set_option("display.width", 200)
        log(P.sort_values(["network", "level", "seconds"]).round(4).to_string(index=False))
        fig_pareto(P)
        fig_learning()


if __name__ == "__main__":
    main()
