"""Analysis for the two experiments vs the pred_year baseline (dense, every-5-min logs).

E1  input window 1 week -> 1 month        pairs  m  vs  m@month   (tcn: tcn@wk20 vs tcn@month)
E2a post-hoc path smoothing               pairs  m  vs  m+ma15m / m+ma1h / m+ma3h
E2b smoothed-target training (1 h MA)     pairs  m  vs  m@starget

Every comparison is paired per link on identical targets (same links, same target
times, same actuals): r_l = MAE_variant / MAE_base, summarised as the median change,
the share of links improved and a two-sided Wilcoxon signed-rank test on log r.

Reads the year run's dense directory ($GR_ARTIFACTS_DIR/pred_year/merged/dense, scored
with ``gr-dense metrics --common``); writes $GR_ARTIFACTS_DIR/exp/results/*.csv and
results/summary.json (everything the report shows).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from functools import lru_cache

from scipy import stats

from green_routing.paths import ARTIFACTS_DIR
from green_routing.prediction.data import load_run_panel

RUN = ARTIFACTS_DIR / "pred_year" / "merged"
DENSE = RUN / "dense"
OUT = ARTIFACTS_DIR / "exp" / "results"
OUT.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _panel():
    """(values [T, L] in the run's link order, timestamps [T]) of the year run."""
    p = load_run_panel(RUN)
    return np.asarray(p.values, np.float64), np.asarray(p.timestamps)
HN = ["15min", "1h", "8h", "day", "week"]
SEQ = ["gru", "tcn", "dlinear", "patchtst"]


def load(mode):
    pl = pd.read_csv(DENSE / mode / "per_link.csv")
    agg = pd.read_csv(DENSE / mode / "aggregate.csv")
    pk = pd.read_csv(DENSE / mode / "peaks.csv")
    return pl, agg, pk


def paired(pl, base, var, mode, exp):
    rows = []
    for hn in HN:
        a = pl[(pl.model == base) & (pl.horizon_name == hn)].set_index("link")
        b = pl[(pl.model == var) & (pl.horizon_name == hn)].set_index("link")
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        ra = a.loc[common, "mae"].to_numpy(float)
        rb = b.loc[common, "mae"].to_numpy(float)
        ok = np.isfinite(ra) & np.isfinite(rb) & (ra > 1e-6)
        lr = np.log(rb[ok] / ra[ok])
        p = stats.wilcoxon(lr).pvalue if np.any(lr != 0) else 1.0
        rows.append({"exp": exp, "mode": mode, "base": base, "variant": var, "horizon": hn,
                     "n_links": int(ok.sum()),
                     "median_change_pct": 100 * (np.exp(np.median(lr)) - 1),
                     "share_improved": float((lr < 0).mean()),
                     "wilcoxon_p": float(p)})
    return rows


def summary_rows(pl, agg, pk, models, mode):
    """per model x horizon: median ratio vs persistence, pooled/median WAPE (5-min,
    hourly, daily), roughness, peak bias/WAPE."""
    out = []
    base = pl[pl.model == "persistence"].set_index(["horizon_name", "link"])["mae"]
    for m in models:
        for hn in HN:
            g = pl[(pl.model == m) & (pl.horizon_name == hn)].set_index("link")
            if g.empty:
                continue
            b = base.loc[hn].reindex(g.index)
            ok = np.isfinite(b) & (b > 1e-3) & np.isfinite(g.mae)
            a = agg[(agg.model == m) & (agg.horizon_name == hn)].iloc[0]
            k = pk[(pk.model == m) & (pk.horizon_name == hn)]
            out.append({"mode": mode, "model": m, "horizon": hn,
                        "median_ratio": float((g.mae[ok] / b[ok]).median()),
                        "share_beats_pers": float((g.mae[ok] / b[ok] < 1).mean()),
                        "pooled_mae": float(a.mae),
                        "pooled_wape": float(a.wape), "pooled_wape_1h": float(a.wape_1h),
                        "pooled_wape_1d": float(a.wape_1d),
                        "median_wape": float(g.wape.median()),
                        "median_wape_1h": float(g.wape_1h.median()),
                        "rough": float(a.rough), "median_rough": float(g.rough.median()),
                        "peak_bias_pct": float(k.peak_bias_pct.iloc[0]) if len(k) else np.nan,
                        "peak_wape": float(k.peak_wape.iloc[0]) if len(k) else np.nan,
                        "off_bias_pct": float(k.off_bias_pct.iloc[0]) if len(k) else np.nan})
    return out


def link_features():
    """Per-link train-window traits: weekly / daily autocorrelation of log traffic,
    mean level, burstiness, and how far the level drifts across the train window."""
    cfg = json.loads((RUN / "config.json").read_text())
    v, ts = _panel()
    tr_end = int(np.searchsorted(ts, pd.Timestamp(cfg["train_end"], tz="UTC").timestamp()))
    lv = np.log1p(np.clip(v[:tr_end], 0, None))

    def acf(x, k):
        a, b = x[k:], x[:-k]
        m = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 100 else np.nan

    rows = []
    for j, l in enumerate(cfg["links"]):
        x = lv[:, j]
        raw = v[:tr_end, j]
        w = 2016
        first, last = np.nanmean(x[:4 * w]), np.nanmean(x[-4 * w:])
        rows.append({"link": l, "acf_week": acf(x, 2016), "acf_day": acf(x, 288),
                     "mean_mbps": float(np.nanmean(raw)),
                     "cv": float(np.nanstd(raw) / max(np.nanmean(raw), 1e-9)),
                     "drift": float(abs(last - first))})
    return pd.DataFrame(rows)


def daily_series(base, var, mode, hidx):
    """Pooled WAPE per test day for base vs variant (identical points), from the logs."""
    meta = json.loads((DENSE / "meta.json").read_text())
    lo = meta["lo"]
    v, ts = _panel()
    Y = v[lo:].T
    ts = ts[lo:]
    day = (ts - ts[0]) // 86400
    A = np.asarray(np.load(DENSE / f"pred_{mode}_{base}.npy", mmap_mode="r")[:, hidx], float)
    B = np.asarray(np.load(DENSE / f"pred_{mode}_{var}.npy", mmap_mode="r")[:, hidx], float)
    ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(Y)
    out = []
    for d in np.unique(day):
        m = ok & (day == d)[None, :]
        den = np.abs(Y[m]).sum()
        if den <= 0 or m.sum() < 1000:
            continue
        out.append({"date": pd.Timestamp(int(ts[day == d][0]), unit="s").strftime("%Y-%m-%d"),
                    "base_wape": 100 * np.abs(A[m] - Y[m]).sum() / den,
                    "var_wape": 100 * np.abs(B[m] - Y[m]).sum() / den})
    return out


def main():
    meta = json.loads((DENSE / "meta.json").read_text())
    have = set(meta["modes"]["rolling"]) | set(meta["modes"]["block"])
    res = {"pairs": [], "summary": []}
    feats = link_features()
    feats.to_csv(OUT / "link_features.csv", index=False)
    for mode in ("rolling", "block"):
        pl, agg, pk = load(mode)
        avail = set(pl.model.unique())
        e1 = [(m, f"{m}@month") for m in ("gbdt", "gru", "dlinear", "patchtst")] + \
             [("tcn@wk20", "tcn@month"), ("tcn", "tcn@wk20")]
        e2a = [(m, f"{m}+ma{w}") for m in SEQ for w in ("15m", "1h", "3h")]
        e2b = [(m, f"{m}@starget") for m in SEQ]
        for exp, pairs in (("E1", e1), ("E2a", e2a), ("E2b", e2b)):
            for b, v in pairs:
                if b in avail and v in avail:
                    res["pairs"] += paired(pl, b, v, mode, exp)
        models = sorted(avail)
        res["summary"] += summary_rows(pl, agg, pk, models, mode)
    pairs = pd.DataFrame(res["pairs"])
    summ = pd.DataFrame(res["summary"])
    pairs.to_csv(OUT / "paired_tests.csv", index=False)
    summ.to_csv(OUT / "summary_by_model.csv", index=False)

    # E1 heterogeneity: which links gain from the month window (rolling, day horizon)
    het = []
    pl, _, _ = load("rolling")
    for b, v in [(m, f"{m}@month") for m in ("gbdt", "gru", "dlinear", "patchtst")] + \
                [("tcn@wk20", "tcn@month")]:
        for hn in ("8h", "day", "week"):
            a = pl[(pl.model == b) & (pl.horizon_name == hn)].set_index("link").mae
            c = pl[(pl.model == v) & (pl.horizon_name == hn)].set_index("link").mae
            if a.empty or c.empty:
                continue
            d = pd.DataFrame({"lr": np.log(c / a)}).join(feats.set_index("link"), how="inner")
            d = d[np.isfinite(d.lr)]
            row = {"base": b, "variant": v, "horizon": hn}
            for f in ("acf_week", "acf_day", "mean_mbps", "cv", "drift"):
                rho, p = stats.spearmanr(d[f], d.lr, nan_policy="omit")
                row[f"rho_{f}"], row[f"p_{f}"] = float(rho), float(p)
            q = pd.qcut(d.acf_week, 3, labels=["low", "mid", "high"])
            for lab in ("low", "mid", "high"):
                row[f"weekly_{lab}_median_change_pct"] = float(
                    100 * (np.exp(d.lr[q == lab].median()) - 1))
            het.append(row)
    pd.DataFrame(het).to_csv(OUT / "e1_heterogeneity.csv", index=False)

    extra = {}
    hidx = {h: i for i, h in enumerate(HN)}
    if "patchtst@month" in have:
        extra["daily_e1_patchtst_day"] = daily_series("patchtst", "patchtst@month",
                                                     "rolling", hidx["day"])
    if "gbdt@month" in have:
        extra["daily_e1_gbdt_week"] = daily_series("gbdt", "gbdt@month", "rolling", hidx["week"])
    extra["daily_key"] = "daily_e1_patchtst_day"
    # example trace: a typical mid-size link (~560 Mbps in train and test, strong
    # daily + weekly pattern, TCN 8-h WAPE at the candidates' median), two April days
    extra["trace"] = {"link": "el3/tengige0_0_0_23_0", "model": "tcn", "h_i": 2,
                      "t_from": "2026-04-14", "n_steps": 576, "mode": "block"}
    json.dump({"pairs": res["pairs"], "summary": res["summary"],
               "heterogeneity": het, **extra}, open(OUT / "summary.json", "w"),
              default=float)
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
