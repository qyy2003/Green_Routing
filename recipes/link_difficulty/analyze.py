"""Why is per-link traffic hard to forecast a day ahead? Diagnostics on the dense block logs.

For one run (a finished gr-benchmark run with dense block logs) and one model at the day
horizon (forecast issued at 00:00 UTC for the 24 hours), on hourly totals per link:

  aggregation  hourly WAPE per link, per router (links summed) and of the network sum, and the
               same with each link's errors shifted by a random number of whole days (keeps each
               link's errors, breaks their timing across links): lower than shuffled = links'
               errors cancel at the same hour (traffic moving between links), higher = shared shocks
  level/shape  the model's hourly WAPE, and again with each day rescaled to the true daily volume
               (what is left is intra-day shape)
  oracle       a weekly profile fitted on the test block itself (hour of week, in-sample: an upper
               bound for any purely periodic forecast), and that profile scaled to the true day
  size         error by mean link load (quartiles, Spearman)
  events       hours far from the link's usual level (< 1/3 or > 3x the median of the same hour of
               week over the 4 weeks before): share of hours vs share of the error
  concentration share of each link's error in its worst 5 % of hours

usage: python analyze.py RUN [--model patchtst] [--name label] [--seeds 20]
       (set GR_DATA_SOURCE / GR_LOCAL_ROOT / GR_ARTIFACTS_DIR as for the run)
writes $GR_ARTIFACTS_DIR/link_difficulty/<name>.json (or next to RUN with --out)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from green_routing.prediction.dense import RunCtx


def hourly(x, ok, tsec):
    """[L, n_t] 5-min values -> [L, n_h] hourly sums over hours with all 12 steps scored."""
    b = (tsec // 3600) - tsec[0] // 3600
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    cnt = np.add.reduceat(ok.astype(float), starts, axis=1)
    s = np.add.reduceat(np.where(ok, x, 0.0), starts, axis=1)
    hrs = pd.to_datetime(tsec[starts], unit="s", utc=True)
    return np.where(cnt == 12, s, np.nan), hrs


def wape(F, A, axis=None):
    m = np.isfinite(F) & np.isfinite(A)
    num = np.where(m, np.abs(F - A), 0).sum(axis=axis)
    den = np.where(m, np.abs(A), 0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default="patchtst")
    ap.add_argument("--name", default=None)
    ap.add_argument("--dense", default=None, help="dense dir (default <run>/dense)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    rc = RunCtx(run, a.dense)
    meta = json.loads((rc.out / "meta.json").read_text())
    hi = [h["name"] for h in meta["horizons"]].index("day")
    Y = rc.values_raw[rc.lo:].T.astype(np.float64)                    # [L, n_t] Mbit/s
    P = np.asarray(np.load(rc.out / f"pred_block_{a.model}.npy", mmap_mode="r")[:, hi, :], np.float64)
    tsec = rc.panel.timestamps[rc.lo:]
    ok = np.isfinite(P) & np.isfinite(Y)
    F, hrs = hourly(P, ok, tsec)
    A, _ = hourly(Y, ok, tsec)
    links = list(rc.panel.links)
    L, nh = A.shape
    res = {"run": str(run), "model": a.model, "links": L, "hours": int(nh)}

    # --- aggregation ------------------------------------------------------------------------
    per_link = wape(F, A, axis=1)
    res["link_hourly_wape_median"] = float(np.nanmedian(per_link))
    res["link_hourly_wape_pooled"] = float(wape(F, A))
    router = [l.split("/")[0] for l in links]
    groups = pd.Series(range(L)).groupby(router).apply(list)
    m = np.isfinite(F) & np.isfinite(A)
    Fz, Az = np.where(m, F, 0), np.where(m, A, 0)

    def agg_wapes(E):
        """WAPE of the per-router and network sums for errors E (F = A + E)."""
        r = [wape((Az[g] + E[g]).sum(0), Az[g].sum(0)) for g in groups if len(g) > 1]
        net = wape((Az + E).sum(0), Az.sum(0))
        return float(np.nanmedian(r)), float(net)

    E = Fz - Az
    res["router_sum_hourly_wape_median"], res["network_sum_hourly_wape"] = agg_wapes(E)
    res["routers_with_2plus_links"] = int(sum(len(g) > 1 for g in groups))
    rng = np.random.default_rng(0)
    sh = []
    days = nh // 24
    for _ in range(a.seeds):
        Es = np.empty_like(E)
        for j in range(L):
            Es[j] = np.roll(E[j], 24 * int(rng.integers(1, days)))
        sh.append(agg_wapes(Es))
    sh = np.array(sh)
    res["shuffled_router_sum_hourly_wape_median"] = float(sh[:, 0].mean())
    res["shuffled_network_sum_hourly_wape"] = float(sh[:, 1].mean())
    # correlation of hourly errors between links of the same router vs any two links
    def mean_corr(pairs):
        c = []
        for i, j in pairs:
            mm = m[i] & m[j]
            if mm.sum() > 48 and E[i][mm].std() > 0 and E[j][mm].std() > 0:
                c.append(np.corrcoef(E[i][mm], E[j][mm])[0, 1])
        return float(np.mean(c)) if c else np.nan, len(c)
    same = [(g[x], g[y]) for g in groups for x in range(len(g)) for y in range(x + 1, len(g))]
    anyp = [tuple(rng.choice(L, 2, replace=False)) for _ in range(len(same))]
    res["error_corr_same_router"], res["pairs_same_router"] = mean_corr(same)
    res["error_corr_random_pairs"], _ = mean_corr(anyp)

    # --- level vs shape -----------------------------------------------------------------------
    d = (hrs.floor("D") - hrs[0].floor("D")).days.to_numpy()
    full_day = np.zeros((L, nh), bool)
    Fd = np.full((L, nh), np.nan)
    for k in np.unique(d):
        c = d == k
        if c.sum() != 24:
            continue
        mk = m[:, c].all(axis=1)                                       # links with the whole day
        fs, as_ = F[:, c].sum(1), A[:, c].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Fd[:, c] = np.where((mk & (fs > 0))[:, None], F[:, c] * (as_ / fs)[:, None], np.nan)
        full_day[:, c] = mk[:, None]
    Fm = np.where(full_day, F, np.nan)
    res["link_hourly_wape_median_full_days"] = float(np.nanmedian(wape(Fm, A, axis=1)))
    res["link_hourly_wape_median_true_daily_volume"] = float(np.nanmedian(wape(Fd, A, axis=1)))
    Fday = np.array([[np.nansum(Fm[j, d == k]) for k in np.unique(d)] for j in range(L)])
    Aday = np.array([[np.nansum(np.where(full_day[j, d == k], A[j, d == k], np.nan)) for k in np.unique(d)] for j in range(L)])
    Fday[Aday == 0] = np.nan
    res["link_daily_wape_median"] = float(np.nanmedian(wape(Fday, Aday, axis=1)))

    # --- oracle weekly profile (in-sample on the test block) ---------------------------------
    how = (hrs.dayofweek * 24 + hrs.hour).to_numpy()
    prof = np.full((L, nh), np.nan)
    for b in range(168):
        c = how == b
        prof[:, c] = np.nanmean(np.where(m[:, c], A[:, c], np.nan), axis=1, keepdims=True)
    prof_m = np.where(full_day, prof, np.nan)
    res["oracle_weekly_profile_hourly_wape_median"] = float(np.nanmedian(wape(prof_m, A, axis=1)))
    Pd = np.full((L, nh), np.nan)
    for k in np.unique(d):
        c = d == k
        if c.sum() != 24:
            continue
        ps, as_ = np.nansum(prof_m[:, c], 1), A[:, c].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Pd[:, c] = np.where((full_day[:, c].all(1) & (ps > 0))[:, None], prof_m[:, c] * (as_ / ps)[:, None], np.nan)
    res["oracle_weekly_profile_true_daily_volume_hourly_wape_median"] = float(np.nanmedian(wape(Pd, A, axis=1)))

    # --- size -----------------------------------------------------------------------------------
    mean_mbps = np.nanmean(np.where(m, A, np.nan), axis=1) / 12
    okl = np.isfinite(per_link) & np.isfinite(mean_mbps)
    rho = spearmanr(mean_mbps[okl], per_link[okl])
    res["size_spearman_rho"] = float(rho.statistic)
    q = pd.qcut(mean_mbps[okl], 4, labels=False)
    res["size_quartile_mean_mbps"] = [float(np.median(mean_mbps[okl][q == i])) for i in range(4)]
    res["size_quartile_hourly_wape_median"] = [float(np.median(per_link[okl][q == i])) for i in range(4)]

    # --- events: hours far from the usual level (history only) -----------------------------
    V = rc.values_raw.astype(np.float64)                                # [T, L], full series
    ts_all = rc.panel.timestamps
    okall = np.isfinite(V.T)
    Ah_all, hrs_all = hourly(V.T, okall, ts_all)
    hist = pd.DataFrame(Ah_all.T, index=hrs_all)
    usual = pd.concat([hist.shift(168 * k) for k in range(1, 5)], axis=1, keys=range(4)).T.groupby(level=1).median().T
    U = usual.reindex(hrs).to_numpy().T                                  # [L, nh]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = A / U
    ev = m & np.isfinite(rel) & ((rel < 1 / 3) | (rel > 3)) & (U > 0)
    base = m & np.isfinite(rel) & (U > 0)
    ae = np.where(base, np.abs(F - A), 0)
    res["event_hour_share"] = float(ev.sum() / base.sum())
    res["event_error_share_pooled"] = float(ae[ev].sum() / ae.sum())
    per_ev = np.where(ae.sum(1) > 0, np.where(ev, ae, 0).sum(1) / np.maximum(ae.sum(1), 1e-12), np.nan)
    res["event_error_share_median_link"] = float(np.nanmedian(per_ev))
    res["links_with_event_hours"] = float((ev.sum(1) > 0).mean())
    res["event_low_hour_share"] = float((ev & (rel < 1 / 3)).sum() / base.sum())
    res["event_high_hour_share"] = float((ev & (rel > 3)).sum() / base.sum())

    # --- concentration ------------------------------------------------------------------------
    conc = []
    for j in range(L):
        e = np.abs(F[j] - A[j])[m[j]]
        if e.size >= 100 and e.sum() > 0:
            k = max(1, int(0.05 * e.size))
            conc.append(np.sort(e)[-k:].sum() / e.sum())
    res["worst_5pct_hours_error_share_median_link"] = float(np.median(conc))

    from green_routing.paths import ARTIFACTS_DIR
    od = Path(a.out) if a.out else ARTIFACTS_DIR / "link_difficulty"
    od.mkdir(parents=True, exist_ok=True)
    name = a.name or run.parent.name
    (od / f"{name}_{a.model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
