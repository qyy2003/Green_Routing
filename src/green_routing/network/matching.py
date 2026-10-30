"""Physical links and bundles from the traffic alone.

A cable shows up twice in the export: what one end receives the other end sends, so
``a.in`` tracks ``b.out`` and ``a.out`` tracks ``b.in``. ``match_ends`` scores every pair
of active interfaces on different routers by min(corr(a.in, b.out), corr(a.out, b.in))
and keeps mutual best matches with score > 0.9 and levels within 10 %. Routers that were
replaced during the year only overlap in some months, hence the extra hourly windows.

A bundle's in and out series are the sums of its members'; ``find_bundles`` adds
physical ports of the same router greedily while the summed series gets closer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 5-minute window of traffic.extract, then hourly windows that cover replaced routers
HOURLY_WINDOWS = (("2025-07-15", "2025-08-15"), ("2025-10-01", "2025-11-01"),
                  ("2026-01-10", "2026-02-10"), ("2026-03-20", "2026-04-20"))
MIN_SCORE, MAX_LOGLEV = 0.9, 0.1
MAX_BUNDLE_ERR = 0.05     # a bundle's members must reproduce its series within 5 %


def _z(X):
    mu = np.nanmean(X, 1, keepdims=True)
    sd = np.nanstd(X, 1, keepdims=True)
    return np.nan_to_num((X - mu) / np.where(sd > 0, sd, np.inf))


def match_ends(traffic: dict, window: tuple[str, str] | None = None) -> pd.DataFrame:
    """Best far-end candidate for every active interface.

    ``window=None`` uses the 5-minute matching window; ``(start, end)`` the hourly
    series between those dates. Columns: a, b, score, second (runner-up score),
    loglev (|log level ratio|), mutual, a_in, a_out.
    """
    links = traffic["links"]
    if window is None:
        WI, WO = traffic["win"].astype(np.float64), traffic["wout"].astype(np.float64)
    else:
        H = pd.to_datetime(traffic["hours"], unit="s", utc=True)
        m = (H >= window[0]) & (H < window[1])
        WI, WO = traffic["hin"][:, m].astype(np.float64), traffic["hout"][:, m].astype(np.float64)
    T = WI.shape[1]
    cov = (np.isfinite(WI) & np.isfinite(WO)).mean(1)
    mi, mo = np.nanmean(WI, 1), np.nanmean(WO, 1)
    idx = np.where((cov > 0.7) & (np.nan_to_num(mi + mo) > 0.2))[0]
    C = _z(WI[idx]) @ _z(WO[idx]).T / T          # C[a, b] = corr(a.in, b.out)
    S = np.minimum(C, C.T)                        # both directions must agree
    lr = np.abs(np.log((mi[idx][:, None] + 1e-3) / (mo[idx][None, :] + 1e-3)))
    L = np.maximum(lr, lr.T)
    dev = np.array([str(x).split("/")[0] for x in links[idx]])
    S[dev[:, None] == dev[None, :]] = -1          # never two ports of one router
    np.fill_diagonal(S, -1)
    rows = []
    for a in range(len(idx)):
        b = int(np.argmax(S[a]))
        rows.append(dict(a=links[idx[a]], b=links[idx[b]], score=S[a, b], second=np.sort(S[a])[-2],
                         loglev=L[a, b], mutual=int(np.argmax(S[b])) == a,
                         a_in=mi[idx[a]], a_out=mo[idx[a]]))
    return pd.DataFrame(rows)


def good(R: pd.DataFrame) -> pd.DataFrame:
    return R[(R.score > MIN_SCORE) & (R.loglev < MAX_LOGLEV) & R.mutual]


def match_all(traffic: dict, out: str | Path, windows=HOURLY_WINDOWS) -> dict[str, tuple]:
    """Run the 5-minute window, then each hourly window; the first window that matches
    an interface wins. Writes ``matches/best_match_<window>.csv``; returns
    {link: (peer, score, window)}."""
    d = Path(out) / "matches"
    d.mkdir(parents=True, exist_ok=True)
    part: dict[str, tuple] = {}
    for key, w in [("w", None)] + [(f"h_{a}", (a, b)) for a, b in windows]:
        R = match_ends(traffic, w)
        R.to_csv(d / f"best_match_{key}.csv", index=False)
        g = good(R)
        print(f"[network] window {key}: {len(R)} active interfaces, {len(g)} matched")
        for r in g.itertuples():
            part.setdefault(r.a, (r.b, r.score, key))
    return part


def find_bundles(stats: pd.DataFrame, traffic: dict) -> pd.DataFrame:
    """Members of every bundle: physical ports of its router whose in and out series
    correlate with the bundle's (r > 0.8) and whose summed series gets closest to it;
    kept when the sum reproduces the bundle within ``MAX_BUNDLE_ERR``.
    ``stats`` needs link, dev, kind. Tries the 5-minute window, then the hourly year."""
    d = traffic
    ix = {str(x): i for i, x in enumerate(d["links"])}

    def series(link, src):
        i = ix[link]
        return (d["win"][i], d["wout"][i]) if src == "w" else (d["hin"][i], d["hout"][i])

    rows = []
    for b in stats[stats.kind == "bundle"].itertuples():
        cands = stats[(stats.dev == b.dev) & (stats.kind == "phys")]
        best = None
        for src in ("w", "h"):
            bi, bo = series(b.link, src)
            m = np.isfinite(bi) & np.isfinite(bo)
            if m.sum() < 100 or np.nanmean(bi[m] + bo[m]) < 0.5:
                continue
            sc = []
            for c in cands.itertuples():
                ci, co = series(c.link, src)
                mm = m & np.isfinite(ci) & np.isfinite(co)
                if mm.sum() < 100 or np.nanstd(ci[mm]) == 0 or np.nanstd(co[mm]) == 0:
                    continue
                r = min(np.corrcoef(ci[mm], bi[mm])[0, 1], np.corrcoef(co[mm], bo[mm])[0, 1])
                if r > 0.8 and np.mean(ci[mm] + co[mm]) <= 1.05 * np.mean(bi[mm] + bo[mm]):
                    sc.append((r, c.link))
            sc.sort(reverse=True)
            chosen, best_err = [], np.inf
            for r, link in sc:          # greedy: add members while the sum gets closer
                trial = chosen + [link]
                si = sum(series(x, src)[0] for x in trial)
                so = sum(series(x, src)[1] for x in trial)
                mm = m & np.isfinite(si) & np.isfinite(so)
                err = ((np.abs(si[mm] - bi[mm]).sum() + np.abs(so[mm] - bo[mm]).sum())
                       / (bi[mm].sum() + bo[mm].sum()))
                if err < best_err - 1e-3:
                    chosen, best_err = trial, err
            if chosen:
                best = (src, chosen, best_err)
                break
        if best and best[2] < MAX_BUNDLE_ERR:
            rows.append(dict(bundle=b.link, src=best[0], members=" ".join(best[1]),
                             n=len(best[1]), rel_err=best[2]))
    return pd.DataFrame(rows, columns=["bundle", "src", "members", "n", "rel_err"])
