"""Timestral per link on the common periods, zero-shot, with 24 weeks of hourly context before every
forecast day, on the per-link runs of ladder_common.sh (same links and test block as the trained
baselines). The runs' 5-minute panels start at periods.TRAIN_START_5MIN (2025-11-15), 19.5 weeks before
the test block, so the hours before the panel come from the hourly counters:

  i2      internet2/data/links_hourly.csv.gz (u->v + v->u = the panel link, as hourly_train.py --hist ext)
  switch  the frozen local S3 cache, hourly means of >= 6 of 12 five-minute values (as switch_history.py)

Inside the panel the input is the panel's hourly means (>= 6 of 12 steps), as gr-dense's foundation path.
Block layout at 1 h (Internet2 only, as in round 1), 8 h, day and week; each hourly forecast is repeated over
its 12 steps, so hourly and daily WAPE are exact. Writes $PAPER_COMMON/links/<net>/dense_fm/
pred_block_timestral@1h_24w.npy (the gr-dense format; the run's links and test block, read from part_baselines,
which has the same data window as every part), the context sweep (1, 2, 4, 12, 24 weeks; day and week) to
$PAPER_COMMON/links/<net>/ctx/pred_<h>_c<ctx>.npy ([L, test hours]); `link` symlinks the merged run's block logs
into dense_fm/ and rewrites its meta.json (after ladder_common.sh dense).

usage (env as ladder_common.sh: data source and panel cache of the network):
  python timestral_links.py NET [main] [ctx] [link]
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import periods as P                                                      # noqa: E402
from green_routing.prediction import dense                               # noqa: E402

REPO_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
PRE_WEEKS = 30
CTXS = (168, 336, 672, 2016, 4032)


def log(*a):
    print(*a, flush=True)


def panel_hourly(rc) -> tuple[np.ndarray, int, np.ndarray]:
    V = rc.values_raw.astype(np.float32)
    ts = rc.panel.timestamps
    off = int(np.flatnonzero(ts % 3600 == 0)[0])
    nh = (rc.T - off) // 12
    blk = V[off:off + nh * 12].reshape(nh, 12, rc.L)
    cnt = np.isfinite(blk).sum(axis=1)
    with np.errstate(invalid="ignore"):
        src = np.where(cnt >= 6, np.nanmean(blk, axis=1), np.nan).astype(np.float32)
    return src, off, ts[off::12][:nh]


def _switch_one(args):
    link, a, b = args
    from green_routing.dataset import loaders
    dev, iface = link.split("/", 1)
    f = Path(os.environ["GR_LOCAL_ROOT"]) / "router-interfaces" / dev / f"{iface}.csv"
    g5 = pd.date_range(a, b, freq="5min", inclusive="left")
    try:
        s = loaders.interface_throughput(str(f))["total_mbps"]
    except Exception as e:                                                 # noqa: BLE001
        return link, np.full(len(g5) // 12, np.nan, np.float32), str(e)
    v = s.reindex(g5).to_numpy(np.float64).reshape(-1, 12)
    cnt = np.isfinite(v).sum(1)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(v, 1)
    return link, np.where(cnt >= 6, m, np.nan).astype(np.float32), ""


def pre_history(net: str, links: list[str], t0: pd.Timestamp) -> np.ndarray:
    """[PRE_WEEKS * 168, L] hourly loads before the panel's first hour t0."""
    a = t0 - pd.Timedelta(weeks=PRE_WEEKS)
    f = P.ROOT / "links" / net / f"pre_history_{PRE_WEEKS}w.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        assert list(z["links"]) == list(links)
        return z["values"]
    H = pd.date_range(a, t0, freq="h", inclusive="left")
    if net == "i2":
        h = pd.read_csv(REPO_ARTIFACTS / "internet2/data/links_hourly.csv.gz", index_col=0, parse_dates=True)
        cols = {}
        for l in links:
            u, v = l.split("/to-")
            cols[l] = h[f"{u}->{v}"].add(h[f"{v}->{u}"])
        V = pd.DataFrame(cols).reindex(H).to_numpy(np.float32)
    else:
        with ProcessPoolExecutor(16) as ex:
            out = list(ex.map(_switch_one, [(l, a, t0) for l in links], chunksize=4))
        bad = [(l, e) for l, _, e in out if e]
        log(f"switch pre-history: {len(bad)} links failed {bad[:3]}")
        V = np.column_stack([r[1] for r in out])
    np.savez_compressed(f, values=V, links=np.array(links), hts=(H.asi8 // 10**9))
    log(f"{net} pre-history {H[0]} -> {H[-1]}: coverage median {np.median(np.isfinite(V).mean(0)):.3f}")
    return V


def forecast(fm, src, origins, ctx, H, L, batch=256):
    """[L, len(origins), H] medians; origins are row indices of src (the first forecast hour)."""
    import torch
    out = np.full((L, len(origins), H), np.nan, np.float32)
    jobs = [(j, k) for j in range(L) for k in range(len(origins))]
    for b0 in range(0, len(jobs), batch):
        jb = jobs[b0:b0 + batch]
        X = np.full((len(jb), 1, ctx), np.nan, np.float32)
        for r, (j, k) in enumerate(jb):
            s = origins[k]
            a = max(0, s - ctx)
            X[r, 0, ctx - (s - a):] = src[a:s, j]
        with torch.inference_mode():
            f = fm.forecast(X, prediction_length=H)
        med = f.median[:, 0, :].float().cpu().numpy()
        for r, (j, k) in enumerate(jb):
            if np.isfinite(X[r, 0, -24:]).any():                          # something seen in the last day
                out[j, k] = med[r]
    return out


def main():
    net = sys.argv[1]
    stages = sys.argv[2:] or ["main", "ctx"]
    from timestral import TimestralForecaster
    fmdir = P.ROOT / "links" / net / "dense_fm"
    if stages == ["link"]:
        run = P.ROOT / "links" / net / "merged"
        rc = dense.RunCtx(run, fmdir)
        for p in (run / "dense").glob("pred_block_*.npy"):
            q = fmdir / p.name
            if not q.exists():
                q.symlink_to(p)
        dense._write_meta(rc)
        return
    run = P.ROOT / "links" / net / "part_baselines"
    rc = dense.RunCtx(run, fmdir)
    src_p, off, hts = panel_hourly(rc)
    t0 = pd.Timestamp(int(hts[0]), unit="s", tz="UTC")
    pre = pre_history(net, rc.panel.links, t0)
    src = np.vstack([pre, src_p])
    n_pre = len(pre)
    hlo = (rc.lo - off) // 12 + n_pre                                     # first test hour in src
    n_test_h = rc.n_t // 12
    fm = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda")
    fmdir.mkdir(parents=True, exist_ok=True)
    if "main" in stages and not (fmdir / "pred_block_timestral@1h_24w.npy").exists():
        arr, tmp = dense._alloc(rc, "block", "timestral@1h_24w")
        t_start = time.time()
        for hi, (hn, Hs) in enumerate(zip(rc.hnames, rc.hsteps)):
            if hn not in ("1h", "8h", "day", "week") or (hn == "1h" and net == "switch"):
                continue
            Hh = Hs // 12
            origins = np.arange(hlo, hlo + n_test_h, Hh)
            f = forecast(fm, src, origins, P.CONTEXT_HOURS, Hh, rc.L).reshape(rc.L, -1)[:, :n_test_h]
            arr[:, hi, :n_test_h * 12] = np.repeat(f, 12, axis=1)
            log(f"[{net}] timestral@1h_24w {hn}: {rc.L} x {len(origins)} forecasts, {time.time() - t_start:.0f}s")
        dense._finish(arr, tmp)
        (P.ROOT / "links" / net / "timestral_seconds.json").write_text(json.dumps({"seconds": time.time() - t_start}))
    if "ctx" in stages:
        od = P.ROOT / "links" / net / "ctx"
        od.mkdir(parents=True, exist_ok=True)
        for ctx in CTXS:
            for hn, Hh in (("day", 24), ("week", 168)):
                f = od / f"pred_{hn}_c{ctx}.npy"
                if f.exists():
                    continue
                origins = np.arange(hlo, hlo + n_test_h, Hh)
                p = forecast(fm, src, origins, ctx, Hh, rc.L).reshape(rc.L, -1)[:, :n_test_h]
                np.save(f, p)
                log(f"[{net}] context {ctx} h, {hn}: done")
    dense._write_meta(rc)


if __name__ == "__main__":
    main()
