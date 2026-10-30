"""One pass over every interface CSV: summary statistics and the in/out series that
the link-end matching needs.

Writes into ``out``:

    iface_stats.csv   per interface: columns present, first/last row, first/last row with
                      data, samples and mean/p99 in/out Mbit/s over the year window,
                      packets and errors where exported
    traffic.npz       links; hourly in/out Mbit/s over the year window (hin, hout);
                      5-minute in/out over the matching window (win, wout)

The year window defaults to the year benchmark's (2025-07-01 -> 2026-07-11), the
matching window to two weeks of 5-minute data (2026-06-01 -> 06-15).
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import loaders

YEAR = ("2025-07-01", "2026-07-11")
MATCH = ("2026-06-01", "2026-06-15")

_ctx: dict = {}


def _one(item):
    dev, iface, uri = item
    Y0, Y1, W0, W1, H, G = (_ctx[k] for k in ("Y0", "Y1", "W0", "W1", "H", "G"))
    try:
        df = loaders.read_series(uri)
    except Exception as e:          # noqa: BLE001 -- recorded and reported by the caller
        return dict(link=f"{dev}/{iface}", err=str(e))
    cols = df.columns

    def col(c):
        return df[c] * 8 / 1e6 if c in cols else pd.Series(np.nan, index=df.index)

    i, o = col("ifHCInOctets"), col("ifHCOutOctets")
    ip = df["ifHCInUcastPkts"] if "ifHCInUcastPkts" in cols else (
        df["ifInUcastPackets"] if "ifInUcastPackets" in cols else None)
    op = df["ifHCOutUcastPkts"] if "ifHCOutUcastPkts" in cols else (
        df["ifOutUcastPackets"] if "ifOutUcastPackets" in cols else None)
    y = (df.index >= Y0) & (df.index < Y1)
    has = df.notna().any(axis=1)
    r = dict(link=f"{dev}/{iface}", cols=";".join(cols), first=str(df.index.min()),
             last=str(df.index.max()),
             first_data=str(df.index[has].min()) if has.any() else "",
             last_data=str(df.index[has].max()) if has.any() else "",
             n_year=int(i[y].notna().sum()), in_mean=float(i[y].mean()), out_mean=float(o[y].mean()),
             in_p99=float(i[y].quantile(.99)) if i[y].notna().any() else np.nan,
             out_p99=float(o[y].quantile(.99)) if o[y].notna().any() else np.nan)
    if ip is not None:
        r["in_pps"] = float(ip[y].mean())
        r["out_pps"] = float(op[y].mean()) if op is not None else np.nan
    for c in ("ifInErrors", "ifOutErrors"):
        if c in cols:
            r[c] = float(df[c][y].mean())
    hi = i[y].resample("1h").mean().reindex(H).to_numpy(np.float32)
    ho = o[y].resample("1h").mean().reindex(H).to_numpy(np.float32)
    w = (df.index >= W0) & (df.index < W1)
    wi = i[w].reindex(G).to_numpy(np.float32)
    wo = o[w].reindex(G).to_numpy(np.float32)
    return r, hi, ho, wi, wo


def extract(out: str | Path, year=YEAR, match=MATCH, workers: int = 16) -> pd.DataFrame:
    """Read every interface CSV once (in parallel) and write iface_stats.csv + traffic.npz."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    Y0, Y1 = (pd.Timestamp(t, tz="UTC") for t in year)
    W0, W1 = (pd.Timestamp(t, tz="UTC") for t in match)
    _ctx.update(Y0=Y0, Y1=Y1, W0=W0, W1=W1,
                H=pd.date_range(Y0, Y1, freq="1h", inclusive="left"),
                G=pd.date_range(W0, W1, freq="5min", inclusive="left"))
    items = sorted(loaders.iter_interface_files(), key=lambda x: x[2])
    loaders.prefetch_interfaces(verbose=True)
    print(f"[network] reading {len(items)} interface CSVs with {workers} workers", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_one, items, chunksize=4)
    stats, HI, HO, WI, WO = [], [], [], [], []
    for x in res:
        if isinstance(x, dict):
            print("[network] unreadable:", x)
            continue
        r, hi, ho, wi, wo = x
        stats.append(r); HI.append(hi); HO.append(ho); WI.append(wi); WO.append(wo)
    S = pd.DataFrame(stats)
    S.to_csv(out / "iface_stats.csv", index=False)
    np.savez_compressed(out / "traffic.npz", links=np.array([s["link"] for s in stats]),
                        hours=_ctx["H"].astype("int64") // 10**9,
                        grid=_ctx["G"].astype("int64") // 10**9,
                        hin=np.stack(HI), hout=np.stack(HO), win=np.stack(WI), wout=np.stack(WO))
    print(f"[network] {len(S)} interfaces -> {out}")
    return S
