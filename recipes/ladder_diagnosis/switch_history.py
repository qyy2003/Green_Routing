"""History of the 534 Switch LAN year-run links before the panel (2024-01 -> 2025-07), read
from the frozen local S3 cache (pitfall 18): hourly means of >= 6 of 12 five-minute values, or
(--res 5m) the raw 5-minute grid.

usage: GR_DATA_SOURCE=local GR_LOCAL_ROOT=~/.cache/green-routing/s3/network-energy-efficiency-research \
       python switch_history.py [--res 5m]
Writes <CACHE>/switch/history_hourly.npz (values [n_hours, L], hour timestamps, links), or
history_5m.npz (values [n_steps, L] float32, step timestamps, links).
"""
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import DS, CACHE

START = "2024-01-01"


def one(args):
    link, end, res = args
    from green_routing.dataset import config, loaders
    dev, iface = link.split("/", 1)
    root = Path(os.environ["GR_LOCAL_ROOT"]) / "router-interfaces" / dev / f"{iface}.csv"
    grid = pd.date_range(START, end, freq="1h" if res == "1h" else "5min", inclusive="left",
                         tz="UTC")
    try:
        s = loaders.interface_throughput(str(root))["total_mbps"]
    except Exception as e:
        return link, np.full(len(grid), np.nan, np.float32), str(e)
    s = s[(s.index >= grid[0]) & (s.index < pd.Timestamp(end, tz="UTC"))]
    g5 = pd.date_range(START, end, freq="5min", inclusive="left", tz="UTC")
    if res == "5m":
        return link, s.reindex(g5).to_numpy(np.float32), ""
    v = s.reindex(g5).to_numpy(np.float64).reshape(-1, 12)
    cnt = np.isfinite(v).sum(1)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(v, 1)
    return link, np.where(cnt >= 6, m, np.nan).astype(np.float32), ""


def main():
    res = "5m" if "--res" in sys.argv and sys.argv[sys.argv.index("--res") + 1] == "5m" else "1h"
    ds = DS("switch")
    end = pd.Timestamp(int(ds.hts[0]), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")
    with ProcessPoolExecutor(16) as ex:
        out_ = list(ex.map(one, [(l, end, res) for l in ds.links], chunksize=4))
    errs = [(l, e) for l, _, e in out_ if e]
    print(f"{len(errs)} links failed: {errs[:3]}")
    V = np.column_stack([r[1] for r in out_])
    hts = pd.date_range(START, end, freq="1h" if res == "1h" else "5min", inclusive="left",
                        tz="UTC")
    out = CACHE / "switch"
    out.mkdir(parents=True, exist_ok=True)
    if res == "5m":
        np.save(out / "history_5m_values.npy", V)
        np.savez(out / "history_5m.npz", ts=(hts.astype("int64") // 10**9).to_numpy(),
                 links=np.array(ds.links))
        print("steps", V.shape[0], "coverage per link: median", np.median(np.isfinite(V).mean(0)))
        return
    np.savez_compressed(out / "history_hourly.npz", values=V,
                        hts=(hts.astype("int64") // 10**9).to_numpy(), links=np.array(ds.links))
    cov = np.isfinite(V).mean(0)
    print("hours", V.shape[0], "coverage per link: median", np.median(cov),
          "share links > 0.8:", (cov > 0.8).mean())
    df = pd.DataFrame(V, index=hts)
    print((df.notna().resample("MS").mean().median(axis=1)).round(2).to_string())


if __name__ == "__main__":
    main()
