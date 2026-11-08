"""Hourly power and traffic of every metered device, for the paper's Fig. 1 (power does not follow
traffic). The same data and definitions as the published numbers:

- SWITCHlan: each router's PSU power (all PSUs summed) and its summed interface traffic, hourly,
  2024-02-01 -> 2024-10-31 (``carbon.inventory.traffic_dependence``: the share of power that follows
  traffic, slope x mean traffic / mean power, is under 1 % on 113 of 116 routers);
- Internet2: each core PoP's router-side DC power (power plants, DWDM shelves excluded) and the
  traffic through the PoP (door ports in + out, backbone links at either end), hourly over the year
  2025-11-15 -> 2026-11-01 (recipes/internet2/carbon.py, dependence.csv: median R^2 0.013).

usage: python power_traffic.py [switch|i2|all]
writes $GR_ARTIFACTS_DIR/paper/power_traffic_{switch,i2}.csv.gz (device, time, W, gbps) and prints
the check against the published summaries.
"""
from __future__ import annotations

import collections
import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
OUT = ARTIFACTS_DIR / "paper"


def _switch_device(item):
    from green_routing.dataset import loaders
    dev, psu_uris, if_uris, start, end = item
    W = pd.concat([loaders.psu_power(u) for u in psu_uris], axis=1).dropna().sum(axis=1)
    tr = None
    for u in if_uris:
        try:
            t = loaders.interface_throughput(u)["total_mbps"]
        except Exception:        # noqa: BLE001 -- a broken interface file is skipped (as inventory.py)
            continue
        tr = t if tr is None else tr.add(t, fill_value=0)
    if tr is None or W.empty:
        return None
    df = pd.concat([W.rename("W"), tr.rename("mbps")], axis=1).loc[start:end].resample("1h").mean().dropna()
    if len(df) < 500:
        return None
    df["gbps"] = df.mbps / 1e3
    return df[["W", "gbps"]].assign(device=dev)


def switch(workers: int = 12, start="2024-02-01", end="2024-10-31"):
    from green_routing.carbon.inventory import full_name
    from green_routing.dataset import loaders
    psu = collections.defaultdict(list)
    for dev, _, uri in loaders.iter_psu_files():
        if "lab" not in dev:
            psu[dev].append(uri)
    ifs = collections.defaultdict(list)
    for d, _, uri in loaders.iter_interface_files():
        ifs[full_name(d)].append(uri)
    items = [(d, psu[d], ifs[d], start, end) for d in sorted(psu) if ifs.get(d)]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = [p for p in pool.map(_switch_device, items) if p is not None]
    df = pd.concat(parts).rename_axis("time").reset_index()
    df.to_csv(OUT / "power_traffic_switch.csv.gz", index=False)
    share = df.groupby("device").apply(lambda d: np.polyfit(d.gbps, d.W, 1)[0] * d.gbps.mean() / d.W.mean())
    print(f"SWITCHlan: {share.size} routers, traffic-dependent share median {100 * share.median():.2f} %, "
          f"under 1 % on {(share.abs() < 0.01).sum()} (published: under 1 % on 113 of 116)", flush=True)


def i2():
    from green_routing.dataset import internet2 as I2
    spec = importlib.util.spec_from_file_location("i2_carbon", HERE.parent / "internet2" / "carbon.py")
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    links, LH, D, DI, DO, _ = c.load()
    H = pd.date_range(c.YEAR0, c.YEAR1, freq="h", inclusive="left", tz="UTC")
    core = sorted(set(links.u) | set(links.v) | set(D.node))
    pop_of = {r: I2.pop_of(r) for r in core}
    core_pops = sorted(set(pop_of.values()))
    door_router = D.set_index("door").node
    full = lambda s: s + I2.FQDN                                                  # noqa: E731
    port_tr = (pd.concat([DI, DO]).groupby(level=0).sum()).T.groupby(door_router.reindex(DI.columns).map(pop_of)).sum().T
    bb_tr = LH.T.groupby([pop_of.get(full(x.split("->")[0])) for x in LH.columns]).sum().T \
        .add(LH.T.groupby([pop_of.get(full(x.split("->")[1])) for x in LH.columns]).sum().T, fill_value=0)
    sw = port_tr.add(bb_tr, fill_value=0).reindex(H)                               # Mbit/s through each PoP
    Wraw = pd.read_csv(c.DATA / "power_hourly.csv.gz", parse_dates=["time"])
    Wraw = Wraw[Wraw.dc & Wraw["pop"].isin(core_pops) & ~Wraw.type.str.startswith("dwdm")]
    PR = Wraw.pivot_table(index="time", columns="pop", values="watts", aggfunc="sum").reindex(H)
    parts, r2 = [], {}
    for p in core_pops:
        if p not in PR or p not in sw:
            continue
        d = pd.DataFrame({"W": PR[p], "gbps": sw[p] / 1e3}).dropna()
        d = d[d.W > 0]
        if len(d) < 24 * 30:
            continue
        r2[p] = float(np.corrcoef(d.gbps, d.W)[0, 1] ** 2) if d.gbps.std() > 0 and d.W.std() > 0 else 0.0
        parts.append(d.assign(device=p))
    df = pd.concat(parts).rename_axis("time").reset_index()
    df.to_csv(OUT / "power_traffic_i2.csv.gz", index=False)
    print(f"Internet2: {len(r2)} PoPs, median R^2 {np.median(list(r2.values())):.3f} (published 0.013)", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    if what in ("i2", "all"):
        i2()
    if what in ("switch", "all"):
        switch()
