"""Download what the Internet2 pipeline needs from GlobalNOC TSDS (cached, rate-limited).

    python recipes/internet2/fetch.py inventory   every port of the 73 core routers: label, capacity,
                                                  mean traffic in October 2026 (one query per router)
    python recipes/internet2/fetch.py hourly      connector ports (PDP: members, peers, every service) and
                                                  backbone ports, hourly, 2024-11-01 ->
                                                  2026-11-01 (one query per router and half-year)
    python recipes/internet2/fetch.py power       every PoP power system, hourly amps, same period
    python recipes/internet2/fetch.py tunnels     old network, 2021-03-01 -> 03-29: the tunnel mesh and
                                                  the old routers' ports, hourly
    python recipes/internet2/fetch.py fivemin     backbone VLAN ports, 5-minute means of the 1-minute data
                                                  (kept about a year), 2025-11-15 -> 2026-11-01, one query per port

Output: $GR_ARTIFACTS_DIR/internet2/raw/*.csv.gz; every server answer is cached in ../cache/,
so an interrupted stage resumes where it stopped.
"""
from __future__ import annotations

import sys

import pandas as pd

from green_routing.dataset import internet2 as i2
from green_routing.paths import ARTIFACTS_DIR

ROOT = ARTIFACTS_DIR / "internet2"
RAW = ROOT / "raw"
H0, H1 = "2024-11-01", "2026-11-01"           # hourly period
F0, F1 = "2025-11-15", "2026-11-01"           # 5-minute period (1-minute data kept ~1 year)
T0, T1 = "2021-03-01", "2021-03-29"           # old network, tunnel mesh
DOOR_OR_BB = '(description like "^PDP" or description like "BACKBONE")'   # connector ports (doors) + backbone


def log(*a):
    print(*a, flush=True)


def core_nodes(c) -> list[str]:
    q = ('get node, intf, aggregate(values.input, 3600, average) as input '
         'between("10/15/2026 00:00:00 UTC","10/15/2026 01:00:00 UTC") by node, intf from interface '
         'where node like "^core" and description like "BACKBONE"')
    return sorted({r["node"] for r in c.query(q)})


def inventory(c):
    nodes = core_nodes(c)
    log(f"[inventory] {len(nodes)} core routers")
    out = []
    for k, n in enumerate(nodes, 1):
        q = (f"get node, intf, description, max_bandwidth, aggregate(values.input, 3600, average) as input, "
             f"aggregate(values.output, 3600, average) as output "
             f'between("{i2.tsds_time("2026-10-01")}","{i2.tsds_time("2026-11-01")}") by node, intf from interface '
             f'where node = "{n}"')
        for r in c.query(q):
            vi = [v for _, v in r.get("input") or [] if v is not None]
            vo = [v for _, v in r.get("output") or [] if v is not None]
            out.append(dict(node=n, intf=r["intf"], description=r.get("description"),
                            max_bandwidth=r.get("max_bandwidth"), hours=len(vi),
                            input=sum(vi) / len(vi) if vi else None, output=sum(vo) / len(vo) if vo else None))
        if k % 10 == 0:
            log(f"[inventory] {k}/{len(nodes)} ({c.live_requests} live)")
    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(RAW / "inventory.csv.gz", index=False)
    log(f"[inventory] {len(out)} ports")


def hourly(c, part: str | None = None):
    """``part`` = "k/n": only fill the cache for every n-th router from the k-th (a helper worker);
    the run without ``part`` reads everything (cached) and writes the file."""
    nodes = core_nodes(c)
    if part:
        k0, n = map(int, part.split("/"))
        nodes = nodes[k0::n]
    halves = list(pd.date_range(H0, H1, freq="6MS")) + [pd.Timestamp(H1)]
    parts = []
    for k, n in enumerate(nodes, 1):
        for a, b in zip(halves[:-1], halves[1:]):
            q = (f"get node, intf, description, aggregate(values.input, 3600, average) as input, "
                 f"aggregate(values.output, 3600, average) as output "
                 f'between("{i2.tsds_time(a)}","{i2.tsds_time(b)}") by node, intf from interface '
                 f'where node = "{n}" and {DOOR_OR_BB}')
            parts.append(i2._long(c.query(q), ["node", "intf", "description"], ["input", "output"]))
        log(f"[hourly{'' if not part else ' ' + part}] {k}/{len(nodes)} {n} ({c.live_requests} live)")
    if part:
        return
    RAW.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(RAW / "hourly_ports.csv.gz", index=False)


def power(c):
    nodes = c.meta_values("power", "node")
    parts = []
    for k, n in enumerate(nodes, 1):
        parts.append(i2.power(c, n, H0, H1))
        if k % 25 == 0:
            log(f"[power] {k}/{len(nodes)} ({c.live_requests} live)")
    RAW.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(RAW / "power_hourly.csv.gz", index=False)


def tunnels(c):
    routers = sorted(c.meta_values("lsp", "from"))
    L, I = [], []
    for k, r in enumerate(routers, 1):
        L.append(i2.lsps(c, r, T0, T1))
        I.append(i2.interfaces(c, r, T0, T1))
        log(f"[tunnels] {k}/{len(routers)} {r}: {L[-1].name.nunique()} tunnels ({c.live_requests} live)")
    RAW.mkdir(parents=True, exist_ok=True)
    pd.concat(L).to_csv(RAW / "tunnels_2021-03.csv.gz", index=False)
    pd.concat(I).to_csv(RAW / "old_ports_2021-03.csv.gz", index=False)


def fivemin(c):
    inv = pd.read_csv(RAW / "inventory.csv.gz")
    bb = inv[inv.description.fillna("").str.upper().str.startswith("BACKBONE")
             & inv.description.str.contains(r"-VLAN-\d+", regex=True) & (inv.hours > 0)].copy()
    # one end per circuit is enough for the panel (the hourly data keep both ends as a check)
    bb["circuit"] = bb.description.str.extract(r"(I2-[A-Z0-9]+-[A-Z0-9]+-VLAN-\d+)", expand=False)
    bb = bb.sort_values(["circuit", "node"]).drop_duplicates("circuit")
    log(f"[fivemin] {len(bb)} backbone circuits (one end each)")
    parts = []
    for k, r in enumerate(bb.itertuples(), 1):
        q = (f"get node, intf, aggregate(values.input, 300, average) as input, "
             f"aggregate(values.output, 300, average) as output "
             f'between("{i2.tsds_time(F0)}","{i2.tsds_time(F1)}") by node, intf from interface '
             f'where node = "{r.node}" and intf = "{r.intf}"')
        parts.append(i2._long(c.query(q), ["node", "intf"], ["input", "output"]))
        if k % 10 == 0:
            log(f"[fivemin] {k}/{len(bb)} ({c.live_requests} live)")
    pd.concat(parts).to_csv(RAW / "backbone_5min.csv.gz", index=False)


def main():
    c = i2.Client(ROOT / "cache", pause=1.0)
    if len(sys.argv) == 3 and sys.argv[1] == "hourly" and "/" in sys.argv[2]:
        return hourly(c, sys.argv[2])
    for stage in sys.argv[1:] or ["inventory", "hourly", "power", "tunnels", "fivemin"]:
        {"inventory": inventory, "hourly": hourly, "power": power, "tunnels": tunnels, "fivemin": fivemin}[stage](c)
        log(f"[{stage}] done ({c.live_requests} live requests so far)")


if __name__ == "__main__":
    main()
