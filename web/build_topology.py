#!/usr/bin/env python3
"""Step: map the SWITCH topology + Cricket dataset onto the map.

Produces:
  data/links.json  — one entry per drawable backbone node-pair, with capacity
                     and the interface CSV(s) whose octets give its traffic.
  data/node_devices.json — per map node: router CSVs + PSU CSVs (for CPU /
                     power / temperature).

Only node-pairs whose *both* endpoints have a position in data/nodes.json are
kept (that's what we can draw). Device -> map-node resolution strips the `swi`
prefix and any trailing digits, then matches the map code or its label alias.
"""

import json
import re
from pathlib import Path

import _source as src

HERE = Path(__file__).resolve().parent
# Data source: the green_routing package's (S3 bucket + object cache, or a local
# mirror); see _source.py. The topology is not in the bucket: the research
# checkout's copy is used when present, else the one committed with the package.
TOPO = src.topology_txt()

nodes = json.load(open(HERE / "data" / "nodes.json"))

# code/label (uppercase) -> canonical map key
CODEMAP = {}
for k, v in nodes.items():
    CODEMAP[k.upper()] = k
    CODEMAP[v["label"].upper()] = k


def dev_short(dev):
    return dev[3:] if dev.startswith("swi") else dev


def dev_to_node(dev):
    d = dev_short(dev)
    m = re.match(r"([a-zA-Z]+)", d)
    return CODEMAP.get(m.group(1).upper()) if m else None


def iface_file(dev, iface):
    """(short-device-dir, filename) for an interface, if the CSV exists."""
    short = dev_short(dev)
    fname = iface.lower().replace("/", "_") + ".csv"
    return (short, fname) if src.exists(f"router-interfaces/{short}/{fname}") else None


def parse_topology():
    phys = []
    for line in open(TOPO):
        if "<=>" not in line:
            continue
        p = line.split()
        devA, ifA, devB, ifB, speed = p[0], p[1], p[3], p[4], float(p[5].strip("()"))
        phys.append((devA, ifA, devB, ifB, speed))
    return phys


def main():
    phys = parse_topology()
    links = {}   # (nodeA,nodeB) sorted -> record
    for devA, ifA, devB, ifB, speed in phys:
        na, nb = dev_to_node(devA), dev_to_node(devB)
        if not na or not nb or na == nb:
            continue
        # pick one interface CSV for this physical link (either end)
        f = iface_file(devA, ifA) or iface_file(devB, ifB)
        key = tuple(sorted([na, nb]))
        rec = links.setdefault(key, {
            "id": "-".join(key), "endpoints": list(key),
            "capacity_bps": 0.0, "members": [],
        })
        rec["capacity_bps"] += speed
        member = {"speed_bps": speed}
        if f:
            member["dir"] = f[0]
            member["file"] = f[1]
        rec["members"].append(member)

    # node devices: routers + PSUs per map node
    node_dev = {code: {"routers": [], "psus": []} for code in nodes}
    for f in src.names("routers"):
        if not f.endswith(".csv"):
            continue
        code = dev_to_node(f[:-4])
        if code in node_dev:
            node_dev[code]["routers"].append(f)
    for f in src.names("router-power"):
        if not f.endswith(".csv"):
            continue
        base = re.sub(r"_psu\d+$", "", f[:-4])
        code = dev_to_node(base)
        if code in node_dev:
            node_dev[code]["psus"].append(f)

    n_with_iface = sum(1 for k in links for m in links[k]["members"] if m.get("file"))
    if n_with_iface == 0:
        # The committed fallback topology (rebuilt from this viewer's own JSON) has no
        # real interface names, so nothing maps to a traffic CSV. Writing would wipe
        # the working links.json — keep it and say what is needed instead.
        raise SystemExit(
            f"no topology link resolved to an interface CSV (topology: {TOPO}). "
            "That happens with the fallback topology file; rebuild data/links.json only "
            "with the network-energy-efficiency-research checkout present "
            "(GR_RESEARCH_ROOT). Keeping the existing data/*.json.")

    with open(HERE / "data" / "links.json", "w") as fp:
        json.dump({
            "count": len(links),
            "links": [links[k] for k in sorted(links)],
        }, fp, indent=2)
    with open(HERE / "data" / "node_devices.json", "w") as fp:
        json.dump(node_dev, fp, indent=2)

    print(f"drawable links: {len(links)}  (physical members with a CSV: {n_with_iface})")
    print("links missing any interface CSV:",
          [links[k]["id"] for k in sorted(links)
           if not any(m.get("file") for m in links[k]["members"])])
    print("nodes with a router CSV:", sum(1 for c in node_dev if node_dev[c]["routers"]))
    print("nodes with a PSU  CSV:", sum(1 for c in node_dev if node_dev[c]["psus"]))


if __name__ == "__main__":
    main()
