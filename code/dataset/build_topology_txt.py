"""Reconstruct a device-level ``switch-network-topology.txt`` from the repo's own
``web/data`` JSON, for machines that don't have the original
``network-energy-efficiency-research`` checkout.

The canonical source is that research repo's ``switch-network-topology.txt`` (SNMP/OSPF
dump, ``swiXX <=> swiYY`` device pairs). When it isn't present, this rebuilds an
equivalent edge list from the derived topology JSON that *is* checked in here:

  * ``web/data/links.json``        physical PoP-to-PoP links (SWITCHlan backbone)
  * ``web/data/node_devices.json`` routers present at each PoP

We lift each PoP-level link to the device level by connecting every router at one
endpoint to every router at the other, and also connect co-located routers within a PoP
(they share the site and are physically adjacent). Speeds come from the link's capacity;
OSPF costs aren't in the JSON, so a nominal cost is emitted (unused by the STGNN, which
only reads the adjacency structure via ``dataset.topology.parse_links``).

Run:  python -m dataset.build_topology_txt   # writes dataset/switch-network-topology.txt
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # .../Green_Routing/Green_Routing
WEB_DATA = REPO / "web" / "data"
OUT = Path(__file__).resolve().parent / "switch-network-topology.txt"

_NOMINAL_COST = 20000                                # OSPF cost placeholder (v4==v6)


def _pop_routers() -> dict[str, list[str]]:
    nd = json.loads((WEB_DATA / "node_devices.json").read_text())
    return {pop: sorted(r[:-4] for r in v.get("routers", []))   # strip ".csv"
            for pop, v in nd.items()}


def build() -> str:
    pops = _pop_routers()
    links = json.loads((WEB_DATA / "links.json").read_text())["links"]
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    def emit(a: str, b: str, speed: float):
        key = tuple(sorted((a, b)))
        if a == b or key in seen:
            return
        seen.add(key)
        # match topology.py's regex: devA ifA <=> devB ifB (speed) (costv4 costv6)
        lines.append(
            f"{a} - <=> {b} - ({speed:g}) ({_NOMINAL_COST} {_NOMINAL_COST})"
        )

    # inter-PoP: every router at one endpoint <=> every router at the other
    for lk in links:
        a_pop, b_pop = lk["endpoints"]
        speed = float(lk.get("capacity_bps", 1e11))
        for da, db in itertools.product(pops.get(a_pop, []), pops.get(b_pop, [])):
            emit(da, db, speed)

    # intra-PoP: co-located routers are physically adjacent
    for routers in pops.values():
        for da, db in itertools.combinations(routers, 2):
            emit(da, db, 1e11)

    return "\n".join(lines) + "\n"


def main():
    OUT.write_text(build())
    n = sum(1 for _ in OUT.read_text().splitlines())
    print(f"[topology] wrote {n} device-level edges -> {OUT}")


if __name__ == "__main__":
    main()
