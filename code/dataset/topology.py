"""Parse the SWITCHlan physical topology into nodes, edges and adjacency.

Source: switch-network-topology.txt, lines of the form
    swiag2 HundredGigE0/0/0/4 <=> swira4 HundredGigE0/0/0/29 (1e+11) (20000 20000)
    devA   ifaceA               dev B iface B            speed  costv4 costv6
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from . import config

_LINE = re.compile(
    r"^(\S+)\s+(\S+)\s+<=>\s+(\S+)\s+(\S+)\s+\(([^)]+)\)\s+\(([^)]+)\)"
)


@dataclass(frozen=True)
class Link:
    a: str
    a_if: str
    b: str
    b_if: str
    speed_bps: float
    cost_v4: float
    cost_v6: float


def parse_links(path=config.TOPOLOGY_TXT) -> list[Link]:
    links: list[Link] = []
    with open(path) as fh:
        for line in fh:
            m = _LINE.match(line.strip())
            if not m:
                continue
            a, a_if, b, b_if, speed, costs = m.groups()
            cost_parts = costs.split()
            v4 = float(cost_parts[0]) if cost_parts else float("nan")
            v6 = float(cost_parts[1]) if len(cost_parts) > 1 else v4
            links.append(Link(a, a_if, b, b_if, float(speed), v4, v6))
    return links


def nodes(links: list[Link]) -> list[str]:
    """Sorted list of distinct device (node) names, full swi* form."""
    s = set()
    for lk in links:
        s.add(lk.a)
        s.add(lk.b)
    return sorted(s)


def node_adjacency(node_index: list[str], links: list[Link] | None = None):
    """Undirected node-level adjacency for a given ordered node list.

    Returns (A, W_speed, W_cost) as float ndarrays [N, N]:
      A         binary adjacency
      W_speed   summed link speed (bps) between node pairs (parallel links added)
      W_cost    min OSPFv4 cost between node pairs (NaN where no link)
    Links whose endpoints are not both in node_index are ignored.
    """
    if links is None:
        links = parse_links()
    idx = {n: i for i, n in enumerate(node_index)}
    n = len(node_index)
    A = np.zeros((n, n), dtype=np.float32)
    W_speed = np.zeros((n, n), dtype=np.float64)
    W_cost = np.full((n, n), np.nan, dtype=np.float64)
    for lk in links:
        if lk.a not in idx or lk.b not in idx:
            continue
        i, j = idx[lk.a], idx[lk.b]
        A[i, j] = A[j, i] = 1.0
        W_speed[i, j] += lk.speed_bps
        W_speed[j, i] += lk.speed_bps
        c = lk.cost_v4
        W_cost[i, j] = c if np.isnan(W_cost[i, j]) else min(W_cost[i, j], c)
        W_cost[j, i] = W_cost[i, j]
    return A, W_speed, W_cost
