"""Internet2 hourly history before the I2 recipe's window: 2021-01 -> 2024-12 (the current core
network reports from the first half of 2021). Same query as recipes/internet2/fetch.py `hourly`
(connector and backbone ports, one query per core router and half-year, cached and rate-limited),
same link construction as make_data.py (each row labelled by the description it had at that time).

usage: python i2_history.py fetch      download (resumes from the TSDS cache)
       python i2_history.py build      directed link loads + door loads, overlap check with the
                                       recipe's links_hourly.csv.gz (2024-11)
Writes <CACHE>/i2/hourly_ports_2021.csv.gz, links_hourly_2021.csv.gz, doors_{in,out}_hourly_2021.csv.gz.
Leaves artifacts/internet2/ untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "internet2"))
import fetch as F                      # noqa: E402  (recipes/internet2/fetch.py)
import make_data as M                  # noqa: E402

from green_routing.dataset import internet2 as i2   # noqa: E402
from common import CACHE                              # noqa: E402

A0, A1 = "2021-01-01", "2024-12-01"     # one month of overlap with the recipe's 2024-11 start
OUT = CACHE / "i2"


def fetch():
    c = i2.Client(F.ROOT / "cache", pause=1.0)
    nodes = F.core_nodes(c)
    halves = list(pd.date_range(A0, "2024-07-01", freq="6MS")) + [pd.Timestamp(A1)]
    parts = []
    for k, n in enumerate(nodes, 1):
        for a, b in zip(halves[:-1], halves[1:]):
            q = (f"get node, intf, description, aggregate(values.input, 3600, average) as input, "
                 f"aggregate(values.output, 3600, average) as output "
                 f'between("{i2.tsds_time(a)}","{i2.tsds_time(b)}") by node, intf from interface '
                 f'where node = "{n}" and {F.DOOR_OR_BB}')
            parts.append(i2._long(c.query(q), ["node", "intf", "description"], ["input", "output"]))
        F.log(f"[i2_history] {k}/{len(nodes)} {n} ({c.live_requests} live)")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(OUT / "hourly_ports_2021.csv.gz", index=False)


def build():
    P = M.ports_table()
    core = set(P.node)
    H = pd.read_csv(OUT / "hourly_ports_2021.csv.gz", parse_dates=["time"])
    R = M.label_rows(H, P, core)
    print("rows by role:", R.role.value_counts().to_dict(), flush=True)
    _, loads = M.directed_links(P, R)
    loads.to_csv(OUT / "links_hourly_2021.csv.gz")
    _, din, dout = M.doors(R)
    din.to_csv(OUT / "doors_in_hourly_2021.csv.gz")
    dout.to_csv(OUT / "doors_out_hourly_2021.csv.gz")
    # overlap with the recipe's file (2024-11-01 -> 2024-12-01)
    ref = pd.read_csv(F.ROOT / "data" / "links_hourly.csv.gz", index_col=0, parse_dates=True)
    t0, t1 = pd.Timestamp("2024-11-01", tz="UTC"), pd.Timestamp("2024-12-01", tz="UTC")
    a = loads[(loads.index >= t0) & (loads.index < t1)]
    b = ref[(ref.index >= t0) & (ref.index < t1)]
    cols = [c for c in a.columns if c in b.columns]
    x, y = a[cols].to_numpy(float), b.loc[a.index, cols].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    rel = np.abs(x[ok] - y[ok]) / np.maximum(np.abs(y[ok]), 1.0)
    print(f"overlap 2024-11: {len(cols)} common links, {ok.sum()} values, median rel diff "
          f"{np.median(rel):.2e}, p99 {np.percentile(rel, 99):.2e}", flush=True)
    first = loads.apply(lambda s: s.first_valid_index())
    print("links:", loads.shape[1], "hours:", len(loads), "first data, quartiles:",
          first.sort_values().iloc[[0, len(first) // 4, len(first) // 2, 3 * len(first) // 4]].astype(str).tolist())


if __name__ == "__main__":
    {"fetch": fetch, "build": build}[sys.argv[1]]()
