"""SWITCHlan's routing check on the test block (periods.TEST): hide a fifth of the busy links, estimate the
traffic matrix from the rest (ECMP over shortest paths weighted by hop count or by inverse capacity), and predict
the hidden links' loads. As recipes/carbon_rates/path_rate.py (its bill window 2026-03-22 -> 07-10), on every
37th hour of the test block, three seeds. Internet2's check is in i2_carbon_common.py (summary.json).

usage: python switch_routing_check.py      -> $PAPER_COMMON/switch_routing_check.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.carbon import path_rate
from green_routing.paths import ARTIFACTS_DIR, reference_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
import periods as P                                                   # noqa: E402

C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
I = pd.read_csv(NET / "interfaces.csv")
doors = pd.read_csv(C / "allocation" / "doors.csv")
traffic = dict(np.load(NET / "traffic.npz"))
sites = pd.read_csv(reference_dir() / "sites.csv")
H = P.hours(P.TEST)
net = path_rate.Network(I, doors, traffic, H)
sample = list(range(0, len(H), 37))
rows = []
for weight in ("hops", "capacity"):
    net.route(weight, sites)
    busy = np.where(net.y[sample].mean(0) > 100)[0]
    for seed in range(3):
        drop = np.random.default_rng(seed).choice(busy, size=len(busy) // 5, replace=False)
        Xd = net.estimate(hours=sample, drop_links=drop)
        ho = np.mean([np.abs(net.A_link[drop] @ Xd[h] - net.y[h][drop]).sum() / net.y[h][drop].sum() for h in sample])
        rows.append(dict(routing=weight, seed=seed, holdout_WAPE=ho, hours=len(sample), links_hidden=len(drop)))
        print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(P.ROOT / "switch_routing_check.csv", index=False)
print(R.groupby("routing").holdout_WAPE.mean().round(4).to_string())
