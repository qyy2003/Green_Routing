"""Build the node-aligned multivariate tensor X[T, N, F] for GNN/ConvLSTM/global models.

Interface throughput is aggregated to the **device (node)** level so the tensor lines up
with the physical node adjacency from topology.py (robust; no interface-name matching).
Per-node features default to in/out Mbps summed over the device's interfaces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, loaders, topology
from .cohort import Coverage, common_grid


def build_node_tensor(
    cohort: list[Coverage],
    start,
    end,
    features=("in_mbps", "out_mbps"),
    fill_limit: int = config.DAILY_STEPS // 24,   # forward-fill up to ~1h of gaps
):
    """Aggregate cohort interfaces to nodes on the common grid.

    Returns dict with:
      X            float32 ndarray [T, N, F]
      timestamps   int64 epoch seconds [T]
      node_index   list[str] of device names (full swi* form) length N
      feature_names list[str] length F
      nan_fraction float, share of missing cells before fill
    """
    grid = common_grid(start, end)
    # group cohort series by device
    by_dev: dict[str, list[Coverage]] = {}
    for c in cohort:
        by_dev.setdefault(c.device, []).append(c)
    node_index = sorted(by_dev)
    F = len(features)
    N = len(node_index)
    T = len(grid)
    X = np.full((T, N, F), np.nan, dtype=np.float32)

    for j, dev in enumerate(node_index):
        acc = pd.DataFrame(0.0, index=grid, columns=list(features))
        counts = pd.Series(0, index=grid)
        for c in by_dev[dev]:
            tp = loaders.interface_throughput(c.path)
            tp = tp.reindex(grid)
            for f in features:
                if f in tp:
                    acc[f] = acc[f].add(tp[f].fillna(0.0), fill_value=0.0)
            counts = counts.add(tp[features[0]].notna().astype(int), fill_value=0)
        # rows with no contributing interface at all -> NaN (genuine gap)
        acc[counts == 0] = np.nan
        X[:, j, :] = acc.to_numpy(dtype=np.float32)

    nan_fraction = float(np.isnan(X).mean())

    if fill_limit and fill_limit > 0:
        # forward/back-fill short gaps per (node, feature) column
        for j in range(N):
            col = pd.DataFrame(X[:, j, :])
            col = col.ffill(limit=fill_limit).bfill(limit=fill_limit)
            X[:, j, :] = col.to_numpy(dtype=np.float32)

    return {
        "X": X,
        "timestamps": (grid.astype("int64") // 10**9).to_numpy(),
        "node_index": node_index,
        "feature_names": list(features),
        "nan_fraction": nan_fraction,
    }


def build_adjacency(node_index: list[str]):
    """Node adjacency aligned to node_index. Returns dict of arrays."""
    links = topology.parse_links()
    A, W_speed, W_cost = topology.node_adjacency(node_index, links)
    in_topo = A.sum(axis=1) > 0
    return {
        "A": A,
        "W_speed": W_speed,
        "W_cost": W_cost,
        "node_index": np.array(node_index),
        "n_isolated": int((~in_topo).sum()),
    }
