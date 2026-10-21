"""Splitting utilities: chronological split with purge, rolling-origin backtest,
and entity (federated / inductive) partitions.

These implement the "universal harness" in benchmark_design.md: one time-ordered
train/val/test split plus rolling origins so every method -- ARIMA through foundation
models -- is scored on identical targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from . import config


@dataclass
class Split:
    train: tuple[int, int]      # [start, end) index positions on the grid
    val: tuple[int, int]
    test: tuple[int, int]
    purge_steps: int
    n_steps: int

    def as_dict(self):
        return {
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "purge_steps": self.purge_steps,
            "n_steps": self.n_steps,
        }


def chronological_split(
    n_steps: int,
    train_frac: float = 0.67,
    val_frac: float = 0.12,
    purge_steps: int | None = None,
    max_lookback: int = config.WEEKLY_STEPS,
    max_horizon: int = config.DAILY_STEPS,
) -> Split:
    """Time-ordered split with an embargo gap between blocks.

    purge_steps defaults to max_lookback + max_horizon so no training/val window can
    straddle a boundary and leak across it.
    """
    if purge_steps is None:
        purge_steps = max_lookback + max_horizon
    train_end = int(n_steps * train_frac)
    val_start = train_end + purge_steps
    val_end = val_start + int(n_steps * val_frac)
    test_start = val_end + purge_steps
    if test_start >= n_steps:
        raise ValueError(
            f"purge_steps={purge_steps} too large for n_steps={n_steps}; "
            "widen the window or shrink lookback/horizon."
        )
    return Split(
        train=(0, train_end),
        val=(val_start, val_end),
        test=(test_start, n_steps),
        purge_steps=purge_steps,
        n_steps=n_steps,
    )


def chronological_split_by_date(
    timestamps: np.ndarray,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
) -> Split:
    """Explicit calendar-boundary split (epoch-second timestamps array)."""
    ts = pd.to_datetime(timestamps, unit="s", utc=True)

    def pos(t):
        return int(ts.searchsorted(pd.Timestamp(t, tz="UTC")))

    tr_end, v_start, v_end, te_start = (
        pos(train_end), pos(val_start), pos(val_end), pos(test_start)
    )
    return Split(
        train=(0, tr_end),
        val=(v_start, v_end),
        test=(te_start, len(ts)),
        purge_steps=v_start - tr_end,
        n_steps=len(ts),
    )


@dataclass
class Origin:
    origin: int                 # index of last observed step (inclusive)
    input_slice: tuple[int, int]
    target_slice: tuple[int, int]


def rolling_origins(
    test_span: tuple[int, int],
    horizon: int,
    lookback: int,
    stride: int | None = None,
) -> Iterator[Origin]:
    """Yield forecast origins across the test block.

    At each origin o: inputs are [o-lookback, o), targets are [o, o+horizon).
    Every method uses only data <= origin, guaranteeing identical targets.
    """
    stride = stride or horizon
    lo, hi = test_span
    o = lo + lookback
    while o + horizon <= hi:
        yield Origin(
            origin=o,
            input_slice=(o - lookback, o),
            target_slice=(o, o + horizon),
        )
        o += stride


# --- Entity partitions -------------------------------------------------------

def _pop_of(device: str) -> str:
    """PoP / site key from a device name, e.g. swiag1 -> ag, swibf-mgmt1 -> bf."""
    core = config.strip_swi(device).split("-")[0]
    return core.rstrip("0123456789") or core


def federated_partition(node_index: list[str], by: str = "device") -> dict[str, list[int]]:
    """Map client -> node positions. by='device' (one client per node) or 'pop'."""
    clients: dict[str, list[int]] = {}
    for i, dev in enumerate(node_index):
        key = dev if by == "device" else _pop_of(dev)
        clients.setdefault(key, []).append(i)
    return clients


def inductive_node_split(
    node_index: list[str], holdout_frac: float = 0.2, seed: int = 0
) -> dict[str, list[int]]:
    """Deterministically hold out whole nodes for inductive (unseen-node) testing."""
    n = len(node_index)
    order = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    k = int(n * holdout_frac)
    return {"holdout": sorted(order[:k].tolist()), "train": sorted(order[k:].tolist())}
