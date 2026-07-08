"""Coverage scanning and cohort selection.

Coverage across the dataset is highly non-uniform (see dataset.md section 5), so before
building any node-aligned tensor we must select a set of series that are all alive and
dense over a common window, and place them on a shared 5-min grid.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, loaders


@dataclass
class Coverage:
    key: str            # identifier, e.g. "ag1/tengige0_0_1_0"
    device: str         # full swi* device name
    path: str
    first: pd.Timestamp
    last: pd.Timestamp
    n_rows: int
    win_coverage: float  # fraction of the requested window's grid that is non-NaN


def _window_coverage(index: pd.DatetimeIndex, start, end) -> float:
    grid = pd.date_range(start, end, freq=f"{config.STEP_SECONDS}s", tz="UTC")
    present = index.intersection(grid)
    return len(present) / len(grid) if len(grid) else 0.0


def scan_interfaces(start, end, value_col: str = "total_mbps") -> list[Coverage]:
    """Scan every interface series and measure coverage over [start, end]."""
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    out: list[Coverage] = []
    for dev_short, iface, path in loaders.iter_interface_files():
        try:
            s = loaders.interface_throughput(path)[value_col].dropna()
        except Exception:
            continue
        if s.empty:
            continue
        out.append(
            Coverage(
                key=f"{dev_short}/{iface}",
                device=config.add_swi(dev_short),
                path=str(path),
                first=s.index.min(),
                last=s.index.max(),
                n_rows=len(s),
                win_coverage=_window_coverage(s.index, start, end),
            )
        )
    return out


def select_cohort(cov: list[Coverage], min_coverage: float = 0.9) -> list[Coverage]:
    """Keep series dense enough over the window; sorted by coverage desc."""
    kept = [c for c in cov if c.win_coverage >= min_coverage]
    return sorted(kept, key=lambda c: c.win_coverage, reverse=True)


def common_grid(start, end) -> pd.DatetimeIndex:
    """The shared 5-min UTC grid all cohort series are reindexed onto."""
    return pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC"),
        freq=f"{config.STEP_SECONDS}s",
    )


def coverage_frame(cov: list[Coverage]) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in cov])
