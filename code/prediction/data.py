"""Build and cache a per-link traffic *panel* for prediction experiments.

A panel is a set of interface-throughput series that all live on one shared 5-min
UTC grid over a chosen window, packed into a dense matrix ``values[T, L]`` (Mbps).
This is the object the whole method ladder consumes: univariate baselines/classical
models read one column; the global GBDT reads all columns; a future GNN would add the
node adjacency on top.

We deliberately do **not** call ``dataset.cohort.scan_interfaces`` here — that reads
all ~19 GB every time. Instead we scan a *subset* of devices (or a device allow-list),
keep the series dense enough over the window, and cache the assembled panel to
``artifacts/`` as an ``.npz`` so repeat runs are instant.

The two schema eras and NaN/gap handling are all delegated to
``dataset.loaders.interface_throughput`` (the canonical coalescing loader).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import config, loaders
from dataset.cohort import common_grid

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


@dataclass
class Panel:
    """A dense per-link traffic matrix on a common 5-min UTC grid.

    values      float32 [T, L]  Mbps (genuine gaps stay NaN)
    timestamps  int64   [T]     epoch seconds
    links       list[str] [L]   "device/iface" keys
    devices     list[str] [L]   full swi* device per link (for topology / federated)
    value_col   which throughput metric the values hold
    """

    values: np.ndarray
    timestamps: np.ndarray
    links: list[str]
    devices: list[str]
    value_col: str

    @property
    def T(self) -> int:
        return self.values.shape[0]

    @property
    def L(self) -> int:
        return self.values.shape[1]

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps, unit="s", utc=True)

    def series(self, j: int) -> pd.Series:
        return pd.Series(self.values[:, j], index=self.index, name=self.links[j])

    def coverage(self) -> np.ndarray:
        """Non-NaN fraction per link."""
        return 1.0 - np.isnan(self.values).mean(axis=0)


def _cache_key(start, end, value_col, min_coverage, max_links, devices) -> str:
    payload = json.dumps(
        {
            "start": str(start),
            "end": str(end),
            "value_col": value_col,
            "min_coverage": min_coverage,
            "max_links": max_links,
            "devices": sorted(devices) if devices else None,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def build_panel(
    start,
    end,
    value_col: str = "total_mbps",
    min_coverage: float = 0.9,
    max_links: int | None = 40,
    devices: list[str] | None = None,
    cache: bool = True,
    verbose: bool = True,
) -> Panel:
    """Assemble (and cache) a dense panel of dense interface series over [start, end].

    devices: optional allow-list of short device names (e.g. ["ag1", "be1"]). If None,
             scan all devices in sorted order until ``max_links`` dense links are found.
    """
    key = _cache_key(start, end, value_col, min_coverage, max_links, devices)
    cache_path = ARTIFACTS / f"panel_{key}.npz"
    if cache and cache_path.exists():
        if verbose:
            print(f"[data] loading cached panel {cache_path.name}")
        return load_panel(cache_path)

    grid = common_grid(start, end)
    grid_len = len(grid)
    keep_dev = set(devices) if devices else None

    cols: list[np.ndarray] = []
    links: list[str] = []
    devs: list[str] = []
    n_scanned = 0
    if verbose:
        print(f"[data] scanning interfaces over {start}..{end} "
              f"(grid={grid_len} steps, min_coverage={min_coverage})")

    for dev_short, iface, path in loaders.iter_interface_files():
        if keep_dev is not None and dev_short not in keep_dev:
            continue
        n_scanned += 1
        try:
            s = loaders.interface_throughput(path)[value_col]
        except Exception:
            continue
        s = s.reindex(grid)
        cov = float(s.notna().mean())
        if cov < min_coverage:
            continue
        cols.append(s.to_numpy(dtype=np.float32))
        links.append(f"{dev_short}/{iface}")
        devs.append(config.add_swi(dev_short))
        if verbose and len(links) % 10 == 0:
            print(f"      kept {len(links)} dense links (scanned {n_scanned})")
        if max_links is not None and len(links) >= max_links:
            break

    if not links:
        raise SystemExit(
            "empty panel; lower --min-coverage, widen the window, or add devices"
        )
    values = np.column_stack(cols) if cols else np.empty((grid_len, 0), np.float32)
    # Robust epoch-seconds: DatetimeIndex.astype("int64") returns the index's *native*
    # resolution (datetime64[us] on pandas 2.x here, not ns), so a fixed //10**9 was
    # off by 1000x. Cast through datetime64[s] to get epoch seconds regardless of unit.
    grid_utc = grid.tz_convert("UTC") if grid.tz is not None else grid.tz_localize("UTC")
    timestamps = grid_utc.tz_localize(None).to_numpy().astype("datetime64[s]").astype("int64")
    panel = Panel(
        values=values,
        timestamps=timestamps,
        links=links,
        devices=devs,
        value_col=value_col,
    )
    if verbose:
        print(f"[data] panel ready: {panel.T} steps x {panel.L} links "
              f"(scanned {n_scanned} files)")
    if cache:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        save_panel(panel, cache_path)
        if verbose:
            print(f"[data] cached -> {cache_path}")
    return panel


def save_panel(panel: Panel, path: str | Path) -> None:
    np.savez_compressed(
        path,
        values=panel.values,
        timestamps=panel.timestamps,
        links=np.array(panel.links),
        devices=np.array(panel.devices),
        value_col=np.array(panel.value_col),
    )


def load_panel(path: str | Path) -> Panel:
    d = np.load(path, allow_pickle=True)
    return Panel(
        values=d["values"],
        timestamps=d["timestamps"],
        links=list(d["links"]),
        devices=list(d["devices"]),
        value_col=str(d["value_col"]),
    )
