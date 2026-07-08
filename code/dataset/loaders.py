"""Load and clean individual SWITCH Cricket CSVs.

The key job here is **coalescing the two schema eras** documented in dataset.md:
every CSV holds the same metrics twice -- raw ``ds0..dsN`` columns (populated for the
first ~1-3 days only) and named datasource columns (populated afterwards). Column
``dsK`` maps to the K-th named column. We merge them into a single clean series.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from . import config


def read_series(path: str | Path) -> pd.DataFrame:
    """Read one Cricket CSV into a clean, UTC-indexed DataFrame.

    - Coalesces ``dsK`` into the K-th named column (ds era + named era -> one series).
    - Drops the raw ``ds*`` columns and any all-empty trailing named columns.
    - Sorts by time and drops duplicate timestamps.

    Returns a DataFrame indexed by a tz-aware (UTC) DatetimeIndex, named columns only.
    """
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: no timestamp column")

    ds_cols = [c for c in df.columns if c.startswith("ds") and c[2:].isdigit()]
    named_cols = [c for c in df.columns if c != "timestamp" and c not in ds_cols]

    # Coalesce: dsK fills gaps in the K-th named column (the two eras never overlap).
    for k, ds in enumerate(ds_cols):
        if k < len(named_cols):
            named = named_cols[k]
            df[named] = df[named].where(df[named].notna(), df[ds])

    df = df[["timestamp"] + named_cols].copy()
    # Drop trailing named columns that are always empty (e.g. ifHCInUcastPkts).
    df = df.dropna(axis=1, how="all")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = (
        df.dropna(subset=["timestamp"])
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    df.index.name = "time"
    return df


# --- Target extraction -------------------------------------------------------

def interface_throughput(path: str | Path) -> pd.DataFrame:
    """Interface CSV -> DataFrame with in_mbps / out_mbps / total_mbps."""
    df = read_series(path)
    out = pd.DataFrame(index=df.index)
    if config.IF_IN_OCTETS in df:
        out["in_mbps"] = df[config.IF_IN_OCTETS] * config.BYTES_TO_MBPS
    if config.IF_OUT_OCTETS in df:
        out["out_mbps"] = df[config.IF_OUT_OCTETS] * config.BYTES_TO_MBPS
    out["total_mbps"] = out.get("in_mbps", 0) + out.get("out_mbps", 0)
    return out


def psu_power(path: str | Path) -> pd.Series:
    """router-power PSU CSV -> power [W] = U * I * PSU_POWER_SCALE.

    NOTE the scaling assumption in config.PSU_POWER_SCALE (mV*mA -> W). Verify for
    your hardware before using absolute wattage.
    """
    df = read_series(path)
    if config.PSU_U not in df or config.PSU_I not in df:
        raise ValueError(f"{path}: expected {config.PSU_U}/{config.PSU_I} columns")
    return (df[config.PSU_U] * df[config.PSU_I] * config.PSU_POWER_SCALE).rename("power_w")


def eci_shelf_power(path: str | Path) -> pd.Series:
    """ECI Apollo shelf CSV -> shelfPowerS0 series."""
    df = read_series(path)
    if config.ECI_SHELF_POWER not in df:
        raise ValueError(f"{path}: no {config.ECI_SHELF_POWER}")
    return df[config.ECI_SHELF_POWER].rename("shelf_power")


# --- Iterators over a subtree ------------------------------------------------

def iter_interface_files() -> Iterator[tuple[str, str, Path]]:
    """Yield (device_short, iface, path) for every interface CSV."""
    root = config.SUBTREES["router-interfaces"]
    for dev_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for csv in sorted(dev_dir.glob("*.csv")):   # ignores stray *.csv.<hex>
            yield dev_dir.name, csv.stem, csv


def iter_psu_files() -> Iterator[tuple[str, str, Path]]:
    """Yield (device_full, psu, path) for every router-power CSV."""
    root = config.SUBTREES["router-power"]
    for csv in sorted(root.glob("*.csv")):
        dev, _, psu = csv.stem.rpartition("_")      # swiag1_psu0 -> (swiag1, psu0)
        yield dev, psu, csv


def iter_eci_files() -> Iterator[tuple[str, Path]]:
    """Yield (shelf, path) for every ECI Apollo CSV."""
    root = config.SUBTREES["eci"]
    for csv in sorted(root.glob("*.csv")):
        yield csv.stem, csv
