"""Load and clean individual SWITCH Cricket CSVs.

The key job here is **coalescing the two schema eras** documented in dataset.md:
every CSV holds the same metrics twice -- raw ``ds0..dsN`` columns (populated for the
first ~1-3 days only) and named datasource columns (populated afterwards). Column
``dsK`` maps to the K-th named column. We merge them into a single clean series.

Every read goes through ``dataset.storage``, so a "path" here is a URI: an
``s3://`` object (the default source) or a local file. Callers pass these straight
back in -- e.g. the paths recorded in ``cohort.csv`` -- and both forms keep working.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from . import config, storage
from .storage import ObjectInfo


def read_series(path: str | Path) -> pd.DataFrame:
    """Read one Cricket CSV into a clean, UTC-indexed DataFrame.

    ``path`` may be a local path or an ``s3://`` URI.

    - Coalesces ``dsK`` into the K-th named column (ds era + named era -> one series).
    - Drops the raw ``ds*`` columns and any all-empty trailing named columns.
    - Sorts by time and drops duplicate timestamps.

    Returns a DataFrame indexed by a tz-aware (UTC) DatetimeIndex, named columns only.
    """
    with storage.open_binary(path) as fh:
        df = pd.read_csv(fh)
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
# One listing covers a whole subtree (a single paginated request on S3, cached per
# config.S3_LIST_TTL), so these iterators cost one round-trip rather than one per
# device directory.

def interface_objects() -> list[ObjectInfo]:
    """Every interface CSV under ``router-interfaces/``, with size/ETag metadata."""
    return storage.walk_objects(config.SUBTREES["router-interfaces"], suffix=".csv")


def iter_interface_files() -> Iterator[tuple[str, str, str]]:
    """Yield (device_short, iface, uri) for every interface CSV.

    Keys/paths look like ``router-interfaces/<device>/<iface>.csv``; anything not at
    that depth is skipped, as is the stray ``*.csv.<hex>`` the export leaves behind.
    """
    root = str(config.SUBTREES["router-interfaces"]).rstrip("/")
    for obj in interface_objects():
        if not obj.uri.startswith(root):
            continue
        parts = obj.uri[len(root):].strip("/").split("/")
        if len(parts) != 2:
            continue
        yield parts[0], storage.stem(obj.uri), obj.uri


def iter_psu_files() -> Iterator[tuple[str, str, str]]:
    """Yield (device_full, psu, uri) for every router-power CSV."""
    for obj in storage.walk_objects(config.SUBTREES["router-power"], suffix=".csv"):
        dev, _, psu = storage.stem(obj.uri).rpartition("_")   # swiag1_psu0 -> swiag1, psu0
        yield dev, psu, obj.uri


def iter_eci_files() -> Iterator[tuple[str, str]]:
    """Yield (shelf, uri) for every ECI Apollo CSV."""
    for obj in storage.walk_objects(config.SUBTREES["eci"], suffix=".csv"):
        yield storage.stem(obj.uri), obj.uri


# --- Bulk warm-up ------------------------------------------------------------

def prefetch_interfaces(
    devices: set[str] | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> int:
    """Download interface CSVs into the local cache in parallel; returns the number
    actually fetched (0 when reading a local mirror or with the cache off).

    Anything about to walk the whole subtree -- the coverage scan, the panel builder
    -- should call this first: serialised GETs over ~2 300 objects are round-trip
    bound, and a warm cache makes reruns local-disk fast.

    ``devices``: restrict to these short device names (e.g. ``{"ag1", "be1"}``).
    ``limit``:   only warm the first N objects in iteration order, for callers that
                 stop early. Objects past the limit still read fine, one GET at a time.
    """
    objs = interface_objects()
    if devices is not None:
        root = str(config.SUBTREES["router-interfaces"]).rstrip("/")
        objs = [
            o for o in objs
            if o.uri.startswith(root)
            and o.uri[len(root):].strip("/").split("/")[0] in devices
        ]
    if limit is not None:
        objs = objs[:limit]
    if verbose and objs and config.USING_S3 and config.S3_CACHE:
        print(f"[storage] warming cache for {len(objs)} interface CSVs "
              f"({config.S3_WORKERS} workers)")
    return storage.prefetch(objs, verbose=verbose)
