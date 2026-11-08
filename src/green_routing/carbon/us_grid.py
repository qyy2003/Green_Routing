"""Hourly US grid factors from EIA-930 (public domain), per balancing authority.

EIA's Hourly Electric Grid Monitor publishes, for every balancing authority (BA), the net
generation by energy source each hour (six-month bulk files ``EIA930_BALANCE_<year>_<half>.csv``
from https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/). Multiplying each source by a
combustion-only emission factor and dividing by the BA's generation gives a production-based,
location-based factor per hour (g CO2/kWh), the closest public analogue of the hourly factors
used for Switch LAN. Imports between BAs are not traced (production-based, not
consumption-based); most Internet2 PoPs sit in large BAs (PJM, MISO, ERCOT, CAISO) where the
difference is small, but it is an approximation.

    load_balance(files)          hourly generation by source per BA (MW), UTC hour starting
    intensity(gen, factors)      hourly g CO2/kWh per BA
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {           # EIA-930 column stem -> our source name
    "Coal": "coal", "Natural Gas": "gas", "Nuclear": "nuclear", "All Petroleum Products": "oil",
    "Hydropower Excluding Pumped Storage": "hydro", "Solar without Integrated Battery Storage": "solar",
    "Solar with Integrated Battery Storage": "solar", "Wind without Integrated Battery Storage": "wind",
    "Wind with Integrated Battery Storage": "wind", "Geothermal": "geothermal",
    "Other Fuel Sources": "other", "Unknown Fuel Sources": "unknown",
}
# files before mid-2024 (e.g. 2021) carry coarser stems; matched exactly, so the newer files'
# "Solar without ..." columns are never counted twice
OLD_SOURCES = {"Hydropower and Pumped Storage": "hydro", "Solar": "solar", "Wind": "wind"}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_balance(files) -> pd.DataFrame:
    """Hourly net generation by source (MW) per BA, from the BALANCE files.

    Index (ba, time) with time = start of the hour, UTC. The "(Adjusted)" columns (EIA's
    cleaned values) are used where present, the raw ones otherwise. Storage is left out
    (it moves energy in time; its charging is counted where the energy was generated)."""
    out = []
    for f in sorted(map(Path, files)):
        d = pd.read_csv(f, low_memory=False)
        t = pd.to_datetime(d["UTC Time at End of Hour"], format="mixed", utc=True) - pd.Timedelta(hours=1)
        g = pd.DataFrame({"ba": d["Balancing Authority"], "time": t})
        for stem, name in SOURCES.items():
            adj = [c for c in d.columns if c.startswith(f"Net Generation (MW) from {stem}") and c.endswith("(Adjusted)")]
            raw = f"Net Generation (MW) from {stem}"
            if not adj and raw not in d.columns:
                continue                    # not in this file (older files use OLD_SOURCES)
            col = _num(d[adj[0]]) if adj else _num(d[raw])
            g[name] = g.get(name, 0) + col.fillna(0) if name in g else col
        for stem, name in OLD_SOURCES.items():
            adj, raw = f"Net Generation (MW) from {stem} (Adjusted)", f"Net Generation (MW) from {stem}"
            if adj in d.columns or raw in d.columns:
                col = _num(d[adj] if adj in d.columns else d[raw])
                g[name] = g[name].fillna(0) + col.fillna(0) if name in g else col
        out.append(g)
    G = pd.concat(out).drop_duplicates(["ba", "time"], keep="last").set_index(["ba", "time"]).sort_index()
    return G.clip(lower=0)


def intensity(gen: pd.DataFrame, factors: pd.Series) -> pd.Series:
    """Hourly production-based intensity (g CO2/kWh) per BA: sum(gen x factor) / sum(gen).

    ``factors``: g CO2/kWh per source name (combustion only). Hours with no reported
    generation are NaN."""
    cols = [c for c in gen.columns if c in factors.index]
    tot = gen[cols].sum(axis=1)
    em = (gen[cols] * factors[cols]).sum(axis=1)
    return (em / tot.where(tot > 0)).rename("g_per_kWh")
