"""Paths and constants for the SWITCH Cricket dataset processing pipeline.

See ../../dataset.md and ../../benchmark_design.md for the data description and the
split/benchmark design this package implements.
"""
from __future__ import annotations

from pathlib import Path

# --- Roots -------------------------------------------------------------------
# The parsed Cricket time-series (output of network-energy-efficiency-research/
# parsing-rrd/parse_rrd.py). ~45 GB, 5-min-averaged CSVs, UTC epoch timestamps.
SWITCH_ROOT = Path("/media/yuyqin/share/switch")

# The research repo that generated the data (topology, metadata, etc.).
RESEARCH_ROOT = Path(
    "/home/yuyqin/ETH_Master_Study/Green_Routing/network-energy-efficiency-research"
)
TOPOLOGY_TXT = RESEARCH_ROOT / "switch-network-topology" / "switch-network-topology.txt"
METADATA_JSON = RESEARCH_ROOT / "parsing-rrd" / "metadata.json"

# --- Subtrees ----------------------------------------------------------------
SUBTREES = {
    "router-interfaces": SWITCH_ROOT / "router-interfaces",   # per-interface traffic
    "transceiver-monitoring": SWITCH_ROOT / "transceiver-monitoring",
    "routers": SWITCH_ROOT / "routers",                       # per-router cpu/mem/temp
    "router-power": SWITCH_ROOT / "router-power",             # PSU U/I -> power
    "eci": SWITCH_ROOT / "eci" / "apollo",                    # optical shelves incl. power
    "cpu-usage": SWITCH_ROOT / "cpu-usage",
}

# --- Sampling ----------------------------------------------------------------
STEP_SECONDS = 300          # native 5-min sampling
DAILY_STEPS = 24 * 3600 // STEP_SECONDS      # 288
WEEKLY_STEPS = 7 * DAILY_STEPS               # 2016

# Network-meaningful forecast horizons (in steps).
HORIZONS = {
    "5min": 1,
    "15min": 3,
    "30min": 6,
    "1h": 12,
    "day": DAILY_STEPS,
    "week": WEEKLY_STEPS,
}

# --- Interface throughput semantics -----------------------------------------
# Coalesced named columns (see dataset.md section 3).
IF_IN_OCTETS = "ifHCInOctets"     # bytes/sec, 5-min average
IF_OUT_OCTETS = "ifHCOutOctets"
BYTES_TO_MBPS = 8 / 1e6           # bytes/sec -> Mbit/s

# PSU power: raw datasources are U (~12000, assumed mV) and I (assumed mA).
# power[W] = U * I * PSU_POWER_SCALE. Default assumes mV*mA -> W (÷1e6).
# CHANGE if you confirm different SNMP scaling for your devices.
PSU_U = "Cisco_PSU_U"
PSU_I = "Cisco_PSU_I"
PSU_POWER_SCALE = 1e-6

# ECI Apollo optical shelf power (direct reading, no scaling applied here).
ECI_SHELF_POWER = "shelfPowerS0"


def strip_swi(name: str) -> str:
    """Full device name -> router-interfaces short name (swiag1 -> ag1)."""
    return name[3:] if name.startswith("swi") else name


def add_swi(name: str) -> str:
    """router-interfaces short name -> full/topology name (ag1 -> swiag1)."""
    return name if name.startswith("swi") else "swi" + name
