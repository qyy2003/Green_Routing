"""Paths, data source and constants for the SWITCH Cricket dataset pipeline.

See docs/data.md in the repository for the data description, the bucket layout and
the setup of credentials.

**Data source.** The time-series are produced daily by the research repo's
``parse_rrd.py`` (a reference copy lives in tools/upstream/), which uploads every
cumulative CSV to SWITCHengines object storage; a local ``~/switch`` tree is only ever a
mirror of that bucket. The uploader's own bucket is ``switchlan-load-timeseries``; the
research project's credentials read the copy in ``network-energy-efficiency-research``,
which is the default here. The package therefore reads S3 by default and takes the filesystem as
an opt-in override, so it runs on a cloud box with nothing but credentials. Object keys
mirror the directory layout exactly (``router-interfaces/ag1/tengige0_0_1_0.csv``), so
``SUBTREES`` is the same shape either way — only the root changes.

Everything below is overridable by environment variable, since a cloud deployment
configures itself through the environment rather than by editing this file:

    GR_DATA_SOURCE   "s3" (default) | "local" | "auto" (local mirror if present)
    GR_LOCAL_ROOT    filesystem mirror root                (default ~/switch)
    GR_S3_BUCKET     bucket holding the CSVs        (network-energy-efficiency-research)
    GR_S3_PREFIX     key prefix inside the bucket          (default: none)
    GR_S3_ENDPOINT   S3 endpoint URL                       (SWITCHengines zhw-b)
    GR_S3_PROFILE    AWS profile name / else AWS_PROFILE   (default: default chain)
    GR_S3_REGION     region                                (eu-central-1)
    GR_S3_ADDRESSING "path" (default) | "virtual"
    GR_S3_CACHE      1 (default) cache objects+listings on disk | 0 no disk writes
    GR_S3_VALIDATE   1 (default) check cached objects against the bucket ETag | 0 trust
    GR_CACHE_DIR     cache location            (default ~/.cache/green-routing/s3)
    GR_S3_LIST_TTL   seconds a cached bucket listing stays valid   (default 43200)
    GR_S3_WORKERS    parallel prefetch/download threads            (default 8)
    GR_RESEARCH_ROOT checkout of network-energy-efficiency-research (topology source;
                     default: a sibling of this repository, used only if present)
"""
from __future__ import annotations

import os
from pathlib import Path

from ..paths import REPO_ROOT


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_flag(name: str, default: bool) -> bool:
    raw = _env(name)
    return default if not raw else raw.lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


# --- Data source -------------------------------------------------------------
DATA_SOURCE = _env("GR_DATA_SOURCE", "s3").lower()      # s3 | local | auto

# Filesystem mirror (used for DATA_SOURCE=local, and by "auto" when it exists).
LOCAL_ROOT = Path(_env("GR_LOCAL_ROOT") or (Path.home() / "switch"))

# SWITCHengines object storage — written daily by parsing-rrd/parse_rrd.py.
S3_BUCKET = _env("GR_S3_BUCKET", "network-energy-efficiency-research")
S3_PREFIX = _env("GR_S3_PREFIX").strip("/")
S3_ENDPOINT = _env("GR_S3_ENDPOINT", "https://zhw-b.s3.cloud.switch.ch")
S3_PROFILE = _env("GR_S3_PROFILE") or _env("AWS_PROFILE") or None
S3_REGION = _env("GR_S3_REGION", "eu-central-1")
S3_ADDRESSING = _env("GR_S3_ADDRESSING", "path")

# Object cache: the coverage scan reads every interface CSV, so downloading each one
# once and reusing it is the difference between a slow first run and slow every run.
S3_CACHE = _env_flag("GR_S3_CACHE", True)
# The bucket is rewritten daily, so a cached object is only good until its ETag moves.
# Listing a subtree already returns every ETag, making validation free for the scans;
# a lone read of an unlisted object pays one HEAD. Turn off to trust the cache blindly.
S3_VALIDATE = _env_flag("GR_S3_VALIDATE", True)
CACHE_DIR = Path(
    _env("GR_CACHE_DIR") or (Path.home() / ".cache" / "green-routing" / "s3")
)
S3_LIST_TTL = _env_int("GR_S3_LIST_TTL", 12 * 3600)     # upstream writes daily
S3_WORKERS = _env_int("GR_S3_WORKERS", 8)

_S3_ROOT = f"s3://{S3_BUCKET}" + (f"/{S3_PREFIX}" if S3_PREFIX else "")

if DATA_SOURCE == "local":
    DATA_ROOT: str = str(LOCAL_ROOT)
elif DATA_SOURCE == "auto":
    DATA_ROOT = str(LOCAL_ROOT) if LOCAL_ROOT.exists() else _S3_ROOT
else:
    DATA_ROOT = _S3_ROOT

USING_S3 = DATA_ROOT.startswith("s3://")

# Kept for callers that still reference the old name; it is the *mirror* root, which
# is only where the data actually lives when USING_S3 is False.
SWITCH_ROOT = LOCAL_ROOT


def describe_source() -> str:
    """One-line summary of where data is being read from (for logs/meta.json)."""
    if not USING_S3:
        return f"local:{DATA_ROOT}"
    profile = S3_PROFILE or "default-chain"
    cache = f"cache={CACHE_DIR}" if S3_CACHE else "cache=off"
    return f"{DATA_ROOT} (endpoint={S3_ENDPOINT}, profile={profile}, {cache})"


# --- Topology / metadata -----------------------------------------------------
# The research repo that generated the data. Not present on a cloud box, hence the
# committed fallback below.
RESEARCH_ROOT = Path(
    _env("GR_RESEARCH_ROOT")
    or (REPO_ROOT.parent / "network-energy-efficiency-research")
).expanduser()
_RESEARCH_TOPOLOGY = RESEARCH_ROOT / "switch-network-topology" / "switch-network-topology.txt"
# Fallback: a device-level topology rebuilt from web/data JSON by
# ``green_routing.dataset.build_topology_txt``, committed alongside this package. The topology is
# *not* in the bucket (parse_rrd.py uploads only time-series), so on a cloud box this
# committed copy is the source of truth — refresh it when the topology changes.
_LOCAL_TOPOLOGY = Path(__file__).resolve().parent / "switch-network-topology.txt"
TOPOLOGY_TXT = str(
    _RESEARCH_TOPOLOGY if _RESEARCH_TOPOLOGY.exists() else _LOCAL_TOPOLOGY
)
METADATA_JSON = str(RESEARCH_ROOT / "parsing-rrd" / "metadata.json")

# --- Subtrees ----------------------------------------------------------------
# Relative layout, identical in the bucket and in the filesystem mirror.
SUBTREE_KEYS = {
    "router-interfaces": "router-interfaces",           # per-interface traffic
    "transceiver-monitoring": "transceiver-monitoring",
    "routers": "routers",                               # per-router cpu/mem/temp
    "router-power": "router-power",                     # PSU U/I -> power
    "eci": "eci/apollo",                                # optical shelves incl. power
    "cpu-usage": "cpu-usage",
}


def _subtree_uri(rel: str) -> str:
    if USING_S3:
        return f"{DATA_ROOT.rstrip('/')}/{rel}"
    return str(Path(DATA_ROOT).joinpath(*rel.split("/")))


SUBTREES = {name: _subtree_uri(rel) for name, rel in SUBTREE_KEYS.items()}

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
    "8h": 96,
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
