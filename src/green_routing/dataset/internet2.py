"""Internet2 telemetry (GlobalNOC TSDS): a current second network with public data.

Internet2 publishes its network measurements through GlobalNOC's Time Series Data Service,
readable without an account (checked 2026-11-02):

    interface   bits/s in and out per router port, labelled: ``BACKBONE: CHIC-KANS |
                I2-CHIC-KANS-VLAN-184514`` for backbone circuits, ``[RE] UOC via CIC ~ AS160``
                for members and peers. Old Juniper routers ``rtsw.<pop>`` 2016-2022, current
                Cisco core ``core1/core2.<pop>`` from 2021-05; hourly back to the start,
                1-minute values for about the last year.
    lsp         bits/s per MPLS tunnel of the old routers' full mesh, ``I2-<A>-<B>-LSP-<id>``,
                with the backbone circuits it crosses (``path``); 1-minute, 2018-03 -> 2021-09.
                Routed over their paths, they are a measured router-to-router traffic matrix
                if all inter-PoP traffic rides them (what recipes/internet2/pilot.py checks).
    power       amps per circuit of each PoP's DC/AC power systems (``dcp1.<pop>``,
                ``rtrdcp.<pop>``, ``dwdmdcp.<pop>``, ...); hourly from 2023-06.

Queries use TSDS's own language, e.g.
``get node, intf, aggregate(values.input, 3600, average) as v between("10/01/2026 00:00:00 UTC",
"11/01/2026 00:00:00 UTC") by node, intf from interface where node = "core1.chic.net.internet2.edu"``.
Unfiltered queries on ``interface`` time out on the server: always name a node.

    Client(cache_dir).query(q)             results of one query (cached on disk, rate-limited)
    Client.meta_values(mtype, field, like) distinct values of a metadata field
    interfaces / lsps / power              one node's (or router's) series as a long frame
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://services.tsds.bldc.net.internet2.edu/i2/services"
NODE_RE = re.compile(r"^([a-z0-9-]+)\.([a-z0-9]+)\.net\.internet2\.edu$")


def tsds_time(ts) -> str:
    """TSDS's date format, from anything pandas reads as a UTC time."""
    return pd.Timestamp(ts, tz="UTC").strftime("%m/%d/%Y %H:%M:%S UTC")


def pop_of(node: str) -> str | None:
    """``core1.chic.net.internet2.edu`` -> ``chic`` (None for names outside that pattern)."""
    m = NODE_RE.match(node)
    return m.group(2) if m else None


class Client:
    """Cached, rate-limited access to the TSDS query and metadata services.

    Every answer is stored under ``cache_dir`` keyed by the request, so a rerun costs the
    server nothing. ``pause`` seconds separate two live requests (the service is shared).
    """

    def __init__(self, cache_dir: str | Path, pause: float = 1.0, timeout: int = 600, retries: int = 3):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pause, self.timeout, self.retries = pause, timeout, retries
        self._last = 0.0
        self.live_requests = 0

    def _get(self, service: str, params: dict) -> dict:
        url = f"{BASE}/{service}?" + urllib.parse.urlencode(params)
        key = self.cache / f"{hashlib.sha1(url.encode()).hexdigest()}.json.gz"
        if key.exists():
            with gzip.open(key, "rt") as f:
                return json.load(f)
        err = None
        for attempt in range(self.retries):
            time.sleep(max(0.0, self.pause - (time.time() - self._last)))
            self._last = time.time()
            self.live_requests += 1
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as r:
                    d = json.loads(r.read())
            except Exception as e:          # network errors: retry with back-off
                err = e
                time.sleep(10 * (attempt + 1))
                continue
            if d.get("error"):
                text = str(d.get("error_text"))
                err = RuntimeError(f"TSDS: {text} [{params}]")
                if "Error getting meta fields values" in text:
                    d = {"results": []}     # what the metadata service says when nothing matches
                elif "Timed out" not in text:
                    raise err               # a query error will not fix itself
                else:
                    time.sleep(10 * (attempt + 1))
                    continue
            with gzip.open(key, "wt") as f:
                json.dump(d, f)
            return d
        raise RuntimeError(f"TSDS request failed after {self.retries} tries: {err}")

    def query(self, q: str) -> list[dict]:
        return self._get("query.cgi", {"method": "query", "query": q}).get("results") or []

    def meta_values(self, mtype: str, field: str, like: str | None = None, limit: int = 1000) -> list[str]:
        p = {"method": "get_meta_field_values", "measurement_type": mtype, "meta_field": field,
             "limit": limit, "offset": 0}
        if like:
            p[f"{field}_like"] = like
        return [r["value"] for r in self._get("metadata.cgi", p).get("results") or []]


def _long(results: list[dict], keys: list[str], values: list[str]) -> pd.DataFrame:
    """TSDS results -> one row per (series, time), columns keys + values."""
    rows = []
    for r in results:
        meta = {k: r.get(k) for k in keys}
        cols = {v: dict((t, x) for t, x in (r.get(v) or [])) for v in values}
        times = sorted(set().union(*[c.keys() for c in cols.values()]))
        for t in times:
            rows.append({**meta, "time": t, **{v: cols[v].get(t) for v in values}})
    df = pd.DataFrame(rows, columns=keys + ["time"] + values)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def interfaces(c: Client, node: str, start, end, step: int = 3600) -> pd.DataFrame:
    """Every port of one router: description, mean input and output bits/s per ``step``."""
    q = (f"get node, intf, description, aggregate(values.input, {step}, average) as input, "
         f"aggregate(values.output, {step}, average) as output "
         f'between("{tsds_time(start)}","{tsds_time(end)}") by node, intf from interface '
         f'where node = "{node}"')
    return _long(c.query(q), ["node", "intf", "description"], ["input", "output"])


def lsps(c: Client, router: str, start, end, step: int = 3600) -> pd.DataFrame:
    """Every tunnel starting at one old router, one row per backbone circuit on its path
    (TSDS splits a tunnel into one series per ``path`` value, each with the tunnel's bits/s)."""
    q = (f"get from, to, name, path, aggregate(values.bps, {step}, average) as bps "
         f'between("{tsds_time(start)}","{tsds_time(end)}") by from, to, name, path from lsp '
         f'where from = "{router}"')
    return _long(c.query(q), ["from", "to", "name", "path"], ["bps"])


FQDN = ".net.internet2.edu"
BB_RE = re.compile(r"^BACKBONE:\s*([A-Z0-9]+)\.([A-Z0-9]+)-([A-Z0-9]+)\.([A-Z0-9]+)")
CIRCUIT_RE = re.compile(r"\bI2-([A-Z0-9]+)-([A-Z0-9]+)-(VLAN|LAG|[0-9]+GE)-(\d+)\b")
PDP_RE = re.compile(r"^PDP\s+([A-Z0-9]+)-(?:CONN-)?(.+?)(?:-\d+)?(?:\s+\d+X?\d*G)?\s*(?:\||$)")


def classify(desc: str) -> str:
    """Kind of a core-router port from its label: ``backbone_l3`` (the routed VLAN of a
    backbone circuit), ``backbone_phys``, ``door`` (a connector port: ``PDP <PoP>-CONN-<name>``,
    physical or bundle, every service on it), ``member`` (a bundle member, ``[BE80] PDP ...``),
    ``mgmt`` (SMN), or ``other`` (services inside a door, spare ports, ...)."""
    d = (desc or "").strip()
    u = d.upper()
    if u.startswith("BACKBONE"):
        return "backbone_l3" if "-VLAN-" in u else "backbone_phys"
    if re.match(r"^\[(BE|AE)\d+\]", u):
        return "member"
    if u.startswith("PDP"):
        return "mgmt" if re.search(r"-SMN\b|SMN-", u) else "door"
    if u.startswith("SMN"):
        return "mgmt"
    return "other"


def backbone_ends(desc: str) -> tuple[str, str] | None:
    """``BACKBONE: CORE1.CHIC-CORE2.KANS | ...`` -> ("core1.chic.net.internet2.edu", "core2.kans...")."""
    m = BB_RE.match((desc or "").strip().upper())
    if not m:
        return None
    a = f"{m.group(1).lower()}.{m.group(2).lower()}{FQDN}"
    b = f"{m.group(3).lower()}.{m.group(4).lower()}{FQDN}"
    return a, b


def party_of(desc: str) -> str | None:
    """``PDP CHIC-CONN-MERIT-2 400G | I2-...`` -> ``MERIT``; ``PDP ATLA-PAS-100G-2`` -> ``PAS-100G``."""
    m = PDP_RE.match((desc or "").strip().upper())
    return re.sub(r"(-\d+)+$", "", m.group(2).strip()) if m else None


def power(c: Client, node: str, start, end, step: int = 3600) -> pd.DataFrame:
    """Every circuit of one power system: mean amps per ``step``."""
    q = (f"get node, intf, aggregate(values.amps, {step}, average) as amps "
         f'between("{tsds_time(start)}","{tsds_time(end)}") by node, intf from power '
         f'where node = "{node}"')
    return _long(c.query(q), ["node", "intf"], ["amps"])
