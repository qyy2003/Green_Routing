"""Router tiers and interface roles of SWITCHlan, from matched link ends.

The bucket has no interface descriptions, so roles are inferred:

- **kind**: phys / bundle / sub (VLAN subinterface) / virtual (BVI, BDI, tunnel).
- **tier** of a router: *core* reaches >= 2 other sites over >= 100G links (or is a
  co-located router with a >= 100G link into the core at a backbone site, or has 100G
  ports at a site that ends a >= 100G trunk of SWITCH's OSPF topology); *regional* sits
  at another DWDM site or topology PoP; *access* is everything else (customer premises).
- **role** of an interface: a matched far end makes it a trunk (core-core, regional,
  intra-site, access uplink/ring, cache, L2 service); an active port with no far end is
  an *edge port* (customer, external network, or a router missing from the export).

SWITCH PoP routers are combined core + provider-edge routers, so the backbone/access
split is per port, not per router.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..paths import REPO_ROOT

LINKS_JSON = REPO_ROOT / "web" / "data" / "links.json"   # SWITCH OSPF topology (trunks)


def site_of(dev: str) -> str:
    """Site code of a router: ``ez2`` -> ``ez``, ``ez-ggc1`` -> ``ez``, lab gear -> ``lab``."""
    return "lab" if dev.startswith("lab") else re.sub(r"(-ggc|-mgmt|-test)?\d*$", "", dev)


def iface_kind(i: str) -> str:
    if "." in i:
        return "sub"
    if i.startswith(("bundle-ether", "port-channel")):
        return "bundle"
    if i.startswith(("bvi", "bdi", "vlan", "tunnel")):
        return "virtual"
    return "phys"


def iface_speed(i: str) -> float:
    """Port speed in Gbit/s from the interface name (NaN when the name does not say)."""
    base = i.split(".")[0]
    for p, g in [("fourhundredgige", 400), ("hundredgige", 100), ("fortygige", 40),
                 ("tengigabitethernet", 10), ("tengige", 10), ("gigabitethernet", 1)]:
        if base.startswith(p):
            return g
    return np.nan


def interfaces(stats: pd.DataFrame, partners: dict, panel_links=()) -> pd.DataFrame:
    """iface_stats + kind, speed, parent, matched peer. ``panel_links`` flags the links of
    a forecasting panel (the year benchmark's, by default none)."""
    s = stats.copy()
    s["dev"] = s.link.str.split("/").str[0]
    s["if"] = s.link.str.split("/").str[1]
    s["panel"] = s.link.isin(set(panel_links))
    s["kind"] = s["if"].map(iface_kind)
    s["speed"] = s["if"].map(iface_speed)
    s["parent"] = np.where(s.kind == "sub", s.dev + "/" + s["if"].str.split(".").str[0], "")
    s["tot"] = s.in_mean.fillna(0) + s.out_mean.fillna(0)
    s["peer"] = s.link.map(lambda x: partners.get(x, ("",))[0])
    s["peer_score"] = s.link.map(lambda x: partners.get(x, (0, np.nan))[1])
    s["peer_src"] = s.link.map(lambda x: partners.get(x, ("", "", ""))[2])
    s["peer_dev"] = s.peer.str.split("/").str[0]
    return s


def _known_trunks():
    LJ = json.load(open(LINKS_JSON))["links"]
    known = {}
    for link in LJ:
        for m in link["members"]:
            known[m["dir"] + "/" + m["file"][:-4]] = (link["id"], link["capacity_bps"] / 1e9,
                                                      link["endpoints"])
    return LJ, known


def tiers(s: pd.DataFrame, bundles: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Add site, bundle, known_trunk, peer_site, link_speed and the router tier.
    ``sites`` is the reference sites table (``kind == "both"`` marks DWDM sites)."""
    s = s.set_index("link", drop=False)
    s["site"] = s.dev.map(site_of)
    s["bundle"] = ""
    for b in bundles.itertuples():
        for m in b.members.split():
            s.loc[m, "bundle"] = b.bundle
    for b in bundles.itertuples():                  # a bundle's speed = its members' sum
        s.loc[b.bundle, "speed"] = s.loc[b.members.split(), "speed"].sum()
    LJ, known = _known_trunks()
    s["known_trunk"] = s.link.map(lambda k: known.get(k, ("",))[0])
    s["peer_site"] = s.peer_dev.map(lambda d: site_of(d) if isinstance(d, str) and d else "")

    def sp(a, b):
        return np.nanmax([a, b]) if not (np.isnan(a) and np.isnan(b)) else np.nan
    s["link_speed"] = [sp(r.speed, s.speed.get(r.peer, np.nan)) if r.peer else r.speed
                       for r in s.itertuples()]
    reach: dict[str, set] = {}
    for r in s.itertuples():
        far = None
        if r.peer and r.peer_site != r.site and r.link_speed >= 100:
            far = r.peer_site
        if r.known_trunk and known[r.link][1] >= 100:
            far = [e for e in known[r.link][2] if e.lower() != r.site][0].lower()
        if far:
            reach.setdefault(r.dev, set()).add(far)
    for r in s.itertuples():                       # far ends of >= 100G links count too
        if r.peer and r.peer_site != r.site and r.link_speed >= 100:
            reach.setdefault(r.peer_dev, set()).add(r.site)
    dwdm = set(sites.loc[sites.kind == "both", "code"])
    trunk100 = {e.lower() for link in LJ if link["capacity_bps"] >= 1e11 for e in link["endpoints"]}
    pop10 = {e.lower() for link in LJ for e in link["endpoints"]} - trunk100
    core = {d for d, v in reach.items() if len(v) >= 2}
    for _ in range(2):      # co-located routers with a >= 100G link into the core
        for r in s.itertuples():
            if r.peer and r.link_speed >= 100 and r.peer_dev in core and (r.site in dwdm or r.site in trunk100):
                core.add(r.dev)
    devsite = s.groupby("dev").site.first()
    big = s[s.speed >= 100].groupby("dev").size()
    core |= {d for d, st in devsite.items() if st in trunk100 and d in big.index}
    regional = {d for d, st in devsite.items()
                if d not in core and (st in dwdm or st in trunk100 or st in pop10)}
    s["tier"] = np.where(s.dev.isin(core), "core", np.where(s.dev.isin(regional), "regional", "access"))
    s.loc[s.dev == "ez-ggc1", "tier"] = "cache"     # Google cache at ETH
    s.loc[s.site == "lab", "tier"] = "lab"
    s.loc[s.dev.str.contains("mgmt"), "tier"] = "mgmt"
    return s


def roles(s: pd.DataFrame) -> pd.DataFrame:
    """Add ``role`` (members and subinterfaces inherit their bundle's / parent's)."""
    tier = s.groupby("dev").tier.first().to_dict()

    def base_role(r):
        if r.kind == "virtual":
            return "logical"
        if r.peer:
            if r.peer_site == r.site:
                return "intra-site link"
            if r.kind == "sub" and s.loc[r.peer].kind == "sub":
                return "L2 service (VLAN, both ends)"
            t = {tier.get(r.dev), tier.get(r.peer_dev)}
            if t == {"core"}:
                return "backbone trunk (core-core)"
            if t <= {"core", "regional"}:
                return "regional trunk"
            if "cache" in t:
                return "cache link"
            return "access uplink/ring"
        if r.known_trunk:
            return "backbone trunk (core-core)"
        if r.n_year == 0 or (r.in_mean + r.out_mean) < 0.01:
            return "idle/absent"
        t = tier.get(r.dev)
        if t in ("core", "regional", "cache"):
            return f"edge port on {t} router"
        if t in ("lab", "mgmt"):
            return t
        return "edge port on access router"
    s = s.copy()
    s["role0"] = [base_role(r) for r in s.itertuples()]
    keep = ("L2 service (VLAN, both ends)", "intra-site link", "backbone trunk (core-core)",
            "regional trunk", "access uplink/ring")

    def role(r):
        if r.kind == "sub" and r.role0 not in keep:
            return s.loc[r.parent].role0 if r.parent in s.index else r.role0
        if r.bundle and r.role0.startswith("edge"):
            return s.loc[r.bundle].role0
        return r.role0
    s["role"] = [role(r) for r in s.itertuples()]
    return s


def panel_roles(s: pd.DataFrame) -> pd.DataFrame:
    """The panel's links with a coarse group (A backbone / B edge of core routers /
    C access-regional / D other) and why a series duplicates another one (``redundant``)."""
    tier = s.groupby("dev").tier.first()
    P = s[s.panel].copy()
    inP = set(P.link)

    def redundant(r):
        if r.kind == "sub" and r.parent in inP:
            return "subinterface of a panel port"
        if r.bundle and r.bundle in inP:
            return "member of a panel bundle"
        if r.kind == "virtual":
            return "logical interface"
        if r.peer in inP and r.peer < r.link:
            return "other end already in panel"
        return ""

    def group(r):
        if r.role == "backbone trunk (core-core)":
            return "A backbone"
        if r.role == "intra-site link":
            return "A backbone" if tier.get(r.peer_dev) == "core" and r.tier == "core" else "C access/regional"
        if r.role in ("edge port on core router", "edge port on cache router"):
            return "B edge of core routers"
        if r.role in ("access uplink/ring", "edge port on access router", "regional trunk",
                      "edge port on regional router"):
            return "C access/regional"
        return "D other"
    P["redundant"] = [redundant(r) for r in P.itertuples()]
    P["group"] = [group(r) for r in P.itertuples()]
    return P


COLUMNS = ["link", "dev", "site", "tier", "kind", "speed", "bundle", "parent", "peer", "peer_score",
           "peer_src", "known_trunk", "role", "panel", "in_mean", "out_mean", "first", "last",
           "first_data", "last_data"]


def routers(s: pd.DataFrame) -> pd.DataFrame:
    return s.groupby("dev").agg(site=("site", "first"), tier=("tier", "first"),
                                n_if=("link", "size"), panel_links=("panel", "sum"))
