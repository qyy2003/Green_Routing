"""Who pays for which watt: energy pools, doors and parties.

Router power barely depends on traffic, so the bill splits *capacity* and *transport*:

- **Access pool** — what exists for one customer: access routers at a customer's
  premises (whole), and on PoP routers the optics of customer-facing ports plus the
  chassis base in proportion to customer-facing port capacity. Allocated by
  **allocated bandwidth** (port speed is the proxy until SWITCH's contracted rates
  are available).
- **Backbone pool** — the shared transport: the trunk share of PoP routers and every
  ECI optical shelf. Allocated by **bytes**.

A **door** is a port where traffic enters or leaves SWITCHlan: a customer port, a
customer site's access router, a link to an external network (ISP, IXP, CDN), or a
trunk into a router that is missing from the export ("hidden:<router>", which stands for
everything behind it). Customers and external networks are both billed; every byte is
billed half at the door where it enters and half where it leaves, i.e. by in + out
volume at each door. A party is a billable group of doors (one port, one site, one
hidden router, one external partner).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import assumptions as A

IGNORED_ROLES = ("idle/absent", "lab", "mgmt", "logical")
TRUNK_ROLES = ("backbone trunk (core-core)", "regional trunk", "intra-site link", "access uplink/ring")


def port_functions(I: pd.DataFrame) -> pd.DataFrame:
    """Active physical ports with ``func``: access (customer-dedicated capacity),
    "access (door)" (an edge port on a PoP router) or backbone; plus typical optics W.
    Far ends named in reverse DNS make a port a trunk whatever its traffic role."""
    role = I.set_index("link").role
    I = I.copy()
    I["eff_role"] = [role.get(b, r) if isinstance(b, str) and b else r for b, r in zip(I.bundle, I.role)]
    P = I[(I.kind == "phys") & ~I.eff_role.isin(IGNORED_ROLES)].copy()
    trunk_dns = set(I.link[I.dns_peer.notna()]) | set(I.bundle[I.dns_peer.notna() & I.bundle.notna()])

    def func(r):
        if r.link in trunk_dns or (isinstance(r.bundle, str) and r.bundle in trunk_dns):
            return "backbone"
        if r.eff_role in ("edge port on access router", "access uplink/ring", "edge port on regional router"):
            return "access"
        if r.eff_role == "intra-site link":
            return "access" if r.tier == "access" else "backbone"
        if r.eff_role == "edge port on core router":
            return "access (door)"
        return "backbone"       # trunks, regional trunks, cache edge, L2 service
    P["func"] = P.apply(func, axis=1)
    P["trx_W"] = P.speed.map(lambda s: A.TRANSCEIVER_W.get(int(s), A.TRANSCEIVER_W_DEFAULT)
                             if s == s else A.TRANSCEIVER_W_NO_SPEED)
    return P


def device_pools(U: pd.DataFrame, P: pd.DataFrame, tiers: pd.Series) -> pd.DataFrame:
    """Split every router's site energy into access and backbone (MWh).

    ``U``: site_energy table (layer == "router" rows: unit = device, W, MWh, site_MWh);
    ``P``: port_functions; ``tiers``: router tier by short name. Access-tier routers are
    access entirely; on others, optics follow their ports' function and the chassis base
    the access share of port capacity."""
    rows = []
    for r in U[U.layer == "router"].itertuples():
        d = r.unit.removeprefix("swi")
        p = P[P.dev == d]
        tier = tiers.get(d, "access")
        if p.empty:
            acc = r.site_MWh if tier == "access" else 0.0
            rows.append(dict(dev=d, tier=tier, site=r.site, access=acc, backbone=r.site_MWh - acc))
            continue
        trx = min(p.trx_W.sum() / r.W * r.site_MWh, A.OPTICS_MAX_SHARE * r.site_MWh)
        is_acc = p.func.str.startswith("access")
        trx_acc = p.trx_W[is_acc].sum() / p.trx_W.sum() * trx
        cap = p.speed[is_acc].sum() / p.speed.sum() if p.speed.sum() > 0 else 0.0
        acc = r.site_MWh if tier == "access" else trx_acc + (r.site_MWh - trx) * cap
        rows.append(dict(dev=d, tier=tier, site=r.site, access=acc, backbone=r.site_MWh - acc, trx_MWh=trx))
    return pd.DataFrame(rows)


def pools(D: pd.DataFrame, U: pd.DataFrame) -> dict[str, float]:
    opt = float(U.loc[U.layer == "optical", "site_MWh"].sum())
    return {"access_MWh": float(D.access.sum()), "backbone_MWh": float(D.backbone.sum()) + opt,
            "optical_MWh": opt}


def doors(I: pd.DataFrame, D: pd.DataFrame, external: pd.DataFrame,
          hidden_before: str = "2026-07-01") -> pd.DataFrame:
    """Every door with its party and party type.

    ``I``: interfaces (roles + DNS); ``D``: device_pools (for which routers are access
    tier); ``external``: reference table door -> partner, partner_type. A trunk whose
    DNS far end stopped exporting before ``hidden_before`` (or never exported) becomes
    the door ``hidden:<router>``."""
    acc_dev = set(D[D.tier == "access"].dev)
    I = I.copy()
    I["door"] = np.where(I.bundle.notna(), I.bundle, I.link)          # bundle members act as one door
    I["internal"] = I.peer.notna() | I.dns_peer.notna() | I.role.isin(TRUNK_ROLES)
    top = I[I.kind.isin(["phys", "bundle"]) & I.bundle.isna() & (I.role != "idle/absent")].copy()
    top["internal"] = top.link.map(I.groupby("door").internal.any()).fillna(top.internal)
    last = pd.to_datetime(I.groupby("dev")["last"].max(), utc=True)
    I["peer_dev"] = I.dns_peer.str.split("/").str[0]
    cut = pd.Timestamp(hidden_before, tz="UTC")

    def hidden(d):
        return isinstance(d, str) and (d not in last.index or last[d] < cut)
    top["hidden_peer"] = top.link.map(I.set_index("link").peer_dev).where(lambda s: s.map(hidden))
    top.loc[top.hidden_peer.notna(), "internal"] = False
    edge = top[(~top.internal & top.role.str.startswith("edge")) | top.hidden_peer.notna()].copy()
    edge["gbps"] = (edge.in_mean.fillna(0) + edge.out_mean.fillna(0)) / 1e3
    edge["speed"] = edge.speed.fillna(0)
    ext = external.set_index("door")
    edge["party"] = np.where(edge.hidden_peer.notna(), "hidden:" + edge.hidden_peer.astype(str),
                    np.where(edge.dev.isin(acc_dev), "site:" + edge.site,
                    np.where(edge.link.isin(ext.index), "ext:" + edge.link.map(ext.partner).astype(str),
                             "port:" + edge.link)))
    edge["ptype"] = np.where(edge.hidden_peer.isin(list(A.BORDER_ROUTERS)), "external, behind hidden border router",
                    np.where(edge.hidden_peer.notna(), "behind hidden router (customers/transit)",
                    np.where(edge.party.str.startswith("ext:"), edge.link.map(ext.partner_type),
                    np.where(edge.dev.isin(acc_dev), "customer site", "customer port"))))
    return edge[["link", "dev", "site", "party", "ptype", "speed", "gbps", "hidden_peer", "role"]]


def access_split(D: pd.DataFrame, E: pd.DataFrame) -> dict:
    """Access energy (or power) of each party: access routers wholly to their site, PoP
    routers by the port speed of their doors. ``D``: dev, tier, site, access per router;
    ``E``: doors. Access shares of routers without doors come back under key None."""
    acc: dict = {}
    for d in D.itertuples():
        a = d.access
        if d.tier == "access":
            acc["site:" + d.site] = acc.get("site:" + d.site, 0) + a
            continue
        p = E[E.dev == d.dev]
        if p.empty or a == 0:
            if a:
                acc[None] = acc.get(None, 0) + a
            continue
        w = p.groupby("party").speed.sum()
        w = w / w.sum() if w.sum() > 0 else w * 0 + 1 / len(w)
        for k, v in w.items():
            acc[k] = acc.get(k, 0) + a * v
    return acc


def annual_bills(D: pd.DataFrame, E: pd.DataFrame, pool: dict, ci_g_per_kwh: float,
                 sites: pd.DataFrame | None = None) -> pd.DataFrame:
    """Annual bill per party with a flat grid factor (the hourly bill is in ``bills``):
    access MWh by allocated bandwidth, backbone MWh by mean in+out volume."""
    acc = access_split(D, E)
    acc.pop(None, None)
    vol = E.groupby("party").gbps.sum()
    T = pd.DataFrame({"access_MWh": pd.Series(acc), "backbone_MWh": vol / vol.sum() * pool["backbone_MWh"]}).fillna(0)
    T["MWh"] = T.access_MWh + T.backbone_MWh
    T["traffic_Gbps"] = vol.reindex(T.index).fillna(0)
    T["alloc_Gbps"] = E.groupby("party").speed.sum().reindex(T.index)
    T["ptype"] = E.groupby("party").ptype.first().reindex(T.index).fillna("customer site")
    T["tCO2"] = T.MWh * ci_g_per_kwh / 1e3
    T["g_per_GB"] = T.tCO2 * 1e6 / (T.traffic_Gbps / 8 * 3.156e7).replace(0, np.nan)
    if sites is not None:
        host = sites.drop_duplicates("code").set_index("code").host_or_facility.fillna("")

        def key(k):
            return k.split(":")[1].split("/")[0].rstrip("0123456789") if k.startswith("port:") else k.split(":", 1)[1]
        T["host"] = [host.get(key(k), "")[:40] if not k.startswith("ext:") else "" for k in T.index]
    return T.sort_values("MWh", ascending=False)
