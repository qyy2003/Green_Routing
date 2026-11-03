"""Build the Internet2 dataset from the fetched raw files (recipes/internet2/fetch.py).

    python recipes/internet2/make_data.py

Ports are classified from their labels (``internet2.classify``). Backbone links are the routed
VLANs of backbone circuits between two routers; a circuit measured at both ends gives each
direction as the mean of the sender's output and the receiver's input. Doors are connector ports
(``PDP <PoP>-CONN-<name>``, all services on them) plus backbone links to routers outside the
measured core (virtual doors, as for Switch LAN's missing routers). Power: DC plant amps x 48 V.

Outputs ($GR_ARTIFACTS_DIR/internet2/data/):
    ports.csv              every core-router port with kind, ends, party, capacity (October 2026)
    links.csv              directed backbone links between core routers, capacity (bits/s)
    doors.csv              doors: router, port, party, kind (connector / virtual), capacity
    links_hourly.csv.gz    directed link loads, Mbit/s, hourly          (2024-11 -> 2026-11)
    doors_in_hourly.csv.gz, doors_out_hourly.csv.gz   door ingress / egress, Mbit/s, hourly
    power_hourly.csv.gz    W per power system, hourly (DC at 48 V; AC kept in amps)
    mirror_links/          5-minute directed backbone links in the SWITCH export format
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from green_routing.dataset import internet2 as i2
from green_routing.paths import ARTIFACTS_DIR

ROOT = ARTIFACTS_DIR / "internet2"
RAW, OUT = ROOT / "raw", ROOT / "data"
DC_VOLTS = 48.0


def log(*a):
    print(*a, flush=True)


def short(node: str) -> str:
    return node.replace(i2.FQDN, "")


def ports_table() -> pd.DataFrame:
    inv = pd.read_csv(RAW / "inventory.csv.gz")
    inv["kind"] = inv.description.map(i2.classify)
    ends = inv.description.map(i2.backbone_ends)
    inv["end_a"] = ends.map(lambda e: e[0] if e else None)
    inv["end_b"] = ends.map(lambda e: e[1] if e else None)
    inv["remote"] = np.where(inv.end_a == inv.node, inv.end_b, inv.end_a)
    inv["party"] = inv.description.map(i2.party_of)
    inv["circuit"] = inv.description.str.extract(r"\b(I2-[A-Z0-9]+-[A-Z0-9]+-(?:VLAN|LAG|[0-9]+GE)-\d+)\b", expand=False)
    # capacity: the port's own max_bandwidth, else its parent's (a VLAN on a physical port or bundle)
    cap = inv.set_index(["node", "intf"]).max_bandwidth
    parent = inv.intf.str.split(".").str[0]
    inv["capacity_bps"] = inv.max_bandwidth.fillna(pd.Series([cap.get((n, p)) for n, p in zip(inv.node, parent)],
                                                             index=inv.index))
    return inv


def label_rows(H: pd.DataFrame, P: pd.DataFrame, core: set) -> pd.DataFrame:
    """Classify every hourly row by the label it carried at that time (ports are relabelled and
    moved over two years; the October 2026 inventory alone misses half the doors of 2024)."""
    lab = H[["node", "intf", "description"]].drop_duplicates()
    lab["kind"] = lab.description.map(i2.classify)
    ends = lab.description.map(i2.backbone_ends)
    lab["remote"] = [(e[1] if e[0] == n else e[0]) if e else None for e, n in zip(ends, lab.node)]
    lab["party"] = lab.description.map(i2.party_of)
    lab["circuit"] = lab.description.str.extract(r"\b(I2-[A-Z0-9]+-[A-Z0-9]+-(?:VLAN|LAG|[0-9]+GE)-\d+)\b", expand=False)
    lab["role"] = np.select(
        [(lab.kind == "backbone_l3") & lab.remote.isin(core) & (lab.remote != lab.node),
         lab.kind == "door",
         (lab.kind == "backbone_l3") & lab.remote.notna() & ~lab.remote.isin(core)],
        ["link", "door", "virtual_door"], "none")
    lab.loc[lab.role == "virtual_door", "party"] = "router " + lab.loc[lab.role == "virtual_door", "remote"].map(short)
    return H.merge(lab, on=["node", "intf", "description"], how="left")


def directed_links(P: pd.DataFrame, R: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Directed loads (Mbit/s) of backbone links between core routers, from the routed VLAN
    ports at both ends: u->v = mean(u's output, v's input); parallel circuits add."""
    h = R[R.role == "link"]
    a = h.assign(u=h.node, v=h.remote, mbps=h.output / 1e6)[["u", "v", "circuit", "time", "mbps"]]
    b = h.assign(u=h.remote, v=h.node, mbps=h.input / 1e6)[["u", "v", "circuit", "time", "mbps"]]
    per_circ = pd.concat([a, b]).groupby(["u", "v", "circuit", "time"]).mbps.mean()
    loads = per_circ.groupby(["u", "v", "time"]).sum().unstack(["u", "v"])
    bb = P[(P.kind == "backbone_l3")]
    caps = bb.groupby(["node", "remote"]).capacity_bps.sum()
    caps_any = {tuple(sorted(k)): v for k, v in caps.items()}
    L = pd.DataFrame([(u, v, caps.get((u, v), caps_any.get(tuple(sorted((u, v))), np.nan))) for u, v in loads.columns],
                     columns=["u", "v", "capacity_bps"])
    loads.columns = [f"{short(u)}->{short(v)}" for u, v in loads.columns]
    L["name"] = [f"{short(u)}->{short(v)}" for u, v in zip(L.u, L.v)]
    return L, loads


def doors(R: pd.DataFrame):
    """Door ingress / egress (Mbit/s): connector ports, and backbone links to routers outside
    the core (the outside router's traffic enters and leaves there). A door is a port; its
    party and kind are taken from its latest label."""
    h = R[R.role.isin(["door", "virtual_door"])].copy()
    h["door"] = h.node.map(short) + ":" + h.intf
    last = h.sort_values("time").groupby("door").last()
    D = pd.DataFrame({"door": last.index, "node": last.node.to_numpy(), "intf": last.intf.to_numpy(),
                      "description": last.description.to_numpy(), "party": last.party.to_numpy(),
                      "door_kind": np.where(last.role == "virtual_door", "virtual", "connector"),
                      "first": h.groupby("door").time.min().reindex(last.index).to_numpy(),
                      "last": last.time.to_numpy()})
    din = h.pivot_table(index="time", columns="door", values="input", aggfunc="sum") / 1e6
    dout = h.pivot_table(index="time", columns="door", values="output", aggfunc="sum") / 1e6
    return D, din, dout


def write_mirror(F: pd.DataFrame, P: pd.DataFrame, core: set, root):
    """5-minute backbone links in the SWITCH export format: one interface per undirected
    router pair (in = v->u, out = u->v, bytes/s), the routed VLANs of every circuit summed."""
    bb = P[(P.kind == "backbone_l3") & P.remote.isin(core) & (P.remote != P.node)].set_index(["node", "intf"])
    f = F.set_index(["node", "intf"]).join(bb[["remote", "circuit"]], how="inner").reset_index()
    a = f.assign(u=f.node, v=f.remote, bps=f.output)
    b = f.assign(u=f.remote, v=f.node, bps=f.input)
    d = pd.concat([a, b]).groupby(["u", "v", "circuit", "time"]).bps.mean().groupby(["u", "v", "time"]).sum()
    S = d.unstack(["u", "v"]).sort_index()
    grid = pd.date_range(S.index.min(), S.index.max(), freq="5min")
    S = S.reindex(grid)
    ts = (grid.astype("int64") // 10**9).to_numpy()
    root = root / "router-interfaces"
    n = 0
    for u, v in S.columns:
        if (short(u), short(v)) > (short(v), short(u)):
            continue
        back = S.get((v, u), pd.Series(np.nan, index=grid))
        out = root / short(u) / f"to-{short(v)}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": ts, "ifHCInOctets": back.to_numpy() / 8,
                      "ifHCOutOctets": S[(u, v)].to_numpy() / 8}).to_csv(out, index=False)
        n += 1
    return n, S


def power_table(W: pd.DataFrame) -> pd.DataFrame:
    W = W.copy()
    W["system"] = W.node.str.split(".").str[0]
    W["type"] = W.system.str.replace(r"\d+$", "", regex=True)
    W["pop"] = W.node.map(lambda n: i2.pop_of(n) or ("manlan" if "manlan" in n else "wix" if "wix" in n else None))
    W["dc"] = W.type.str.contains("dcp")
    W["watts"] = np.where(W.dc, W.amps * DC_VOLTS, np.nan)
    return W


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = ports_table()
    core = set(P.node)
    P.to_csv(OUT / "ports.csv", index=False)
    log(f"{len(core)} core routers, {len(P)} ports:", P.kind.value_counts().to_dict())
    H = pd.read_csv(RAW / "hourly_ports.csv.gz", parse_dates=["time"])
    R = label_rows(H, P, core)
    log("hourly rows by role:", R.role.value_counts().to_dict())
    L, loads = directed_links(P, R)
    L.to_csv(OUT / "links.csv", index=False)
    loads.to_csv(OUT / "links_hourly.csv.gz")
    D, din, dout = doors(R)
    D.to_csv(OUT / "doors.csv", index=False)
    din.to_csv(OUT / "doors_in_hourly.csv.gz")
    dout.to_csv(OUT / "doors_out_hourly.csv.gz")
    tot_in, tot_out = din.sum(axis=1, min_count=1), dout.sum(axis=1, min_count=1)
    info = dict(core_routers=len(core), directed_links=len(L), doors=len(D),
                connector_doors=int((D.door_kind == "connector").sum()), virtual_doors=int((D.door_kind == "virtual").sum()),
                parties=int(D.party.nunique()), hours=len(din),
                doors_reporting_median=float(din.notna().sum(axis=1).median()),
                total_in_gbps_mean=float(tot_in.mean() / 1e3), total_out_gbps_mean=float(tot_out.mean() / 1e3),
                in_out_ratio=float(tot_in.sum() / tot_out.sum()),
                backbone_gbps_mean=float(loads.sum(axis=1, min_count=1).mean() / 1e3))
    if (RAW / "power_hourly.csv.gz").exists():
        W = power_table(pd.read_csv(RAW / "power_hourly.csv.gz", parse_dates=["time"]))
        W.to_csv(OUT / "power_hourly.csv.gz", index=False)
        info["power_systems"] = int(W.node.nunique())
    if (RAW / "backbone_5min.csv.gz").exists():
        F = pd.read_csv(RAW / "backbone_5min.csv.gz", parse_dates=["time"])
        n, S = write_mirror(F, P, core, OUT / "mirror_links")
        info["mirror_link_files"] = n
        info["mirror_steps"] = len(S)
    (OUT / "info.json").write_text(json.dumps(info, indent=1))
    log(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
