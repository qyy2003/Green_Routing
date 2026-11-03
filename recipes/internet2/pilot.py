"""Pilot: is Internet2's public telemetry a usable second network? (green_routing.dataset.internet2)

A. Old network, one week (2019-10-07 -> 10-14, hourly). Route every tunnel (LSP) of the full
   mesh over the backbone circuits on its path and compare with the circuits' own interface
   counters. If the tunnels explain the circuits, they are a measured traffic matrix.
B. Current network, one month (2026-10-01 -> 11-01, hourly). Core routers' ports by label
   (backbone circuits, members and peers), circuits seen at both ends, power per PoP.

    python recipes/internet2/pilot.py download [a|b]     fetch (cached; rate-limited)
    python recipes/internet2/pilot.py analyse  [a|b]     tables + summary.json

Output: $GR_ARTIFACTS_DIR/internet2/pilot/ (raw answers cached in ../cache/).
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from green_routing.dataset import internet2 as i2
from green_routing.paths import ARTIFACTS_DIR

ROOT = ARTIFACTS_DIR / "internet2"
OUT = ROOT / "pilot"
A0, A1 = "2019-10-07", "2019-10-14"
B0, B1 = "2026-10-01", "2026-11-01"
CIRCUIT = re.compile(r"\bI2-([A-Z0-9]+)-([A-Z0-9]+)-(VLAN|LAG|100GE|400GE|10GE)-(\d+)\b")
ASN = re.compile(r"~\s*AS(\d+)")
TAG = re.compile(r"^\s*\[([^\]]+)\]")


def log(*a):
    print(*a, flush=True)


def client():
    return i2.Client(ROOT / "cache", pause=1.0)


# ---------------------------------------------------------------- A: old network, LSP mesh
def download_a(c):
    routers = sorted(c.meta_values("lsp", "from"))
    log(f"[a] {len(routers)} LSP head-end routers")
    L, I = [], []
    for k, r in enumerate(routers, 1):
        L.append(i2.lsps(c, r, A0, A1))
        I.append(i2.interfaces(c, r, A0, A1))
        log(f"[a] {k}/{len(routers)} {r}: {L[-1].name.nunique()} tunnels, {I[-1].intf.nunique()} ports "
            f"({c.live_requests} live requests)")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(L).to_csv(OUT / "a_lsp.csv.gz", index=False)
    pd.concat(I).to_csv(OUT / "a_interfaces.csv.gz", index=False)


def analyse_a():
    L = pd.read_csv(OUT / "a_lsp.csv.gz", parse_dates=["time"])
    I = pd.read_csv(OUT / "a_interfaces.csv.gz", parse_dates=["time"])
    # One series per (tunnel, circuit on its path), each carrying the tunnel's bits/s.
    spread = L.groupby(["name", "time"]).bps.agg(lambda s: s.max() - s.min())
    tun = L.groupby(["name", "time"]).bps.max().rename("bps").reset_index()
    paths = L.groupby("name").path.apply(lambda s: sorted(set(s.dropna())))
    # Measured: backbone ports labelled with a VLAN circuit id (the layer the tunnels name).
    bb = I[I.description.fillna("").str.contains("BACKBONE", case=False)].copy()
    bb["circuit"] = bb.description.str.extract(r"\b(I2-[A-Z0-9]+-[A-Z0-9]+-VLAN-\d+)\b", expand=False)
    bb = bb.dropna(subset=["circuit"])
    per_end = bb.groupby(["circuit", "node", "time"])[["input", "output"]].sum().sum(axis=1)
    measured = per_end.groupby(["circuit", "time"]).mean().rename("measured")      # mean of the two ends
    ends = bb.groupby("circuit").node.nunique()
    # Predicted: every tunnel's bits/s on every circuit of its path (both directions summed).
    rows = [(n, p) for n, ps in paths.items() for p in ps]
    use = pd.DataFrame(rows, columns=["name", "circuit"]).merge(tun, on="name")
    predicted = use.groupby(["circuit", "time"]).bps.sum().rename("predicted")
    J = pd.concat([measured, predicted], axis=1).fillna({"predicted": 0.0}).dropna(subset=["measured"])
    per = J.groupby("circuit").agg(measured_gbps=("measured", "mean"), predicted_gbps=("predicted", "mean"))
    per[["measured_gbps", "predicted_gbps"]] /= 1e9
    per["ratio"] = per.predicted_gbps / per.measured_gbps.replace(0, np.nan)
    per["hourly_wape"] = J.assign(e=(J.predicted - J.measured).abs()).groupby("circuit").e.sum() / \
        J.groupby("circuit").measured.sum().replace(0, np.nan)
    per["ends_seen"] = ends.reindex(per.index)
    per = per.sort_values("measured_gbps", ascending=False)
    per.to_csv(OUT / "a_circuits.csv")
    big = per[per.measured_gbps > 1]
    w = big.measured_gbps / big.measured_gbps.sum()
    s = dict(
        week=f"{A0} -> {A1}", tunnels=int(tun.name.nunique()), head_ends=int(L["from"].nunique()),
        tunnel_pairs=int(L[["from", "to"]].drop_duplicates().shape[0]),
        path_rows_disagree=int((spread > 1).sum()), circuits_on_paths=int(len(set(use.circuit))),
        backbone_circuits_measured=int(per.shape[0]), measured_both_ends=int((per.ends_seen >= 2).sum()),
        circuits_over_1gbps=int(big.shape[0]),
        tunnel_traffic_gbps=float(tun.groupby("time").bps.sum().mean() / 1e9),
        backbone_measured_gbps_sum=float(per.measured_gbps.sum()),
        backbone_predicted_gbps_sum=float(per.predicted_gbps.sum()),
        ratio_total=float(per.predicted_gbps.sum() / per.measured_gbps.sum()),
        ratio_median_over_1gbps=float(big.ratio.median()),
        ratio_weighted_over_1gbps=float((big.ratio * w).sum()),
        hourly_wape_weighted_over_1gbps=float((big.hourly_wape * w).sum()),
    )
    return s, per


# ---------------------------------------------------------------- B: current network
def pops(c):
    names = c.meta_values("power", "node")
    return sorted({p for p in map(i2.pop_of, names) if p}), names


def download_b(c):
    P, power_nodes = pops(c)
    log(f"[b] {len(P)} PoPs, {len(power_nodes)} power systems")
    core = []
    for p in P:
        core += [n for n in c.meta_values("interface", "node", like=p)
                 if i2.pop_of(n) == p and re.match(r"^core\d*\.", n)]
    core = sorted(set(core))
    log(f"[b] {len(core)} core routers: {', '.join(n.split('.net')[0] for n in core)}")
    I = []
    for k, n in enumerate(core, 1):
        I.append(i2.interfaces(c, n, B0, B1))
        log(f"[b] {k}/{len(core)} {n}: {I[-1].intf.nunique()} ports ({c.live_requests} live)")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(I).to_csv(OUT / "b_interfaces.csv.gz", index=False)
    W = []
    for k, n in enumerate(power_nodes, 1):
        W.append(i2.power(c, n, B0, B1))
        if k % 25 == 0:
            log(f"[b] power {k}/{len(power_nodes)} ({c.live_requests} live)")
    pd.concat(W).to_csv(OUT / "b_power.csv.gz", index=False)


def label(desc: str) -> str:
    d = desc or ""
    if "CONSTITUENT" in d.upper():
        return "constituent"
    if d.upper().startswith("BACKBONE"):
        return "backbone"
    m = TAG.match(d)
    if m:
        return f"tag:{m.group(1)}"
    return "other" if d.strip() else "unlabelled"


def analyse_b():
    I = pd.read_csv(OUT / "b_interfaces.csv.gz", parse_dates=["time"])
    W = pd.read_csv(OUT / "b_power.csv.gz", parse_dates=["time"])
    ports = I.groupby(["node", "intf"]).agg(description=("description", "last"), input=("input", "mean"),
                                           output=("output", "mean")).reset_index()
    ports["kind"] = ports.description.map(label)
    ports["sub"] = ports.intf.str.contains(r"\.")
    ports["asn"] = ports.description.str.extract(ASN, expand=False)
    ports["party"] = ports.description.str.extract(r"^\s*\[[^\]]+\]\s*([^~|]+?)\s*(?:~|\||$)", expand=False)
    bbv = ports[(ports.kind == "backbone")].copy()
    bbv["circuit"] = bbv.description.str.extract(r"\b(I2-[A-Z0-9]+-[A-Z0-9]+-VLAN-\d+)\b", expand=False)
    circ = bbv.dropna(subset=["circuit"]).groupby("circuit").agg(ends=("node", "nunique"),
                                                               gbps=("input", lambda s: s.sum() / 1e9))
    edge = ports[ports.kind.str.startswith("tag:")]
    tags = edge.groupby("kind").agg(ports=("intf", "size"), parties=("party", "nunique"),
                                    asns=("asn", "nunique"), in_gbps=("input", lambda s: s.sum() / 1e9),
                                    out_gbps=("output", lambda s: s.sum() / 1e9)).sort_values("in_gbps", ascending=False)
    ports.to_csv(OUT / "b_ports.csv.gz", index=False)
    tags.to_csv(OUT / "b_tags.csv")
    # Power: amps per circuit, summed per system; DC plants taken at 48 V (an assumption).
    amps = W.groupby(["node", "intf"]).amps.mean().groupby("node").sum()
    kind = amps.index.str.split(".").str[0].str.replace(r"\d+$", "", regex=True)
    pw = pd.DataFrame({"amps": amps, "type": kind, "pop": [i2.pop_of(n) for n in amps.index]})
    pw.to_csv(OUT / "b_power_systems.csv")
    dc = pw[pw.type.str.contains("dcp")]
    s = dict(
        month=f"{B0} -> {B1}", core_routers=int(ports.node.nunique()), ports=int(len(ports)),
        kinds={k: int(v) for k, v in ports.kind.value_counts().head(15).items()},
        backbone_vlan_circuits=int(len(circ)), circuits_both_ends=int((circ.ends >= 2).sum()),
        edge_ports_tagged=int(len(edge)), edge_parties=int(edge.party.nunique()), edge_asns=int(edge.asn.nunique()),
        edge_in_gbps=float(edge.input.sum() / 1e9), edge_out_gbps=float(edge.output.sum() / 1e9),
        power_systems=int(len(pw)), power_systems_with_data=int((pw.amps > 0).sum()),
        power_types={k: int(v) for k, v in pw.type.value_counts().items()},
        dc_amps=float(dc.amps.sum()), dc_kw_at_48v=float(dc.amps.sum() * 48 / 1e3),
        dc_pops=int(dc[dc.amps > 0]["pop"].nunique()),
    )
    return s, tags


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    parts = sys.argv[2:] or ["a", "b"]
    OUT.mkdir(parents=True, exist_ok=True)
    if step == "download":
        c = client()
        for p in parts:
            {"a": download_a, "b": download_b}[p](c)
        log(f"done, {c.live_requests} live requests")
    elif step == "analyse":
        summary = {}
        for p in parts:
            s, t = {"a": analyse_a, "b": analyse_b}[p]()
            summary[p] = s
            log(json.dumps(s, indent=1))
            log(t.head(25).round(3).to_string())
        prev = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
        (OUT / "summary.json").write_text(json.dumps({**prev, **summary}, indent=1))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
