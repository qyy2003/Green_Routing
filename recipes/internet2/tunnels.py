"""Measured traffic matrix: Internet2's old network, whose routers carried all inter-PoP traffic
in a full mesh of MPLS tunnels (LSPs) with per-tunnel counters (March 2021, hourly).

1. Do the tunnels explain the backbone? Route every tunnel over the circuits of its path
   (oriented from its head end) and compare with the circuits' measured directed loads.
2. Is the shortest-path routing model right? The same, routing the PoP matrix over ECMP shortest
   paths (hop weights; capacities of the old ports are not in the export).
3. Counters vs flows (as Abilene before): from counters made by routing the measured matrix,
   estimate it with the gravity prior + EM (``carbon.tomography``) and compare the path bills.
   Then the hard case: the estimate from the *measured* counters under the shortest-path model.

Energy per PoP for the bills: the current network's mean measured power (router side and optical,
``carbon/energy_pops.csv``) and each PoP's mean grid factor; only the bills' relative split matters.

Output: $GR_ARTIFACTS_DIR/internet2/tunnels/{pairs.csv, summary.json}
"""
from __future__ import annotations

import collections
import json
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from green_routing.carbon import path_rate, tomography
from green_routing.paths import ARTIFACTS_DIR

ROOT = ARTIFACTS_DIR / "internet2"
RAW, OUT = ROOT / "raw", ROOT / "tunnels"
PAIR = re.compile(r"I2-([A-Z0-9]+)-([A-Z0-9]+)-VLAN-\d+")


def log(*a):
    print(*a, flush=True)


def pop(router: str) -> str:
    return router.split(".")[1].upper()


def main(tunnels=RAW / "tunnels_2021-03.csv.gz", ports=RAW / "old_ports_2021-03.csv.gz", out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(tunnels, parse_dates=["time"])
    Pt = pd.read_csv(ports, parse_dates=["time"])
    L["s"], L["t"] = L["from"].map(pop), L["to"].map(pop)
    tun = L.groupby(["name", "s", "t", "time"]).bps.max().reset_index()          # one value per tunnel-hour
    hops = L.dropna(subset=["path"]).groupby("name").path.apply(
        lambda s: [tuple(PAIR.match(p).groups()) for p in set(s) if PAIR.match(p)])

    # orient each tunnel's circuits by walking from its head-end PoP
    def walk(name, s, t):
        adj = collections.defaultdict(list)
        for a, b in hops.get(name, []):
            if a != b:
                adj[a].append(b)
                adj[b].append(a)
        prev, st, seen = {}, [s], {s}
        while st:
            u = st.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    prev[v] = u
                    st.append(v)
        if t not in seen:
            return None
        out, v = [], t
        while v != s:
            out.append((prev[v], v))
            v = prev[v]
        return out[::-1]
    meta = tun.drop_duplicates("name")[["name", "s", "t"]]
    route = {r.name: walk(r.name, r.s, r.t) for r in meta.itertuples() if r.s != r.t}
    bytes_by = tun[tun.s != tun.t].groupby("name").bps.sum()
    routed_share = float(bytes_by[[n for n, r in route.items() if r]].sum() / bytes_by.sum())

    # measured directed PoP-pair loads from the old routers' routed backbone VLANs
    bb = Pt[Pt.description.fillna("").str.upper().str.startswith("BACKBONE")].copy()
    bb["pair"] = bb.description.str.extract(r"(I2-[A-Z0-9]+-[A-Z0-9]+-VLAN-\d+)", expand=False)
    bb = bb.dropna(subset=["pair"])
    ab = bb.pair.str.extract(r"I2-([A-Z0-9]+)-([A-Z0-9]+)-VLAN")
    bb["a"], bb["b"], bb["here"] = ab[0], ab[1], bb.node.map(pop)
    bb = bb[(bb.a != bb.b) & ((bb.here == bb.a) | (bb.here == bb.b))]
    bb["there"] = np.where(bb.here == bb.a, bb.b, bb.a)
    out_ = bb.assign(u=bb.here, v=bb.there, bps=bb.output)[["pair", "node", "u", "v", "time", "bps"]]
    in_ = bb.assign(u=bb.there, v=bb.here, bps=bb.input)[["pair", "node", "u", "v", "time", "bps"]]
    m = pd.concat([out_, in_]).groupby(["pair", "u", "v", "time"]).bps.mean().groupby(["u", "v", "time"]).sum()
    meas = m.unstack(["u", "v"])
    times = meas.index

    # 1. tunnels routed over their own paths
    pred = collections.defaultdict(lambda: pd.Series(0.0, index=times))
    tw = tun[tun.s != tun.t].pivot_table(index="time", columns="name", values="bps").reindex(times).fillna(0)
    for n, r in route.items():
        if r and n in tw:
            for e in r:
                pred[e] = pred[e] + tw[n]
    rows = []
    for e in meas.columns:
        y, p = meas[e], pred.get(e, pd.Series(0.0, index=times))
        ok = y.notna()
        rows.append(dict(u=e[0], v=e[1], measured_gbps=y[ok].mean() / 1e9, tunnels_gbps=p[ok].mean() / 1e9,
                         wape=float((p[ok] - y[ok]).abs().sum() / max(y[ok].sum(), 1))))
    PR = pd.DataFrame(rows).sort_values("measured_gbps", ascending=False)
    PR["ratio"] = PR.tunnels_gbps / PR.measured_gbps
    PR.to_csv(out / "pairs.csv", index=False)
    big = PR[PR.measured_gbps > 1]
    w = big.measured_gbps / big.measured_gbps.sum()

    # PoP matrix (hourly), PoP graph from the measured pairs
    pops_ = sorted(set(meas.columns.get_level_values(0)) | set(meas.columns.get_level_values(1)) | set(tun.s) | set(tun.t))
    T = tun[tun.s != tun.t].groupby(["time", "s", "t"]).bps.sum() / 1e6                # Mbit/s
    edges = {e: 1.0 for e in meas.columns}
    ends = sorted(set(tun.s) | set(tun.t))
    lks, A_link, A_node, od = tomography.ecmp_routing(pops_, edges, ends)
    oix = {o: i for i, o in enumerate(od)}
    Xt = np.zeros((len(times), len(od)))
    for (t_, s_, d_), v in T.items():
        if t_ in times and (s_, d_) in oix:
            Xt[times.get_loc(t_), oix[(s_, d_)]] = v
    Ym = np.column_stack([meas[e].to_numpy() / 1e6 if e in meas else np.full(len(times), np.nan) for e in lks])
    ok_h = np.isfinite(Ym).mean(axis=1) > 0.9

    # 2. shortest paths vs measured loads
    Ys = (A_link @ Xt.T).T
    sp_wape = float(np.nansum(np.abs(Ys - Ym)[ok_h]) / np.nansum(Ym[ok_h]))

    # 3. counters vs flows: synthetic counters from the routed truth (the easy case) and the measured ones
    Xo, Yo, Ymo = Xt[ok_h], Ys[ok_h], np.nan_to_num(Ym[ok_h], nan=np.nan)
    n = len(ends)
    ing, egr = Xo.reshape(-1, n, n).sum(2), Xo.reshape(-1, n, n).sum(1)
    X_em = np.array([tomography.em(A_link, Yo[h], ing[h], egr[h]) for h in range(len(Xo))])
    X_gr = np.array([tomography.gravity(ing[h], egr[h]) for h in range(len(Xo))])
    X_real = np.array([tomography.em(A_link, Ymo[h], ing[h], egr[h]) for h in range(len(Xo))])
    # resources: PoPs (router side) and undirected spans, energy from the current network
    f = ROOT / "carbon" / "energy_pops.csv"
    EP = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame({"router_W": [1.0], "optical_W": [1.0]}, index=["x"])
    EP.index = EP.index.str.upper()
    spans = sorted({tuple(sorted(e)) for e in lks})
    Wr = np.array([EP.router_W.get(p, EP.router_W.median()) for p in pops_])
    Wo = {p: EP.optical_W.get(p, EP.optical_W.median()) for p in pops_}
    deg = collections.Counter(x for s in spans for x in s)
    Ws = np.array([Wo[a] / deg[a] + Wo[b] / deg[b] for a, b in spans])
    E_res = np.r_[Wr, Ws]                                                    # relative weights, constant
    Sx = sp.lil_matrix((len(spans), len(lks)))
    for j, e in enumerate(lks):
        Sx[spans.index(tuple(sorted(e))), j] = 1.0
    A_res = sp.vstack([A_node, Sx.tocsr() @ A_link]).tocsr()
    Eo = np.tile(E_res, (len(Xo), 1))
    truth = path_rate.node_bills(A_res, Eo, Xo, n)
    est = path_rate.node_bills(A_res, Eo, X_em, n)
    grav = path_rate.node_bills(A_res, Eo, X_gr, n)
    real = path_rate.node_bills(A_res, Eo, X_real, n)
    V = ing + egr
    flat = V / V.sum(1, keepdims=True) * Eo.sum(1, keepdims=True)

    def wape(b, ref):
        return float(np.abs(b.sum(0) - ref.sum(0)).sum() / ref.sum())

    s = dict(period=f"{L.time.min():%Y-%m-%d %H:%M} -> {L.time.max():%Y-%m-%d %H:%M}", hours=int(ok_h.sum()), tunnels=int(tun.name.nunique()),
             head_ends=int(L["from"].nunique()), pops=len(ends), pop_pairs_measured=int(meas.shape[1]),
             tunnel_bytes_with_a_connected_path=routed_share,
             tunnels_total_gbps=float(tw.sum(axis=1).mean() / 1e9),
             own_paths_total_ratio=float(PR.tunnels_gbps.sum() / PR.measured_gbps.sum()),
             own_paths_median_ratio_over_1gbps=float(big.ratio.median()),
             own_paths_ratio_p10_p90=[float(big.ratio.quantile(.1)), float(big.ratio.quantile(.9))],
             own_paths_weighted_hourly_wape=float((big.wape * w).sum()),
             shortest_paths_hourly_wape=sp_wape,
             counters_vs_flows_path_bill_WAPE=wape(est, truth),
             gravity_vs_flows_path_bill_WAPE=wape(grav, truth),
             measured_counters_shortest_paths_vs_flows_WAPE=wape(real, truth),
             flat_vs_path_WAPE=wape(flat, truth), moved_share_M=wape(flat, truth) / 2,
             tm_rel_l2_median_em=float(np.median(np.linalg.norm(X_em - Xo, axis=1) / np.linalg.norm(Xo, axis=1))),
             tm_rel_l2_median_gravity=float(np.median(np.linalg.norm(X_gr - Xo, axis=1) / np.linalg.norm(Xo, axis=1))))
    json.dump(s, open(out / "summary.json", "w"), indent=1)
    log(json.dumps(s, indent=1))
    log(PR.head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
