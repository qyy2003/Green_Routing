"""Inputs for simulate.py from a measured traffic matrix: Internet2's old network, March 2021, whose
routers carried all inter-PoP traffic in a full mesh of MPLS tunnels with per-tunnel counters
(the data of recipes/internet2/tunnels.py).

- Graph: PoPs (the old network's ``rtsw`` routers, one node per PoP) and the PoP pairs whose routed
  backbone VLANs report counters; capacity per direction from the ports' ``max_bandwidth``
  (TSDS metadata, raw/old_backbone_ports_2021-03.csv.gz; VLAN, else the pair's LAG or 100GE port).
- Matrix ``X_meas``: the tunnels' hourly bytes by head-end and tail-end PoP (measured).
  ``X_est``: the same hours estimated from link counters and PoP totals (EM under the ECMP model,
  as on the current network), to tell how much the estimate matters. The counters are the measured
  matrix routed over the tunnels' own paths: the measured backbone counters also carry ~20 % of
  traffic outside the tunnels (layer-2 and multipoint circuits) that the tunnel totals miss, and an
  estimate from inconsistent inputs would test that gap, not the method (on the current network
  doors and links balance to 1-2 %).
- Today's routing, two versions: ``ecmp`` (shortest paths, inverse-capacity or hop weights,
  whichever fits the measured link loads better) and ``own`` (each tunnel's path in each hour from
  the circuits it crossed that hour, where they join its two ends; ECMP for the rest: one matrix
  per hour).
- Emissions per hour: power per PoP from the *current* network (router side and optical,
  carbon/energy_pops.csv; the old network's power is not in the export), PUE 1.5, each PoP's
  balancing authority's hourly g/kWh in March 2021 (EIA-930); amplifier huts (the current
  network's 12.6 kW) over the spans by capacity at the network-mean factor, as in carbon.py.

Output: $GR_ARTIFACTS_DIR/internet2/green_paths/tm2021/{setup.npz, A_{res,link}_ecmp.npz, A_own_hourly.npz, info.json}
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from green_routing.carbon import tomography, us_grid
from green_routing.paths import ARTIFACTS_DIR, reference_dir

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "internet2"))
from tunnels import PAIR, pop  # noqa: E402

ROOT = ARTIFACTS_DIR / "internet2"
RAW, OUT = ROOT / "raw", ROOT / "green_paths" / "tm2021"
PUE = 1.5
TINY = 1e-3                                    # Mbit/s, as carbon.py


def log(*a):
    print(*a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = pd.read_csv(RAW / "tunnels_2021-03.csv.gz", parse_dates=["time"])
    Pt = pd.read_csv(RAW / "old_ports_2021-03.csv.gz", parse_dates=["time"])
    L["s"], L["t"] = L["from"].map(pop), L["to"].map(pop)
    tun = L[L.s != L.t].groupby(["name", "s", "t", "time"]).bps.max().reset_index()

    # measured directed PoP-pair loads (as tunnels.py)
    bb = Pt[Pt.description.fillna("").str.upper().str.startswith("BACKBONE")].copy()
    bb["pair"] = bb.description.str.extract(r"(I2-[A-Z0-9]+-[A-Z0-9]+-VLAN-\d+)", expand=False)
    bb = bb.dropna(subset=["pair"])
    ab = bb.pair.str.extract(r"I2-([A-Z0-9]+)-([A-Z0-9]+)-VLAN")
    bb["a"], bb["b"], bb["here"] = ab[0], ab[1], bb.node.map(pop)
    bb = bb[(bb.a != bb.b) & ((bb.here == bb.a) | (bb.here == bb.b))]
    bb["there"] = np.where(bb.here == bb.a, bb.b, bb.a)
    out_ = bb.assign(u=bb.here, v=bb.there, bps=bb.output)[["pair", "u", "v", "time", "bps"]]
    in_ = bb.assign(u=bb.there, v=bb.here, bps=bb.input)[["pair", "u", "v", "time", "bps"]]
    meas = pd.concat([out_, in_]).groupby(["pair", "u", "v", "time"]).bps.mean().groupby(["u", "v", "time"]).sum().unstack(["u", "v"])
    times = meas.index

    nodes = sorted(set(meas.columns.get_level_values(0)) | set(meas.columns.get_level_values(1)) | set(tun.s) | set(tun.t))
    n = len(nodes)
    ni = {p: i for i, p in enumerate(nodes)}
    lks = sorted(meas.columns)

    # capacity per direction (Mbit/s)
    B = pd.read_csv(RAW / "old_backbone_ports_2021-03.csv.gz")
    m = B.description.str.extract(r"I2-([A-Z0-9]+)-([A-Z0-9]+)-([A-Z0-9]+)-(\d+)")
    B["key"] = [tuple(sorted((a, b))) if isinstance(a, str) else None for a, b in zip(m[0], m[1])]
    B["kind"], B["cid"] = m[2], m[3]
    cap_pair = {}
    for key, g in B.dropna(subset=["key"]).groupby("key"):
        v = g[g.kind == "VLAN"].groupby("cid").max_bandwidth.max().dropna()
        if len(v):
            cap_pair[key] = v.sum()
        elif g[g.kind == "LAG"].max_bandwidth.notna().any():
            cap_pair[key] = g[g.kind == "LAG"].max_bandwidth.max()
        elif g.max_bandwidth.notna().any():
            cap_pair[key] = g.max_bandwidth.max()
    known = [cap_pair[tuple(sorted(e))] for e in lks if tuple(sorted(e)) in cap_pair]
    cap = np.array([cap_pair.get(tuple(sorted(e)), np.median(known)) for e in lks]) / 1e6
    cap_imputed = int(sum(tuple(sorted(e)) not in cap_pair for e in lks))

    # hourly matrix (Mbit/s), measured link loads
    T = tun.groupby(["time", "s", "t"]).bps.sum() / 1e6
    X = np.zeros((len(times), n * n))
    for (t_, s_, d_), v in T.items():
        if t_ in times:
            X[times.get_loc(t_), ni[s_] * n + ni[d_]] = v
    Ym = np.column_stack([meas[e].to_numpy() / 1e6 for e in lks])
    ok = (np.isfinite(Ym).mean(axis=1) > 0.9) & (X.sum(axis=1) > 0)
    X, Ym, times = X[ok], Ym[ok], times[ok]

    # today's routing: ECMP (the weights that fit the measured loads better), and the tunnels' own paths
    fit = {}
    for wname in ("capacity", "hops"):
        w = {e: (1e5 / c if wname == "capacity" else 1.0) for e, c in zip(lks, cap)}
        lk_, Al, An, od = tomography.ecmp_routing(nodes, w, nodes)
        assert lk_ == lks
        Ys = (Al @ X.T).T
        fit[wname] = (float(np.nansum(np.abs(Ys - Ym)) / np.nansum(Ym)), Al.tocsc(), An.tocsc())
    wbest = min(fit, key=lambda k: fit[k][0])
    _, A_link_e, A_node_e = fit[wbest]
    log(f"[2021] ECMP link-load WAPE: capacity {fit['capacity'][0]:.3f}, hops {fit['hops'][0]:.3f} -> {wbest}")

    # the tunnels' own paths, hour by hour: 1,061 of the 1,977 traceable tunnels change path within
    # the month, so the union of a tunnel's circuits is no path. Each tunnel-hour's named circuits
    # (multipoint "I2-MP-MP" circuits name no PoPs) give its path by breadth-first search from the
    # head end (sorted: deterministic); a tunnel-hour whose circuits do not join s to t over the
    # graph's links falls back to ECMP.
    lix = {e: j for j, e in enumerate(lks)}
    th = (L[L.s != L.t].dropna(subset=["path"]).groupby(["name", "s", "t", "time"])
          .path.apply(lambda c: tuple(sorted({PAIR.match(x).groups() for x in c if PAIR.match(x)}))))
    bps = tun.set_index(["name", "s", "t", "time"]).bps
    th = th.reindex(bps.index)

    def bfs(circuits, s, t):
        adj = collections.defaultdict(list)
        for a, b in circuits:
            if a != b:
                adj[a].append(b)
                adj[b].append(a)
        prev, q, seen = {}, collections.deque([s]), {s}
        while q:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    prev[v] = u
                    q.append(v)
        if t not in seen:
            return None
        out, v = [], t
        while v != s:
            out.append((prev[v], v))
            v = prev[v]
        out = out[::-1]
        return out if all(e in lix for e in out) else None

    cache = {}
    tix = {t_: i for i, t_ in enumerate(times)}
    od_tot = collections.defaultdict(float)
    for (name, s_, t_, time), v in bps.items():
        if time in tix and v > 0:
            od_tot[(tix[time], s_, t_)] += v
    hl, rl, cl, vl, hn, rn, cn, vn = ([] for _ in range(8))
    share = np.zeros((len(times), n * n))
    for (name, s_, t_, time), v in bps.items():
        h = tix.get(time)
        if h is None or not v > 0:
            continue
        circ = th.get((name, s_, t_, time))
        key = (circ, s_, t_)
        if key not in cache:
            cache[key] = bfs(circ, s_, t_) if isinstance(circ, tuple) else None
        path = cache[key]
        if not path:
            continue
        k = ni[s_] * n + ni[t_]
        f = v / od_tot[(h, s_, t_)]
        share[h, k] += f
        for e in path:
            hl.append(h); rl.append(lix[e]); cl.append(k); vl.append(f)
        for p_ in [s_] + [e[1] for e in path]:
            hn.append(h); rn.append(ni[p_]); cn.append(k); vn.append(f)
    hl, rl, cl, vl, hn, rn, cn, vn = map(np.array, (hl, rl, cl, vl, hn, rn, cn, vn))
    own_bytes = float((X * share).sum() / X.sum())

    # resources: PoP routers, then undirected spans "A|B"
    spans = sorted({tuple(sorted(e)) for e in lks})
    res = nodes + [f"{a}|{b}" for a, b in spans]
    sidx = {s: len(nodes) + j for j, s in enumerate(spans)}
    S = sp.csr_matrix((np.ones(len(lks)), ([sidx[tuple(sorted(e))] for e in lks], range(len(lks)))),
                      shape=(len(res), len(lks)))
    Rn = sp.csr_matrix((np.ones(n), (range(n), range(n))), shape=(len(res), n))
    A_res_e = (Rn @ A_node_e + S @ A_link_e).tocsc()

    # hourly own-path matrices (traced part + ECMP for the rest), stored as triplets with the hour
    own_l, own_r = [], []
    Ys_own = np.zeros_like(Ym)
    for h in range(len(times)):
        mh, mn = hl == h, hn == h
        rest = sp.diags(1 - share[h])
        Al = (sp.csr_matrix((vl[mh], (rl[mh], cl[mh])), shape=(len(lks), n * n)) + A_link_e @ rest).tocoo()
        An = sp.csr_matrix((vn[mn], (rn[mn], cn[mn])), shape=(n, n * n)) + A_node_e @ rest
        Ar = (Rn @ An + S @ Al).tocoo()
        own_l.append(np.c_[np.full(Al.nnz, h), Al.row, Al.col, Al.data])
        own_r.append(np.c_[np.full(Ar.nnz, h), Ar.row, Ar.col, Ar.data])
        Ys_own[h] = Al.tocsr() @ X[h]
    own_wape = float(np.nansum(np.abs(Ys_own - Ym)) / np.nansum(Ym))
    log(f"[2021] own hourly paths for {own_bytes:.1%} of bytes; link-load WAPE {own_wape:.3f}")


    # emissions (g/h) per resource and hour
    EP = pd.read_csv(ROOT / "carbon" / "energy_pops.csv", index_col=0)
    EP.index = EP.index.str.upper()
    Wr = pd.Series({p: EP.router_W.get(p, EP.router_W.median()) for p in nodes})
    Wo = pd.Series({p: EP.optical_W.get(p, EP.optical_W.median()) for p in nodes})
    power_imputed = sorted(p for p in nodes if p not in EP.index)
    pops = pd.read_csv(HERE.parent / "internet2" / "pops.csv")
    ba = dict(zip(pops["pop"].str.upper(), pops.ba))
    F = pd.read_csv(HERE.parent / "internet2" / "fuel_factors.csv").set_index("source").g_per_kWh
    G = us_grid.load_balance([reference_dir() / "eia930" / "EIA930_BALANCE_2021_Jan_Jun.csv"])
    I = us_grid.intensity(G, F).unstack("ba").reindex(times).interpolate(limit=6, limit_direction="both")
    g = pd.DataFrame({p: I[ba[p]] if isinstance(ba.get(p), str) and ba[p] in I else np.nan for p in nodes}, index=times)
    no_grid = sorted(p for p in nodes if g[p].isna().all())
    mean_g = g.mean(axis=1)
    g = g.apply(lambda c: c.fillna(mean_g))
    span_cap = np.array([sum(c for e, c in zip(lks, cap) if tuple(sorted(e)) == s) / 2 for s in spans])
    E = np.zeros((len(times), len(res)))
    for i, p in enumerate(nodes):
        E[:, i] = Wr[p] * PUE / 1000 * g[p].to_numpy()
    for p in nodes:
        at = [j for j, s in enumerate(spans) if p in s]
        if not at:
            continue
        c = span_cap[at] / span_cap[at].sum()
        for j, f in zip(at, c):
            E[:, len(nodes) + j] += Wo[p] * f * PUE / 1000 * g[p].to_numpy()
    summ = json.loads((ROOT / "carbon" / "summary.json").read_text())
    huts_W = summ["it_kW_other_sites"] * 1000
    E[:, len(nodes):] += (huts_W * span_cap / span_cap.sum())[None, :] * PUE / 1000 * mean_g.to_numpy()[:, None]

    # the estimate from counters (EM under the ECMP model, margins = measured PoP totals; counters =
    # the measured matrix on its own paths, see the docstring)
    ing, egr = X.reshape(-1, n, n).sum(2), X.reshape(-1, n, n).sum(1)
    Y_own = Ys_own
    X_est = np.array([tomography.em(A_link_e, Y_own[h], ing[h], egr[h]) for h in range(len(X))])
    X_est[X_est < TINY] = 0.0
    tm_err = float(np.median(np.abs(X_est - X).sum(axis=1) / X.sum(axis=1)))

    np.savez_compressed(OUT / "setup.npz", T=times.asi8, ends=np.array(nodes), res=np.array(res),
                        lks=np.array([f"{u}->{v}" for u, v in lks]), cap=cap, X_meas=X, X_est=X_est, E=E, Y=Ym)
    for name, M in [("A_res_ecmp", A_res_e), ("A_link_ecmp", A_link_e)]:
        sp.save_npz(OUT / f"{name}.npz", M.tocsr())
    np.savez_compressed(OUT / "A_own_hourly.npz", link=np.vstack(own_l), res=np.vstack(own_r),
                        shape_link=[len(lks), n * n], shape_res=[len(res), n * n])
    info = dict(period=f"{times.min()} -> {times.max()}", hours=len(times), pops=n, directed_links=len(lks),
                spans=len(spans), mean_gbps=float(X.sum(1).mean() / 1e3), ecmp_weights=wbest,
                ecmp_link_wape={k: v[0] for k, v in fit.items()}, own_path_bytes=own_bytes, own_link_wape=own_wape,
                capacity_imputed_links=cap_imputed, capacity_gbps_counts=pd.Series(cap / 1e3).round().value_counts().to_dict(),
                power_imputed_pops=power_imputed, pops_without_grid=no_grid,
                g_per_kWh_mean_by_pop=g.mean().round(0).to_dict(), E_t_per_h=float(E.sum(1).mean() / 1e6),
                span_E_share=float(E[:, len(nodes):].sum() / E.sum()),
                tm_est_hourly_wape_median=tm_err, util_max_measured=float(np.nanmax(Ym / cap)))
    (OUT / "info.json").write_text(json.dumps(info, indent=1, default=str))
    log(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
