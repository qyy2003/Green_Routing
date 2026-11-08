"""Greenest-path routing on Internet2, a first cut: what one party saves by sending its traffic on
the path with the lowest carbon rate, and what the backbone looks like when every party does.

Inputs (written by recipes/internet2/carbon.py, test block 2026-07-19 -> 11-01, 2,512 hours):
the hourly router-to-router traffic matrix estimated from the counters (EM + IPF; Internet2
publishes no flow data), every resource's emissions per hour (73 core routers and 107 inter-PoP
spans: measured PoP power x PUE x the balancing authority's hourly g/kWh), today's routing as
modelled there (ECMP shortest paths, inverse-capacity weights), and link capacities.

Path carbon rate (the paper's usage pricing): a resource charges E / U, its emissions over the
traffic that crosses it; a path's rate is the sum over its routers and spans. Two definitions of
"greenest":

- ``usage``: the usage rate above. It depends on everyone's routing: a resource's rate falls as
  traffic joins it and becomes infinite when the last byte leaves (E / 0).
- ``static``: E / capacity, the emissions of a resource per unit of capacity at the hour's grid
  factor; independent of traffic, so greenest paths are plain shortest paths with hourly weights.

Experiments, every hour of the test block:

1. ``unilateral``: one origin-destination flow (a router pair: all members behind one router
   towards one other router) moves to its greenest usage-rate path while every other flow stays
   on today's routing. Its bill is evaluated after the move (its own bytes change the rates).
   Physically nothing changes (power is static: PoP power vs traffic median R^2 0.013), so its
   saving is paid by the flows left behind.
2. ``everyone`` (usage): all flows re-route to their greenest path given the rates posted from the
   current loads; in each round a random quarter of the flows that would save >= 1 % switch
   (simultaneous switching oscillates); repeat until no flow wants to move. This is best-response
   dynamics in a network-design game with proportional cost sharing (Anshelevich et al. 2004).
   Each hour starts again from today's routing, unless ``--carry`` (then from the previous hour's
   end state: path dependence).
3. ``everyone`` (static): all flows on their static greenest path (no feedback, one shot).

For each end state: links and spans left empty, the emissions of the empty spans (what switching
their optics off would save: the only physical gain), the highest link utilisation, hops per byte,
and how many flows pay more or less than today.

Output: $GR_ARTIFACTS_DIR/internet2/green_paths/{hourly_<exp>.csv, spans_<exp>.csv, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra

from green_routing.paths import ARTIFACTS_DIR

ROOT = ARTIFACTS_DIR / "internet2"
CARBON, DATA, OUT = ROOT / "carbon", ROOT / "data", ROOT / "green_paths"
GB = 3600 / 8 / 1000              # GB per Mbit/s-hour
U_MIN = 1e-3                      # Mbit/s: an idle resource's rate is E / 1 kbit/s (as TINY in carbon.py)
SWITCH_GAIN = 0.01                # a flow moves only if it saves at least 1 % (no churn on ties)
SWITCH_P = 0.25                   # share of the willing flows that move in one round
MAX_ROUNDS = 200
EMPTY = 1e-3                      # Mbit/s: a link below this is empty


def log(*a):
    print(*a, flush=True)


class Net:
    """The core graph, resource maps and today's routing (shared by the worker processes)."""

    def __init__(self, setup: str = "", matrix: str = "meas", baseline: str = "own"):
        """``setup`` = "" (the current network, recipes/internet2/carbon.py) or the name of a
        directory under OUT written by a setup script (setup_2021.py: ``tm2021``), with its
        ``matrix`` (meas / est) and today's routing ``baseline`` (own / ecmp)."""
        if setup:
            d = OUT / setup
            z = np.load(d / "setup.npz")
            self.X = z[f"X_{matrix}"]
            if baseline == "own":                  # one matrix per hour (tunnels change path)
                o = np.load(d / "A_own_hourly.npz")
                self.A_hourly = {}
                for key in ("res", "link"):
                    a, shape = o[key], tuple(o[f"shape_{key}"])
                    hh = a[:, 0].astype(int)
                    cut = np.searchsorted(hh, np.arange(len(self.X) + 1))
                    self.A_hourly[key] = [sp.csc_matrix((a[i:j, 3], (a[i:j, 1].astype(int), a[i:j, 2].astype(int))), shape=shape)
                                          for i, j in zip(cut[:-1], cut[1:])]
                self.A_res, self.A_link = self.A_hourly["res"][0], self.A_hourly["link"][0]
            else:
                self.A_res = sp.load_npz(d / f"A_res_{baseline}.npz").tocsc()
                self.A_link = sp.load_npz(d / f"A_link_{baseline}.npz").tocsc()
        else:
            z = np.load(CARBON / "test_setup.npz")
            self.X = z["X_em"]
            self.A_res = sp.load_npz(CARBON / "A_res.npz").tocsc()
            self.A_link = sp.load_npz(CARBON / "A_link.npz").tocsc()
        self.T = pd.to_datetime(z["T"], utc=True)
        self.ends = [str(e) for e in z["ends"]]
        self.res = [str(r) for r in z["res"]]
        self.E = z["E"]
        self.lks = [tuple(s.split("->")) for s in z["lks"]]
        n, nl, nr = len(self.ends), len(self.lks), len(self.res)
        self.n, self.nl, self.nr = n, nl, nr
        ni = {e: i for i, e in enumerate(self.ends)}
        ri = {r: i for i, r in enumerate(self.res)}
        self.eu = np.array([ni[u] for u, _ in self.lks])
        self.ev = np.array([ni[v] for _, v in self.lks])
        self.eidx = -np.ones((n, n), int)
        self.eidx[self.eu, self.ev] = np.arange(nl)
        self.r_of_node = np.array([ri[e] for e in self.ends])
        self.span_of_link = np.array([ri.get("|".join(sorted(k)), -1) for k in self.lks])
        self.is_span = np.zeros(nr, bool)
        self.is_span[self.span_of_link[self.span_of_link >= 0]] = True
        if setup:
            self.cap = z["cap"]                                                    # Mbit/s per direction
        else:
            L = pd.read_csv(DATA / "links.csv")
            c = L.set_index(["u", "v"]).capacity_bps
            c = c.fillna(c.median())
            self.cap = np.array([c[k] for k in self.lks]) / 1e6
        # resource capacity for the static rate: a router's outgoing backbone ports, a span both directions
        rc = np.zeros(nr)
        np.add.at(rc, self.r_of_node[self.eu], self.cap)
        m = self.span_of_link >= 0
        np.add.at(rc, self.span_of_link[m], self.cap[m])
        self.res_cap = rc
        K = n * n
        self.src, self.dst = np.divmod(np.arange(K), n)                            # OD order s-major, as tomography

    def at(self, h: int):
        """Today's routing in hour ``h``: (A_res, A_link)."""
        if getattr(self, "A_hourly", None):
            return self.A_hourly["res"][h], self.A_hourly["link"][h]
        return self.A_res, self.A_link

    def link_matrix(self, w_link: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix((w_link, (self.eu, self.ev)), shape=(self.n, self.n))

    def weights(self, rate: np.ndarray) -> np.ndarray:
        """Cost of crossing directed link j: its span (if inter-PoP) + the router it enters."""
        w = rate[self.r_of_node[self.ev]].copy()
        m = self.span_of_link >= 0
        w[m] += rate[self.span_of_link[m]]
        return w * (1 + 1e-6) + 1e-12 * np.median(w)     # tiny hop term: fewer hops win ties

    def shortest(self, rate: np.ndarray, link_extra=None):
        """Greenest paths for every OD: (cost per Mbit/s-h [K], resource incidence [nr, K],
        link incidence [nl, K]); ``link_extra`` adds a cost per directed link (capacity penalty)."""
        w = self.weights(rate) if link_extra is None else self.weights(rate) + link_extra
        D, P = dijkstra(self.link_matrix(w), directed=True, return_predecessors=True)
        cost = (rate[self.r_of_node][self.src] + D[self.src, self.dst])
        K = self.n * self.n
        # core1.phoe3 has no backbone link in the inventory (2e-7 of the traffic): its pairs
        # get infinite cost (never chosen) and touch only their two routers
        reach = np.isfinite(cost)
        cur = np.where(reach, self.dst, self.src)
        rows_n, cols_n = [self.dst.copy(), self.src[~reach]], [np.arange(K), np.nonzero(~reach)[0]]
        rows_l, cols_l = [], []
        active = cur != self.src
        while active.any():
            k = np.nonzero(active)[0]
            prev = P[self.src[k], cur[k]]
            if (prev < 0).any():
                raise RuntimeError("unreachable pair")
            rows_l.append(self.eidx[prev, cur[k]])
            cols_l.append(k)
            rows_n.append(prev)
            cols_n.append(k)
            cur[k] = prev
            active = cur != self.src
        rn, cn = np.concatenate(rows_n), np.concatenate(cols_n)
        rl, cl = np.concatenate(rows_l), np.concatenate(cols_l)
        B_node = sp.csr_matrix((np.ones(len(rn)), (self.r_of_node[rn], cn)), shape=(self.nr, K))
        B_link = sp.csr_matrix((np.ones(len(rl)), (rl, cl)), shape=(self.nl, K))
        sl = self.span_of_link[rl]
        m = sl >= 0
        B_span = sp.csr_matrix((np.ones(m.sum()), (sl[m], cl[m])), shape=(self.nr, K))
        return cost, (B_node + B_span).tocsc(), B_link.tocsc()


NET: Net | None = None


def _net() -> Net:
    global NET
    if NET is None:
        NET = Net()
    return NET


def usage_rate(E, U):
    return E / np.maximum(U, U_MIN)                    # g per Mbit/s-h


def bridges(net: Net, L) -> int:
    """Router pairs whose loss would split the routers joined by the lit links (L >= EMPTY)."""
    lit = L >= EMPTY
    pairs = sorted({tuple(sorted((u, v))) for u, v in zip(net.eu[lit], net.ev[lit])})

    def comps(skip=None):
        e = [p for p in pairs if p != skip]
        if not e:
            return net.n
        a, b = np.array(e).T
        return connected_components(sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(net.n, net.n)), directed=False)[0]
    c0 = comps()
    return sum(comps(p) > c0 for p in pairs)


def spread(E, U, x):
    """Emissions of idle resources (no bytes to charge), spread flat per Mbit/s-h."""
    return E[U < EMPTY].sum() / x.sum()


def state_metrics(net: Net, h: int, x, U, L, C_link, cost_new, cost_old):
    """Summary of an end state (loads U per resource, L per directed link); costs per OD include
    the flat spread of idle resources' emissions."""
    E = net.E[h]
    tot = E.sum()
    empty_l = L < EMPTY
    span_empty = net.is_span & (U < EMPTY)
    util = L / net.cap
    hops = np.asarray(C_link.sum(axis=0)).ravel()
    xb = x.sum()
    pay = x > 0
    return {
        "links_empty": int(empty_l.sum()),
        "spans_empty": int(span_empty.sum()),
        "span_E_empty_share": float(E[span_empty].sum() / tot),
        "util_max": float(util.max()),
        "links_over_50": int((util > 0.5).sum()),
        "links_over_100": int((util > 1.0).sum()),
        "hops_per_byte": float((hops * x).sum() / xb),
        "bytes_paying_more": float(x[pay & (cost_new > cost_old * 1.001)].sum() / xb),
        "bytes_paying_less": float(x[pay & (cost_new < cost_old * 0.999)].sum() / xb),
        "unallocated_share": float(E[U < EMPTY].sum() / tot),
        "bridges": bridges(net, L),
    }


def run_hour(args):
    h, exp, carry_state, cap_util = args
    net = _net()
    x = net.X[h]
    E = net.E[h]
    A_res, A_link = net.at(h)
    U0 = A_res @ x
    L0 = A_link @ x
    r0 = usage_rate(E, U0)
    c0 = A_res.T @ r0                                      # g per Mbit/s-h, today's path, per OD
    out = {"hour": str(net.T[h]), "total_gbps": float(x.sum() / 1e3), "E_g": float(E.sum())}
    c0e = c0 + spread(E, U0, x)
    base = state_metrics(net, h, x, U0, L0, A_link, c0e, c0e)
    out.update({f"base_{k}": v for k, v in base.items() if not k.startswith("bytes_")})

    if exp == "unilateral":
        cost, B, BL = net.shortest(r0)
        k = np.nonzero((x > 0) & (net.src != net.dst))[0]
        # exact bill after the move: own bytes leave today's path and join the new one
        Ak, Bk = A_res[:, k], B[:, k]
        Uk = U0[:, None] - Ak.multiply(x[k]).toarray() + Bk.multiply(x[k]).toarray()
        c_new = np.asarray(Bk.multiply(E[:, None] / np.maximum(Uk, U_MIN)).sum(axis=0)).ravel()
        c_new = np.minimum(c_new, c0[k])                       # staying is always possible
        save = (c0[k] - c_new) * x[k]                          # g per hour
        bill = c0[k] * x[k]
        better = c_new < c0[k] * (1 - SWITCH_GAIN)
        rel = np.where(c0[k] > 0, 1 - c_new / np.where(c0[k] > 0, c0[k], 1), 0.0)
        w = x[k]
        o = np.argsort(rel)
        cw = np.cumsum(w[o]) / w.sum()
        out.update({
            "uni_bytes_with_greener_path": float(w[better].sum() / w.sum()),
            "uni_saving_bytes_weighted": float(save.sum() / bill.sum()),
            "uni_saving_median_byte": float(np.interp(0.5, cw, rel[o])),
            "uni_saving_p90_byte": float(np.interp(0.9, cw, rel[o])),
            "uni_rate_today_g_per_GB": float(bill.sum() / (w.sum() * GB)),
            "uni_hops_ratio_movers": float((np.asarray(BL[:, k].sum(axis=0)).ravel() * w)[better].sum()
                                           / max((np.asarray(A_link[:, k].sum(axis=0)).ravel() * w)[better].sum(), 1e-9)),
        })
        per_od = pd.DataFrame({"h": h, "k": k, "x": x[k], "c0": c0[k], "c_new": c_new})
        return out, per_od, None

    if exp == "static":
        rate = E / np.maximum(net.res_cap, 1.0)
        _, B, BL = net.shortest(rate)
        U, L = B @ x, BL @ x
        cn = B.T @ usage_rate(E, U) + spread(E, U, x)
        s = state_metrics(net, h, x, U, L, BL, cn, c0e)
        out.update({f"green_{k}": v for k, v in s.items()})
        sig = tuple(np.nonzero(L >= EMPTY)[0])
        return out, None, {"span_empty": (net.is_span & (U < EMPTY)), "used_links": sig}

    # everyone, usage rates, best-response rounds
    rng = np.random.default_rng(h)
    K = len(x)
    if carry_state is not None:
        on_ecmp, Bcur, BLcur = carry_state
    else:
        on_ecmp = np.ones(K, bool)
        Bcur = sp.csc_matrix((net.nr, K))
        BLcur = sp.csc_matrix((net.nl, K))
    live = x > 0
    rounds = 0
    for rounds in range(1, MAX_ROUNDS + 1):
        C = A_res @ sp.diags(on_ecmp.astype(float)) + Bcur
        U = C @ x
        r = usage_rate(E, U)
        c_cur = C.T @ r
        extra = None
        if cap_util:
            # links above the cap cost a penalty far above any rate: flows on them want to leave,
            # and nobody joins them (a flow that would push a link over is caught next round)
            CL = A_link @ sp.diags(on_ecmp.astype(float)) + BLcur
            over = (CL @ x) / net.cap > cap_util
            extra = np.where(over, 1e3 * r.max(), 0.0)
            c_cur = c_cur + CL.T @ extra
        cost, B, BL = net.shortest(r, extra)
        want = live & (cost < c_cur * (1 - SWITCH_GAIN)) & (net.src != net.dst)
        if not want.any():
            break
        move = want & (rng.random(K) < SWITCH_P)
        if not move.any():
            move = want & (np.arange(K) == np.nonzero(want)[0][rng.integers(want.sum())])
        keep = sp.diags((~move).astype(float))
        mv = sp.diags(move.astype(float))
        Bcur = (Bcur @ keep + B @ mv).tocsc()
        BLcur = (BLcur @ keep + BL @ mv).tocsc()
        on_ecmp = on_ecmp & ~move
    C = A_res @ sp.diags(on_ecmp.astype(float)) + Bcur
    CL = A_link @ sp.diags(on_ecmp.astype(float)) + BLcur
    U, L = C @ x, CL @ x
    cn = C.T @ usage_rate(E, U) + spread(E, U, x)
    s = state_metrics(net, h, x, U, L, CL, cn, c0e)
    out.update({f"green_{k}": v for k, v in s.items()})
    out.update({"rounds": rounds, "converged": rounds < MAX_ROUNDS, "flows_still_ecmp": float(x[on_ecmp & live & (net.src != net.dst)].sum() / x[live & (net.src != net.dst)].sum())})
    return out, None, {"span_empty": (net.is_span & (U < EMPTY)), "used_links": tuple(np.nonzero(L >= EMPTY)[0]),
                       "state": (on_ecmp, Bcur, BLcur)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["unilateral", "everyone", "static"])
    ap.add_argument("--stride", type=int, default=1, help="every n-th hour of the test block")
    ap.add_argument("--hours", type=int, default=0, help="only the first n hours (after the stride)")
    ap.add_argument("--carry", action="store_true", help="everyone: start each hour from the previous end state")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap-util", type=float, default=0.0, help="everyone: keep link utilisation under this (0 = capacity-blind)")
    ap.add_argument("--setup", default="", help="inputs from OUT/<setup> (e.g. tm2021 from setup_2021.py); default the current network")
    ap.add_argument("--matrix", default="meas", choices=["meas", "est"], help="with --setup: measured or estimated matrix")
    ap.add_argument("--baseline", default="own", choices=["own", "ecmp"], help="with --setup: today's routing")
    a = ap.parse_args()
    global NET
    outdir = OUT / a.setup if a.setup else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    NET = Net(a.setup, a.matrix, a.baseline)                # forked workers inherit it
    net = NET
    hrs = list(range(0, len(net.T), a.stride))
    if a.hours:
        hrs = hrs[: a.hours]
    suffix = f"_{a.matrix}_{a.baseline}" if a.setup else ""
    tag = a.exp + ("_carry" if a.carry else "") + (f"_cap{int(a.cap_util * 100)}" if a.cap_util else "") + suffix
    log(f"[green] {tag}: {len(hrs)} hours, {net.n} routers, {net.nl} links, {net.is_span.sum()} spans")
    rows, per_od, span_empty, used = [], [], [], []
    if a.carry:
        state = None
        for i, h in enumerate(hrs):
            o, _, extra = run_hour((h, "everyone", state, a.cap_util))
            state = extra["state"]
            rows.append(o)
            span_empty.append(extra["span_empty"])
            used.append(extra["used_links"])
            if i % 24 == 0:
                log(f"  {o['hour']} rounds {o['rounds']} spans empty {o['green_spans_empty']} util max {o['green_util_max']:.2f}")
    else:
        with Pool(a.workers) as pool:
            for i, (o, p, extra) in enumerate(pool.imap(run_hour, [(h, a.exp, None, a.cap_util) for h in hrs], chunksize=4)):
                rows.append(o)
                if p is not None:
                    per_od.append(p)
                if extra is not None:
                    span_empty.append(extra["span_empty"])
                    used.append(extra["used_links"])
                if i % 200 == 0:
                    log(f"  {i}/{len(hrs)} {o['hour']}")
    R = pd.DataFrame(rows)
    R.to_csv(outdir / f"hourly_{tag}.csv", index=False)
    summ = {"exp": tag, "hours": len(R), "stride": a.stride}
    num = R.select_dtypes("number")
    summ["mean"] = num.mean().round(5).to_dict()
    summ["p10"] = num.quantile(0.1).round(5).to_dict()
    summ["p90"] = num.quantile(0.9).round(5).to_dict()
    if per_od:
        P = pd.concat(per_od)
        P["src"], P["dst"] = net.src[P.k], net.dst[P.k]
        # per source router over the block: what all its flows would save, each moving alone
        P["b0"], P["b1"] = P.c0 * P.x, P.c_new * P.x
        g = P.groupby("src")[["b0", "b1", "x"]].sum()
        g["saving"] = 1 - g.b1 / g.b0
        g["router"] = [".".join(net.ends[i].split(".")[:2]) for i in g.index]
        g["traffic_share"] = g.x / g.x.sum()
        g.sort_values("x", ascending=False).to_csv(outdir / f"unilateral_by_router{suffix}.csv")
        od = P.groupby("k")[["b0", "b1", "x"]].sum()
        od["saving"] = 1 - od.b1 / od.b0
        w = od.x / od.x.sum()
        o = np.argsort(od.saving.to_numpy())
        cw = np.cumsum(w.to_numpy()[o])
        summ["block"] = {
            "saving_total_if_each_alone": float(1 - P.b1.sum() / P.b0.sum()),
            "od_saving_p10_p50_p90_bytes": [float(np.interp(q, cw, od.saving.to_numpy()[o])) for q in (0.1, 0.5, 0.9)],
            "od_pairs": int(len(od)),
            "router_saving_p10_p50_p90": g.saving.quantile([0.1, 0.5, 0.9]).round(4).tolist(),
        }
    if span_empty:
        S = np.array(span_empty)
        names = np.array(net.res)
        sp_idx = np.nonzero(net.is_span)[0]
        mean_E = net.E[hrs].mean(axis=0)
        pd.DataFrame({"span": names[sp_idx], "share_hours_empty": S[:, sp_idx].mean(axis=0),
                      "mean_E_g_per_h": mean_E[sp_idx]}).sort_values("share_hours_empty", ascending=False
                      ).to_csv(outdir / f"spans_{tag}.csv", index=False)
        summ["spans_always_empty"] = int((S[:, sp_idx].mean(axis=0) == 1).sum())
        summ["spans_never_empty"] = int((S[:, sp_idx].mean(axis=0) == 0).sum())
        summ["spans_empty_any_hour"] = int((S[:, sp_idx].mean(axis=0) > 0).sum())
        summ["distinct_link_sets"] = len(set(used))
        summ["link_set_changes_between_consecutive_hours"] = int(sum(a_ != b_ for a_, b_ in zip(used, used[1:])))
    js = outdir / "summary.json"
    allsum = json.loads(js.read_text()) if js.exists() else {}
    allsum[tag] = summ
    js.write_text(json.dumps(allsum, indent=1))
    log(json.dumps(summ, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
