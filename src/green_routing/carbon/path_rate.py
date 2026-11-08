"""Path-based carbon rate: a byte pays for the equipment it actually crosses.

Resources are the backbone share of every router and the optical layer, mapped onto the
fibre spans between sites (each site's ECI shelves split over the inter-site spans that
end there, by capacity). Every hour each resource has emissions E (site power x the
zone's grid factor) and a throughput X, hence its own rate E / X (g per GB). A byte from
edge node s to edge node t pays the sum of the rates along its route:

    rate(s, t) = sum_routers share_r(s, t) * E_r / X_r  +  sum_spans share_l(s, t) * E_l / X_l

where share is the fraction of (s, t) traffic crossing the resource (shortest paths,
equal-cost split). Two ways to price a resource:

- **usage**: rate = E / X, its emissions over the bytes that cross it. A backup link that
  carries almost nothing then charges its few bytes for all of its power.
- **capacity**: rate = E / capacity, so a byte pays for the share of the equipment it
  occupies; the idle remainder E * (1 - utilisation) is shared by all bytes, like the
  flat rate. These rates need only counters and routing. Bills also need to know
who sends what to whom, the traffic matrix x_st: from flow data when there is some,
otherwise estimated from the counters (``tomography.tomogravity``). Each resource's
emissions are split over the matrix's use of it, so the bills add up to the emissions
exactly, and each flow's cost is billed half to the doors where it enters and half to the
doors where it leaves (as in the flat plan).

Edge nodes: every exported router with customer or external doors, and every router
missing from the export that our trunks reach (standing for its whole group; it may not
relay traffic between two of our routers, as in the door accounting). Access-tier routers
stay in the access pool, as in the flat plan, so both plans split the same backbone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..network.roles import site_of
from . import assumptions as A, tomography

SKIP_ROLES = ("idle/absent", "lab", "mgmt", "logical")
REF_GBPS = 1000.0            # inverse-capacity link cost = REF_GBPS / speed (OSPF-style)


def _series(traffic: dict, links, key: str, H) -> pd.DataFrame:
    ix = {str(x): i for i, x in enumerate(traffic["links"])}
    T = pd.to_datetime(traffic["hours"], unit="s", utc=True)
    X = pd.DataFrame({x: traffic[key][ix[x]] for x in links if x in ix}, index=T).astype(float)
    return X.interpolate(limit=6, limit_area="inside").reindex(H)


class Network:
    """SWITCHlan as a router graph with hourly link loads, and its doors as edge traffic."""

    def __init__(self, I: pd.DataFrame, doors: pd.DataFrame, traffic: dict, H: pd.DatetimeIndex):
        self.H = H
        top = I[I.kind.isin(["phys", "bundle"]) & I.bundle.isna() & ~I.role.isin(SKIP_ROLES)]
        top = top[~top.dev.str.contains("lab|mgmt")]
        real = doors[doors.hidden_peer.isna()]
        real_links = set(real.link)
        hidden_door = doors.dropna(subset=["hidden_peer"]).set_index("link").hidden_peer
        exported = set(I.dev)
        pair_ports: dict = {}
        for r in top.itertuples():
            if r.link in real_links:
                continue
            far = hidden_door.get(r.link)
            if not isinstance(far, str):
                far = r.peer.split("/")[0] if isinstance(r.peer, str) and r.peer else (
                    r.dns_peer.split("/")[0] if isinstance(r.dns_peer, str) else None)
            if not far or far == r.dev or "lab" in far or "mgmt" in far:
                continue
            pair_ports.setdefault((r.dev, far), []).append(r.link)
        self.hidden = sorted(set(hidden_door.values))
        # directed load u -> v: u's ports towards v (out), or v's ports towards u (in); the
        # larger where both report (a side that misses a port reads low)
        ports = sorted({p for v in pair_ports.values() for p in v})
        hin, hout = _series(traffic, ports, "hin", H), _series(traffic, ports, "hout", H)
        speed = I.set_index("link").speed
        loads, cap = {}, {}
        for a, b in {tuple(sorted(k)) for k in pair_ports}:
            if a in self.hidden and b in self.hidden:
                continue                                    # unobserved: neither end reports
            for u, v in ((a, b), (b, a)):
                sides = []
                if u in exported and (u, v) in pair_ports:
                    sides.append(hout[[p for p in pair_ports[(u, v)] if p in hout]].sum(axis=1, min_count=1))
                if v in exported and (v, u) in pair_ports:
                    sides.append(hin[[p for p in pair_ports[(v, u)] if p in hin]].sum(axis=1, min_count=1))
                if sides:
                    loads[(u, v)] = pd.concat(sides, axis=1).max(axis=1)
            sp_ = np.array([speed.get(p, np.nan) for p in pair_ports.get((a, b), []) + pair_ports.get((b, a), [])], float)
            cap[(a, b)] = cap[(b, a)] = float(np.nanmax(sp_)) if np.isfinite(sp_).any() else 10.0
        # a link that has not reported yet did not exist: 0; inner gaps interpolated
        self.loads = pd.DataFrame(loads).reindex(H).interpolate(limit_area="inside").fillna(0)
        self.cap = cap
        self.nodes = sorted({n for k in loads for n in k})
        # every door's hourly in / out (all doors: the flat plan's key)
        self.din = _series(traffic, doors.link, "hin", H).fillna(0)
        self.dout = _series(traffic, doors.link, "hout", H).fillna(0)
        self.door_party = doors.set_index("link").party
        # edge doors: real doors on routers in the graph, plus one per hidden router
        rd = real[real.dev.isin(self.nodes)]
        self.isolated_doors = sorted(set(real.link) - set(rd.link))
        ed_in, ed_out = self.din[rd.link].copy(), self.dout[rd.link].copy()
        node, party = list(rd.dev), list(rd.party)
        for h in self.hidden:
            into = [k for k in loads if k[0] == h]
            outof = [k for k in loads if k[1] == h]
            if not into and not outof:
                continue
            ed_in["hidden:" + h] = self.loads[into].sum(axis=1) if into else 0.0
            ed_out["hidden:" + h] = self.loads[outof].sum(axis=1) if outof else 0.0
            node.append(h)
            party.append("hidden:" + h)
        # bytes a router sends more than it receives on the ports we see entered through
        # ports the export misses (and vice versa): one "unseen" door per router absorbs the
        # hourly imbalance, so the equations stay consistent; its cost is spread flat
        dev_in = self.din[rd.link].T.groupby(rd.set_index("link").dev.reindex(rd.link).values).sum().T
        dev_out = self.dout[rd.link].T.groupby(rd.set_index("link").dev.reindex(rd.link).values).sum().T
        extra_in, extra_out = {}, {}
        for r in self.nodes:
            if r in self.hidden:
                continue
            into = self.loads[[k for k in loads if k[1] == r]].sum(axis=1)
            outof = self.loads[[k for k in loads if k[0] == r]].sum(axis=1)
            bal = (into + dev_in.get(r, 0.0)) - (outof + dev_out.get(r, 0.0))
            extra_in["unseen:" + r] = (-bal).clip(lower=0)
            extra_out["unseen:" + r] = bal.clip(lower=0)
            node.append(r)
            party.append("unseen:" + r)
        self.ed_in = pd.concat([ed_in, pd.DataFrame(extra_in)], axis=1).fillna(0)
        self.ed_out = pd.concat([ed_out, pd.DataFrame(extra_out)], axis=1).fillna(0)
        self.ed_node, self.ed_party = np.array(node), np.array(party)
        self.ends = sorted(set(node))
        ni = np.array([self.ends.index(n) for n in node])
        G = sp.csr_matrix((np.ones(len(ni)), (ni, np.arange(len(ni)))), shape=(len(self.ends), len(ni)))
        self.ed_to_end = G
        self.ingress = (G @ self.ed_in.to_numpy().T).T          # [hours, ends]
        self.egress = (G @ self.ed_out.to_numpy().T).T
        # capacity (Mbit/s, one direction) of every router: its ports; of a router missing
        # from the export: our ports facing it
        self.router_cap = (top.groupby("dev").speed.sum() * 1e3).to_dict()
        for r in self.nodes:            # routers missing from the export, or without port data
            if r in self.hidden or not self.router_cap.get(r):
                self.router_cap[r] = 1e3 * sum(self.cap[k] for k in cap if k[0] == r)
        # measured throughput of each exported router: bytes entering it on any port
        rp = top[top.dev.isin(self.nodes)]
        rin = _series(traffic, rp.link, "hin", H).fillna(0)
        self.router_throughput = rin.T.groupby(rp.set_index("link").dev.reindex(rin.columns).values).sum().T

    def route(self, weight: str = "capacity", sites: pd.DataFrame | None = None):
        """ECMP shortest paths; ``weight`` = 'capacity' (REF_GBPS / speed), 'hops', or
        'distance' (km between the sites, +1 per hop; needs ``sites`` with lat/lon)."""
        if weight == "distance":
            ll = sites.drop_duplicates("code").set_index("code")[["lat", "lon"]]

            def km(a, b):
                p, q = ll.reindex([site_of(a)]).to_numpy()[0], ll.reindex([site_of(b)]).to_numpy()[0]
                if not (np.isfinite(p).all() and np.isfinite(q).all()):
                    return 50.0
                la1, lo1, la2, lo2 = np.radians([p[0], p[1], q[0], q[1]])
                return 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2
                                                    + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
            w = {k: km(*k) + 1.0 for k in self.loads.columns}
        else:
            w = {k: (REF_GBPS / max(self.cap[k], 1.0) if weight == "capacity" else 1.0) for k in self.loads.columns}
        self.links, self.A_link, self.A_node, self.od = tomography.ecmp_routing(
            self.nodes, w, self.ends, no_transit=self.hidden)
        self.y = self.loads[self.links].to_numpy()
        self.allowed = np.asarray(self.A_node.sum(axis=0)).ravel() > 0
        self.weight = weight
        return self

    def estimate(self, hours=None, drop_links=(), method: str = "em") -> np.ndarray:
        """Traffic matrix [hours, OD] (Mbit/s) from the counters: ``method`` = 'em'
        (non-negative, default) or 'tomogravity'; ``drop_links`` hides link loads from the
        estimator (hold-out tests); hours not asked for stay zero."""
        idx = range(len(self.H)) if hours is None else hours
        mask = np.ones(len(self.links), bool)
        mask[list(drop_links)] = False
        out = np.zeros((len(self.H), len(self.od)))
        for h in idx:
            y = np.where(mask, self.y[h], np.nan)
            f = tomography.em if method == "em" else tomography.tomogravity
            out[h] = f(self.A_link, y, self.ingress[h], self.egress[h], allowed=self.allowed)
        return out


def resources(net: Network, R: pd.DataFrame, E: pd.DataFrame, doors: pd.DataFrame):
    """Backbone power (W at the site) of every router node and fibre span of the graph.

    ``R``/``E`` from ``bills.window_units``. A router's backbone power is its non-access
    share, plus its access share when it has no doors (as in the flat plan); access-tier
    routers are access entirely. Each site's optical power goes to the inter-site spans
    ending there, by capacity. Returns (router_W by node, span_W by undirected span and
    the grid zone the power draws from, unplaced W by zone: power no route can reach,
    spread flat)."""
    has_doors = set(doors.dev)
    bb = R.W_site * (1 - R.f_acc) + np.where(~R.d.isin(has_doors) & (R.tier != "access"), R.W_site * R.f_acc, 0)
    bb = pd.Series(np.where(R.tier == "access", 0.0, bb), index=R.d)
    zone_r = pd.Series(R.zone.values, index=R.d)
    bb = bb.groupby(level=0).sum()
    router_W = bb.reindex(net.nodes).fillna(0.0)
    spans = sorted({tuple(sorted(k)) for k in net.links if site_of(k[0]) != site_of(k[1])})
    site_spans: dict = {}
    for a, b in spans:
        for s in (site_of(a), site_of(b)):
            site_spans.setdefault(s, []).append((a, b))
    zones = sorted(set(E.zone) | set(R.zone))
    span_W = pd.DataFrame(0.0, index=pd.MultiIndex.from_tuples(spans), columns=zones)   # W by the zone it draws from
    unplaced = {}
    for s, g in E.groupby("site"):
        ss = site_spans.get(s)
        if not ss:
            for z, w in g.groupby("zone").W_site.sum().items():
                unplaced[z] = unplaced.get(z, 0.0) + w
            continue
        c = np.array([net.cap[k] for k in ss], float)
        for z, w in g.groupby("zone").W_site.sum().items():
            for k, frac in zip(ss, c / c.sum()):
                span_W.loc[k, z] += w * frac
    off = bb[~bb.index.isin(net.nodes)]
    for d, w in off.items():
        z = zone_r.groupby(level=0).first().get(d, A.DEFAULT_ZONE)
        unplaced[z] = unplaced.get(z, 0.0) + w
    return router_W, span_W, unplaced


class Allocation:
    """Hourly backbone bills under the flat and the path-based rate, from one traffic matrix."""

    def __init__(self, net: Network, router_W: pd.Series, span_W: pd.Series, unplaced: dict,
                 grid: pd.DataFrame):
        self.net = net
        lix = {k: i for i, k in enumerate(net.links)}
        rows, cols = [], []
        for j, (a, b) in enumerate(span_W.index):               # a span carries both directions
            for k in ((a, b), (b, a)):
                if k in lix:
                    rows.append(j)
                    cols.append(lix[k])
        S = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(span_W), len(net.links)))
        nidx = [net.nodes.index(n) for n in router_W.index]
        self.A_res = sp.vstack([net.A_node[nidx], S @ net.A_link]).tocsr()
        self.names = [("router", n) for n in router_W.index] + [("span", f"{a}-{b}") for a, b in span_W.index]
        Zr = [A.zone_of(site_of(n)) for n in router_W.index]
        E_r = router_W.to_numpy()[None, :] / 1000 * grid[Zr].to_numpy()
        E_s = sum(span_W[z].to_numpy()[None, :] / 1000 * grid[[z]].to_numpy() for z in span_W.columns)
        self.E = np.c_[E_r, E_s]                                  # g per hour and resource
        self.unplaced_g = sum(w / 1000 * grid[z].to_numpy() for z, w in unplaced.items()) if unplaced else 0 * grid.iloc[:, 0].to_numpy()
        # measured throughput per resource (counters only): routers' entering bytes, spans' both directions
        thr_r = net.router_throughput.reindex(columns=router_W.index).to_numpy()
        thr_s = np.stack([(net.loads.get((a, b), 0) + net.loads.get((b, a), 0)).to_numpy() * np.ones(len(net.H))
                          for a, b in span_W.index], axis=1) if len(span_W) else np.zeros((len(net.H), 0))
        self.X_meas = np.c_[thr_r, thr_s]
        self.capacity = np.r_[[net.router_cap.get(n, np.nan) for n in router_W.index],
                              [2e3 * net.cap[(a, b)] for a, b in span_W.index]]       # Mbit/s, both directions for spans
        self.parties = sorted(set(net.door_party.values) | set(net.ed_party))
        self.P_ed = sp.csr_matrix((np.ones(len(net.ed_party)),
                                   (np.arange(len(net.ed_party)), [self.parties.index(p) for p in net.ed_party])),
                                  shape=(len(net.ed_party), len(self.parties)))
        V = net.din.add(net.dout)
        self.V = V.T.groupby(net.door_party.reindex(V.columns).values).sum().T.reindex(columns=self.parties, fill_value=0)

    def rate_counters(self) -> np.ndarray:
        """Per-resource rate from counters only, g per Mbit/s-hour [hours, resources]
        (NaN where the counters see no throughput, e.g. routers missing from the export)."""
        return np.divide(self.E, self.X_meas, out=np.full_like(self.E, np.nan), where=self.X_meas > 0)

    def bills(self, X: np.ndarray, basis: str = "usage", cap_mult=None) -> dict:
        """Backbone bills (g per party and hour) from traffic matrices X [hours, OD];
        ``basis`` = 'usage' (E / bytes) or 'capacity' (E / capacity, idle part flat); with
        ``cap_mult`` [hours] (usage only) each resource charges at most the idle-link cap of
        ``capped_rates``, the rest is spread flat."""
        net, H = self.net, self.net.H
        Ee = len(net.ends)
        ni = self.net.ed_to_end
        path = np.zeros((len(H), len(self.parties)))
        unalloc = np.array(self.unplaced_g, float).copy()
        od_cost = np.zeros(len(net.od))
        od_bytes = np.zeros(len(net.od))
        end_of = np.asarray(ni.argmax(axis=0)).ravel()           # each edge door's end node
        for h in range(len(H)):
            x = X[h]
            U = self.A_res @ x
            used = U > 0
            if basis == "capacity":
                per = np.divide(self.E[h], np.maximum(self.capacity, U), out=np.zeros(len(U)),
                                where=np.isfinite(self.capacity) & used)
                per = np.where(np.isfinite(self.capacity), per, np.divide(self.E[h], U, out=np.zeros(len(U)), where=used))
                unalloc[h] += (self.E[h] - per * U)[used].sum()
            elif cap_mult is not None:
                per = np.where(used, capped_rates(self.E[h], U, x.sum(), cap_mult[h])[0], 0.0)
                unalloc[h] += (self.E[h] - per * U)[used].sum()
            else:
                per = np.divide(self.E[h], U, out=np.zeros(len(U)), where=used)
            c = self.A_res.T @ per                              # g per (Mbit/s . h) of each OD
            C = (x * c).reshape(Ee, Ee)
            od_cost += x * c
            od_bytes += x
            unalloc[h] += self.E[h][~used].sum()
            src, dst = C.sum(axis=1), C.sum(axis=0)
            di, do = net.ed_in.iloc[h].to_numpy(), net.ed_out.iloc[h].to_numpy()
            node_in, node_out = ni @ di, ni @ do
            s_in = np.divide(di, node_in[end_of], out=np.zeros_like(di), where=node_in[end_of] > 0)
            s_out = np.divide(do, node_out[end_of], out=np.zeros_like(do), where=node_out[end_of] > 0)
            door_cost = (s_in * src[end_of] + s_out * dst[end_of]) / 2
            path[h] = self.P_ed.T @ door_cost
        path = pd.DataFrame(path, index=H, columns=self.parties)
        unseen = path[[p for p in self.parties if p.startswith("unseen:")]].sum(axis=1)
        path[[p for p in self.parties if p.startswith("unseen:")]] = 0.0
        unalloc = unalloc + unseen.to_numpy()
        Vt = self.V.sum(axis=1).replace(0, np.nan)
        spread = self.V.mul(unalloc / Vt, axis=0).fillna(0)
        total = self.E.sum(axis=1) + np.asarray(self.unplaced_g, float)
        flat = self.V.mul(total / Vt, axis=0).fillna(0)
        keep = [p for p in self.parties if not p.startswith("unseen:")]
        return {"path_g": (path + spread)[keep], "flat_g": flat[keep], "unallocated_g": pd.Series(unalloc, index=H),
                "unseen_g": unseen,
                "total_g": pd.Series(total, index=H),
                "od_g_per_GB": pd.Series(np.divide(od_cost, od_bytes * A.GB_PER_MBPS_HOUR, out=np.full(len(od_cost), np.nan),
                                                   where=od_bytes > 0), index=pd.MultiIndex.from_tuples(net.od)),
                "od_GB": pd.Series(od_bytes * A.GB_PER_MBPS_HOUR, index=pd.MultiIndex.from_tuples(net.od))}


def idle_cap(E: np.ndarray, U: np.ndarray, hours: pd.DatetimeIndex, k: float = A.IDLE_CAP_K,
             days: int = A.IDLE_CAP_DAYS, u_min: float = A.IDLE_CAP_MIN_MBPS) -> np.ndarray:
    """The idle-link cap of every hour, as a multiple of that hour's flat rate [hours].

    E, U: emissions (g/h) and loads (Mbit/s) of every resource [hours, resources]. A resource's
    rate relative to the hour's reference is (E / U) / (sum E / sum U); for day d the cap is
    10 ** (median + k * 1.4826 * MAD) of the log10 relative rates of every resource-hour with
    U >= u_min in the ``days`` days before d (causal: it can be published with the day-ahead
    rates). Days with less than a week of history before them use the first week of the record.
    ``capped_rates`` applies the multiple to the same reference, sum E / sum U."""
    hours = pd.DatetimeIndex(hours)
    day = hours.floor("D")
    udays = day.unique()
    out = np.full(len(hours), np.nan)
    for d in udays:
        win = (day >= d - pd.Timedelta(days=days)) & (day < d)
        if (day < d).sum() < 24 * 7:
            win = day < udays[min(7, len(udays) - 1)]
        out[day == d] = cap_multiple(E[win], U[win], k, u_min)
    return out


def cap_multiple(E: np.ndarray, U: np.ndarray, k: float = A.IDLE_CAP_K, u_min: float = A.IDLE_CAP_MIN_MBPS) -> float:
    """10 ** (median + k * 1.4826 * MAD) of log10((E / U) / (sum E / sum U)) over the
    resource-hours of one window [hours, resources] with U >= u_min (see ``idle_cap``)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.log10(E / U / (E.sum(1) / U.sum(1))[:, None])
    v = rel[(U >= u_min) & (E > 0) & np.isfinite(rel)]
    med = np.median(v)
    return float(10 ** (med + k * 1.4826 * np.median(np.abs(v - med))))


def capped_rates(E: np.ndarray, U: np.ndarray, x_total: float, mult: float):
    """Per-resource usage rates of one hour (g per Mbit/s-h) under the idle-link cap, and the
    flat add-on that spreads the rest: (rate [resources], add-on per Mbit/s-h of traffic).

    rate_j = min(E_j / U_j, mult * sum E / sum U), idle resources (U_j = 0) at the cap; the
    emissions the resource rates do not charge (above the cap, or on idle resources) are spread
    over all ``x_total`` Mbit/s-h of the hour, so that sum_j rate_j U_j + add-on x_total = sum E."""
    cap = mult * E.sum() / max(U.sum(), 1e-12)
    per = np.where(U > 0, np.minimum(np.divide(E, U, out=np.full(len(U), np.inf), where=U > 0), cap), cap)
    rest = E.sum() - (per * U).sum()
    return per, rest / max(x_total, 1e-12)


def node_bills(A_res: sp.csr_matrix, E: np.ndarray, X: np.ndarray, n: int, capacity=None,
               cap_mult=None) -> np.ndarray:
    """Generic path-based bills when every edge node is one party: each resource's
    emissions E [hours, resources] split over the traffic matrices' use of it (or, with
    ``capacity``, charged at E / capacity with the idle part spread over all bytes; or, with
    ``cap_mult`` [hours], at the capped usage rates of ``capped_rates``), each flow's cost half
    to its source node and half to its destination; what is left is spread over all bytes.
    Returns [hours, n]."""
    out = np.zeros((len(X), n))
    for h in range(len(X)):
        U = A_res @ X[h]
        den = U if capacity is None else np.maximum(capacity, U)
        per = np.divide(E[h], den, out=np.zeros(len(U)), where=den > 0)
        if cap_mult is not None:
            per = np.where(U > 0, capped_rates(E[h], U, X[h].sum(), cap_mult[h])[0], 0.0)
        C = (X[h] * (A_res.T @ per)).reshape(n, n)
        b = (C.sum(axis=1) + C.sum(axis=0)) / 2
        idle = E[h].sum() - b.sum()
        Xn = X[h].reshape(n, n)
        v = Xn.sum(1) + Xn.sum(0)
        out[h] = b + idle * v / v.sum()
    return out
