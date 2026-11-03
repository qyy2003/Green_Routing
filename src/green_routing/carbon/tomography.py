"""Who talks to whom, from link counters: routing matrices and the tomogravity estimate.

SNMP counters give, every hour, the bytes entering and leaving at each edge node (the
"marginals") and the bytes on every internal link, but not the traffic matrix x_st (bytes
from node s to node t). Network tomography estimates it (Zhang, Roughan, Duffield and
Greenberg, SIGMETRICS 2003, "Fast accurate computation of large-scale IP traffic matrices
from link loads"):

1. **gravity prior**: x_st proportional to in_s * out_t / total (who sends a lot talks to
   who receives a lot);
2. **tomogravity**: the smallest change to the prior, weighted by the prior, that makes
   the routed matrix reproduce the measured link loads and marginals:
   x = x_g + W A^T (A W A^T)^+ (y - A x_g), W = diag(x_g);
3. negative entries clipped, then iterative proportional fitting (IPF) restores the
   marginals.

When the routing model does not match the network exactly, step 2 needs large negative
flows and clipping them destroys the fit. ``em`` keeps every flow non-negative throughout:
multiplicative EM updates from the gravity prior (Vardi, JASA 1996), which converge to the
fit closest to the prior in Kullback-Leibler terms, the minimum-information estimate of
Zhang, Roughan, Lund and Donoho (SIGCOMM 2003).

Routing: shortest paths with equal-cost multipath split at every hop (as OSPF does).
"""
from __future__ import annotations

import heapq

import numpy as np
import scipy.sparse as sp


def ecmp_routing(nodes: list, edges: dict, ends: list, no_transit=()) -> tuple:
    """Fractions of each origin-destination pair's traffic on every link and node.

    ``edges``: {(u, v): weight} for directed links; ``ends``: the edge nodes (OD
    endpoints); ``no_transit``: nodes that may only be endpoints. Returns
    (links, A_link, A_node, od): ``links`` the list of (u, v), ``A_link`` sparse
    [len(links), len(od)] with the share of OD traffic on each link, ``A_node`` sparse
    [len(nodes), len(od)] with the share crossing each node (endpoints included), and
    ``od`` the list of (s, t) in column order (all ordered pairs of ``ends``, s == t
    included: local traffic crosses only its node)."""
    links = sorted(edges)
    lix = {e: i for i, e in enumerate(links)}
    nix = {n: i for i, n in enumerate(nodes)}
    out = {n: [] for n in nodes}
    inc = {n: [] for n in nodes}
    for (u, v), w in edges.items():
        out[u].append((v, w))
        inc[v].append((u, w))
    blocked = set(no_transit)
    E = len(ends)
    od = [(s, t) for s in ends for t in ends]
    rl, cl, vl, rn, cn, vn = [], [], [], [], [], []
    for ti, t in enumerate(ends):
        # distances to t over reversed links; blocked nodes relay nothing
        dist = {t: 0.0}
        pq = [(0.0, t)]
        while pq:
            d, v = heapq.heappop(pq)
            if d > dist.get(v, np.inf) or (v in blocked and v != t):
                continue
            for u, w in inc[v]:
                nd = d + w
                if nd < dist.get(u, np.inf) - 1e-9:
                    dist[u] = nd
                    heapq.heappush(pq, (nd, u))
        nh = {u: [v for v, w in out[u] if v in dist and abs(dist[u] - w - dist[v]) < 1e-6
                  and (v == t or v not in blocked)] for u in dist if u != t}
        order = sorted(dist, key=lambda n: -dist[n])
        F = {n: np.zeros(E) for n in dist}            # flow at node n from every source (unit each)
        for si, s in enumerate(ends):
            if s in F:
                F[s][si] += 1.0
        for u in order:
            if u == t or not nh.get(u):
                continue
            share = F[u] / len(nh[u])
            for v in nh[u]:
                F[v] += share
                nz = np.nonzero(share)[0]
                rl += [lix[(u, v)]] * len(nz)
                cl += list(nz * E + ti)
                vl += list(share[nz])
        for n, f in F.items():
            nz = np.nonzero(f)[0]
            rn += [nix[n]] * len(nz)
            cn += list(nz * E + ti)
            vn += list(f[nz])
    A_link = sp.csr_matrix((vl, (rl, cl)), shape=(len(links), E * E))
    A_node = sp.csr_matrix((vn, (rn, cn)), shape=(len(nodes), E * E))
    return links, A_link, A_node, od


def marginal_rows(E: int) -> sp.csr_matrix:
    """Rows summing the OD vector (s-major order) into ingress (E rows) and egress (E rows)."""
    I = sp.identity(E, format="csr")
    one = sp.csr_matrix(np.ones((1, E)))
    return sp.vstack([sp.kron(I, one), sp.kron(one, I)]).tocsr()


def gravity(ingress: np.ndarray, egress: np.ndarray) -> np.ndarray:
    tot = (ingress.sum() + egress.sum()) / 2
    return np.outer(ingress, egress).ravel() / tot if tot > 0 else np.zeros(len(ingress) ** 2)


def ipf(x: np.ndarray, ingress: np.ndarray, egress: np.ndarray, iters: int = 50) -> np.ndarray:
    E = len(ingress)
    X = x.reshape(E, E).copy()
    eg = egress * ingress.sum() / egress.sum() if egress.sum() > 0 else egress
    for _ in range(iters):
        r = X.sum(1)
        X *= np.divide(ingress, r, out=np.zeros_like(r), where=r > 0)[:, None]
        c = X.sum(0)
        X *= np.divide(eg, c, out=np.zeros_like(c), where=c > 0)[None, :]
    return X.ravel()


def tomogravity(A_link: sp.csr_matrix, y: np.ndarray, ingress: np.ndarray, egress: np.ndarray,
                ridge: float = 1e-6, allowed: np.ndarray | None = None) -> np.ndarray:
    """Estimate the OD vector from link loads ``y`` (NaN = unobserved link) and marginals.
    ``allowed`` (bool per OD) zeroes the prior of pairs that have no route."""
    E = len(ingress)
    xg = gravity(ingress, egress)
    if allowed is not None:
        xg = ipf(xg * allowed, ingress, egress)
    ok = np.isfinite(y)
    A = sp.vstack([A_link[ok], marginal_rows(E)]).tocsr()
    b = np.r_[y[ok], ingress, egress * ingress.sum() / max(egress.sum(), 1e-12)]
    keep = np.asarray(A.getnnz(axis=1) > 0)
    A, b = A[keep], b[keep]
    W = sp.diags(xg)
    M = (A @ W @ A.T).toarray()
    M += ridge * np.trace(M) / len(M) * np.eye(len(M))
    lam = np.linalg.solve(M, b - A @ xg)
    x = np.clip(xg + W @ (A.T @ lam), 0, None)
    return ipf(x, ingress, egress)


def em(A_link: sp.csr_matrix, y: np.ndarray, ingress: np.ndarray, egress: np.ndarray,
       allowed: np.ndarray | None = None, iters: int = 300) -> np.ndarray:
    """Non-negative estimate fitting link loads (NaN = unobserved) and marginals:
    x <- x * A^T(b / Ax) / A^T 1, started from the gravity prior; IPF then makes every
    node's row and column sums equal its measured ingress and egress."""
    E = len(ingress)
    x = gravity(ingress, egress)
    if allowed is not None:
        x = ipf(x * allowed, ingress, egress)
    ok = np.isfinite(y)
    A = sp.vstack([A_link[ok], marginal_rows(E)]).tocsr()
    b = np.r_[y[ok], ingress, egress * ingress.sum() / max(egress.sum(), 1e-12)]
    At = A.T.tocsr()
    norm = np.asarray(At.sum(axis=1)).ravel()
    norm[norm == 0] = 1.0
    for _ in range(iters):
        Ax = A @ x
        r = np.divide(b, Ax, out=np.zeros_like(b), where=Ax > 0)
        x *= (At @ r) / norm
    return ipf(x, ingress, egress)        # the marginals are measured: match them exactly
