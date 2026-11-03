"""Abilene (Internet2, 2004): the second dataset, with real traffic matrices from NetFlow.

Yin Zhang's archive (www.cs.utexas.edu/~yzhang/research/AbileneTM/) holds 24 weeks of
5-minute origin-destination matrices between Abilene's 12 routers, built from sampled
NetFlow (1 in 100 packets), with the OSPF routing matrix and the 2003 topology (link
capacities, OSPF weights). Unlike SWITCHlan, the traffic matrix is measured, so every
link load and every router total can be rebuilt exactly from it.

Weeks (first day, per the archive's readme) are not all contiguous: X06 (2004-05-01) to
X24 (09-04) form one run except that X21 repeats X20's last day (dropped) and 08-20 is
missing. Times are taken as UTC (the archive does not say). One corrupt interval in X24
(70 Gbit/s on a 10 Gbit/s router) is removed by ``clean``.

    download(dest)                    fetch the archive's files (~170 MB)
    load(dest, weeks)                 topology, routing, and the 5-minute matrices
    write_mirror(ds, root, kind)      the links / door totals / network total as CSVs in the
                                      SWITCH export's format, so gr-benchmark runs unchanged
"""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

URL = "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/"
WEEK_START = {1: "03-01", 2: "03-08", 3: "04-02", 4: "04-09", 5: "04-22", 6: "05-01", 7: "05-08", 8: "05-15",
              9: "05-22", 10: "05-29", 11: "06-05", 12: "06-12", 13: "06-19", 14: "06-26", 15: "07-03", 16: "07-10",
              17: "07-17", 18: "07-24", 19: "07-31", 20: "08-07", 21: "08-13", 22: "08-21", 23: "08-28", 24: "09-04"}
CONTIGUOUS = list(range(6, 25))          # 2004-05-01 .. 2004-09-10, one gap day
UNIT_TO_MBPS = 100 * 8 / 300 / 1e6       # (100 bytes / 5 min) -> Mbit/s


def download(dest: str | Path, weeks=range(1, 25)) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in ["A", "links", "demands", "topo-2003-04-10.txt", "readme.txt"] + [f"X{w:02d}.gz" for w in weeks]:
        if not (dest / f).exists():
            urllib.request.urlretrieve(URL + f, dest / f)
    return dest


def _rows(path):
    return [l.split() for l in open(path) if l.strip() and not l.startswith("#")]


def load(dest: str | Path, weeks=CONTIGUOUS) -> dict:
    """Everything about the dataset: nodes and coordinates, internal links with capacity
    (Mbit/s) and OSPF weight, the routing matrix restricted to internal links (A_link,
    [links, OD]), the routers on each path (A_node, [nodes, OD]), and the true 5-minute
    matrices X (Mbit/s, [time, OD], OD s-major) on a regular grid (gaps = NaN)."""
    dest = Path(dest)
    # the topology file has a "router" section (name city lat lon) and a "link" section
    # (x y capacity_kbps ospf_weight); both have four fields per line
    section, coord, cap, ospf = None, {}, {}, {}
    for line in open(dest / "topo-2003-04-10.txt"):
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        if len(f) == 1:
            section = f[0]
        elif section == "router":
            coord[f[0]] = (float(f[2]), float(f[3]))
        elif section == "link":
            cap[(f[0], f[1])] = float(f[2]) / 1e3            # Mbit/s
            ospf[(f[0], f[1])] = float(f[3])
    links = {int(i): (name, int(k)) for name, i, k in _rows(dest / "links")}
    dem = {int(i): tuple(name.split(",")) for name, i in _rows(dest / "demands")}
    nodes = sorted({s for s, _ in dem.values()})
    n = len(nodes)
    od = [(s, t) for s in nodes for t in nodes]
    col = {p: i for i, p in enumerate(od)}
    internal = sorted(i for i, (_, k) in links.items() if k == 0)
    lrow = {i: j for j, i in enumerate(internal)}
    link_xy = [tuple(links[i][0].split(",")) for i in internal]
    r_, c_, v_ = [], [], []
    for _, _, li, di, frac in _rows(dest / "A"):
        if int(li) in lrow:
            r_.append(lrow[int(li)]); c_.append(col[dem[int(di)]]); v_.append(float(frac))
    A_link = sp.csr_matrix((v_, (r_, c_)), shape=(len(internal), n * n))
    A_node = sp.lil_matrix((n, n * n))
    for j, (s, t) in enumerate(od):
        A_node[nodes.index(s), j] = 1.0
        for li in A_link[:, j].nonzero()[0]:
            A_node[nodes.index(link_xy[li][1]), j] = 1.0
    order = [col[dem[i]] for i in sorted(dem)]
    frames = []
    for w in weeks:
        M = np.loadtxt(gzip.open(dest / f"X{w:02d}.gz"))[:, 0::5]      # the real matrix (column 0 of 5)
        X = np.zeros((len(M), n * n))
        X[:, order] = M * UNIT_TO_MBPS
        t = pd.date_range(f"2004-{WEEK_START[w]}", periods=len(M), freq="5min", tz="UTC")
        frames.append(pd.DataFrame(X, index=t))
    X = pd.concat(frames)
    X = X[~X.index.duplicated(keep="first")]                           # X21 repeats X20's last day
    grid = pd.date_range(X.index.min(), X.index.max(), freq="5min")
    X = clean(X.reindex(grid))
    return dict(nodes=nodes, coord=coord, od=od, link_xy=link_xy, cap={k: cap.get(k, np.nan) for k in link_xy},
                ospf=ospf, A_link=A_link, A_node=A_node.tocsr(), X=X)


def clean(X: pd.DataFrame, factor: float = 5.0) -> pd.DataFrame:
    """Drop snapshots whose network total exceeds ``factor`` x the median of the
    surrounding day (corrupt records); short gaps (<= 1 h) are interpolated."""
    tot = X.sum(axis=1, min_count=1)
    med = tot.rolling(288, center=True, min_periods=12).median()
    X = X.copy()
    X.loc[tot > factor * med] = np.nan
    return X.interpolate(limit=12, limit_area="inside")


def series(ds: dict) -> dict:
    """Per-direction internal link loads, router ingress / egress and the network total
    (Mbit/s, 5 minutes), all exact sums of the measured matrix."""
    X = ds["X"].to_numpy()
    n = len(ds["nodes"])
    links = pd.DataFrame((ds["A_link"] @ np.nan_to_num(X).T).T, index=ds["X"].index,
                         columns=[f"{a}->{b}" for a, b in ds["link_xy"]])
    links[ds["X"].isna().all(axis=1).to_numpy()] = np.nan
    R = X.reshape(len(X), n, n)
    ing = pd.DataFrame(R.sum(2), index=ds["X"].index, columns=ds["nodes"])
    egr = pd.DataFrame(R.sum(1), index=ds["X"].index, columns=ds["nodes"])
    return dict(links=links, ingress=ing, egress=egr, total=ing.sum(axis=1, min_count=1))


def write_mirror(ds: dict, root: str | Path, kind: str = "links") -> list[Path]:
    """Write Abilene in the SWITCH export's CSV format under ``root/router-interfaces``.

    ``kind="links"``: one interface per undirected internal link (out = the first router's
    direction) and one "access" interface per router (in = bytes entering Abilene there,
    out = bytes leaving); ``kind="total"``: one series, the network total (in = out =
    half of it, so the panel's in + out equals the bytes carried)."""
    S = series(ds)
    root = Path(root) / "router-interfaces"
    ts = (S["total"].index.astype("int64") // 10**9).to_numpy()
    out = []

    def write(dev, iface, inn, outt):
        f = root / dev / f"{iface}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": ts, "ifHCInOctets": np.asarray(inn) * 1e6 / 8,
                      "ifHCOutOctets": np.asarray(outt) * 1e6 / 8}).to_csv(f, index=False)
        out.append(f)
    if kind == "total":
        write("abilene", "total", S["total"] / 2, S["total"] / 2)
        return out
    done = set()
    for a, b in ds["link_xy"]:
        if (b, a) in done or (a, b) in done:
            continue
        done.add((a, b))
        back = S["links"].get(f"{b}->{a}", pd.Series(np.nan, index=S["links"].index))
        write(a.lower(), f"to-{b.lower()}", back, S["links"][f"{a}->{b}"])
    for v in ds["nodes"]:
        write(v.lower(), "access", S["ingress"][v], S["egress"][v])
    return out
