#!/usr/bin/env python3
"""
Schematic map of the SWITCH backbone network.

Nodes are placed at (approximate) Swiss geographic coordinates.
Edges are drawn as straight lines with width proportional to bandwidth.
Output: topology_map.svg  +  topology_map.png
"""

import re
import os
import subprocess
import pathlib
import random
import math

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatch
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = pathlib.Path(
    "../network-energy-efficiency-research/switch-network-topology"
)
TXT_FILE      = BASE / "switch-network-topology.txt"
JSON_FILE     = BASE / "switch-network-topology.json"
OBSERVIUM_DIR = pathlib.Path(
    "../network-energy-efficiency-research/observium-data"
)
BB_DIR = pathlib.Path(
    "../network-energy-efficiency-research/lan-mon2021/bb-usage-logs"
)
OUT_SVG = pathlib.Path(__file__).parent / "topology_map.svg"
OUT_PNG = pathlib.Path(__file__).parent / "topology_map.png"

# ---------------------------------------------------------------------------
# Swiss geographic coordinates  lon, lat  (WGS84)
# Only longitude / latitude matter — we project them linearly to x/y.
# ---------------------------------------------------------------------------
CITY_COORDS = {
    # ---- known SWITCH PoPs ------------------------------------------------
    "ag":    (8.05, 47.39),   # Aarau
    "avp":   (6.88, 46.82),   # Avenches / Payerne area (HEIG-VD)
    "ba":    (7.59, 47.56),   # Basel (alt abbreviation)
    "bf":    (7.47, 46.96),   # Bern-Felsenau / BFH
    "bi":    (7.24, 47.14),   # Biel/Bienne
    "bis":   (9.25, 47.50),   # Bischofszell
    "blu":   (9.53, 47.15),   # Buchs SG
    "br":    (8.21, 47.53),   # Brugg
    "bs":    (7.59, 47.56),   # Basel
    "bu":    (7.68, 47.06),   # Burgdorf
    "bul":   (8.31, 47.07),   # ?  (near Lucerne)
    "bz":    (9.02, 46.19),   # Bellinzona
    "car":   (6.14, 46.18),   # Carouge (near Geneva)
    "caw":   (9.03, 46.20),   # Cadenazzo area
    "ce":    (6.13, 46.20),   # Centre/Carouge Geneva
    "cma":   (8.97, 46.02),   # Chiasso / Mendrisio
    "cmb":   (8.98, 46.02),   # Chiasso area
    "cr":    (6.15, 46.21),   # Geneva area
    "cvb":   (7.39, 46.95),   # Bern area
    "cvl":   (7.40, 46.95),   # Bern area
    "del":   (7.34, 47.37),   # Delémont
    "dgd":   (9.40, 47.42),   # near St. Gallen
    "due":   (7.18, 46.80),   # Düdingen (near Fribourg)
    "duf":   (7.87, 46.47),   # Goms/Leuk area
    "dv":    (8.22, 47.40),   # near Aarau / Brugg
    "eb":    (8.33, 47.08),   # Ebikon (near Lucerne)
    "ehblg": (8.17, 47.40),   # Lenzburg area
    "ehbzo": (8.32, 47.41),   # near Lucerne
    "ehl":   (6.24, 46.22),   # Geneva area (western)
    "eic":   (9.50, 47.47),   # Eichberg, near St. Gallen
    "el":    (9.10, 47.10),   # Glarus area
    "ensi":  (7.57, 47.54),   # Ensi near Basel
    "ez":    (8.57, 47.36),   # Zürich (EZ = Zürich south?)
    "ff":    (8.90, 47.56),   # Frauenfeld
    "fga":   (9.36, 47.42),   # near St. Gallen
    "fhsg":  (9.38, 47.43),   # FH St. Gallen
    "fr":    (7.15, 46.80),   # Fribourg
    "gbl":   (6.18, 46.22),   # Geneva area
    "ge":    (6.15, 46.20),   # Geneva
    "gl":    (9.07, 47.04),   # Glarus
    "gla":   (9.06, 47.03),   # Glarus
    "gno":   (6.16, 46.24),   # Geneva north
    "go":    (9.39, 47.44),   # near St. Gallen
    "gov":   (6.19, 46.22),   # Geneva area
    "gr":    (9.53, 46.85),   # Chur / Graubünden
    "gsa":   (9.37, 47.41),   # St. Gallen
    "gsb":   (9.38, 47.41),   # St. Gallen
    "gva":   (6.12, 46.24),   # Geneva Airport
    "gvb":   (6.12, 46.23),   # Geneva area
    "hepvd": (6.64, 46.78),   # HEIG-VD Yverdon area
    "hfh":   (8.68, 47.38),   # near Zürich (Horgen/Thalwil)
    "ho":    (8.80, 46.12),   # near Lugano (Morcote / Lugano south)
    "hug":   (6.14, 46.19),   # HUG Geneva hospital
    "ibm":   (8.50, 47.31),   # IBM Rüschlikon
    "imd":   (6.57, 46.52),   # IMD Lausanne
    "itis":  (8.72, 47.50),   # near Winterthur
    "ix":    (8.53, 47.37),   # Internet Exchange Zürich
    "ixg":   (6.16, 46.23),   # Internet Exchange Geneva
    "jfj":   (7.99, 46.55),   # Jungfraujoch
    "kas":   (8.90, 47.10),   # near Glarus
    "kl":    (8.58, 47.45),   # Kloten (Zürich airport)
    "kr":    (9.18, 47.65),   # Kreuzlingen
    "lab":   (8.54, 47.38),   # Lab (Zürich)
    "lg":    (8.95, 46.00),   # Lugano
    "lgd":   (8.93, 46.01),   # Lugano area
    "li":    (9.56, 47.14),   # Liechtenstein
    "lio":   (8.96, 46.01),   # Lugano area
    "lo":    (8.80, 46.17),   # Locarno
    "ls":    (6.63, 46.52),   # Lausanne
    "lsm":   (6.62, 46.52),   # Lausanne area
    "lz":    (8.94, 46.02),   # Lugano area
    "maa":   (7.07, 46.10),   # Martigny area
    "mc":    (8.93, 46.03),   # Mendrisio/Chiasso
    "mdlp":  (6.63, 46.53),   # Lausanne area
    "metas": (7.47, 46.92),   # METAS Bern-Wabern
    "my":    (6.94, 46.25),   # Monthey
    "naz":   (8.61, 47.36),   # Zürich area (Nationales Zentrum?)
    "ne":    (6.93, 47.00),   # Neuchâtel
    "nea":   (6.91, 47.01),   # Neuchâtel area
    "neb":   (6.93, 47.01),   # Neuchâtel area
    "npa":   (8.54, 47.38),   # Zürich area
    "npb":   (8.55, 47.38),   # Zürich area
    "nyr":   (9.37, 47.42),   # near St. Gallen
    "ol":    (7.91, 47.35),   # Olten
    "ps":    (8.52, 47.37),   # Zürich area
    "ra":    (7.53, 46.25),   # Sierre / Sion area
    "rot":   (8.45, 47.14),   # Rotkreuz
    "sa":    (7.36, 46.24),   # Sion
    "sg":    (9.37, 47.42),   # St. Gallen
    "sgd":   (8.63, 47.70),   # Schaffhausen
    "sge":   (9.38, 47.44),   # St. Gallen East
    "sgm":   (9.39, 47.43),   # St. Gallen area
    "si":    (7.36, 46.23),   # Sion
    "snf":   (8.64, 47.70),   # Schaffhausen
    "snl":   (8.36, 47.21),   # near Lucerne
    "snm":   (8.62, 47.71),   # Schaffhausen area
    "sno":   (8.61, 47.72),   # Schaffhausen area
    "soa":   (7.37, 46.25),   # Sion area
    "sob":   (7.38, 46.25),   # Sion area
    "sp":    (7.68, 46.69),   # Spiez
    "svf":   (7.50, 47.02),   # near Bern
    "svg":   (7.50, 47.03),   # near Bern
    "thu":   (7.63, 46.75),   # Thun
    "to":    (6.62, 46.51),   # Tolochenaz / Lausanne west
    "ukb":   (8.59, 47.37),   # UniversitätsSpital Zürich
    "vi":    (7.88, 46.29),   # Visp
    "wsl":   (8.46, 47.36),   # WSL Birmensdorf
    "yv":    (6.64, 46.78),   # Yverdon-les-Bains
    "zar":   (9.95, 46.70),   # Zernez area
    "zbe":   (7.45, 46.95),   # Bern area
    "zbu":   (8.53, 47.37),   # Zürich area
    "zgl":   (8.52, 47.17),   # Zug
    "zh":    (8.55, 47.37),   # Zürich
    "zob":   (8.54, 47.37),   # Zürich area
    "zop":   (8.55, 47.38),   # Zürich area
}

# Mercator-like projection: scale longitude so distances are isometric at 47°N
import math as _math
LON_SCALE = _math.cos(_math.radians(46.9))  # ≈ 0.682

def _project(lon, lat):
    """Project (lon, lat) to (x, y) with equal-distance units."""
    return lon * LON_SCALE, lat

# Per-city index for jitter
_city_count: dict[str, int] = {}

# jitter in raw lon/lat degrees (before projection)
JITTER = [
    (0.00,  0.00),
    (0.12,  0.00),
    (-0.12, 0.00),
    (0.00,  0.07),
    (0.12,  0.07),
    (-0.12, 0.07),
    (0.06, -0.07),
    (-0.06,-0.07),
]

def city_code(node_id: str) -> str:
    s = node_id
    if s.startswith("swi"):
        s = s[3:]
    s = re.sub(r"\d+$", "", s)
    s = re.sub(r"-.*$", "", s)
    return s.lower()


def geo_pos(node_id: str) -> tuple[float, float]:
    code = city_code(node_id)
    base = CITY_COORDS.get(code)
    if base is None:
        rng = random.Random(hash(node_id))
        base = (rng.uniform(6.1, 10.4), rng.uniform(45.8, 47.8))

    idx = _city_count.get(code, 0)
    _city_count[code] = idx + 1
    jx, jy = JITTER[idx % len(JITTER)]
    return _project(base[0] + jx, base[1] + jy)


# ---------------------------------------------------------------------------
# Parse text topology
# ---------------------------------------------------------------------------
EDGE_PAT = re.compile(
    r"^(\S+)\s+(\S+)\s+<=>\s+(\S+)\s+(\S+)\s+\(([^)]+)\)\s+\((\S+)\s+(\S+)\)"
)

def parse_txt(path):
    edges = []
    with open(path) as fh:
        for line in fh:
            m = EDGE_PAT.match(line.strip())
            if m:
                src, src_if, dst, dst_if, bw_s, m1, m2 = m.groups()
                edges.append(dict(src=src, src_if=src_if,
                                  dst=dst, dst_if=dst_if,
                                  bw=float(bw_s),
                                  metric1=int(m1), metric2=int(m2)))
    return edges


# ---------------------------------------------------------------------------
# Bandwidth styling
# ---------------------------------------------------------------------------
# Default edge colors when no traffic-monitoring data is available.
# These match the reference diagram: backbone black, 10GE blue, GE dashed blue.
BW_STYLE = [
    # (min_bw, linewidth, default_color,  linestyle, legend_label)
    (4e11, 7.0, "#111111", "solid",  "400 GE"),
    (2e11, 5.0, "#222222", "solid",  "200 GE"),
    (1e11, 3.2, "#333333", "solid",  "100 GE"),
    (2e10, 2.0, "#555555", "solid",  "20 GE"),
    (1e10, 1.2, "#2980b9", "solid",  "10 GE"),
    (1e9,  0.8, "#2980b9", "dashed", "1 GE"),
    (0,    0.6, "#aaaaaa", "dotted", "<1 GE"),
]

def edge_style(bw):
    for min_bw, lw, color, ls, label in BW_STYLE:
        if bw >= min_bw:
            return lw, color, ls, label
    return 0.6, "#aaaaaa", "dotted", "<1 GE"

def short_label(node_id: str) -> str:
    s = node_id
    if s.startswith("swi"):
        s = s[3:]
    return s


# ---------------------------------------------------------------------------
# Build graph + two-tier hierarchical layout
# ---------------------------------------------------------------------------
def build_and_layout(edges):
    import math
    import collections
    from collections import defaultdict

    G = nx.Graph()
    for e in edges:
        u, v = e["src"], e["dst"]
        if G.has_edge(u, v):
            if e["bw"] > G[u][v]["bw"]:
                G[u][v].update(bw=e["bw"], src_if=e["src_if"], dst_if=e["dst_if"],
                               metric1=e["metric1"], metric2=e["metric2"])
        else:
            G.add_edge(u, v, **{k: e[k] for k in
                                ("bw", "src_if", "dst_if", "metric1", "metric2")})

    # ------------------------------------------------------------------
    # Tier classification
    #
    # A node is "backbone" only if it is a genuine multi-homed hub:
    #   • at least MIN_BB_LINKS neighbours connected via ≥ 100 GE
    #
    # This excludes sites that have exactly one high-capacity uplink
    # (those are access/edge nodes despite their link speed) and keeps
    # only true crossroads (e.g. ez2 has 7 × 100G neighbours, ls1 has
    # 4 × 100G and degree 13).  The threshold of 3 yields ~25 nodes.
    # ------------------------------------------------------------------
    BACKBONE_BW       = 1e11   # 100 GE — minimum link BW to count
    MIN_BB_LINKS      = 3      # must have this many ≥100G neighbours

    bb_link_count: dict[str, int] = defaultdict(int)
    for u, v, d in G.edges(data=True):
        if d["bw"] >= BACKBONE_BW:
            bb_link_count[u] += 1
            bb_link_count[v] += 1

    backbone_nodes: set[str] = {n for n, cnt in bb_link_count.items()
                                 if cnt >= MIN_BB_LINKS}
    access_nodes = set(G.nodes()) - backbone_nodes
    print(f"  backbone: {len(backbone_nodes)} nodes  "
          f"(criterion: ≥{MIN_BB_LINKS} neighbours at ≥100 GE)")

    # ------------------------------------------------------------------
    # Step 1 — Place backbone nodes at Swiss geographic coordinates,
    # then run Kamada-Kawai *initialised from those positions* so KK
    # can only make local adjustments to reduce crossings while staying
    # close to the geographic arrangement.
    # ------------------------------------------------------------------
    geo_raw = {n: geo_pos(n) for n in backbone_nodes}
    # Normalise geographic positions to [-1, 1] for KK
    xs_g = [p[0] for p in geo_raw.values()]
    ys_g = [p[1] for p in geo_raw.values()]
    xr = max(xs_g) - min(xs_g) or 1.0
    yr = max(ys_g) - min(ys_g) or 1.0
    geo_seed = {
        n: (2 * (geo_raw[n][0] - min(xs_g)) / xr - 1,
            2 * (geo_raw[n][1] - min(ys_g)) / yr - 1)
        for n in backbone_nodes
    }
    B = G.subgraph(backbone_nodes).copy()
    print("Running Kamada-Kawai on backbone (geographic seed) …")
    pos_b = nx.kamada_kawai_layout(B, pos=geo_seed, weight=None)
    # Scale backbone positions outward so clusters have more breathing room
    # and backbone links spread apart more before access nodes are attached.
    BB_SCALE = 2.2
    pos: dict[str, tuple[float, float]] = {
        n: (x * BB_SCALE, y * BB_SCALE) for n, (x, y) in pos_b.items()
    }
    print(f"  backbone placed ({len(backbone_nodes)} nodes)")

    # ------------------------------------------------------------------
    # Step 2 — Assign each access node to its nearest backbone node
    # via BFS over the full graph.
    # ------------------------------------------------------------------
    backbone_parent: dict[str, str | None] = {}
    for start in access_nodes:
        queue: collections.deque[str] = collections.deque([start])
        visited: set[str] = {start}
        found = None
        while queue:
            node = queue.popleft()
            if node in backbone_nodes:
                found = node
                break
            for nb in G.neighbors(node):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        backbone_parent[start] = found

    # ------------------------------------------------------------------
    # Step 3 — Recursive radial-tree placement for each access cluster.
    #
    # Instead of a flat fan, we build a BFS tree from each backbone node
    # through its access cluster and allocate angle sectors proportional
    # to subtree size — exactly like a Reingold-Tilford radial layout.
    # This keeps access→access chains in their own angular branch and
    # prevents them from overlapping siblings.
    # ------------------------------------------------------------------
    STEP_R = 0.09   # radial distance per tree level — keep clusters tight

    # backbone centroid (used to pick "outward" base angle per bb node)
    cx = sum(pos[n][0] for n in backbone_nodes) / len(backbone_nodes)
    cy = sum(pos[n][1] for n in backbone_nodes) / len(backbone_nodes)

    # group access nodes by nearest backbone parent
    groups: dict[str | None, list[str]] = defaultdict(list)
    for node, parent in backbone_parent.items():
        groups[parent].append(node)

    # Build BFS spanning trees per cluster (no cycles → safe recursion)
    def _bfs_children(root: str, cluster: set[str]) -> dict[str, list[str]]:
        """Return {node: [children]} BFS tree from root through cluster."""
        ch: dict[str, list[str]] = {root: []}
        q = collections.deque([root])
        while q:
            node = q.popleft()
            for nb in G.neighbors(node):
                if nb in cluster and nb not in ch:
                    ch[node].append(nb)
                    ch[nb] = []
                    q.append(nb)
        return ch

    def _subtree_size(node: str, ch: dict[str, list[str]]) -> int:
        return 1 + sum(_subtree_size(c, ch) for c in ch.get(node, []))

    def _place_radial(node: str, node_pos: tuple[float, float],
                      ch: dict[str, list[str]],
                      base_angle: float, half_span: float,
                      radius: float) -> None:
        """
        Recursively place the BFS subtree of `node` in a radial layout.
        Angle sector: [base_angle − half_span, base_angle + half_span].
        Uses subtree size for proportional angle allocation (Reingold-Tilford).
        """
        children = ch.get(node, [])
        if not children:
            return
        sizes = {c: _subtree_size(c, ch) for c in children}
        total = sum(sizes.values())
        rx, ry = node_pos
        cumfrac = 0.0
        for child in children:
            frac = sizes[child] / total
            angle = (base_angle - half_span) + (cumfrac + frac / 2) * 2 * half_span
            cumfrac += frac
            cx2 = rx + radius * math.cos(angle)
            cy2 = ry + radius * math.sin(angle)
            pos[child] = (cx2, cy2)
            child_span = max(math.radians(12), half_span * frac * 2.2)
            _place_radial(child, (cx2, cy2), ch,
                          angle, child_span, radius * 0.72)

    for bb_node, group in groups.items():
        if bb_node is None:
            for i, n in enumerate(group):
                angle = 2 * math.pi * i / max(len(group), 1)
                pos[n] = (1.9 * math.cos(angle), 1.9 * math.sin(angle))
            continue

        px, py = pos[bb_node]
        dx, dy = px - cx, py - cy
        d_len = (dx * dx + dy * dy) ** 0.5 or 1.0
        base_angle = math.atan2(dy / d_len, dx / d_len)
        half_span = min(math.radians(130), math.radians(20) * len(group))

        ch_map = _bfs_children(bb_node, set(group))
        _place_radial(bb_node, (px, py), ch_map,
                      base_angle, half_span, STEP_R)

    # ------------------------------------------------------------------
    # Step 4 — Overlap removal across ALL nodes.
    # Backbone nodes are nudged gently; access nodes more aggressively.
    # ------------------------------------------------------------------
    all_nodes = list(G.nodes())
    for _ in range(80):
        for i, n1 in enumerate(all_nodes):
            for n2 in all_nodes[i + 1:]:
                x1, y1 = pos[n1]
                x2, y2 = pos[n2]
                dx2, dy2 = x2 - x1, y2 - y1
                dist = (dx2 * dx2 + dy2 * dy2) ** 0.5
                min_d = 0.055
                if 1e-9 < dist < min_d:
                    push = (min_d - dist) / (2.0 * dist)
                    # backbone positions are anchors — move them less
                    w1 = 0.15 if n1 in backbone_nodes else 1.0
                    w2 = 0.15 if n2 in backbone_nodes else 1.0
                    pos[n1] = (x1 - dx2 * push * w1, y1 - dy2 * push * w1)
                    pos[n2] = (x2 + dx2 * push * w2, y2 + dy2 * push * w2)

    # clusters: backbone_node → list of all nodes in that site (incl. backbone node itself)
    clusters: dict[str, list[str]] = {}
    for bb_node in backbone_nodes:
        members = [bb_node] + groups.get(bb_node, [])
        clusters[bb_node] = members

    return G, pos, backbone_nodes, clusters


# ---------------------------------------------------------------------------
# RRD data loader  (Observium directory layout)
# ---------------------------------------------------------------------------
# NodeData keys:  power_W (float), traffic_in_bps (float), traffic_out_bps (float)
NodeData = dict   # {node_id: {power_W, traffic_in_bps, traffic_out_bps}}


def _rrd_lastupdate(path: pathlib.Path) -> int:
    """Return the last-update Unix timestamp of an RRD file."""
    out = subprocess.check_output(["rrdtool", "last", str(path)],
                                  stderr=subprocess.DEVNULL).decode().strip()
    return int(out)


def _rrd_fetch_avg(path: pathlib.Path, ds_index: int,
                   start: int, end: int) -> float:
    """Return the mean AVERAGE value for one datasource over [start, end]."""
    out = subprocess.check_output(
        ["rrdtool", "fetch", str(path), "AVERAGE",
         "--start", str(start), "--end", str(end)],
        stderr=subprocess.DEVNULL,
    ).decode()
    values = []
    for line in out.splitlines():
        parts = line.split(":")
        if len(parts) != 2:
            continue
        cols = parts[1].split()
        if ds_index >= len(cols):
            continue
        try:
            v = float(cols[ds_index])
            if math.isfinite(v):
                values.append(v)
        except ValueError:
            pass
    return sum(values) / len(values) if values else 0.0


def load_observium_data(obs_dir: pathlib.Path) -> NodeData:
    """
    Walk obs_dir looking for per-device subdirectories that contain RRD files.
    For each device found, aggregate:
      - power_W          : sum of last values of sensor-power-*.rrd
      - traffic_in_bps   : sum of avg INOCTETS  (bytes/s → bits/s ×8) across port-*.rrd
      - traffic_out_bps  : sum of avg OUTOCTETS across port-*.rrd

    Returns a dict keyed by lowercase device name (e.g. "swine2").
    """
    result: NodeData = {}

    for dev_dir in sorted(obs_dir.iterdir()):
        if not dev_dir.is_dir():
            continue

        dev_name = dev_dir.name.lower()
        # Strip .switch.ch suffix if present
        dev_name = dev_name.split(".")[0]

        power_W = 0.0
        traffic_in_bps = 0.0
        traffic_out_bps = 0.0

        # ---- Power sensors ------------------------------------------------
        for rrd in dev_dir.glob("sensor-power-*.rrd"):
            try:
                out = subprocess.check_output(
                    ["rrdtool", "lastupdate", str(rrd)],
                    stderr=subprocess.DEVNULL,
                ).decode()
                # last line: "timestamp: v1 v2 ..."
                last_line = [l for l in out.splitlines() if ":" in l][-1]
                val_str = last_line.split(":")[-1].strip().split()[0]
                power_W += float(val_str)
            except Exception:
                pass

        # ---- Port traffic -------------------------------------------------
        port_rrds = [r for r in dev_dir.glob("port-*.rrd")
                     if "ipv6" not in r.name]
        if port_rrds:
            # Use the most recent last-update across all ports as reference
            try:
                ref_ts = max(_rrd_lastupdate(r) for r in port_rrds)
            except Exception:
                ref_ts = 0
            window = 3600  # 1 h of history

            for rrd in port_rrds:
                try:
                    # ds[0]=INOCTETS ds[1]=OUTOCTETS (bytes/s)
                    in_Bps  = _rrd_fetch_avg(rrd, 0, ref_ts - window, ref_ts)
                    out_Bps = _rrd_fetch_avg(rrd, 1, ref_ts - window, ref_ts)
                    traffic_in_bps  += in_Bps  * 8
                    traffic_out_bps += out_Bps * 8
                except Exception:
                    pass

        if power_W > 0 or traffic_in_bps > 0:
            result[dev_name] = dict(
                power_W=power_W,
                traffic_in_bps=traffic_in_bps,
                traffic_out_bps=traffic_out_bps,
            )
            print(f"  {dev_name}: power={power_W:.0f}W  "
                  f"in={traffic_in_bps/1e9:.2f}Gbps  "
                  f"out={traffic_out_bps/1e9:.2f}Gbps")

    return result


# ---------------------------------------------------------------------------
# Backbone traffic loader  (bb-usage-logs)
# ---------------------------------------------------------------------------
# EdgeUtil keys: util_pct, avg_mbps, max_mbps, cap_mbps
# Keyed by frozenset of the two city codes, e.g. frozenset({'be','fr'})
EdgeUtil = dict


def _parse_bb_file(path: pathlib.Path, max_mbps_sanity: float = 1e6) -> tuple[float, float]:
    """Return (mean_Mbps, max_Mbps) for max(in, out) across all 5-min samples."""
    vals: list[float] = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            try:
                iv, ov = float(p[1]), float(p[3])
                v = max(iv, ov)
                if 0 < v <= max_mbps_sanity:
                    vals.append(v)
            except ValueError:
                pass
    if not vals:
        return 0.0, 0.0
    return sum(vals) / len(vals), max(vals)


def _pop_code(node_id: str) -> str:
    """node_id like 'swibe1' → city code 'be'."""
    s = node_id[3:] if node_id.startswith("swi") else node_id
    return re.sub(r"\d+$", "", s).lower()


def load_bb_traffic(bb_dir: pathlib.Path, topo_txt: pathlib.Path) -> EdgeUtil:
    """
    Load the most-recent monthly file for every bb-usage-log link pair.
    Returns a dict keyed by frozenset({code1, code2}) with utilisation stats.
    Also attaches 'node_pairs': list of (u, v) topology node tuples that match.
    """
    if not bb_dir.is_dir():
        return {}

    # Build topology edge capacity index  {frozenset({c1,c2}): cap_bps}
    PAT = re.compile(r"^(\S+)\s+\S+\s+<=>\s+(\S+)\s+\S+\s+\(([^)]+)\)")
    edge_cap: dict[frozenset, float] = {}
    edge_nodes: dict[frozenset, list[tuple[str, str]]] = {}
    with open(topo_txt) as fh:
        for line in fh:
            m = PAT.match(line.strip())
            if not m:
                continue
            u, v, bw = m.group(1), m.group(2), float(m.group(3))
            key: frozenset = frozenset({_pop_code(u), _pop_code(v)})
            if bw > edge_cap.get(key, 0):
                edge_cap[key] = bw
            edge_nodes.setdefault(key, []).append((u, v))

    # Load latest file per link pair
    pairs = sorted(set(f.rsplit(".", 1)[0] for f in os.listdir(bb_dir)))
    result: EdgeUtil = {}

    for pair in pairs:
        parts = pair.split("-")
        c1, c2 = parts[0].lower(), parts[1].lower()
        key = frozenset({c1, c2})
        cap_bps = edge_cap.get(key)
        if not cap_bps:
            continue  # no matching topology edge

        files = sorted(f for f in os.listdir(bb_dir) if f.startswith(pair + "."))
        if not files:
            continue
        avg_mbps, max_mbps = _parse_bb_file(bb_dir / files[-1])
        if max_mbps == 0:
            continue

        cap_mbps = cap_bps / 1e6
        util_pct = max_mbps / cap_mbps * 100.0

        # Merge with any existing entry (keep whichever has higher util)
        existing = result.get(key)
        if existing is None or util_pct > existing["util_pct"]:
            result[key] = dict(
                util_pct=util_pct,
                avg_mbps=avg_mbps,
                max_mbps=max_mbps,
                cap_mbps=cap_mbps,
                pair=pair,
                node_pairs=edge_nodes.get(key, []),
            )

    for key, v in sorted(result.items(), key=lambda x: -x[1]["util_pct"]):
        print(f"  {v['pair']:<15}  util={v['util_pct']:5.1f}%  "
              f"avg={v['avg_mbps']:.0f}  max={v['max_mbps']:.0f}  "
              f"cap={v['cap_mbps']/1000:.0f}G Mbps")

    return result


def _util_color(util_pct: float) -> str:
    """Return a hex colour for a given utilisation percentage."""
    if util_pct < 10:   return "#1a9850"   # dark green
    if util_pct < 25:   return "#91cf60"   # light green
    if util_pct < 50:   return "#fee08b"   # yellow
    if util_pct < 75:   return "#fc8d59"   # orange
    return "#d73027"                        # red


def _fmt_bps(bps: float) -> str:
    if bps >= 1e9:
        return f"{bps/1e9:.1f}G"
    if bps >= 1e6:
        return f"{bps/1e6:.0f}M"
    return f"{bps/1e3:.0f}k"


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------
def draw(G, pos, backbone_nodes: set,
         clusters: dict | None = None,
         edge_util: EdgeUtil | None = None):
    fig, ax = plt.subplots(figsize=(28, 18))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    eu = edge_util or {}
    LS_MAP = {"solid": "-", "dashed": "--", "dotted": ":"}

    # =======================================================================
    # BLOCKS — one rounded box per site (backbone node + its access cluster).
    # Drawn first so they sit behind edges and nodes.
    # =======================================================================
    BLOCK_PAD  = 0.055   # padding around the tightest bounding box
    BLOCK_FC   = "#f0f4f8"   # light blue-grey fill
    BLOCK_EC   = "#9aabbb"   # muted blue-grey border

    if clusters:
        for bb_node, members in clusters.items():
            pts = [pos[n] for n in members if n in pos]
            if not pts:
                continue
            xs_b = [p[0] for p in pts]
            ys_b = [p[1] for p in pts]
            x0b = min(xs_b) - BLOCK_PAD
            y0b = min(ys_b) - BLOCK_PAD
            w    = max(xs_b) - min(xs_b) + 2 * BLOCK_PAD
            h    = max(ys_b) - min(ys_b) + 2 * BLOCK_PAD
            # ensure a minimum size even for single-node clusters
            w = max(w, BLOCK_PAD * 3)
            h = max(h, BLOCK_PAD * 3)
            ax.add_patch(mpatch.FancyBboxPatch(
                (x0b, y0b), w, h,
                boxstyle="round,pad=0.012",
                facecolor=BLOCK_FC, edgecolor=BLOCK_EC,
                linewidth=1.2, zorder=1,
            ))
            # site label — short backbone-node name, top-left of block
            label = short_label(bb_node)
            ax.text(x0b + 0.008, y0b + h - 0.006, label,
                    fontsize=7, fontweight="bold",
                    color="#4a6278", va="top", ha="left", zorder=2)

    # =======================================================================
    # EDGES — thinnest first so thick backbone lines render on top.
    # Backbone↔backbone links use a quadratic Bézier curve so parallel
    # paths through the core fan apart and are easier to follow.
    # Access links stay straight (they are short and local).
    # =======================================================================
    bw_order = sorted(G.edges(data=True), key=lambda e: e[2]["bw"])

    for u, v, data in bw_order:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        lw, default_color, ls, _ = edge_style(data["bw"])

        ekey = frozenset({_pop_code(u), _pop_code(v)})
        util_info = eu.get(ekey)
        # When traffic data exists use utilisation colour; otherwise
        # fall back to the per-bandwidth-tier default (black for backbone,
        # blue for 10 GE) — matching the reference diagram style.
        color = _util_color(util_info["util_pct"]) if util_info else default_color

        ax.plot([x0, x1], [y0, y1],
                color=color, linewidth=lw, linestyle=LS_MAP[ls],
                solid_capstyle="round", zorder=2)

    # =======================================================================
    # NODES — backbone: larger circle + bold label; access: smaller + normal
    # =======================================================================
    BB_R  = 0.030   # backbone node radius
    ACC_R = 0.018   # access node radius

    # draw access first so backbone circles sit on top
    for node in sorted(G.nodes(), key=lambda n: n not in backbone_nodes):
        x, y = pos[node]
        is_bb = node in backbone_nodes
        r   = BB_R  if is_bb else ACC_R
        lw  = 2.0   if is_bb else 0.8
        fs  = 7.5   if is_bb else 5.5
        fw  = "bold" if is_bb else "normal"

        ax.add_patch(Circle((x, y), r,
                            facecolor="white", edgecolor="#222222",
                            linewidth=lw, zorder=4))
        ax.text(x, y + r + 0.008, short_label(node),
                ha="center", va="bottom",
                fontsize=fs, fontweight=fw,
                color="#111111", zorder=5)

    # =======================================================================
    # UTILISATION LABELS — only on links that have monitoring data
    # =======================================================================
    for u, v, data in G.edges(data=True):
        ekey = frozenset({_pop_code(u), _pop_code(v)})
        util_info = eu.get(ekey)
        if not util_info:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        cap_g  = util_info["cap_mbps"] / 1000
        avg_g  = util_info["avg_mbps"] / 1000
        label  = f"{util_info['util_pct']:.0f}%\n({avg_g:.1f}/{cap_g:.0f} G)"
        ax.text(mx, my, label,
                fontsize=4.8, ha="center", va="center",
                color="#222222", zorder=6, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.80))

    # =======================================================================
    # LEGENDS
    # =======================================================================

    # --- 1. Node type -------------------------------------------------------
    node_items = [
        mlines.Line2D([], [], marker="o", markersize=10,
                      markerfacecolor="white", markeredgecolor="#222222",
                      markeredgewidth=2.0, linestyle="none",
                      label=f"Backbone router  ({len(backbone_nodes)} nodes)"),
        mlines.Line2D([], [], marker="o", markersize=6,
                      markerfacecolor="white", markeredgecolor="#222222",
                      markeredgewidth=0.8, linestyle="none",
                      label=f"Access router  ({len(G.nodes())-len(backbone_nodes)} nodes)"),
    ]
    node_legend = ax.legend(
        handles=node_items,
        title="Node type",
        title_fontsize=8,
        loc="upper left",
        fontsize=7.5,
        frameon=True, framealpha=0.95, edgecolor="#aaaaaa",
    )
    ax.add_artist(node_legend)

    # --- 2. Link capacity (width) -------------------------------------------
    cap_items = []
    seen_cap: set[str] = set()
    for _, lw, _, ls, label in BW_STYLE:
        if label in seen_cap:
            continue
        seen_cap.add(label)
        cap_items.append(
            mlines.Line2D([], [], color="#555555", linewidth=lw,
                          linestyle=LS_MAP[ls], label=label)
        )
    cap_legend = ax.legend(
        handles=cap_items,
        title="Line width = Link capacity",
        title_fontsize=8,
        loc="lower left",
        fontsize=7.5,
        frameon=True, framealpha=0.95, edgecolor="#aaaaaa",
    )
    ax.add_artist(cap_legend)

    # --- 3. Utilisation colour ----------------------------------------------
    util_bands = [
        ("<10 %  — low",           "#1a9850"),
        ("10–25 %",                "#91cf60"),
        ("25–50 %  — moderate",    "#fee08b"),
        ("50–75 %  — high",        "#fc8d59"),
        (">75 %  — critical",      "#d73027"),
        ("No data — black/blue default", "#555555"),
    ]
    util_items = [
        mpatch.Patch(facecolor=clr, edgecolor="#666", linewidth=0.5, label=lbl)
        for lbl, clr in util_bands
    ]
    ax.legend(
        handles=util_items,
        title="Line colour = Peak utilisation (Aug 2023)",
        title_fontsize=8,
        loc="lower right",
        fontsize=7.5,
        frameon=True, framealpha=0.95, edgecolor="#aaaaaa",
    )

    # =======================================================================
    # DATA-SOURCE ANNOTATION
    # =======================================================================
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad_x, pad_y = 0.30, 0.22
    x_min = min(xs) - pad_x
    y_min = min(ys) - pad_y

    ax.text(
        x_min + 0.02, y_min + 0.01,
        ("Traffic: SWITCH bb-usage-logs (Aug 2023)  ·  "
         "Topology: switch-network-topology.json\n"
         "Layout: Kamada-Kawai backbone (25 nodes) + radial-tree access clusters  ·  "
         "Utilisation = peak 5-min sample ÷ link capacity"),
        fontsize=5.5, color="#666666", va="bottom", linespacing=1.5,
    )

    # =======================================================================
    # AXIS / TITLE / SAVE
    # =======================================================================
    ax.set_xlim(x_min, max(xs) + pad_x)
    ax.set_ylim(y_min, max(ys) + pad_y)
    ax.axis("off")
    ax.set_title(
        "SWITCH Backbone Network — Capacity & Traffic Utilisation  (Aug 2023)",
        fontsize=14, fontweight="bold", pad=14,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
    fig.savefig(str(OUT_SVG), format="svg", bbox_inches="tight")
    fig.savefig(str(OUT_PNG), format="png", dpi=220, bbox_inches="tight")
    print(f"Saved → {OUT_SVG}")
    print(f"Saved → {OUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    edges = parse_txt(TXT_FILE)
    G, pos, backbone_nodes, clusters = build_and_layout(edges)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges  "
          f"({len(backbone_nodes)} backbone, "
          f"{G.number_of_nodes()-len(backbone_nodes)} access)")

    print(f"Loading backbone traffic from {BB_DIR} …")
    edge_util = load_bb_traffic(BB_DIR, TXT_FILE) if BB_DIR.is_dir() else {}
    if not edge_util:
        print("  (no bb-usage-log data found)")

    draw(G, pos, backbone_nodes, clusters, edge_util)


if __name__ == "__main__":
    main()
