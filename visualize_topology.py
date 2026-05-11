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
NO_DATA_COLOR = "#bbbbbb"   # gray used for ALL links that have no utilisation data

BW_STYLE = [
    # (min_bw,  linewidth, linestyle, legend_label)
    # Color is intentionally absent — color encodes utilisation, NOT capacity.
    (4e11,   7.0,  "solid",  "400 GE"),
    (2e11,   5.0,  "solid",  "200 GE"),
    (1e11,   3.2,  "solid",  "100 GE"),
    (2e10,   2.0,  "solid",  "20 GE"),
    (1e10,   1.2,  "solid",  "10 GE"),
    (1e9,    0.8,  "dashed", "1 GE"),
    (0,      0.6,  "dotted", "<1 GE"),
]

def edge_style(bw):
    for min_bw, lw, ls, label in BW_STYLE:
        if bw >= min_bw:
            return lw, ls, label
    return 0.6, "dotted", "<1 GE"

def short_label(node_id: str) -> str:
    s = node_id
    if s.startswith("swi"):
        s = s[3:]
    return s


# ---------------------------------------------------------------------------
# Build graph + assign geographic positions
# ---------------------------------------------------------------------------
def build_and_layout(edges):
    G = nx.Graph()
    for e in edges:
        # keep max bandwidth if parallel edges exist
        u, v = e["src"], e["dst"]
        if G.has_edge(u, v):
            if e["bw"] > G[u][v]["bw"]:
                G[u][v].update(bw=e["bw"], src_if=e["src_if"], dst_if=e["dst_if"],
                               metric1=e["metric1"], metric2=e["metric2"])
        else:
            G.add_edge(u, v, **{k: e[k] for k in ("bw","src_if","dst_if","metric1","metric2")})

    # ------------------------------------------------------------------
    # Layout strategy: Kamada-Kawai minimises a stress function that
    # makes Euclidean distances proportional to graph-theoretic
    # shortest-path distances.  This naturally separates nodes that are
    # topologically far apart and pulls connected clusters together,
    # producing far fewer edge crossings than a pure geographic or
    # spring layout.
    #
    # We seed KK with geographic positions (scaled to [-1,1]) so the
    # result stays roughly oriented like a Swiss map while still being
    # free to optimise for readability.
    # ------------------------------------------------------------------

    # Build geographic seed positions and normalise to [-1, 1]
    geo = {}
    for node in G.nodes():
        geo[node] = geo_pos(node)

    xs = [p[0] for p in geo.values()]
    ys = [p[1] for p in geo.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_rng = x_max - x_min or 1.0
    y_rng = y_max - y_min or 1.0

    seed_pos = {
        n: (2 * (geo[n][0] - x_min) / x_rng - 1,
            2 * (geo[n][1] - y_min) / y_rng - 1)
        for n in G.nodes()
    }

    print("Running Kamada-Kawai layout …")
    pos = nx.kamada_kawai_layout(G, pos=seed_pos, weight=None)

    return G, pos


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
def draw(G, pos, edge_util: EdgeUtil | None = None):
    fig, ax = plt.subplots(figsize=(26, 16))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # --- Edges (draw thinner first so thick ones appear on top) ---
    eu = edge_util or {}
    bw_order = sorted(G.edges(data=True), key=lambda e: e[2]["bw"])

    for u, v, data in bw_order:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        lw, ls, _ = edge_style(data["bw"])
        mls = {"solid": "-", "dashed": "--", "dotted": ":"}[ls]

        # look up utilisation by PoP-code pair
        ekey = frozenset({_pop_code(u), _pop_code(v)})
        util_info = eu.get(ekey)
        # Color encodes utilisation ONLY; gray for any link with no data
        color = _util_color(util_info["util_pct"]) if util_info else NO_DATA_COLOR

        ax.plot([x0, x1], [y0, y1],
                color=color, linewidth=lw, linestyle=mls,
                solid_capstyle="round", zorder=2)

    # --- Nodes (uniform style, label above) --------------------------------
    NODE_R = 0.018
    for node in G.nodes():
        x, y = pos[node]
        ax.add_patch(Circle((x, y), NODE_R,
                            facecolor="white", edgecolor="#333333",
                            linewidth=1.0, zorder=4))
        ax.text(x, y + NODE_R + 0.004, short_label(node),
                ha="center", va="bottom",
                fontsize=6.0, color="#111111", zorder=5)

    # --- Utilisation % label on each coloured link -------------------------
    for u, v, data in G.edges(data=True):
        ekey = frozenset({_pop_code(u), _pop_code(v)})
        util_info = eu.get(ekey)
        if not util_info:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, f"{util_info['util_pct']:.0f}%",
                fontsize=4.5, ha="center", va="center", color="#111111",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec="none", alpha=0.75))

    # =======================================================================
    # LEGEND  (two separate boxes, placed outside the map area)
    # =======================================================================

    # -- 1. Capacity legend (line width) ------------------------------------
    cap_items = []
    seen_cap: set[str] = set()
    for _, lw, ls, label in BW_STYLE:
        if label in seen_cap:
            continue
        seen_cap.add(label)
        cap_items.append(
            mlines.Line2D([], [], color="#555555", linewidth=lw,
                          linestyle={"solid": "-", "dashed": "--",
                                     "dotted": ":"}[ls],
                          label=label)
        )
    cap_legend = ax.legend(
        handles=cap_items,
        title="Line width = Link capacity",
        title_fontsize=7.5,
        loc="lower left",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        edgecolor="#aaaaaa",
    )
    ax.add_artist(cap_legend)   # keep it when we add the second legend

    # -- 2. Utilisation colour legend  --------------------------------------
    util_bands = [
        ("< 10 %  (low)",       "#1a9850"),
        ("10 – 25 %",           "#91cf60"),
        ("25 – 50 %  (moderate)","#fee08b"),
        ("50 – 75 %  (high)",   "#fc8d59"),
        ("> 75 %  (critical)",  "#d73027"),
        ("No data",             "#aaaaaa"),
    ]
    util_items = [
        mpatch.Patch(facecolor=clr, edgecolor="#666666", linewidth=0.6, label=lbl)
        for lbl, clr in util_bands
    ]
    ax.legend(
        handles=util_items,
        title="Link colour = Utilisation (max in month)",
        title_fontsize=7.5,
        loc="upper left",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        edgecolor="#aaaaaa",
    )

    # =======================================================================
    # DATA SOURCE NOTE  (bottom-left corner)
    # =======================================================================
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, y_min = min(xs) - 0.25, min(ys) - 0.18

    latest_month = "Aug 2023"   # most-recent bb-usage-log file
    ax.text(
        x_min + 0.02, y_min + 0.02,
        (f"Traffic data: SWITCH bb-usage-logs, latest month ({latest_month})\n"
         f"Topology: SWITCH backbone ({len(G.nodes())} routers, "
         f"{len(G.edges())} links)  ·  Layout: Swiss geographic coordinates\n"
         f"Utilisation = peak 5-min interval over the month / link capacity"),
        fontsize=5.5, color="#555555", va="bottom", ha="left",
        linespacing=1.5, zorder=10,
    )

    # =======================================================================
    # AXIS / TITLE / SAVE
    # =======================================================================
    ax.set_xlim(x_min, max(xs) + 0.25)
    ax.set_ylim(y_min, max(ys) + 0.20)
    ax.axis("off")
    ax.set_title(
        "SWITCH Backbone Network — Link Capacity & Traffic Utilisation",
        fontsize=13, fontweight="bold", pad=12,
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
    G, pos = build_and_layout(edges)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"Loading backbone traffic from {BB_DIR} …")
    edge_util = load_bb_traffic(BB_DIR, TXT_FILE) if BB_DIR.is_dir() else {}
    if not edge_util:
        print("  (no bb-usage-log data found)")

    draw(G, pos, edge_util)


if __name__ == "__main__":
    main()
