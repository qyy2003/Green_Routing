#!/usr/bin/env python3
"""
Interactive Dash web app: SWITCH backbone network with 5-minute traffic timeline.

  python3.10 web_topology.py
  open http://localhost:8050

Key design:
  • Figure sent to browser once; only edge line colours are patched per tick.
  • All 45 bb-usage-log link pairs are shown:
      – 24 pairs have direct router-level edges in the topology → coloured normally.
      – ~21 pairs are city-level backbone circuits with no direct L3 edge → drawn as
        dashed virtual edges connecting the two city anchor-node positions.
  • Capacity for virtual edges is inferred as min(max-adjacent-bw) of the two cities.
  • Sanity filter (≤1 000 000 Mbps) removes corrupted samples (e.g. CE-LS raw Bytes/s
    values mislabelled as Mbps).
  • Values shown are instantaneous 5-minute samples; the SVG showed peak-of-month,
    so a slider control lets you add a rolling-max window (raw / 1 h / 6 h).
"""

import re, os, math, pathlib, random, collections
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import networkx as nx
import dash
from dash import dcc, html, Input, Output, State, Patch
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = pathlib.Path("../network-energy-efficiency-research/switch-network-topology")
TXT_FILE = BASE / "switch-network-topology.txt"
BB_DIR   = pathlib.Path("../network-energy-efficiency-research/lan-mon2021/bb-usage-logs")

# ---------------------------------------------------------------------------
# Geographic coordinates
# ---------------------------------------------------------------------------
CITY_COORDS = {
    "ag":    (8.05, 47.39), "avp":   (6.88, 46.82), "ba":    (7.59, 47.56),
    "bf":    (7.47, 46.96), "bi":    (7.24, 47.14), "bis":   (9.25, 47.50),
    "blu":   (9.53, 47.15), "br":    (8.21, 47.53), "bs":    (7.59, 47.56),
    "bu":    (7.68, 47.06), "bul":   (8.31, 47.07), "bz":    (9.02, 46.19),
    "car":   (6.14, 46.18), "caw":   (9.03, 46.20), "ce":    (6.13, 46.20),
    "cma":   (8.97, 46.02), "cmb":   (8.98, 46.02), "cr":    (6.15, 46.21),
    "cvb":   (7.39, 46.95), "cvl":   (7.40, 46.95), "del":   (7.34, 47.37),
    "dgd":   (9.40, 47.42), "due":   (7.18, 46.80), "duf":   (7.87, 46.47),
    "dv":    (8.22, 47.40), "eb":    (8.33, 47.08), "ehblg": (8.17, 47.40),
    "ehbzo": (8.32, 47.41), "ehl":   (6.24, 46.22), "eic":   (9.50, 47.47),
    "el":    (9.10, 47.10), "ensi":  (7.57, 47.54), "ez":    (8.57, 47.36),
    "ff":    (8.90, 47.56), "fga":   (9.36, 47.42), "fhsg":  (9.38, 47.43),
    "fr":    (7.15, 46.80), "gbl":   (6.18, 46.22), "ge":    (6.15, 46.20),
    "gl":    (9.07, 47.04), "gla":   (9.06, 47.03), "gno":   (6.16, 46.24),
    "go":    (9.39, 47.44), "gov":   (6.19, 46.22), "gr":    (9.53, 46.85),
    "gsa":   (9.37, 47.41), "gsb":   (9.38, 47.41), "gva":   (6.12, 46.24),
    "gvb":   (6.12, 46.23), "hepvd": (6.64, 46.78), "hfh":   (8.68, 47.38),
    "ho":    (8.80, 46.12), "hug":   (6.14, 46.19), "ibm":   (8.50, 47.31),
    "imd":   (6.57, 46.52), "itis":  (8.72, 47.50), "ix":    (8.53, 47.37),
    "ixg":   (6.16, 46.23), "jfj":   (7.99, 46.55), "kas":   (8.90, 47.10),
    "kl":    (8.58, 47.45), "kr":    (9.18, 47.65), "lab":   (8.54, 47.38),
    "lg":    (8.95, 46.00), "lgd":   (8.93, 46.01), "li":    (9.56, 47.14),
    "lio":   (8.96, 46.01), "lo":    (8.80, 46.17), "ls":    (6.63, 46.52),
    "lsm":   (6.62, 46.52), "lz":    (8.94, 46.02), "maa":   (7.07, 46.10),
    "mc":    (8.93, 46.03), "mdlp":  (6.63, 46.53), "metas": (7.47, 46.92),
    "my":    (6.94, 46.25), "naz":   (8.61, 47.36), "ne":    (6.93, 47.00),
    "nea":   (6.91, 47.01), "neb":   (6.93, 47.01), "npa":   (8.54, 47.38),
    "npb":   (8.55, 47.38), "nyr":   (9.37, 47.42), "ol":    (7.91, 47.35),
    "ps":    (8.52, 47.37), "ra":    (7.53, 46.25), "rot":   (8.45, 47.14),
    "sa":    (7.36, 46.24), "sg":    (9.37, 47.42), "sgd":   (8.63, 47.70),
    "sge":   (9.38, 47.44), "sgm":   (9.39, 47.43), "si":    (7.36, 46.23),
    "snf":   (8.64, 47.70), "snl":   (8.36, 47.21), "snm":   (8.62, 47.71),
    "sno":   (8.61, 47.72), "soa":   (7.37, 46.25), "sob":   (7.38, 46.25),
    "sp":    (7.68, 46.69), "svf":   (7.50, 47.02), "svg":   (7.50, 47.03),
    "thu":   (7.63, 46.75), "to":    (6.62, 46.51), "ukb":   (8.59, 47.37),
    "vi":    (7.88, 46.29), "wsl":   (8.46, 47.36), "yv":    (6.64, 46.78),
    "zar":   (9.95, 46.70), "zbe":   (7.45, 46.95), "zbu":   (8.53, 47.37),
    "zgl":   (8.52, 47.17), "zh":    (8.55, 47.37), "zob":   (8.54, 47.37),
    "zop":   (8.55, 47.38),
}
LON_SCALE = math.cos(math.radians(46.9))

def _project(lon, lat):
    return lon * LON_SCALE, lat

_city_count: dict = {}
JITTER = [(0.00,0.00),(0.12,0.00),(-0.12,0.00),(0.00,0.07),
          (0.12,0.07),(-0.12,0.07),(0.06,-0.07),(-0.06,-0.07)]

def _city_code(node_id):
    s = node_id[3:] if node_id.startswith("swi") else node_id
    s = re.sub(r"\d+$", "", s)
    return re.sub(r"-.*$", "", s).lower()

def _geo_pos(node_id):
    code = _city_code(node_id)
    base = CITY_COORDS.get(code)
    if base is None:
        rng = random.Random(hash(node_id)); base = (rng.uniform(6.1,10.4), rng.uniform(45.8,47.8))
    idx = _city_count.get(code, 0); _city_count[code] = idx+1
    jx, jy = JITTER[idx % len(JITTER)]
    return _project(base[0]+jx, base[1]+jy)

# ---------------------------------------------------------------------------
# Topology parsing
# ---------------------------------------------------------------------------
EDGE_PAT = re.compile(r"^(\S+)\s+(\S+)\s+<=>\s+(\S+)\s+(\S+)\s+\(([^)]+)\)\s+\((\S+)\s+(\S+)\)")

def parse_txt(path):
    edges = []
    with open(path) as fh:
        for line in fh:
            m = EDGE_PAT.match(line.strip())
            if m:
                src,src_if,dst,dst_if,bw_s,m1,m2 = m.groups()
                edges.append(dict(src=src,src_if=src_if,dst=dst,dst_if=dst_if,
                                  bw=float(bw_s),metric1=int(m1),metric2=int(m2)))
    return edges

# ---------------------------------------------------------------------------
# Graph layout  (same algorithm as visualize_topology.py)
# ---------------------------------------------------------------------------
def build_and_layout(edges):
    G = nx.Graph()
    for e in edges:
        u, v = e["src"], e["dst"]
        if G.has_edge(u, v):
            if e["bw"] > G[u][v]["bw"]: G[u][v].update(bw=e["bw"])
        else:
            G.add_edge(u, v, **{k: e[k] for k in ("bw","src_if","dst_if","metric1","metric2")})

    BACKBONE_BW, MIN_BB_LINKS = 1e11, 3
    bb_cnt: dict = defaultdict(int)
    for u, v, d in G.edges(data=True):
        if d["bw"] >= BACKBONE_BW: bb_cnt[u]+=1; bb_cnt[v]+=1
    backbone_nodes = {n for n,c in bb_cnt.items() if c >= MIN_BB_LINKS}

    geo_raw = {n: _geo_pos(n) for n in backbone_nodes}
    xs_g=[p[0] for p in geo_raw.values()]; ys_g=[p[1] for p in geo_raw.values()]
    xr=max(xs_g)-min(xs_g) or 1.0; yr=max(ys_g)-min(ys_g) or 1.0
    geo_seed={n:(2*(geo_raw[n][0]-min(xs_g))/xr-1, 2*(geo_raw[n][1]-min(ys_g))/yr-1)
              for n in backbone_nodes}
    pos_b=nx.kamada_kawai_layout(G.subgraph(backbone_nodes).copy(), pos=geo_seed, weight=None)
    BB_SCALE=2.2; pos={n:(x*BB_SCALE,y*BB_SCALE) for n,(x,y) in pos_b.items()}

    access_nodes=set(G.nodes())-backbone_nodes
    backbone_parent={}
    for start in access_nodes:
        q=collections.deque([start]); vis={start}; found=None
        while q:
            node=q.popleft()
            if node in backbone_nodes: found=node; break
            for nb in G.neighbors(node):
                if nb not in vis: vis.add(nb); q.append(nb)
        backbone_parent[start]=found

    groups=defaultdict(list)
    for node,parent in backbone_parent.items(): groups[parent].append(node)
    cx=sum(pos[n][0] for n in backbone_nodes)/len(backbone_nodes)
    cy=sum(pos[n][1] for n in backbone_nodes)/len(backbone_nodes)

    def _bfs_ch(root,cluster):
        ch={root:[]}; q=collections.deque([root])
        while q:
            node=q.popleft()
            for nb in G.neighbors(node):
                if nb in cluster and nb not in ch: ch[node].append(nb); ch[nb]=[]; q.append(nb)
        return ch
    def _sz(n,ch): return 1+sum(_sz(c,ch) for c in ch.get(n,[]))
    def _radial(node,npos,ch,base,span,r):
        ch_=ch.get(node,[])
        if not ch_: return
        sizes={c:_sz(c,ch) for c in ch_}; total=sum(sizes.values()); rx,ry=npos; cum=0.0
        for child in ch_:
            frac=sizes[child]/total; angle=(base-span)+(cum+frac/2)*2*span; cum+=frac
            pos[child]=(rx+r*math.cos(angle), ry+r*math.sin(angle))
            _radial(child,pos[child],ch,angle,max(math.radians(12),span*frac*2.2),r*0.72)

    for bb,group in groups.items():
        if bb is None:
            for i,n in enumerate(group):
                a=2*math.pi*i/max(len(group),1); pos[n]=(1.9*math.cos(a),1.9*math.sin(a))
            continue
        px,py=pos[bb]; dx,dy=px-cx,py-cy; d=(dx*dx+dy*dy)**0.5 or 1.0
        _radial(bb,(px,py),_bfs_ch(bb,set(group)),
                math.atan2(dy/d,dx/d), min(math.radians(130),math.radians(20)*len(group)), 0.09)

    all_nodes=list(G.nodes())
    for _ in range(80):
        for i,n1 in enumerate(all_nodes):
            for n2 in all_nodes[i+1:]:
                x1,y1=pos[n1]; x2,y2=pos[n2]; dx2,dy2=x2-x1,y2-y1
                dist=(dx2*dx2+dy2*dy2)**0.5
                if 1e-9<dist<0.055:
                    push=(0.055-dist)/(2*dist)
                    w1=0.15 if n1 in backbone_nodes else 1.0
                    w2=0.15 if n2 in backbone_nodes else 1.0
                    pos[n1]=(x1-dx2*push*w1,y1-dy2*push*w1)
                    pos[n2]=(x2+dx2*push*w2,y2+dy2*push*w2)

    clusters = {bb: [bb]+groups.get(bb,[]) for bb in backbone_nodes}
    return G, pos, backbone_nodes, clusters

# ---------------------------------------------------------------------------
# Bandwidth / utilisation helpers
# ---------------------------------------------------------------------------
BW_STYLE = [
    (4e11,7.0,"#111111","400 GE"),(2e11,5.0,"#222222","200 GE"),
    (1e11,3.2,"#333333","100 GE"),(2e10,2.0,"#555555"," 20 GE"),
    (1e10,1.2,"#2980b9"," 10 GE"),(1e9, 0.8,"#2980b9","  1 GE"),
    (0,   0.6,"#aaaaaa","<1  GE"),
]
def _edge_style(bw):
    for min_bw,lw,color,label in BW_STYLE:
        if bw>=min_bw: return lw,color,label
    return 0.6,"#aaaaaa","<1 GE"

def _short(n): return n[3:] if n.startswith("swi") else n

def _pop_code(node_id):
    s = node_id[3:] if node_id.startswith("swi") else node_id
    return re.sub(r"\d+$","",s).lower()

def _util_color(p):
    if p<10:  return "#1a9850"
    if p<25:  return "#91cf60"
    if p<50:  return "#fee08b"
    if p<75:  return "#fc8d59"
    return "#d73027"

# ---------------------------------------------------------------------------
# Build link capacity index from topology + infer for unmatched bb-log pairs
# ---------------------------------------------------------------------------
def build_link_caps(topo_txt, bb_dir):
    """
    Returns:
      edge_cap   – {frozenset: bw_bps} from topology (direct edges only)
      link_cap   – edge_cap + inferred capacities for bb-log pairs without direct edges
      bb_pairs   – all frozenset pairs that appear in bb-log filenames
      virt_keys  – frozensets that need virtual edges (in link_cap but not edge_cap)
    """
    PAT = re.compile(r"^(\S+)\s+\S+\s+<=>\s+(\S+)\s+\S+\s+\(([^)]+)\)")
    edge_cap: dict = {}
    with open(topo_txt) as fh:
        for line in fh:
            m = PAT.match(line.strip())
            if m:
                u,v,bw = m.group(1),m.group(2),float(m.group(3))
                key=frozenset({_pop_code(u),_pop_code(v)})
                if bw>edge_cap.get(key,0): edge_cap[key]=bw

    bb_pairs: set = set()
    for fname in os.listdir(bb_dir):
        m2 = re.match(r'^([A-Z]+-[A-Z]+)\.\d{6}$', fname)
        if m2:
            c1,c2 = m2.group(1).split('-')
            bb_pairs.add(frozenset({c1.lower(),c2.lower()}))

    # Per-city max adjacent bw (for inferring virtual edge capacity)
    city_max: dict = {}
    for key,bw in edge_cap.items():
        for c in key: city_max[c]=max(city_max.get(c,0),bw)

    link_cap = dict(edge_cap)
    virt_keys: set = set()
    for key in bb_pairs:
        if key in edge_cap: continue
        c_list=list(key); bw1=city_max.get(c_list[0],0); bw2=city_max.get(c_list[1],0)
        if bw1>0 and bw2>0:
            link_cap[key]=min(bw1,bw2); virt_keys.add(key)
        # else: city code not in topology at all — skip

    return edge_cap, link_cap, bb_pairs, virt_keys


# ---------------------------------------------------------------------------
# 5-minute data loading
# ---------------------------------------------------------------------------
MAX_MBPS_SANITY = 1_000_000   # drop samples above 1 Tbps (corrupted files, e.g. CE-LS)

def load_all_5min(bb_dir, link_cap):
    """
    Parse all bb-usage-log files for pairs that are in link_cap.
    Returns:
      timestamps  – sorted int64 numpy array of Unix timestamps
      link_utils  – {frozenset: float32 array}  util_pct per timestamp (NaN = no data)
    """
    by_key: dict = {}
    for fname in sorted(os.listdir(bb_dir)):
        m = re.match(r'^([A-Z]+-[A-Z]+)\.(\d{6})$', fname)
        if not m: continue
        c1,c2 = m.group(1).split('-')
        key = frozenset({c1.lower(),c2.lower()})
        if key in link_cap:
            by_key.setdefault(key,[]).append(bb_dir/fname)

    print(f"  collecting timestamps from {sum(len(v) for v in by_key.values())} files …", flush=True)
    all_ts: set = set()
    for files in by_key.values():
        for fpath in files:
            with open(fpath) as fh:
                for line in fh:
                    p=line.split()
                    if p:
                        try: all_ts.add(int(p[0]))
                        except ValueError: pass

    timestamps=np.array(sorted(all_ts),dtype=np.int64)
    ts_idx={int(ts):i for i,ts in enumerate(timestamps)}
    N=len(timestamps)
    print(f"  {N:,} unique timestamps  ({len(by_key)} monitored links)", flush=True)

    link_utils: dict = {}
    for key,files in by_key.items():
        cap_mbps=link_cap[key]/1e6
        arr=np.full(N,np.nan,dtype=np.float32)
        for fpath in files:
            with open(fpath) as fh:
                for line in fh:
                    p=line.split()
                    if len(p)<4: continue
                    try:
                        v=max(float(p[1]),float(p[3]))
                        if v<=0 or v>MAX_MBPS_SANITY: continue   # sanity filter
                        i=ts_idx[int(p[0])]
                        val=v/cap_mbps*100.0
                        if np.isnan(arr[i]) or val>arr[i]: arr[i]=val
                    except (ValueError,KeyError): pass
        link_utils[key]=arr
    return timestamps,link_utils


# ---------------------------------------------------------------------------
# Base figure builder  (called once at startup)
# ---------------------------------------------------------------------------
def make_base_figure(G, pos, backbone_nodes, link_cap, edge_cap, virt_keys):
    """
    Trace index layout
    ------------------
    0 … N_CLUSTERS-1          cluster box polygons (one per backbone site)
    N_CLUSTERS                 cluster label text trace
    N_CLUSTERS+1 … +N_TOPO    topology edges
    +N_TOPO+1 … +N_VIRT       virtual edges
    HOVER_IDX                  invisible hover markers
    HOVER_IDX+1                access nodes
    HOVER_IDX+2                backbone nodes
    """
    n_cl = len(CLUSTER_RECTS)
    topo_edges = sorted(G.edges(data=True), key=lambda e: e[2]["bw"])
    traces = []

    # ---- cluster boxes (drawn first so they sit behind everything) ------------
    for _bb, x0c, y0c, x1c, y1c in CLUSTER_RECTS:
        traces.append(go.Scatter(
            x=[x0c, x1c, x1c, x0c, x0c],
            y=[y0c, y0c, y1c, y1c, y0c],
            mode="lines", fill="toself",
            fillcolor="#f0f4f8",
            line=dict(color="#9aabbb", width=1.2),
            hoverinfo="skip", showlegend=False,
        ))

    # ---- cluster labels (one combined text trace at index n_cl) ---------------
    traces.append(go.Scatter(
        x=[x0c+0.012 for _,x0c,_,_,y1c in CLUSTER_RECTS],
        y=[y1c-0.012 for _,_,_,_,y1c in CLUSTER_RECTS],
        mode="text",
        text=[f"<b>{_short(bb)}</b>" for bb,*_ in CLUSTER_RECTS],
        textposition="bottom right",
        textfont=dict(size=9, color="#4a6278"),
        hoverinfo="skip", showlegend=False,
    ))

    # ---- topology edges -------------------------------------------------------
    for u, v, data in topo_edges:
        x0,y0=pos[u]; x1,y1=pos[v]
        lw,default_color,_ = _edge_style(data["bw"])
        traces.append(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",
                                 line=dict(color=default_color,width=lw),
                                 hoverinfo="skip",showlegend=False))
    N_TOPO = len(topo_edges)
    TOPO_START = n_cl + 1   # cluster boxes (n_cl) + label trace (1)

    # ---- virtual edges (dashed, city-to-city backbone circuits) ----------------
    # Find anchor node per city code (prefer backbone node)
    city_anchor: dict = {}
    for node in G.nodes():
        code = _pop_code(node)
        existing = city_anchor.get(code)
        if existing is None or (node in backbone_nodes and existing not in backbone_nodes):
            city_anchor[code] = node
    city_pos = {code: pos[node] for code,node in city_anchor.items()}

    virt_edges = []
    for key in sorted(virt_keys, key=str):
        c_list = list(key)
        c1,c2 = c_list[0],c_list[1]
        if c1 not in city_pos or c2 not in city_pos: continue
        x0,y0=city_pos[c1]; x1,y1=city_pos[c2]
        virt_edges.append((c1,c2,x0,y0,x1,y1,key))
        traces.append(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",
                                 line=dict(color="#bbbbbb",width=1.5,dash="dot"),
                                 hoverinfo="skip",showlegend=False))
    N_VIRT = len(virt_edges)

    # ---- hover markers at all edge midpoints ----------------------------------
    hover_x,hover_y,hover_text=[],[],[]
    for u,v,data in topo_edges:
        x0,y0=pos[u]; x1,y1=pos[v]
        hover_x.append((x0+x1)/2); hover_y.append((y0+y1)/2)
        _,_,cap_label=_edge_style(data["bw"])
        hover_text.append(f"<b>{_short(u)} ↔ {_short(v)}</b><br>Cap: {cap_label}")
    for c1,c2,x0,y0,x1,y1,key in virt_edges:
        hover_x.append((x0+x1)/2); hover_y.append((y0+y1)/2)
        cap_g=link_cap[key]/1e9
        hover_text.append(f"<b>{c1.upper()} ↔ {c2.upper()}</b>"
                          f"<br>Cap: {cap_g:.0f} Gbps (inferred)<br><i>virtual edge</i>")
    HOVER_IDX = TOPO_START + N_TOPO + N_VIRT
    traces.append(go.Scatter(x=hover_x,y=hover_y,mode="markers",
                             marker=dict(size=12,color="rgba(0,0,0,0)"),
                             text=hover_text,
                             hovertemplate="%{text}<extra></extra>",
                             showlegend=False))

    # ---- access nodes (static) ------------------------------------------------
    acc = [n for n in G.nodes() if n not in backbone_nodes]
    traces.append(go.Scatter(
        x=[pos[n][0] for n in acc], y=[pos[n][1] for n in acc],
        mode="markers+text",
        marker=dict(size=8,color="white",line=dict(color="#888",width=0.8)),
        text=[_short(n) for n in acc], textposition="top center",
        textfont=dict(size=8,color="#555"),
        hovertemplate="<b>%{text}</b> (access)<extra></extra>",
        name="Access router",
    ))
    # ---- backbone nodes (static) ----------------------------------------------
    bb_list=list(backbone_nodes)
    traces.append(go.Scatter(
        x=[pos[n][0] for n in bb_list], y=[pos[n][1] for n in bb_list],
        mode="markers+text",
        marker=dict(size=16,color="white",line=dict(color="#222",width=2.2)),
        text=[_short(n) for n in bb_list], textposition="top center",
        textfont=dict(size=12,color="#111",family="Arial Black"),
        hovertemplate="<b>%{text}</b> (backbone)<extra></extra>",
        name="Backbone router",
    ))

    xs=[p[0] for p in pos.values()]; ys=[p[1] for p in pos.values()]
    fig=go.Figure(data=traces)
    fig.update_layout(
        uirevision="static",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[min(xs)-0.30,max(xs)+0.30],visible=False),
        yaxis=dict(range=[min(ys)-0.25,max(ys)+0.25],visible=False,
                   scaleanchor="x",scaleratio=1),
        margin=dict(l=4,r=4,t=4,b=4),
        legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#ccc",borderwidth=1,font=dict(size=11)),
        hovermode="closest",
        dragmode="pan",
        modebar=dict(orientation="v",
                     add=["zoom","pan","zoomin","zoomout","autoscale","resetscale"]),
    )
    return fig, topo_edges, virt_edges, HOVER_IDX, acc, bb_list


# ---------------------------------------------------------------------------
# Hover-text helper
# ---------------------------------------------------------------------------
def _make_hover(topo_edges, virt_edges, link_utils, link_cap, ts_idx, window, agg="peak"):
    texts = []
    for u, v, data in topo_edges:
        _,_,cap_label = _edge_style(data["bw"])
        tip = f"<b>{_short(u)} ↔ {_short(v)}</b><br>Cap: {cap_label}"
        ekey = frozenset({_pop_code(u),_pop_code(v)})
        arr = link_utils.get(ekey)
        util = _window_val(arr, ts_idx, window, agg) if arr is not None else np.nan
        if not np.isnan(util):
            mbps = util * link_cap.get(ekey,1e11) / 1e6 / 100
            tip += f"<br>Util ({agg}): {util:.1f}%  ({mbps:.0f} Mbps)"
        texts.append(tip)
    for c1,c2,x0,y0,x1,y1,key in virt_edges:
        cap_g = link_cap[key]/1e9
        tip = f"<b>{c1.upper()} ↔ {c2.upper()}</b><br>Cap: {cap_g:.0f} Gbps (inferred)"
        arr = link_utils.get(key)
        util = _window_val(arr, ts_idx, window, agg) if arr is not None else np.nan
        if not np.isnan(util):
            mbps = util * link_cap[key] / 1e6 / 100
            tip += f"<br>Util ({agg}): {util:.1f}%  ({mbps:.0f} Mbps)"
        texts.append(tip)
    return texts

def _window_val(arr, ts_idx, window, agg="peak"):
    """Return aggregated util over the past `window` samples."""
    if arr is None: return np.nan
    start = max(0, ts_idx - window + 1)
    chunk = arr[start:ts_idx+1]
    valid = chunk[~np.isnan(chunk)]
    if len(valid) == 0: return np.nan
    if agg == "mean":   return float(np.mean(valid))
    if agg == "median": return float(np.median(valid))
    return float(np.max(valid))   # "peak"


# ---------------------------------------------------------------------------
# Pre-compute at startup
# ---------------------------------------------------------------------------
print("Parsing topology …")
_city_count.clear()
_raw_edges = parse_txt(TXT_FILE)
G, pos, backbone_nodes, CLUSTERS = build_and_layout(_raw_edges)
print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({len(backbone_nodes)} backbone)")

# Pre-compute cluster bounding boxes (unrotated).
# Each entry: (bb_node, x0, y0, x1, y1, label)
_CPAD = 0.060
CLUSTER_RECTS = []
for bb_node in sorted(backbone_nodes):
    members = CLUSTERS.get(bb_node, [bb_node])
    pts = [pos[n] for n in members if n in pos]
    if not pts: continue
    xs_c = [p[0] for p in pts]; ys_c = [p[1] for p in pts]
    x0c = min(xs_c)-_CPAD;  y0c = min(ys_c)-_CPAD
    x1c = max(xs_c)+_CPAD;  y1c = max(ys_c)+_CPAD
    x1c = max(x1c, x0c+_CPAD*3); y1c = max(y1c, y0c+_CPAD*3)
    CLUSTER_RECTS.append((bb_node, x0c, y0c, x1c, y1c))
N_CLUSTERS = len(CLUSTER_RECTS)

print("Building link capacity index …")
EDGE_CAP, LINK_CAP, BB_PAIRS, VIRT_KEYS = build_link_caps(TXT_FILE, BB_DIR)
print(f"  {len(BB_PAIRS)} bb-log pairs: "
      f"{len(BB_PAIRS & set(EDGE_CAP))} direct topology edges + "
      f"{len(VIRT_KEYS)} virtual edges")

print("Loading 5-minute traffic data …")
TIMESTAMPS, LINK_UTILS = load_all_5min(BB_DIR, LINK_CAP)
N_TS = len(TIMESTAMPS)

print("Building base figure …")
BASE_FIG, TOPO_EDGES, VIRT_EDGES, HOVER_IDX, ACC_NODES, BB_LIST = \
    make_base_figure(G, pos, backbone_nodes, LINK_CAP, EDGE_CAP, VIRT_KEYS)
N_TOPO       = len(TOPO_EDGES)
N_VIRT       = len(VIRT_EDGES)
TOPO_START   = N_CLUSTERS + 1          # after N_CLUSTERS box traces + 1 label trace
VIRT_START   = TOPO_START + N_TOPO
LABEL_IDX    = N_CLUSTERS              # cluster label text trace

# Graph centroid (rotation pivot) and default axis padding
_CX = sum(pos[n][0] for n in G.nodes()) / G.number_of_nodes()
_CY = sum(pos[n][1] for n in G.nodes()) / G.number_of_nodes()
_PAD_X, _PAD_Y = 0.30, 0.25

# Slider marks at month boundaries
_mmarks: dict = {}
for i, ts in enumerate(TIMESTAMPS):
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    mk = (dt.year, dt.month)
    if mk not in _mmarks:
        _mmarks[mk] = {"idx": i, "label": dt.strftime("%b\n%y")}
SLIDER_MARKS = {v["idx"]: {"label": v["label"], "style": {"fontSize":"10px","whiteSpace":"pre"}}
                for v in _mmarks.values()}

_t0 = datetime.fromtimestamp(int(TIMESTAMPS[0]),  tz=timezone.utc).strftime('%Y-%m-%d')
_t1 = datetime.fromtimestamp(int(TIMESTAMPS[-1]), tz=timezone.utc).strftime('%Y-%m-%d')
print(f"Ready — {N_TS:,} steps  {_t0} → {_t1}  "
      f"| topology edges: {N_TOPO}  virtual edges: {N_VIRT}")

# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------
def _rxy(x, y, cos_r, sin_r):
    """Rotate point (x,y) around the graph centroid."""
    dx, dy = x - _CX, y - _CY
    return _CX + dx*cos_r - dy*sin_r, _CY + dx*sin_r + dy*cos_r


def _apply_rotation_patch(patched, rotation_deg):
    """Patch all trace x/y coordinates for a new rotation angle (degrees)."""
    rad = math.radians(rotation_deg or 0)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    # Cluster boxes (traces 0..N_CLUSTERS-1)
    for i, (_bb, x0c, y0c, x1c, y1c) in enumerate(CLUSTER_RECTS):
        corners = [(x0c,y0c),(x1c,y0c),(x1c,y1c),(x0c,y1c),(x0c,y0c)]
        rc = [_rxy(x,y,cos_r,sin_r) for x,y in corners]
        patched["data"][i]["x"] = [p[0] for p in rc]
        patched["data"][i]["y"] = [p[1] for p in rc]

    # Cluster labels (trace N_CLUSTERS)
    patched["data"][LABEL_IDX]["x"] = [_rxy(x0c+0.012,y1c-0.012,cos_r,sin_r)[0]
                                        for _,x0c,_,_,y1c in CLUSTER_RECTS]
    patched["data"][LABEL_IDX]["y"] = [_rxy(x0c+0.012,y1c-0.012,cos_r,sin_r)[1]
                                        for _,x0c,_,_,y1c in CLUSTER_RECTS]

    # Topology edges
    for i, (u, v, _) in enumerate(TOPO_EDGES):
        rx0,ry0 = _rxy(pos[u][0],pos[u][1],cos_r,sin_r)
        rx1,ry1 = _rxy(pos[v][0],pos[v][1],cos_r,sin_r)
        patched["data"][TOPO_START+i]["x"] = [rx0, rx1]
        patched["data"][TOPO_START+i]["y"] = [ry0, ry1]

    # Virtual edges
    for j, (c1,c2,x0,y0,x1,y1,key) in enumerate(VIRT_EDGES):
        rx0,ry0 = _rxy(x0,y0,cos_r,sin_r)
        rx1,ry1 = _rxy(x1,y1,cos_r,sin_r)
        patched["data"][VIRT_START+j]["x"] = [rx0, rx1]
        patched["data"][VIRT_START+j]["y"] = [ry0, ry1]

    # Hover midpoints
    h_x, h_y = [], []
    for u, v, _ in TOPO_EDGES:
        mx,my = (pos[u][0]+pos[v][0])/2, (pos[u][1]+pos[v][1])/2
        rx,ry = _rxy(mx,my,cos_r,sin_r); h_x.append(rx); h_y.append(ry)
    for _,_,x0,y0,x1,y1,_ in VIRT_EDGES:
        rx,ry = _rxy((x0+x1)/2,(y0+y1)/2,cos_r,sin_r); h_x.append(rx); h_y.append(ry)
    patched["data"][HOVER_IDX]["x"] = h_x
    patched["data"][HOVER_IDX]["y"] = h_y

    # Access nodes
    patched["data"][HOVER_IDX+1]["x"] = [_rxy(pos[n][0],pos[n][1],cos_r,sin_r)[0] for n in ACC_NODES]
    patched["data"][HOVER_IDX+1]["y"] = [_rxy(pos[n][0],pos[n][1],cos_r,sin_r)[1] for n in ACC_NODES]
    # Backbone nodes
    patched["data"][HOVER_IDX+2]["x"] = [_rxy(pos[n][0],pos[n][1],cos_r,sin_r)[0] for n in BB_LIST]
    patched["data"][HOVER_IDX+2]["y"] = [_rxy(pos[n][0],pos[n][1],cos_r,sin_r)[1] for n in BB_LIST]

    all_rxy = [_rxy(pos[n][0],pos[n][1],cos_r,sin_r) for n in G.nodes()]
    rxs = [p[0] for p in all_rxy]; rys = [p[1] for p in all_rxy]
    patched["layout"]["xaxis"]["range"] = [min(rxs)-_PAD_X, max(rxs)+_PAD_X]
    patched["layout"]["yaxis"]["range"] = [min(rys)-_PAD_Y, max(rys)+_PAD_Y]


# ---------------------------------------------------------------------------
# Legend panel builder
# ---------------------------------------------------------------------------
def _legend():
    def _line(w, color, dash="solid", label=""):
        style = {"display":"inline-block","width":"36px","height":f"{max(2,w)}px",
                 "background":color,"verticalAlign":"middle","marginRight":"6px",
                 "borderTop":f"{max(1,w//2)}px {'dashed' if dash=='dash' else 'dotted' if dash=='dot' else 'solid'} {color}",
                 "height":"0"}
        return html.Span([html.Span(style=style), label],
                         style={"display":"flex","alignItems":"center","marginBottom":"3px","fontSize":"11px"})

    def _swatch(color, label):
        return html.Span([
            html.Span(style={"background":color,"display":"inline-block","width":"12px",
                             "height":"12px","borderRadius":"2px","border":"1px solid #bbb",
                             "verticalAlign":"middle","marginRight":"5px"}),
            label
        ], style={"display":"block","fontSize":"11px","marginBottom":"3px"})

    def _node(size, label):
        return html.Span([
            html.Span("●", style={"fontSize":f"{size}px","color":"#333","marginRight":"5px",
                                  "verticalAlign":"middle"}),
            label
        ], style={"display":"block","fontSize":"11px","marginBottom":"3px"})

    col_style = {"flex":"1","minWidth":"140px","padding":"0 10px"}
    head_style = {"fontWeight":"bold","fontSize":"11px","color":"#444",
                  "borderBottom":"1px solid #ddd","marginBottom":"5px","paddingBottom":"2px"}

    return html.Details([
        html.Summary("📖  Legend & Notes",
                     style={"cursor":"pointer","fontWeight":"bold","fontSize":"12px",
                            "padding":"4px 0","userSelect":"none","color":"#1a1a2e"}),
        html.Div([
            # Column 1 – link width = capacity
            html.Div([
                html.Div("Link width = capacity", style=head_style),
                _line(7,"#111","solid","400 GE"),
                _line(5,"#222","solid","200 GE"),
                _line(3,"#333","solid","100 GE"),
                _line(2,"#555","solid"," 20 GE"),
                _line(1,"#2980b9","solid"," 10 GE"),
                _line(1,"#2980b9","dash","  1 GE"),
                _line(1,"#aaa","dot",  "< 1 GE"),
            ], style=col_style),
            # Column 2 – line colour = utilisation
            html.Div([
                html.Div("Line colour = utilisation", style=head_style),
                _swatch("#1a9850","< 10 %  — low"),
                _swatch("#91cf60","10 – 25 %"),
                _swatch("#fee08b","25 – 50 %  — moderate"),
                _swatch("#fc8d59","50 – 75 %  — high"),
                _swatch("#d73027","> 75 %  — critical"),
                _swatch("#888","No data  (default bw colour)"),
            ], style=col_style),
            # Column 3 – line style & nodes
            html.Div([
                html.Div("Line style", style=head_style),
                html.Span("━━━  Solid = direct L3 router edge",
                          style={"display":"block","fontSize":"11px","marginBottom":"3px"}),
                html.Span("···  Dotted = backbone circuit without direct L3 edge "
                          "(capacity inferred from adjacent links)",
                          style={"display":"block","fontSize":"11px","marginBottom":"8px"}),
                html.Div("Node type", style=head_style),
                _node(16,"Large circle = backbone router (≥ 3 × 100 GE neighbours)"),
                _node(10,"Small circle = access / edge router"),
            ], style=col_style),
            # Column 4 – controls cheat-sheet
            html.Div([
                html.Div("Controls", style=head_style),
                html.Span("🖱 Drag  = pan  (default)",
                          style={"display":"block","fontSize":"11px","marginBottom":"3px"}),
                html.Span("🔍 Scroll = zoom  (or use toolbar ▶)",
                          style={"display":"block","fontSize":"11px","marginBottom":"3px"}),
                html.Span("↺ ↻  = rotate layout 15° steps",
                          style={"display":"block","fontSize":"11px","marginBottom":"3px"}),
                html.Span("⊕  = reset rotation to North-up",
                          style={"display":"block","fontSize":"11px","marginBottom":"3px"}),
                html.Span("Step = play advance per tick",
                          style={"display":"block","fontSize":"11px","marginBottom":"3px"}),
                html.Span("Window = look-back period for aggregation",
                          style={"display":"block","fontSize":"11px","marginBottom":"3px"}),
                html.Span("Agg = peak / mean / median over window",
                          style={"display":"block","fontSize":"11px","marginBottom":"3px"}),
            ], style=col_style),
        ], style={"display":"flex","flexWrap":"wrap","padding":"8px 0 4px 0",
                  "borderTop":"1px solid #eee","marginTop":"4px"}),
    ], style={"padding":"4px 18px","borderBottom":"1px solid #e0e0e0","background":"#fafcff"})


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
# 5-min samples per period
_S = {"5min":1,"1h":12,"6h":72,"1d":288,"1w":2016,"1mo":8640}
STEP_OPTIONS   = [{"label":"5 min",   "value":_S["5min"]},{"label":"1 hour", "value":_S["1h"]},
                  {"label":"6 hours", "value":_S["6h"]}, {"label":"1 day",  "value":_S["1d"]},
                  {"label":"1 week",  "value":_S["1w"]}, {"label":"1 month","value":_S["1mo"]}]
WINDOW_OPTIONS = STEP_OPTIONS   # same time periods
AGG_OPTIONS    = [{"label":"Peak (max)","value":"peak"},
                  {"label":"Mean",      "value":"mean"},
                  {"label":"Median",    "value":"median"}]

_BTN = {"border":"1px solid #aaa","borderRadius":"4px","background":"#f5f5f5",
        "cursor":"pointer","fontSize":"13px","padding":"5px 11px"}
_LBL = {"fontSize":"12px","whiteSpace":"nowrap","alignSelf":"center"}

app = dash.Dash(__name__, title="SWITCH Network Traffic",
                meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}])

app.layout = html.Div([

    # ── header ──────────────────────────────────────────────────────────────
    html.Div([
        html.H2("SWITCH Backbone — Traffic Volume  (5-min resolution)",
                style={"margin":"0","fontSize":"clamp(14px,2vw,18px)","fontWeight":"bold",
                       "color":"#1a1a2e"}),
    ], style={"padding":"10px 16px 6px 16px","borderBottom":"1px solid #e0e0e0",
              "display":"flex","alignItems":"center"}),

    # ── collapsible legend ───────────────────────────────────────────────────
    _legend(),

    # ── graph (fills remaining vertical space) ───────────────────────────────
    dcc.Graph(id="graph", figure=BASE_FIG,
              style={"flex":"1","minHeight":"300px"},
              config={"scrollZoom":True,
                      "modeBarButtonsToAdd":["zoom2d","pan2d","zoomIn2d","zoomOut2d","resetScale2d"],
                      "displaylogo":False}),

    # ── controls bar ────────────────────────────────────────────────────────
    html.Div([
        # play / pause
        html.Button("▶  Play", id="play-btn", n_clicks=0,
                    style={**_BTN,"fontSize":"14px","marginRight":"8px","whiteSpace":"nowrap"}),
        # rotation
        html.Span([
            html.Button("↺", id="rot-left",  n_clicks=0, title="Rotate -15°", style=_BTN),
            html.Button("⊕", id="rot-reset", n_clicks=0, title="Reset rotation", style=_BTN),
            html.Button("↻", id="rot-right", n_clicks=0, title="Rotate +15°", style=_BTN),
        ], style={"display":"inline-flex","gap":"3px","marginRight":"12px"}),
        html.Span(id="rot-label",
                  style={"fontSize":"11px","color":"#666","marginRight":"12px","whiteSpace":"nowrap"}),
        # step
        html.Label("Step:",  style=_LBL), html.Div(style={"width":"4px"}),
        dcc.Dropdown(id="step-dd",   options=STEP_OPTIONS,  value=_S["1h"],   clearable=False,
                     style={"width":"95px","fontSize":"12px","marginRight":"10px"}),
        # window
        html.Label("Window:", style=_LBL), html.Div(style={"width":"4px"}),
        dcc.Dropdown(id="window-dd", options=WINDOW_OPTIONS, value=_S["5min"], clearable=False,
                     style={"width":"95px","fontSize":"12px","marginRight":"10px"}),
        # aggregation
        html.Label("Agg:",   style=_LBL), html.Div(style={"width":"4px"}),
        dcc.Dropdown(id="agg-dd",    options=AGG_OPTIONS,   value="peak",     clearable=False,
                     style={"width":"110px","fontSize":"12px","marginRight":"10px"}),
        # time slider
        html.Div(dcc.Slider(id="time-slider",min=0,max=N_TS-1,step=1,value=0,
                            marks=SLIDER_MARKS,updatemode="drag",
                            tooltip={"placement":"top","always_visible":False}),
                 style={"flex":"1","minWidth":"120px","padding":"0 8px"}),
        # timestamp label
        html.Div(id="ts-label",
                 style={"minWidth":"130px","fontWeight":"bold","fontSize":"12px",
                        "color":"#1a1a2e","textAlign":"right","whiteSpace":"nowrap",
                        "alignSelf":"center"}),
    ], style={"display":"flex","alignItems":"center","flexWrap":"wrap","gap":"4px",
              "padding":"6px 16px 8px 16px","borderTop":"1px solid #e0e0e0","background":"#fafafa"}),

    dcc.Store(id="playing",  data=False),
    dcc.Store(id="rotation", data=0),
    dcc.Interval(id="ticker",interval=600,n_intervals=0,disabled=True),

], style={"fontFamily":"Arial,sans-serif","display":"flex","flexDirection":"column",
          "height":"100dvh","width":"100%","boxSizing":"border-box","overflow":"hidden"})

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("graph","figure"),
    Output("ts-label","children"),
    Input("time-slider","value"),
    Input("window-dd","value"),
    Input("agg-dd","value"),
    Input("rotation","data"),
    prevent_initial_call=True,
)
def update_figure(ts_idx, window, agg, rotation_deg):
    """Patch edge colours; also patch positions when rotation changes."""
    from dash import ctx
    w   = window or 1
    agg = agg    or "peak"
    patched = Patch()

    # Reposition all traces when rotation changes
    if ctx.triggered_id == "rotation":
        _apply_rotation_patch(patched, rotation_deg or 0)

    # Always update edge colours and hover texts
    for i,(u,v,data) in enumerate(TOPO_EDGES):
        ekey = frozenset({_pop_code(u),_pop_code(v)})
        util = _window_val(LINK_UTILS.get(ekey), ts_idx, w, agg)
        color = _util_color(util) if not np.isnan(util) else _edge_style(data["bw"])[1]
        patched["data"][TOPO_START+i]["line"]["color"] = color

    for j,(*_,key) in enumerate(VIRT_EDGES):
        util = _window_val(LINK_UTILS.get(key), ts_idx, w, agg)
        patched["data"][VIRT_START+j]["line"]["color"] = \
            _util_color(util) if not np.isnan(util) else "#bbbbbb"

    patched["data"][HOVER_IDX]["text"] = _make_hover(
        TOPO_EDGES, VIRT_EDGES, LINK_UTILS, LINK_CAP, ts_idx, w, agg)

    ts    = int(TIMESTAMPS[ts_idx])
    label = datetime.fromtimestamp(ts,tz=timezone.utc).strftime("%Y-%m-%d  %H:%M")
    return patched, label


@app.callback(
    Output("rotation","data"),
    Output("rot-label","children"),
    Input("rot-left","n_clicks"),
    Input("rot-reset","n_clicks"),
    Input("rot-right","n_clicks"),
    State("rotation","data"),
    prevent_initial_call=True,
)
def rotate(left, reset, right, current):
    from dash import ctx
    tid = ctx.triggered_id
    if tid == "rot-reset": deg = 0
    elif tid == "rot-left":  deg = ((current or 0) - 15) % 360
    else:                    deg = ((current or 0) + 15) % 360
    return deg, f"{deg}°"


@app.callback(
    Output("playing","data"),
    Output("play-btn","children"),
    Output("ticker","disabled"),
    Input("play-btn","n_clicks"),
    State("playing","data"),
)
def toggle_play(n_clicks, playing):
    if n_clicks == 0: return False, "▶  Play", True
    new = not playing
    return new, "⏸  Pause" if new else "▶  Play", not new


@app.callback(
    Output("time-slider","value"),
    Input("ticker","n_intervals"),
    State("time-slider","value"),
    State("playing","data"),
    State("step-dd","value"),
    prevent_initial_call=True,
)
def advance(_, current, playing, step):
    if not playing: return current
    return min(current + (step or 1), N_TS - 1)


if __name__ == "__main__":
    print("Starting server → http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
