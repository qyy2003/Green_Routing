#!/usr/bin/env python3
"""Viewer backend (Cricket dataset).

Serves the static site plus a time-indexed traffic/telemetry API backed by the
compact per-month frames produced by build_frames.py. Month files are loaded
lazily and cached, so scrubbing the time slider within a month is instant.

Endpoints
  GET /api/meta                -> links (with capacity), node positions, months, span
  GET /api/frame?t=<unixts>    -> { t, slot_ts, links:{id:{in,out}}, nodes:{code:{cpu,pw,temp}} }

Run:  python3 server.py [port]   (default 8137)
"""

import json
import os
import sys
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FRAMES = os.path.join(DATA, "frames")

LINKS = json.load(open(os.path.join(DATA, "links.json")))["links"]
NODES = json.load(open(os.path.join(DATA, "nodes.json")))
try:
    MANIFEST = json.load(open(os.path.join(FRAMES, "manifest.json")))["months"]
except FileNotFoundError:
    MANIFEST = sorted(f[:-5] for f in os.listdir(FRAMES)
                      if f.endswith(".json") and f != "manifest.json") if os.path.isdir(FRAMES) else []

STEP = 300
_MONTH_CACHE = {}


def month_key(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m")


def load_month(ym):
    if ym in _MONTH_CACHE:
        return _MONTH_CACHE[ym]
    path = os.path.join(FRAMES, f"{ym}.json")
    data = None
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    _MONTH_CACHE[ym] = data
    return data


def span():
    if not MANIFEST:
        return {"first": 0, "last": 0}
    first_mo = load_month(MANIFEST[0])
    last_mo = load_month(MANIFEST[-1])
    first = first_mo["t0"] if first_mo else 0
    last = (last_mo["t0"] + (last_mo["n"] - 1) * STEP) if last_mo else 0
    return {"first": first, "last": last}


def frame_at(t):
    ym = month_key(t)
    mo = load_month(ym)
    out = {"t": t, "slot_ts": None, "links": {}, "nodes": {}}
    if not mo:
        return out
    i = (t - mo["t0"]) // STEP
    if i < 0 or i >= mo["n"]:
        return out
    out["slot_ts"] = mo["t0"] + i * STEP
    for lid, rec in mo["links"].items():
        d = {}
        for k in ("in", "out"):
            arr = rec.get(k)
            if arr is not None and arr[i] is not None:
                d[k] = arr[i]
        if d:
            out["links"][lid] = d
    for code, rec in mo["nodes"].items():
        d = {}
        for k in ("cpu", "pw", "temp"):
            arr = rec.get(k)
            if arr is not None and arr[i] is not None:
                d[k] = arr[i]
        if d:
            out["nodes"][code] = d
    return out


import statistics

AGG_FN = {
    "mean": statistics.fmean,
    "median": statistics.median,
    "max": max,
    "min": min,
}


def yms_in(t0, t1):
    """Year-month keys of every month overlapping [t0, t1)."""
    d = datetime.fromtimestamp(t0, tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    while d.timestamp() < t1:
        out.append(d.strftime("%Y%m"))
        d = d.replace(year=d.year + 1, month=1) if d.month == 12 else d.replace(month=d.month + 1)
    return out or [datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%Y%m")]


def _slot_bounds(mo, t0, t1):
    lo = max(0, (t0 - mo["t0"]) // STEP)
    hi = min(mo["n"], -(-(t1 - mo["t0"]) // STEP))  # ceil
    return lo, hi


def agg_window(t0, t1, aggname):
    """Aggregate every link/node metric over [t0, t1) with the given function."""
    fn = AGG_FN.get(aggname, statistics.fmean)
    L = {}   # lid -> {'in':[...vals], 'out':[...]}
    N = {}   # code -> {'cpu':[], 'pw':[], 'temp':[]}
    for ym in yms_in(t0, t1):
        mo = load_month(ym)
        if not mo:
            continue
        lo, hi = _slot_bounds(mo, t0, t1)
        if hi <= lo:
            continue
        for lid, rec in mo["links"].items():
            d = L.setdefault(lid, {"in": [], "out": []})
            for k in ("in", "out"):
                arr = rec.get(k)
                if arr:
                    d[k].extend(v for v in arr[lo:hi] if v is not None)
        for code, rec in mo["nodes"].items():
            d = N.setdefault(code, {"cpu": [], "pw": [], "temp": []})
            for k in ("cpu", "pw", "temp"):
                arr = rec.get(k)
                if arr:
                    d[k].extend(v for v in arr[lo:hi] if v is not None)
    links = {}
    for lid, d in L.items():
        out = {k: round(fn(d[k]), 2) for k in ("in", "out") if d[k]}
        if out:
            links[lid] = out
    nodes = {}
    for code, d in N.items():
        out = {k: round(fn(d[k]), 2) for k in ("cpu", "pw", "temp") if d[k]}
        if out:
            nodes[code] = out
    return {"t": t0, "t_end": t1, "agg": aggname, "links": links, "nodes": nodes}


def _stats(vals, ts):
    v = [(x, ts[i]) for i, x in enumerate(vals) if x is not None]
    if not v:
        return None
    xs = [x for x, _ in v]
    peak_x, peak_t = max(v, key=lambda p: p[0])
    return {
        "min": round(min(xs), 2), "max": round(peak_x, 2),
        "mean": round(statistics.fmean(xs), 2), "median": round(statistics.median(xs), 2),
        "peak_t": peak_t, "n": len(xs),
    }


def _downsample(ts, seriesmap, maxpts=240):
    n = len(ts)
    if n == 0:
        return {"t": [], **{k: [] for k in seriesmap}}
    step = max(1, -(-n // maxpts))
    out = {"t": []}
    for k in seriesmap:
        out[k] = []
    for i in range(0, n, step):
        out["t"].append(ts[i])
        for k, arr in seriesmap.items():
            chunk = [arr[j] for j in range(i, min(i + step, n)) if arr[j] is not None]
            out[k].append(round(statistics.fmean(chunk), 2) if chunk else None)
    return out


def series(kind, sid, t0, t1, metric):
    ts = []
    cols = {}
    keys = ("in", "out") if kind == "link" else (metric,)
    for k in keys:
        cols[k] = []
    for ym in yms_in(t0, t1):
        mo = load_month(ym)
        if not mo:
            continue
        lo, hi = _slot_bounds(mo, t0, t1)
        if hi <= lo:
            continue
        recs = (mo["links"] if kind == "link" else mo["nodes"]).get(sid)
        base = mo["t0"]
        for i in range(lo, hi):
            ts.append(base + i * STEP)
            if recs:
                for k in keys:
                    arr = recs.get(k)
                    cols[k].append(arr[i] if arr else None)
            else:
                for k in keys:
                    cols[k].append(None)
    stats = {k: _stats(cols[k], ts) for k in keys}
    ds = _downsample(ts, cols)
    cap = None
    if kind == "link":
        for l in LINKS:
            if l["id"] == sid:
                cap = l["capacity_bps"]
                break
    return {"kind": kind, "id": sid, "t0": t0, "t1": t1, "metric": metric,
            "capacity_bps": cap, "series": ds, "stats": stats}


META = {
    "links": LINKS,
    "nodes": NODES,
    "months": MANIFEST,
    "interval_seconds": STEP,
    "units": {"traffic": "Mbps", "capacity": "bps", "cpu": "%", "pw": "W", "temp": "C"},
    "time_span": span(),
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def end_headers(self):
        # Never let browsers cache the app/data (avoids stale app.js calling
        # removed API routes -> HTML 404 -> JSON.parse errors).
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/meta":
            return self._json(META)
        q = parse_qs(u.query)
        def iarg(name, default=0):
            return int(float(q.get(name, [str(default)])[0]))
        if u.path == "/api/frame":
            try:
                return self._json(frame_at(iarg("t")))
            except ValueError:
                return self._json({"error": "bad t"}, 400)
        if u.path == "/api/agg":
            try:
                t = iarg("t"); w = max(STEP, iarg("w", STEP))
            except ValueError:
                return self._json({"error": "bad args"}, 400)
            agg = q.get("agg", ["mean"])[0]
            if w <= STEP:                       # single slot = instant
                return self._json(frame_at(t))
            return self._json(agg_window(t, t + w, agg))
        if u.path == "/api/series":
            kind = q.get("kind", ["link"])[0]
            sid = q.get("id", [""])[0]
            metric = q.get("metric", ["cpu"])[0]
            try:
                t = iarg("t"); w = max(STEP, iarg("w", STEP))
            except ValueError:
                return self._json({"error": "bad args"}, 400)
            return self._json(series(kind, sid, t, t + w, metric))
        return super().do_GET()

    def log_message(self, *a):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8137
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"serving http://localhost:{port}/  (months: {len(MANIFEST)})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
