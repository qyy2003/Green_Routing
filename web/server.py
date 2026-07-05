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
        if u.path == "/api/frame":
            q = parse_qs(u.query)
            try:
                t = int(float(q.get("t", ["0"])[0]))
            except ValueError:
                return self._json({"error": "bad t"}, 400)
            return self._json(frame_at(t))
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
