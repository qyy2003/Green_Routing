#!/usr/bin/env python3
"""Step 3 — viewer backend.

Serves the static web/ folder and a small on-demand traffic API backed by the
index built in step 2. Month files are parsed lazily and cached in memory, so
the first query for a month is a one-off cost and every later query (e.g.
dragging the time slider within that month) is an in-memory binary search.

Endpoints
  GET /api/meta                  -> index.json (links, span, interval, units)
  GET /api/nodes                 -> nodes.json (map coordinates per node)
  GET /api/traffic?t=<unixts>    -> { t, matched, links: { id: {in, out, ts} } }

Run:  python3 server.py [port]   (default 8137)
"""

import bisect
import json
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # data repo lives here
INDEX_PATH = os.path.join(HERE, "data", "index.json")
NODES_PATH = os.path.join(HERE, "data", "nodes.json")

with open(INDEX_PATH) as f:
    INDEX = json.load(f)

INTERVAL = INDEX["interval_seconds"]
# Accept a sample as "the value at t" only if within this many seconds of t.
TOLERANCE = INTERVAL * 2

# link id -> sorted list of (first, last, abspath)
MONTHS = {}
for link in INDEX["links"]:
    MONTHS[link["id"]] = sorted(
        (m["first"], m["last"], os.path.join(ROOT, m["file"]))
        for m in link["months"]
    )

# Cache: abspath -> (times[], ins[], outs[])
_FILE_CACHE = {}


def load_file(path):
    cached = _FILE_CACHE.get(path)
    if cached is not None:
        return cached
    times, ins, outs = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            # format: <ts> <in> inMbps <out> outMbps
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            times.append(int(parts[0]))
            ins.append(float(parts[1]))
            outs.append(float(parts[3]))
    cached = (times, ins, outs)
    _FILE_CACHE[path] = cached
    return cached


def sample_at(link_id, t):
    """Nearest in/out (Mbps) for link at unix time t, or None if no data."""
    months = MONTHS.get(link_id)
    if not months:
        return None
    # Find a month whose span contains t (fall back to nearest by edge).
    chosen = None
    for first, last, path in months:
        if first <= t <= last:
            chosen = path
            break
    if chosen is None:
        # nearest month by distance to its [first,last] span
        best, bestd = None, None
        for first, last, path in months:
            d = 0 if first <= t <= last else min(abs(t - first), abs(t - last))
            if bestd is None or d < bestd:
                best, bestd = path, d
        if bestd is None or bestd > TOLERANCE:
            return None
        chosen = best

    times, ins, outs = load_file(chosen)
    if not times:
        return None
    i = bisect.bisect_left(times, t)
    cands = []
    if i < len(times):
        cands.append(i)
    if i > 0:
        cands.append(i - 1)
    best = min(cands, key=lambda j: abs(times[j] - t))
    if abs(times[best] - t) > TOLERANCE:
        return None
    return {"in": round(ins[best], 1), "out": round(outs[best], 1), "ts": times[best]}


def traffic_frame(t):
    out = {}
    matched = 0
    for link in INDEX["links"]:
        s = sample_at(link["id"], t)
        if s is not None:
            out[link["id"]] = s
            matched += 1
    return {"t": t, "matched": matched, "links": out}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/meta":
            return self._json(INDEX)
        if parsed.path == "/api/nodes":
            with open(NODES_PATH) as f:
                return self._json(json.load(f))
        if parsed.path == "/api/traffic":
            q = parse_qs(parsed.query)
            try:
                t = int(float(q.get("t", ["0"])[0]))
            except ValueError:
                return self._json({"error": "bad t"}, 400)
            return self._json(traffic_frame(t))
        return super().do_GET()

    def log_message(self, *a):  # quieter console
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8137
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"serving http://localhost:{port}/  (root: {HERE})")
    print(f"data repo: {ROOT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
