#!/usr/bin/env python3
"""Preprocess the Cricket dataset into compact per-month frames for the viewer.

For every drawable link (data/links.json) and map node (data/node_devices.json)
this reads only the CSVs we need, handles the two-era ds/named column quirk by
positional coalescing, and resamples onto each month's 5-minute grid.

Output: data/frames/<YYYYMM>.json
  {
    "t0": <unix month start, UTC>, "step": 300, "n": <slots>,
    "links": { "<id>": {"in":[mbps|null,...], "out":[...]} },
    "nodes": { "<code>": {"cpu":[%|null], "pw":[W|null], "temp":[C|null]} }
  }

Run:  python3 build_frames.py           (all months; slow, reads ~250 CSVs)
      python3 build_frames.py 2024       (restrict to year 2024, for testing)
"""

import csv
import json
import math
import os
import re
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SW = Path("/media/yuyqin/share/switch")
FRAMES = HERE / "data" / "frames"
STEP = 300
NAN = float("nan")

LINKS = json.load(open(HERE / "data" / "links.json"))["links"]
NODE_DEV = json.load(open(HERE / "data" / "node_devices.json"))

YEAR_FILTER = sys.argv[1] if len(sys.argv) > 1 else None

# ---------- CSV era-coalescing reader ----------

def parse_header(header):
    """Return (num_ds, named_names, named_start_index). Col 0 is timestamp."""
    ds = [h for h in header if re.fullmatch(r"ds\d+", h)]
    num_ds = len(ds)
    named = header[1 + num_ds:]
    return num_ds, named, 1 + num_ds


def coalesce_getter(header, wanted_named):
    """Build a function row->value for a named column, coalescing ds<->named."""
    num_ds, named, named_start = parse_header(header)
    if wanted_named not in named:
        return None
    j = named.index(wanted_named)          # position within named block
    named_col = named_start + j
    ds_col = 1 + j if j < num_ds else None

    def get(row):
        v = row[named_col] if named_col < len(row) else ""
        if v == "" and ds_col is not None and ds_col < len(row):
            v = row[ds_col]
        if v == "":
            return None
        try:
            return float(v)
        except ValueError:
            return None
    return get


def read_series(path, cols):
    """Yield (ts, {name: value|None}) for the requested named columns."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return
        getters = {c: coalesce_getter(header, c) for c in cols}
        getters = {c: g for c, g in getters.items() if g}
        if not getters:
            return
        for row in r:
            if not row or not row[0].isdigit():
                continue
            ts = int(row[0])
            yield ts, {c: g(row) for c, g in getters.items()}


# ---------- month grid ----------

class Month:
    def __init__(self, ym):
        self.ym = ym
        y, m = int(ym[:4]), int(ym[4:])
        self.t0 = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        t1 = int(datetime(ny, nm, 1, tzinfo=timezone.utc).timestamp())
        self.n = (t1 - self.t0) // STEP
        self.link_in = {}
        self.link_out = {}
        self.node_cpu = {}
        self.node_cpu_cnt = {}
        self.node_pw = {}
        self.node_temp = {}

    def slot(self, ts):
        i = (ts - self.t0) // STEP
        return i if 0 <= i < self.n else None

    def farr(self):
        return array("f", [NAN]) * self.n if False else array("f", [NAN] * self.n)

    def iarr(self):
        return array("i", [0] * self.n)


MONTHS = {}

def month_for(ts):
    ym = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m")
    if YEAR_FILTER and not ym.startswith(YEAR_FILTER):
        return None
    mo = MONTHS.get(ym)
    if mo is None:
        mo = MONTHS[ym] = Month(ym)
    return mo


def add_sum(store, key, mo, i, v):
    a = store.get(key)
    if a is None:
        a = store[key] = mo.farr()
    if math.isnan(a[i]):
        a[i] = v
    else:
        a[i] += v


def add_max(store, key, mo, i, v):
    a = store.get(key)
    if a is None:
        a = store[key] = mo.farr()
    if math.isnan(a[i]) or v > a[i]:
        a[i] = v


# ---------- ingest ----------

def ingest_links():
    for link in LINKS:
        lid = link["id"]
        for m in link["members"]:
            if not m.get("file"):
                continue
            path = SW / "router-interfaces" / m["dir"] / m["file"]
            if not path.exists():
                continue
            for ts, vals in read_series(path, ["ifHCInOctets", "ifHCOutOctets"]):
                mo = month_for(ts)
                if mo is None:
                    continue
                i = mo.slot(ts)
                if i is None:
                    continue
                vi, vo = vals.get("ifHCInOctets"), vals.get("ifHCOutOctets")
                if vi is not None:
                    add_sum(mo.link_in, lid, mo, i, vi * 8 / 1e6)
                if vo is not None:
                    add_sum(mo.link_out, lid, mo, i, vo * 8 / 1e6)
        print(f"  link {lid} done", flush=True)


def pick_cpu_col(named):
    lc = [(n, n.lower()) for n in named]
    for want in ("cpu5min", "cpu 5min", "cpu5"):
        for n, l in lc:
            if want in l and "mem" not in l:
                return n
    for n, l in lc:
        if "cpu" in l and "1min" in l and "mem" not in l:
            return n
    for n, l in lc:
        if "cpu" in l and "mem" not in l:
            return n
    return None


def temp_cols(named):
    return [n for n in named if n.lower().startswith("tmp") or "temp" in n.lower()]


def scaled_temp(v):
    return v / 10.0 if v is not None and v > 120 else v


def ingest_nodes():
    for code, dev in NODE_DEV.items():
        # CPU + temperature from routers/
        for rf in dev["routers"]:
            if "-test" in rf:
                continue
            path = SW / "routers" / rf
            if not path.exists():
                continue
            with open(path, newline="") as f:
                header = next(csv.reader(f), None)
            if not header:
                continue
            _, named, _ = parse_header(header)
            cpu = pick_cpu_col(named)
            temps = temp_cols(named)
            want = ([cpu] if cpu else []) + temps
            if not want:
                continue
            for ts, vals in read_series(path, want):
                mo = month_for(ts)
                if mo is None:
                    continue
                i = mo.slot(ts)
                if i is None:
                    continue
                if cpu and vals.get(cpu) is not None:
                    add_sum(mo.node_cpu, code, mo, i, vals[cpu])
                    cnt = mo.node_cpu_cnt.get(code)
                    if cnt is None:
                        cnt = mo.node_cpu_cnt[code] = mo.iarr()
                    cnt[i] += 1
                tvals = [scaled_temp(vals[t]) for t in temps if vals.get(t) is not None]
                if tvals:
                    add_max(mo.node_temp, code, mo, i, max(tvals))
        # power from router-power/ (U*I over PSUs)
        for pf in dev["psus"]:
            path = SW / "router-power" / pf
            if not path.exists():
                continue
            for ts, vals in read_series(path, ["Cisco_PSU_U", "Cisco_PSU_I"]):
                u, cur = vals.get("Cisco_PSU_U"), vals.get("Cisco_PSU_I")
                if u is None or cur is None:
                    continue
                mo = month_for(ts)
                if mo is None:
                    continue
                i = mo.slot(ts)
                if i is None:
                    continue
                add_sum(mo.node_pw, code, mo, i, u * cur / 1e6)
        print(f"  node {code} done", flush=True)


# ---------- write ----------

def clean(a):
    return [None if math.isnan(x) else round(x, 2) for x in a]


def write_months():
    FRAMES.mkdir(parents=True, exist_ok=True)
    for ym, mo in sorted(MONTHS.items()):
        links = {}
        for lid in set(mo.link_in) | set(mo.link_out):
            rec = {}
            if lid in mo.link_in:
                rec["in"] = clean(mo.link_in[lid])
            if lid in mo.link_out:
                rec["out"] = clean(mo.link_out[lid])
            links[lid] = rec
        nodes = {}
        codes = set(mo.node_cpu) | set(mo.node_pw) | set(mo.node_temp)
        for c in codes:
            rec = {}
            if c in mo.node_cpu:
                cpu = mo.node_cpu[c]
                cnt = mo.node_cpu_cnt[c]
                rec["cpu"] = [None if cnt[i] == 0 else round(cpu[i] / cnt[i], 2) for i in range(mo.n)]
            if c in mo.node_pw:
                rec["pw"] = clean(mo.node_pw[c])
            if c in mo.node_temp:
                rec["temp"] = clean(mo.node_temp[c])
            nodes[c] = rec
        out = {"t0": mo.t0, "step": STEP, "n": mo.n, "ym": ym,
               "links": links, "nodes": nodes}
        with open(FRAMES / f"{ym}.json", "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"wrote {ym}.json  (slots={mo.n}, links={len(links)}, nodes={len(nodes)})", flush=True)


def main():
    print("ingesting links...", flush=True)
    ingest_links()
    print("ingesting nodes...", flush=True)
    ingest_nodes()
    print("writing months...", flush=True)
    write_months()
    # also emit a manifest of available months
    manifest = sorted(MONTHS)
    with open(FRAMES / "manifest.json", "w") as f:
        json.dump({"months": manifest}, f)
    print("done. months:", len(manifest))


if __name__ == "__main__":
    main()
