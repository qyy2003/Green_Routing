#!/usr/bin/env python3
"""Step 2 — build the traffic-data index for the SWITCHlan map viewer.

Scans the backbone usage logs (lan-mon2021/bb-usage-logs/) and produces
web/data/index.json describing, for every backbone link:

  - link id and its two endpoint node codes (matching the map labels)
  - the parallel-link instance, if any (filenames like BE-FR-1)
  - the monthly data files, each with path, sample count, and exact
    first/last unix timestamps
  - the overall time span and the sampling interval

This is the "where to read from" index that step 3 consumes to decide which
file to open for a chosen time, and which map link to color.
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../ETH_Master_Study/Green_Routing
LOGS = ROOT / "network-energy-efficiency-research" / "lan-mon2021" / "bb-usage-logs"
OUT = Path(__file__).resolve().parent / "data" / "index.json"

MONTH_RE = re.compile(r"^(?P<link>.+)\.(?P<ym>\d{6})$")


def parse_link_id(link_id):
    """Split a link id like 'BE-FR-1' into (endA, endB, instance)."""
    parts = link_id.split("-")
    instance = None
    if parts[-1].isdigit():
        instance = int(parts[-1])
        parts = parts[:-1]
    if len(parts) != 2:
        # Unexpected shape; keep the whole thing as endpoint A for visibility.
        return link_id, "", instance
    return parts[0], parts[1], instance


def first_last_ts(path):
    """Return (first_ts, last_ts, n_samples) reading the file once."""
    first = last = None
    n = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts = line.split(None, 1)[0]
            if not ts.isdigit():
                continue
            ts = int(ts)
            if first is None:
                first = ts
            last = ts
            n += 1
    return first, last, n


def main():
    files = sorted(os.listdir(LOGS))
    links = {}  # link_id -> record

    for name in files:
        m = MONTH_RE.match(name)
        if not m:
            continue
        link_id = m.group("link")
        ym = m.group("ym")
        path = LOGS / name
        first, last, n = first_last_ts(path)
        if first is None:
            continue

        rec = links.get(link_id)
        if rec is None:
            endA, endB, inst = parse_link_id(link_id)
            rec = {
                "id": link_id,
                "endpoints": [endA, endB],
                "instance": inst,
                "months": [],
            }
            links[link_id] = rec

        rec["months"].append({
            "ym": ym,
            "file": str(path.relative_to(ROOT)),
            "first": first,
            "last": last,
            "samples": n,
        })

    # Finalize: sort months and compute per-link span + global stats.
    interval = 300  # 5 minutes, confirmed from the logs
    g_first = g_last = None
    for rec in links.values():
        rec["months"].sort(key=lambda x: x["ym"])
        rec["first"] = rec["months"][0]["first"]
        rec["last"] = rec["months"][-1]["last"]
        g_first = rec["first"] if g_first is None else min(g_first, rec["first"])
        g_last = rec["last"] if g_last is None else max(g_last, rec["last"])

    index = {
        "source": str(LOGS.relative_to(ROOT)),
        "interval_seconds": interval,
        "units": "Mbps",
        "fields": ["timestamp", "inMbps", "outMbps"],
        "time_span": {"first": g_first, "last": g_last},
        "link_count": len(links),
        "links": [links[k] for k in sorted(links)],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(index, f, indent=2)

    # Console summary.
    import datetime as dt
    fmt = lambda t: dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else "?"
    print(f"links: {len(links)}")
    print(f"span : {fmt(g_first)}  ->  {fmt(g_last)} (UTC)")
    print(f"out  : {OUT}")
    print("link ids:", ", ".join(sorted(links)))


if __name__ == "__main__":
    main()
