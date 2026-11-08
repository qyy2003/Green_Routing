"""Side-by-side tables of the green_paths runs: ``python tables.py [setup]`` (default the current
network; ``tm2021`` for the measured 2021 matrix). Prints a unilateral table and an everyone table."""
from __future__ import annotations

import json
import sys

import pandas as pd

from simulate import OUT

KEYS = ["links_empty", "spans_empty", "span_E_empty_share", "util_max", "hops_per_byte", "bytes_paying_more",
        "unallocated_share", "bridges"]


def main(setup: str = ""):
    d = OUT / setup if setup else OUT
    S = json.loads((d / "summary.json").read_text())
    suffixes = sorted({t.split("unilateral", 1)[1] for t in S if t.startswith("unilateral")})
    uni = {}
    for sfx in suffixes:
        s = S[f"unilateral{sfx}"]
        m = s["mean"]
        uni[sfx or "current"] = {
            "bytes with a greener path": m["uni_bytes_with_greener_path"],
            "saving, bytes-weighted (block)": s["block"]["saving_total_if_each_alone"],
            "saving p90 of bytes": s["block"]["od_saving_p10_p50_p90_bytes"][2],
            "saving per source node p10": s["block"]["router_saving_p10_p50_p90"][0],
            "saving per source node p50": s["block"]["router_saving_p10_p50_p90"][1],
            "saving per source node p90": s["block"]["router_saving_p10_p50_p90"][2],
            "movers' hops / today's": m["uni_hops_ratio_movers"],
            "rate today g/GB": m["uni_rate_today_g_per_GB"],
        }
    print("UNILATERAL (one OD pair moves, everyone else on today's routing)")
    print(pd.DataFrame(uni).round(3).to_string())
    for sfx in suffixes:
        rows = {}
        R0 = pd.read_csv(d / f"hourly_everyone{sfx}.csv")
        rows["today"] = {k: R0[f"base_{k}"].mean() for k in KEYS if f"base_{k}" in R0}
        rows["today"].update({"hours_over_100": (R0.base_util_max > 1).mean(), "hours_over_80": (R0.base_util_max > 0.8).mean()})
        for exp in ["static", "everyone", "everyone_cap80", "everyone_carry", "everyone_carry_cap80"]:
            f = d / f"hourly_{exp}{sfx}.csv"
            if not f.exists():
                continue
            R = pd.read_csv(f)
            r = {k: R[f"green_{k}"].mean() for k in KEYS}
            r.update({"hours_over_100": (R.green_util_max > 1).mean(), "hours_over_80": (R.green_util_max > 0.8).mean()})
            if "converged" in R:
                r["hours_not_settled"] = (~R.converged).mean()
                r["bytes_on_todays_path"] = R.flows_still_ecmp.mean()
            s = S[f"{exp}{sfx}"]
            r["spans_always_dark"] = s.get("spans_always_empty")
            r["link_set_changes"] = s.get("link_set_changes_between_consecutive_hours")
            rows[exp] = r
        print(f"\nEVERYONE{sfx or ' (current network)'}  hours {len(R0)}")
        print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
