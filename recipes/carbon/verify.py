"""Check a carbon run against the reference numbers in expected.json.

usage: python verify.py              compare $GR_ARTIFACTS_DIR/{network,carbon} with expected.json
       python verify.py --snapshot   write expected.json from the current outputs (trusted runs only)

Counts must match exactly; energies, emissions and errors within the tolerance stored
next to each value. A mismatch usually means the data changed (the bucket is rewritten
daily, the reverse DNS and Electricity Maps' estimates change over time), so read which
check failed before suspecting the code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
NET, C = ARTIFACTS_DIR / "network", ARTIFACTS_DIR / "carbon"


def measure() -> dict:
    """{name: (value, tolerance)}; tolerance 0 = exact, < 1 relative, 'abs:x' absolute."""
    I = pd.read_csv(NET / "interfaces.csv")
    R = pd.read_csv(NET / "routers.csv")
    year = int(sorted((C / "inventory").glob("routers_*.csv"))[-1].stem.split("_")[1])
    RY = pd.read_csv(C / "inventory" / f"routers_{year}.csv")
    U = pd.read_csv(C / "inventory" / f"site_energy_{year}.csv")
    s2 = pd.read_csv(C / "inventory" / f"scope2_{year}.csv").set_index(["pue", "basis"]).tCO2e
    pools = json.load(open(C / "allocation" / "pools.json"))
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    S = json.load(open(C / "bills" / "summary.json"))
    E = pd.read_csv(C / "bills" / "bill_errors.csv").set_index("scenario").WAPE_daily
    m = {
        "network.interfaces": (len(I), 0),
        "network.core_routers": (int((R.tier == "core").sum()), 0),
        "network.bundles": (len(pd.read_csv(NET / "bundles.csv")), 0),
        "network.dns_far_ends": (int(I.dns_peer.notna().sum()), 0),
        "inventory.year": (year, 0),
        "inventory.routers_in_service": (len(RY), 0),
        "inventory.router_MWh": (float(RY.MWh_W.sum()), 0.005),
        "inventory.optical_MWh": (float(U[U.layer == "optical"].MWh.sum()), 0.01),
        "inventory.site_MWh": (float(U.site_MWh.sum()), 0.01),
        "inventory.scope2_direct_t": (float(s2["central", "direct"]), 0.02),
        "allocation.access_MWh": (pools["access_MWh"], 0.01),
        "allocation.backbone_MWh": (pools["backbone_MWh"], 0.01),
        "allocation.doors": (len(doors), 0),
        "bills.parties": (S["parties"], 0),
        "bills.model_doors": (S["model_doors"], 0),
        "bills.access_kW": (S["access_kW"], 0.01),
        "bills.backbone_kW": (S["backbone_kW"], 0.01),
        "bills.settled_t": (S["settled_t_total"], 0.02),
        "bills.best_scenario": (S["best_scenario"], 0),
    }
    for sc in ("patchtst + grid actual", "persistence + grid actual", "patchtst + grid fc", "traffic actual + grid fc"):
        m[f"bills.WAPE_daily[{sc}]"] = (float(E[sc]), "abs:0.005")
    return m


def _fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def main():
    m = measure()
    path = HERE / "expected.json"
    if "--snapshot" in sys.argv:
        json.dump({k: {"value": v, "tol": t} for k, (v, t) in m.items()}, open(path, "w"), indent=1)
        print("wrote", path)
        return 0
    exp = json.load(open(path))
    fails = 0
    for k, e in exp.items():
        got, want, tol = m.get(k, (None,))[0], e["value"], e["tol"]
        if got is None:
            ok = False
        elif isinstance(tol, str):
            ok = abs(got - want) <= float(tol.split(":")[1])
        elif tol == 0:
            ok = got == want
        else:
            ok = abs(got - want) <= tol * abs(want)
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {k:42s} {_fmt(got):>20}  expected {_fmt(want)} (tol {tol})")
    print(f"{len(exp) - fails} PASS, {fails} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
