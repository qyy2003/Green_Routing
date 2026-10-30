"""``gr-carbon`` — energy inventory, grid factors, allocation and carbon bills of SWITCHlan.

    grid        Electricity Maps history: download (needs EM_API, --download) and
                process into hourly CSVs; score the day-ahead grid-factor rules
    inventory   PSU survey, router models (Zoo match + rules), router power, optical
                shelves, site energy and location-based Scope 2 for --year
    dependence  how much of each metered router's power follows its traffic (2024)
    allocate    port functions, access / backbone pools, doors and parties, annual bills
    bills       hourly settled and day-ahead forecast bills over a forecasting run's
                test block (--run), with error decomposition and the scheduling test
    total       total network traffic (each byte once): conservation and turnaround
                checks, top-down vs bottom-up day-ahead forecasts, the network g/GB signal
    all         grid, inventory, dependence, allocate, bills, total

Needs ``gr-network all`` first (interfaces.csv, routers.csv, dns_links.csv). Outputs go to
``$GR_ARTIFACTS_DIR/carbon/<step>/`` (``--out``); curated inputs come from
``$GR_REFERENCE_DIR`` (sites.csv, external_doors.csv, zoo/example.zip, dns/sweep_date.txt).
Electricity Maps data is licensed: never commit or publish the grid/ directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..network import dns
from ..paths import artifacts_dir, reference_dir
from . import allocation, bills, grid, inventory, total

DEFAULT_RUN = "pred_year/merged"


class Dirs:
    def __init__(self, a):
        base = Path(a.out) if a.out else artifacts_dir() / "carbon"
        self.inv, self.grid, self.alloc, self.bills = (base / x for x in ("inventory", "grid", "allocation", "bills"))
        self.net = Path(a.network) if a.network else artifacts_dir() / "network"
        self.ref = reference_dir()
        self.run = Path(a.run) if a.run else artifacts_dir() / DEFAULT_RUN
        for d in (self.inv, self.grid, self.alloc, self.bills):
            d.mkdir(parents=True, exist_ok=True)

    def sites(self) -> pd.DataFrame:
        return pd.read_csv(self.ref / "sites.csv")

    def sweep_date(self) -> str:
        f = self.ref / "dns" / "sweep_date.txt"
        return f.read_text().strip() if f.exists() else pd.Timestamp.now().strftime("%Y-%m-%d")


def cmd_grid(a, d: Dirs):
    if a.download:
        print("[grid] requests made:", grid.download(d.grid / "raw"))
    ci = grid.process(d.grid / "raw", d.grid)
    for z, s in ci.items():
        m = s.groupby(s.index.year).ci_direct.mean().round(1).to_dict()
        print(f"[grid] {z}: direct g/kWh by year {m}")
    if (d.run / "dense" / "meta.json").exists():
        H = bills.window(d.run)
        R = grid.evaluate_rules(grid.load(d.grid, "CH"), H[0].strftime("%Y-%m-%d"),
                                (H[-1] + pd.Timedelta("1h")).strftime("%Y-%m-%d"))
        R.to_csv(d.grid / "rules.csv", index=False)
        print("[grid] day-ahead rules over the bill window (CH, g/kWh):")
        print(R.round(2).to_string(index=False))


def cmd_inventory(a, d: Dirs):
    stats = pd.read_csv(d.net / "iface_stats.csv")
    stats[["first_data", "last_data"]] = stats[["first_data", "last_data"]].fillna("")
    print("[inventory] PSU survey ...", flush=True)
    psu = inventory.psu_survey()
    psu.to_csv(d.inv / "psu_survey.csv", index=False)
    print("[inventory] router schemas ...", flush=True)
    schemas = inventory.router_schemas()
    print("[inventory] Zoo match ...", flush=True)
    zoo = inventory.zoo_match(d.ref / "zoo" / "example.zip", workers=a.workers)
    zoo.to_csv(d.inv / "zoo_match.csv", index=False)
    print(f"[inventory] Zoo routers {len(zoo)}, matched {zoo.dev.notna().sum()} ({zoo.dev.dropna().nunique()} unique)")
    T = inventory.devices(stats, schemas, psu, zoo)
    T.to_csv(d.inv / "devices.csv", index=False)
    live = dns.live_routers(pd.read_csv(d.net / "dns_links.csv"))
    R = inventory.router_power(T, live, d.sweep_date())
    R.to_csv(d.inv / "routers.csv", index=False)
    S = inventory.router_energy(R, a.year)
    S.to_csv(d.inv / f"routers_{a.year}.csv", index=False)
    print(f"[inventory] {len(S)} routers in service in {a.year}: {S.MWh_W.sum():.1f} MWh "
          f"({S.MWh_Wlo.sum():.1f}-{S.MWh_Whi.sum():.1f}); measured share "
          f"{S.loc[S.W_src == 'measured 2024', 'MWh_W'].sum() / S.MWh_W.sum():.0%}")
    print("[inventory] optical shelves ...", flush=True)
    E = inventory.shelves(workers=min(a.workers, 8))
    E.to_csv(d.inv / "shelves.csv", index=False)
    U = inventory.site_energy(S, E, d.sites(), a.year)
    U.to_csv(d.inv / f"site_energy_{a.year}.csv", index=False)
    print(f"[inventory] {a.year}: routers {U[U.layer == 'router'].MWh.sum():.1f} MWh, optical "
          f"{U[U.layer == 'optical'].MWh.sum():.1f} MWh IT; site {U.site_MWh.sum():.1f} MWh "
          f"({U.site_MWh_lo.sum():.0f}-{U.site_MWh_hi.sum():.0f})")
    if (d.grid / "em_CH_hourly.csv").exists():
        s2 = inventory.scope2(U, grid.annual_means(d.grid, a.year, zones=sorted(U.zone.unique())))
        s2.to_csv(d.inv / f"scope2_{a.year}.csv", index=False)
        print(s2.round(1).to_string(index=False))


def cmd_dependence(a, d: Dirs):
    T = inventory.traffic_dependence(workers=a.workers)
    T.to_csv(d.inv / "power_vs_traffic.csv", index=False)
    print(f"[dependence] {len(T)} routers: traffic-dependent share of power median "
          f"{T.dyn_share.median():.2%}, max {T.dyn_share.max():.2%}; under 1 % on {(T.dyn_share.abs() < 0.01).sum()}")


def cmd_allocate(a, d: Dirs):
    I = pd.read_csv(d.net / "interfaces.csv")
    tiers = pd.read_csv(d.net / "routers.csv").set_index("dev").tier
    U = pd.read_csv(d.inv / f"site_energy_{a.year}.csv")
    P = allocation.port_functions(I)
    P.to_csv(d.alloc / "ports.csv", index=False)
    D = allocation.device_pools(U, P, tiers)
    D.to_csv(d.alloc / "device_pools.csv", index=False)
    pools = allocation.pools(D, U)
    json.dump(pools, open(d.alloc / "pools.json", "w"), indent=1)
    E = allocation.doors(I, D, pd.read_csv(d.ref / "external_doors.csv"))
    E.to_csv(d.alloc / "doors.csv", index=False)
    ci = grid.annual_means(d.grid, a.year, zones=["CH"])["CH"]["direct"] if (d.grid / "em_CH_hourly.csv").exists() else float("nan")
    T = allocation.annual_bills(D, E, pools, ci, d.sites())
    T.rename_axis("party").to_csv(d.alloc / f"annual_bills_{a.year}.csv")
    print(f"[allocate] access pool {pools['access_MWh']:.0f} MWh, backbone pool {pools['backbone_MWh']:.0f} MWh "
          f"(optical {pools['optical_MWh']:.0f}); {len(E)} doors, {len(T)} parties")
    print(T.groupby("ptype").agg(n=("MWh", "size"), traffic_Gbps=("traffic_Gbps", "sum"),
                                 access=("access_MWh", "sum"), backbone=("backbone_MWh", "sum")).round(1).to_string())


def cmd_bills(a, d: Dirs):
    bills.run(d.run, d.inv, d.alloc, d.grid, d.net, d.sites(), d.bills, grid_rule=a.grid_rule)


def cmd_total(a, d: Dirs):
    total.run(d.run, d.net, d.alloc, d.inv, d.grid, d.sites(), d.bills.parent / "total", grid_rule=a.grid_rule)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gr-carbon", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["grid", "inventory", "dependence", "allocate", "bills", "total", "all"])
    ap.add_argument("--year", type=int, default=2025, help="inventory / allocation year (default 2025)")
    ap.add_argument("--run", default=None, help=f"forecasting run with dense logs (default: {DEFAULT_RUN})")
    ap.add_argument("--out", default=None, help="default: $GR_ARTIFACTS_DIR/carbon")
    ap.add_argument("--network", default=None, help="gr-network output (default: $GR_ARTIFACTS_DIR/network)")
    ap.add_argument("--download", action="store_true", help="grid: fetch from Electricity Maps first")
    ap.add_argument("--grid-rule", default=grid.DEFAULT_RULE, choices=list(grid.RULES),
                    help="day-ahead grid-factor forecast for the bills")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    d = Dirs(a)
    steps = {"grid": [cmd_grid], "inventory": [cmd_inventory], "dependence": [cmd_dependence],
             "allocate": [cmd_allocate], "bills": [cmd_bills], "total": [cmd_total],
             "all": [cmd_grid, cmd_inventory, cmd_dependence, cmd_allocate, cmd_bills, cmd_total]}[a.command]
    for f in steps:
        f(a, d)


if __name__ == "__main__":
    main()
