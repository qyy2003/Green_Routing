"""``gr-network`` — link ends, router tiers and interface roles of SWITCHlan.

    extract     read every interface CSV once -> iface_stats.csv, traffic.npz
    roles       match link ends, find bundle members, classify tiers and roles
                -> interfaces.csv, routers.csv, bundles.csv, matches/, panel_links.csv
    dns-sweep   refresh the reverse-DNS snapshots in the reference directory (needs dig;
                65,536 IPv4 PTR queries, then the IPv6 walk)
    dns         far ends from the DNS snapshots -> dns_links.csv, and dns_* columns
                in interfaces.csv
    all         extract, roles, dns

Outputs go to ``$GR_ARTIFACTS_DIR/network`` (``--out``). Inputs that are not in the
bucket come from ``$GR_REFERENCE_DIR`` (sites.csv, dns/) and ``web/data/links.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import artifacts_dir, reference_dir
from . import dns, matching, roles, traffic

DEFAULT_RUN = "pred_year/merged"     # the year benchmark: its links get a panel table
PTR_V4, PTR_V6 = "ptr_130.59.0.0_16.txt", "ptr_v6_2001-620-0.txt"


def cmd_extract(a):
    traffic.extract(a.out, workers=a.workers)


def cmd_roles(a):
    out = Path(a.out)
    stats = pd.read_csv(out / "iface_stats.csv")
    T = dict(np.load(out / "traffic.npz"))
    partners = matching.match_all(T, out)
    run = Path(a.run) if a.run else artifacts_dir() / DEFAULT_RUN
    panel = json.load(open(run / "config.json"))["links"] if (run / "config.json").exists() else []
    s = roles.interfaces(stats, partners, panel)
    B = matching.find_bundles(s, T)
    B.to_csv(out / "bundles.csv", index=False)
    sites = pd.read_csv(reference_dir() / "sites.csv")
    s = roles.roles(roles.tiers(s, B, sites)).reset_index(drop=True)
    s[roles.COLUMNS].sort_values("link").to_csv(out / "interfaces.csv", index=False)
    roles.routers(s).to_csv(out / "routers.csv")
    if panel:
        P = roles.panel_roles(s)
        P[roles.COLUMNS + ["group", "redundant"]].sort_values("link").to_csv(out / "panel_links.csv", index=False)
    T_ = s.groupby("dev").tier.first()
    print("[network] routers by tier:", T_.value_counts().to_dict())
    print("[network] interfaces by role:", s.role.value_counts().to_dict())


def cmd_dns_sweep(a):
    d = reference_dir() / "dns"
    d.mkdir(parents=True, exist_ok=True)
    print("[dns] IPv4 names:", dns.sweep_v4(d / PTR_V4, workers=a.workers))
    print("[dns] IPv6 names:", dns.walk_v6(d / PTR_V6, workers=a.workers))
    (d / "sweep_date.txt").write_text(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d") + "\n")


def cmd_dns(a):
    out, d = Path(a.out), reference_dir() / "dns"
    v6 = dns.read_ptr(d / PTR_V6) if (d / PTR_V6).exists() else []
    L = dns.link_map(dns.read_ptr(d / PTR_V4), v6)
    L.to_csv(out / "dns_links.csv", index=False)
    I = pd.read_csv(out / "interfaces.csv")
    I = dns.join(I.drop(columns=[c for c in I if c.startswith("dns_")]), L)
    I.to_csv(out / "interfaces.csv", index=False)
    act = I[I.role != "idle/absent"]
    m = act[act.peer.notna() & act.dns_peer.notna()]
    same = (m.peer.str.split("/").str[0] == m.dns_peer.str.split("/").str[0]).sum()
    print(f"[dns] {L.link.notna().sum()} named router interfaces, {L.peer_link.notna().sum()} with a "
          f"SWITCH far end; traffic-matched links also in DNS {len(m)}, same far-end router {same}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gr-network", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["extract", "roles", "dns-sweep", "dns", "all"])
    ap.add_argument("--out", default=None, help="default: $GR_ARTIFACTS_DIR/network")
    ap.add_argument("--run", default=None,
                    help=f"forecasting run whose links get panel_links.csv (default: {DEFAULT_RUN})")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    a.out = str(Path(a.out) if a.out else artifacts_dir() / "network")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    steps = {"extract": [cmd_extract], "roles": [cmd_roles], "dns-sweep": [cmd_dns_sweep],
             "dns": [cmd_dns], "all": [cmd_extract, cmd_roles, cmd_dns]}[a.command]
    for f in steps:
        f(a)


if __name__ == "__main__":
    main()
