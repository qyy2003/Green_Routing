"""Switch LAN, test block (periods.TEST): path rates under the paper's idle-link cap,
per city pair and per entity (paper Fig. 2), next to the flat rate. Same network, routing
(ECMP, inverse capacity), counter-estimated matrices and resources as recipes/carbon_rates/path_rate.py
(it must have run: carbon_rates/path/tm_capacity.npz, the counter-estimated matrices of the year run's bill
window 2026-03-22 -> 07-10, which contains the test block). The cap of day d uses the 28 days before d within
that window (for the first days of the test block, the days from 2026-03-22 on).

Entity bills add each party's dedicated access equipment (carbon/bills/party_bills.csv, access_g)
to its backbone bill; annualised (x 8760 / hours).

usage: python switch_rates.py      writes $GR_ARTIFACTS_DIR/paper/switch/{citypairs.csv, entity_bills.csv, summary.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.carbon import assumptions as A, bills, grid, path_rate
from green_routing.network.roles import site_of
from green_routing.paths import ARTIFACTS_DIR, reference_dir

C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
OUT = ARTIFACTS_DIR / "paper" / "switch"
RUN = ARTIFACTS_DIR / "pred_year" / "merged"
GB = A.GB_PER_MBPS_HOUR
sys.path.insert(0, str(Path(__file__).resolve().parent))
import periods as PER                                                # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    I = pd.read_csv(NET / "interfaces.csv")
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    traffic = dict(np.load(NET / "traffic.npz"))
    sites = pd.read_csv(reference_dir() / "sites.csv")
    Hf = bills.window(RUN)
    Xf = np.load(ARTIFACTS_DIR / "carbon_rates" / "path" / "tm_capacity.npz")["X"]

    def allocation(H):
        net = path_rate.Network(I, doors, traffic, H)
        net.route("capacity", sites)
        R, E = bills.window_units(pd.read_csv(C / "inventory" / "routers.csv"), pd.read_csv(C / "allocation" / "device_pools.csv"),
                                  pd.read_csv(C / "inventory" / "shelves.csv"), sites,
                                  pd.read_csv(NET / "routers.csv").set_index("dev").tier, H)
        router_W, span_W, unplaced = path_rate.resources(net, R, E, doors)
        IA = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in ("CH", "FR")})
        return net, path_rate.Allocation(net, router_W, span_W, unplaced, IA)

    _, alf = allocation(Hf)
    multf = path_rate.idle_cap(alf.E, (alf.A_res @ Xf.T).T, Hf)          # 28 days back within the bill window
    sel = Hf.isin(PER.hours(PER.TEST, tz=Hf.tz))
    assert sel.sum() == len(PER.hours(PER.TEST)), "test block not inside the bill window"
    H, X, mult = Hf[sel], Xf[sel], multf[sel]
    net, al = allocation(H)
    U = (al.A_res @ X.T).T
    res = al.bills(X, cap_mult=mult)
    total = res["total_g"].sum()
    flat_rate = total / (X.sum() * GB)
    capped = (U > 0) & (al.E > 0)
    cap_val = mult[:, None] * al.E.sum(1, keepdims=True) / U.sum(1, keepdims=True)
    share_capped = float(((al.E / np.where(U > 0, U, 1)) > cap_val)[capped].mean())

    # city pairs: end nodes -> site -> town
    town = sites.set_index("code").town.str.split(" \\(").str[0]
    city = {e: town.get(site_of(e), site_of(e)) for e in net.ends}
    od = res["od_g_per_GB"]
    gb = res["od_GB"]
    df = pd.DataFrame({"s": [city[s] for s, _ in net.od], "t": [city[t] for _, t in net.od],
                       "g": (od.fillna(0) * gb).to_numpy(), "GB": gb.to_numpy()})
    df = df[(df.s != df.t) & (df.GB > 0)]
    cp = df.groupby(["s", "t"])[["g", "GB"]].sum()
    # the flat add-on (emissions spread over every byte) is part of every path's rate
    cp["g_per_GB"] = cp.g / cp.GB + res["unallocated_g"].sum() / (X.sum() * GB)
    cp = cp.reset_index()[["s", "t", "GB", "g_per_GB"]]
    cp.to_csv(OUT / "citypairs.csv", index=False)

    # entity bills: backbone (flat or path, capped) + dedicated access equipment
    pb = pd.read_csv(C / "bills" / "party_bills.csv", index_col=0)
    eb = pd.DataFrame({"backbone_flat_g": res["flat_g"].sum(), "backbone_path_g": res["path_g"].sum(),
                       "GB_in_out": al.V.sum() * GB})
    eb["type"] = pb.type.reindex(eb.index)
    # dedicated access equipment over the test block: static site power per party and zone x the zone's
    # hourly factor (carbon.bills.compute's access part; party_bills.csv holds the year run's bill window)
    ACC, _, _ = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"), pd.read_csv(C / "allocation" / "device_pools.csv"),
                                   pd.read_csv(C / "inventory" / "shelves.csv"), sites, doors,
                                   pd.read_csv(NET / "routers.csv").set_index("dev").tier, H)
    IA = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in ("CH", "FR")})
    acc = pd.Series({k: v / 1e3 * float(IA[z].sum()) for (k, z), v in ACC.items()}).groupby(level=0).sum()
    eb["access_g"] = acc.reindex(eb.index).fillna(0)
    eb = eb[eb.GB_in_out > 0]
    scale = 8760 / len(H)
    eb["flat_kg_per_year"] = (eb.backbone_flat_g + eb.access_g) * scale / 1e3
    eb["path_kg_per_year"] = (eb.backbone_path_g + eb.access_g) * scale / 1e3
    eb = eb.sort_values("flat_kg_per_year", ascending=False)
    eb.to_csv(OUT / "entity_bills.csv")
    summ = {"hours": len(H), "window": [str(H[0]), str(H[-1])], "flat_rate_g_per_GB": float(flat_rate),
            "cap_multiple_p10_p50_p90": np.quantile(mult, [.1, .5, .9]).tolist(),
            "cap_over_flat_rate_median": float(np.median(mult * (al.E.sum(1) / U.sum(1)) / (res["total_g"].to_numpy() / X.sum(1)))),
            "resource_hours_capped_share": share_capped,
            "spread_flat_share": float(res["unallocated_g"].sum() / total),
            # without the cap: idle resources and power no route reaches (carbon_rates' spread_flat_share)
            "spread_flat_share_uncapped": float(al.bills(X)["unallocated_g"].sum() / total),
            "resources": int(U.shape[1]), "resources_below_1_mbps_mean": int((U.mean(0) < 1.0).sum()),
            "path_t": float(res["path_g"].sum().sum() / 1e6), "flat_t": float(res["flat_g"].sum().sum() / 1e6),
            "citypairs": int(len(cp)), "citypair_g_per_GB_p10_p50_p90": cp.g_per_GB.quantile([.1, .5, .9]).tolist(),
            "entities": int(len(eb)),
            "entity_kg_per_year_p10_p50_p90": eb.flat_kg_per_year.quantile([.1, .5, .9]).tolist(),
            "entity_top10_share": float(eb.flat_kg_per_year.nlargest(10).sum() / eb.flat_kg_per_year.sum()),
            "moved_share_path_vs_flat": float((eb.backbone_path_g - eb.backbone_flat_g).abs().sum() / eb.backbone_flat_g.sum() / 2),
            "access_share_of_bills": float(eb.access_g.sum() / (eb.access_g.sum() + eb.backbone_flat_g.sum())),
            "access_g_parties_without_traffic": float(acc.drop(eb.index, errors="ignore").sum())}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(eb.head(12).round(2).to_string())


if __name__ == "__main__":
    main()
