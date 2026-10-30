"""Unit tests of the network and carbon packages: small synthetic inputs, no data source."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --- network ------------------------------------------------------------------

@pytest.mark.parametrize("name,link", [
    ("swiez2-400ge-0-0-0-3.switch.ch", "ez2/fourhundredgige0_0_0_3"),
    ("swice2-100ge-0-0-1-2.switch.ch", "ce2/hundredgige0_0_1_2"),
    ("swiag1-10ge-0-0-1-0.switch.ch", "ag1/tengige0_0_1_0"),
    ("swilg2-10ge-0-0-24.switch.ch", "lg2/tengigabitethernet0_0_24"),
    ("swizh3-b3.switch.ch", "zh3/bundle-ether3"),
    ("swiez31-p3-200.switch.ch", "ez31/port-channel3.200"),
    ("swiix3-e29-4-2005.switch.ch", "ix3/ethernet29_4.2005"),
    ("router.example.org", None),
])
def test_dns_names_map_to_export_interface_names(name, link):
    from green_routing.network import dns
    assert dns.iface(name) == link


def test_dns_link_map_pairs_31_30_and_64():
    from green_routing.network import dns
    v4 = [("130.59.1.0", "swia1-b1.switch.ch"), ("130.59.1.1", "swib1-b2.switch.ch"),          # /31
          ("130.59.2.0", "net-bb-p2p-c1-d1.switch.ch"), ("130.59.2.1", "swic1-b1.switch.ch"),
          ("130.59.2.2", "swid1-b1.switch.ch"),                                                # named /30
          ("130.59.3.0", "swie1-b9.switch.ch"), ("130.59.3.1", "customer.example.ch")]        # outside peer
    v6 = [("2001:620:0:10::1", "swif1-b1.switch.ch"), ("2001:620:0:10::2", "swig1-b4.switch.ch")]
    L = dns.link_map(v4, v6).set_index("link")
    assert L.loc["a1/bundle-ether1", "peer_link"] == "b1/bundle-ether2"
    assert L.loc["c1/bundle-ether1", "peer_link"] == "d1/bundle-ether1"
    assert L.loc["c1/bundle-ether1", "p2p_net"].startswith("net-bb-p2p")
    assert pd.isna(L.loc["e1/bundle-ether9", "peer_link"])
    assert L.loc["e1/bundle-ether9", "peer_name"] == "customer.example.ch"
    assert L.loc["f1/bundle-ether1", "peer_link"] == "g1/bundle-ether4"
    assert dns.live_routers(L.reset_index()) == {"a1", "b1", "c1", "d1", "e1", "f1", "g1"}


def test_match_ends_finds_the_two_ends_of_a_cable():
    from green_routing.network import matching
    rng = np.random.default_rng(0)
    T = 500
    x, y, z = (rng.gamma(2, 50, T) for _ in range(3))
    traffic = {"links": np.array(["r1/a", "r2/b", "r3/c"]),
               "win": np.stack([x, y, z]).astype(np.float32),       # r1 receives x, r2 receives y
               "wout": np.stack([y, x, rng.gamma(2, 50, T)]).astype(np.float32)}
    R = matching.good(matching.match_ends(traffic)).set_index("a")
    assert R.loc["r1/a", "b"] == "r2/b" and R.loc["r2/b", "b"] == "r1/a"
    assert "r3/c" not in R.index


def test_site_kind_speed_from_names():
    from green_routing.network import roles
    assert roles.site_of("ez2") == "ez" and roles.site_of("ez-ggc1") == "ez" and roles.site_of("lab-x1") == "lab"
    assert roles.iface_kind("bundle-ether1.200") == "sub" and roles.iface_kind("bvi2") == "virtual"
    assert roles.iface_speed("fourhundredgige0_0_0_1") == 400 and roles.iface_speed("tengigabitethernet0_0_24") == 10
    assert np.isnan(roles.iface_speed("bundle-ether1"))


# --- inventory ------------------------------------------------------------------

@pytest.mark.parametrize("cols,values,kind,watts", [
    (["Cisco_PSU_Pin"], [250.0], "Pin", 250.0),
    (["Cisco_PSU_Pin_mW"], [250000.0], "Pin_mW", 250.0),
    (["Cisco_PSU_U", "Cisco_PSU_I"], [12000.0, 15000.0], "UI", 180.0),
    (["PSU_Uin", "PSU_Iin"], [230.0, 5000.0], "UinIin", 115.0),
])
def test_psu_power_handles_all_four_schemas(tmp_path, cols, values, kind, watts):
    from green_routing.dataset import loaders
    ts = 1735689600 + 300 * np.arange(10)
    df = pd.DataFrame({"timestamp": ts, **{c: v for c, v in zip(cols, values)}})
    p = tmp_path / "x_psu0.csv"
    df.to_csv(p, index=False)
    s = loaders.psu_power(p)
    assert s.attrs["kind"] == kind
    assert s.iloc[0] == pytest.approx(watts)


def test_router_energy_counts_service_days_and_dns_live_routers():
    from green_routing.carbon import inventory
    dev = pd.DataFrame({"dev": ["swia1", "swib1", "swic1"], "first": ["2024-01-01", "2025-07-01", "2024-01-01"],
                        "last": ["2025-12-31", "2026-10-01", "2024-06-01"], "W_meas": [100.0, np.nan, 50.0],
                        "kind": ["Pin", np.nan, "UI"], "model": ["m1", "m1", "m2"]})
    T = inventory.router_power(dev, live={"c1", "x9"}, live_until="2026-10-30")
    S = inventory.router_energy(T, 2025).set_index("dev")
    assert S.loc["swia1", "days"] == 365 and S.loc["swia1", "MWh_W"] == pytest.approx(100 * 365 * 24 / 1e6)
    assert S.loc["swib1", "W"] == 100.0 and S.loc["swib1", "W_src"] == "model peers"   # peer median
    assert S.loc["swic1", "days"] == 365                                               # alive in DNS
    assert S.loc["swic1", "Whi"] == pytest.approx(50 / 0.85)                           # UI = DC side
    assert S.loc["swix9", "model"] == "unknown"                                        # never exported


# --- allocation and bills ----------------------------------------------------------

def _doors():
    return pd.DataFrame({"link": ["p1/x", "p1/y", "p1/z", "s1/u"], "dev": ["p1", "p1", "p1", "s1"],
                         "party": ["port:p1/x", "port:p1/y", "hidden:ix3", "site:s"],
                         "ptype": ["customer port", "customer port", "external, behind hidden border router", "customer site"],
                         "speed": [10.0, 30.0, 100.0, 1.0], "gbps": [1.0, 2.0, 10.0, 0.1]})


def test_access_split_by_door_speed_and_whole_access_routers():
    from green_routing.carbon import allocation
    D = pd.DataFrame({"dev": ["p1", "s1", "p2"], "tier": ["core", "access", "core"], "site": ["p", "s", "q"],
                      "access": [14.0, 5.0, 3.0]})
    acc = allocation.access_split(D, _doors())
    assert acc["site:s"] == 5.0                                      # access router: wholly its site's
    assert acc["port:p1/x"] == pytest.approx(14 * 10 / 140)          # PoP router: by door speed
    assert acc["hidden:ix3"] == pytest.approx(14 * 100 / 140)
    assert acc[None] == 3.0                                          # no doors: back to the shared pool


def test_hourly_bill_adds_up_to_the_network_emissions():
    from green_routing.carbon import bills
    H = pd.date_range("2026-04-01", periods=48, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    doors = _doors()
    V = pd.DataFrame(rng.gamma(2, 100, (48, 4)), index=H, columns=doors.link)
    I = pd.DataFrame({"CH": rng.uniform(5, 30, 48), "FR": rng.uniform(10, 40, 48)}, index=H)
    ACC = pd.Series({("site:s", "CH"): 80.0, ("port:p1/x", "CH"): 20.0, ("hidden:ix3", "FR"): 50.0})
    Pbb = {"CH": 1000.0, "FR": 300.0}
    parties = sorted(set(doors.party))
    acc_g, bb_g, Vp = bills.compute(V, I, ACC, Pbb, doors, parties)
    total = (150 * 0 + 100 / 1000 * I.CH + 50 / 1000 * I.FR + 1000 / 1000 * I.CH + 300 / 1000 * I.FR)
    np.testing.assert_allclose((acc_g + bb_g).sum(axis=1), total)                # nothing lost or double-counted
    share = bb_g.div(bb_g.sum(axis=1), axis=0)
    np.testing.assert_allclose(share.values, Vp.div(Vp.sum(axis=1), axis=0).values)   # backbone by bytes


def test_bill_errors_are_zero_for_the_settled_bill_itself():
    from green_routing.carbon import bills
    H = pd.date_range("2026-04-01", periods=72, freq="h", tz="UTC")
    a = pd.DataFrame(np.ones((72, 2)), index=H, columns=["k1", "k2"])
    res = {"settled": (a, a, a), "same": (a, a, a), "double": (2 * a, 2 * a, a)}
    M = bills.errors(res).set_index("scenario")
    assert M.loc["same", "WAPE_daily"] == 0
    assert M.loc["double", "WAPE_daily"] == pytest.approx(1.0) and M.loc["double", "total_bill_error"] == pytest.approx(1.0)


def test_day_ahead_grid_rules_use_only_the_previous_day():
    from green_routing.carbon import grid
    H = pd.date_range("2026-04-01", periods=24 * 10, freq="h", tz="UTC")
    s = pd.Series(np.arange(len(H), dtype=float), index=H)
    f = grid.day_ahead(s, "yesterday same hour")
    assert f.iloc[24 * 5 + 7] == s.iloc[24 * 4 + 7]
    for rule in grid.RULES:                       # changing today's data never changes today's forecast
        s2 = s.copy()
        s2.loc["2026-04-06"] += 1000
        np.testing.assert_allclose(grid.day_ahead(s2, rule).loc["2026-04-06"], grid.day_ahead(s, rule).loc["2026-04-06"])
