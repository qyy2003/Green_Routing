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


# --- total network traffic -------------------------------------------------------------

def test_total_counts_each_byte_once_and_bounds_local_turnaround():
    from green_routing.carbon import total
    H = pd.date_range("2026-04-01", periods=48, freq="h", tz="UTC")
    rng = np.random.default_rng(2)
    ab, local = rng.uniform(100, 200, 48), rng.uniform(10, 20, 48)    # A -> B via a trunk; A -> C locally
    # router r1: door a (customer A), door c (customer C), trunk t1; router r2: trunk t2, door b
    series = {"r1/a": (ab + local, 0 * ab), "r1/c": (0 * ab, local), "r1/t1": (0 * ab, ab),
              "r2/t2": (ab, 0 * ab), "r2/b": (0 * ab, ab)}
    traffic = {"links": np.array(list(series)), "hours": H.astype("int64") // 10**9,
               "hin": np.stack([v[0] for v in series.values()]), "hout": np.stack([v[1] for v in series.values()])}
    doors = pd.DataFrame({"link": ["r1/a", "r1/c", "r2/b"], "hidden_peer": [np.nan] * 3})
    T, din, dout, cov = total.total_series(traffic, doors)
    np.testing.assert_allclose(T * 1e3, ab + local)              # one-way bytes, trunks not counted
    c = total.conservation(din, dout, H)
    assert c["in_over_out"] == pytest.approx(1.0) and c["hourly_imbalance_median"] == pytest.approx(0)
    I = pd.DataFrame({"link": list(series), "dev": [x.split("/")[0] for x in series], "kind": "phys",
                      "bundle": np.nan, "role": "x"})
    TA = total.turnaround(traffic, I, doors, H).set_index("dev")
    assert TA.loc["r1", "local_lb_gbps"] == pytest.approx(local.mean() / 1e3)   # exact here: one trunk
    assert TA.loc["r2", "local_lb_gbps"] == pytest.approx(0)


def test_top_down_design_uses_only_the_week_before_the_issue_time():
    from green_routing.carbon import total
    H = pd.date_range("2026-04-01", periods=24 * 10, freq="h", tz="UTC")
    s = pd.Series(np.linspace(100, 200, len(H)), index=H)
    X, Y, base, t = total._design(s)
    assert len(t) == 24 * 3 and t[0] == pd.Timestamp("2026-04-08", tz="UTC")     # 7 days of history first
    s2 = s.copy()
    s2.loc["2026-04-08"] *= 5                                    # the target day's own values
    X2, _, base2, _ = total._design(s2)
    np.testing.assert_allclose(X[:24], X2[:24])
    np.testing.assert_allclose(base[:24], base2[:24])


# --- path-based rate --------------------------------------------------------------------

def test_ecmp_routing_splits_equal_cost_paths_and_respects_no_transit():
    from green_routing.carbon import tomography
    nodes = ["a", "b", "c", "d"]                    # a square: a-b-d and a-c-d cost the same
    edges = {(u, v): 1.0 for u, v in [("a", "b"), ("b", "a"), ("a", "c"), ("c", "a"),
                                      ("b", "d"), ("d", "b"), ("c", "d"), ("d", "c")]}
    links, A_link, A_node, od = tomography.ecmp_routing(nodes, edges, ["a", "d"])
    j = od.index(("a", "d"))
    col = dict(zip(links, A_link[:, j].toarray().ravel()))
    assert col[("a", "b")] == pytest.approx(0.5) and col[("c", "d")] == pytest.approx(0.5)
    assert A_node[:, j].toarray().ravel().tolist() == [1.0, 0.5, 0.5, 1.0]
    _, A2, _, od2 = tomography.ecmp_routing(nodes, edges, ["a", "d"], no_transit=["b"])
    assert dict(zip(links, A2[:, od2.index(("a", "d"))].toarray().ravel()))[("c", "d")] == pytest.approx(1.0)


def test_em_estimate_reproduces_counters_and_stays_non_negative():
    from green_routing.carbon import tomography
    nodes = ["a", "b", "c"]                          # a line a - b - c
    edges = {(u, v): 1.0 for u, v in [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]}
    links, A_link, _, od = tomography.ecmp_routing(nodes, edges, nodes)
    X = np.array([0, 5, 3, 2, 0, 4, 6, 1, 0], float)  # true matrix, s-major
    y = A_link @ X
    ing, egr = X.reshape(3, 3).sum(1), X.reshape(3, 3).sum(0)
    x = tomography.em(A_link, y, ing, egr, iters=2000)
    assert (x >= 0).all()
    np.testing.assert_allclose(x.reshape(3, 3).sum(1), ing, rtol=1e-6)
    np.testing.assert_allclose(A_link @ x, y, rtol=0.02)


@pytest.mark.parametrize("capacity", [None, np.array([50.0, 50.0, 50.0, 80.0, 80.0])])
def test_path_bills_add_up_to_the_emissions(capacity):
    from green_routing.carbon import path_rate, tomography
    import scipy.sparse as sp
    nodes = ["a", "b", "c"]
    edges = {(u, v): 1.0 for u, v in [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]}
    links, A_link, A_node, od = tomography.ecmp_routing(nodes, edges, nodes)
    S = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1]], float))   # two spans, both directions
    A_res = sp.vstack([A_node, S @ A_link]).tocsr()
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 10, (24, 9))
    E = rng.uniform(1, 5, (24, 5))
    b = path_rate.node_bills(A_res, E, X, 3, capacity=capacity)
    np.testing.assert_allclose(b.sum(axis=1), E.sum(axis=1))


# --- Abilene loader ----------------------------------------------------------------------

def test_abilene_clean_drops_corrupt_snapshots():
    from green_routing.dataset import abilene
    t = pd.date_range("2004-05-01", periods=288 * 3, freq="5min", tz="UTC")
    X = pd.DataFrame(np.ones((len(t), 4)), index=t)
    X.iloc[400] = 1000.0                                  # one corrupt snapshot
    Y = abilene.clean(X)
    assert Y.iloc[400].tolist() == [1.0] * 4              # removed, then interpolated
    assert np.isclose(Y.to_numpy(), 1.0).all()


def test_abilene_topology_sections_are_parsed_apart(tmp_path):
    from green_routing.dataset import abilene
    (tmp_path / "topo-2003-04-10.txt").write_text(
        "router\n#name city latitude longitude\nA Atl 33.7 -84.4\nB Chi 41.8 -87.6\n"
        "link\n#x y capacity(kbps) OSPF\nA B 9920000 1176\nB A 9920000 1176\n")
    (tmp_path / "links").write_text("A,B 1 0\nB,A 2 0\n*,A 3 1\nA,* 4 2\n*,B 5 1\nB,* 6 2\n")
    (tmp_path / "demands").write_text("A,A 1\nA,B 2\nB,A 3\nB,B 4\n")
    (tmp_path / "A").write_text("A,B A,B 1 2 1\nB,A B,A 2 3 1\n")
    import gzip
    rows = np.tile(np.r_[[10, 0, 0, 0, 0], [20, 0, 0, 0, 0], [30, 0, 0, 0, 0], [40, 0, 0, 0, 0]], (2016, 1))
    with gzip.open(tmp_path / "X06.gz", "wt") as fh:
        np.savetxt(fh, rows)
    ds = abilene.load(tmp_path, weeks=[6])
    assert ds["coord"]["A"] == (33.7, -84.4)              # router rows, not overwritten by link rows
    assert ds["cap"][("A", "B")] == pytest.approx(9920.0)
    S = abilene.series(ds)
    assert S["links"]["A->B"].iloc[0] == pytest.approx(20 * abilene.UNIT_TO_MBPS)
    assert S["total"].iloc[0] == pytest.approx(100 * abilene.UNIT_TO_MBPS)


# --- granularity ------------------------------------------------------------------------

def test_block_rates_conserve_emissions_and_bills_move_only_between_parties():
    from green_routing.carbon import granularity as G
    rng = np.random.default_rng(4)
    n = 288 * 2
    T = rng.gamma(4, 50, n)
    B = np.repeat(rng.uniform(1, 3, n // 12), 12)                  # hourly grid steps
    V = rng.gamma(2, 20, (n, 5))
    V = V / V.sum(axis=1, keepdims=True) * 2 * T[:, None]          # parties' in + out add up to 2 T
    r5 = B / (T * G.GB5)
    for k in G.GRAINS.values():
        rk = G.block_rate(B, T, k)
        np.testing.assert_allclose((rk * T * G.GB5).sum(), B.sum())          # same grams as the emissions
        np.testing.assert_allclose(G.bills(rk, V).sum(), G.bills(r5, V).sum())
    assert G.bill_error(r5, r5, V)[0] == 0 and G.daily_bill_error(r5, r5, V) == 0


def test_fourier_fit_recovers_a_daily_harmonic_and_schedule_finds_the_cheap_block():
    from green_routing.carbon import granularity as G
    x = np.arange(288) / 288 * 2 * np.pi
    path = 3 + np.cos(x) + 0.5 * np.sin(2 * x)
    np.testing.assert_allclose(G.fourier_fit(path[None, :], 2)[0], path, atol=1e-9)
    rate = np.tile(np.r_[np.full(144, 2.0), np.full(144, 1.0)], 3)       # cheap second half every day
    out = G.schedule(rate, rate, 12)
    assert out["realised"] == pytest.approx(1.0) and out["captured"] == pytest.approx(1.0)


def test_resolution_wape_falls_when_unforecastable_wiggle_averages_out():
    from green_routing.carbon import granularity as G
    rng = np.random.default_rng(1)
    hourly = np.repeat(rng.uniform(100, 200, 48), 12)
    actual = hourly + rng.normal(0, 20, hourly.size)               # within-hour noise nobody can forecast
    assert G.wape(hourly, actual, 12) < G.wape(hourly, actual, 1)
    assert abs(G.within_block_skill(hourly, actual)) < 1e-12       # a flat hour explains none of the wiggle
    assert G.within_block_skill(actual, actual) == pytest.approx(1.0)


# --- day-ahead forecasts of the total -------------------------------------------------------

def _hourly(days=70, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-09-01", periods=24 * days, freq="h", tz="UTC")
    lt = t.tz_convert("Europe/Zurich")
    v = 200 + 80 * np.sin((lt.hour - 6) / 24 * 2 * np.pi) - 60 * (lt.dayofweek >= 5) + rng.normal(0, 5, len(t))
    return pd.Series(v, index=t)


def test_calendar_marks_lecture_weeks_holidays_and_christmas():
    from green_routing.carbon import dayahead as DA
    c = DA.calendar_frame(pd.DatetimeIndex(["2025-09-14", "2025-09-15", "2025-12-19", "2025-12-24", "2026-05-14", "2026-06-04"]))
    assert c.lect.tolist() == [0, 1, 1, 0, 1, 0]                   # ISO weeks 38-51 and 8-22
    assert c.hol.tolist()[-2:] == [1.0, 0.27]                       # Ascension everywhere, Corpus Christi in catholic cantons
    assert c.xmas.tolist() == [0, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError):
        DA.calendar_frame(pd.DatetimeIndex(["2027-03-01"]))


def test_day_ahead_features_never_see_the_target_day():
    from green_routing.carbon import dayahead as DA
    s = _hourly()
    spec = DA.Spec(hist=48, lags=(1, 7, 14), daily=7, cal="full", holidays="weighted")
    X, y, base, t, day = DA.design(s, spec)
    s2 = s.copy()
    d = day[len(day) // 2]
    s2[d: d + pd.Timedelta(hours=23)] *= 10                         # change the target day only
    X2, y2, *_ = DA.design(s2, spec)
    upto = (day <= d)                                               # issue days up to the changed day
    pd.testing.assert_frame_equal(X[upto], X2[upto])                # see the same features
    assert not np.allclose(y[day == d], y2[day == d]) and not X[day > d].equals(X2[day > d])


def test_backtest_refits_and_recovers_a_regular_series():
    from green_routing.carbon import dayahead as DA
    s = _hourly()
    days = pd.date_range(s.index[-1].normalize() - pd.Timedelta(days=13), periods=14, freq="D")
    f = DA.backtest(s, s, DA.Spec(hist=24, lags=(1, 7, 14), daily=7, cal="time"), days, refit_every=7)
    assert len(f) == 14 * 24 and f.notna().all()
    idx = DA.common_hours({"f": f}, s, days)
    assert DA.score({"f": f}, s, idx).hourly_WAPE.iloc[0] < 0.05
    p = DA.paired(f, s.shift(168), s, idx)
    assert p["diff"] > 0 and p["lo"] <= p["diff"] <= p["hi"]


def test_cleaned_history_replaces_holidays_by_the_week_before():
    from green_routing.carbon import dayahead as DA
    s = _hourly()
    s["2025-10-02":"2025-10-03"] = 1.0                              # pretend it was a holiday ...
    table = (("2025-10-02", 1.0),)
    c = DA.cleaned(s, "table", table, tz="UTC")
    assert (c["2025-10-02"] == s["2025-09-25"].to_numpy()).all()    # ... replaced by the same day a week earlier
    assert (c["2025-10-03"] == 1.0).all()                           # other days untouched


# --- Internet2 (TSDS labels) and US grid factors -----------------------------------------

@pytest.mark.parametrize("desc,kind,ends,party", [
    ("BACKBONE: CORE1.CHIC-CORE2.KANS | I2-CHIC-KANS-VLAN-193447", "backbone_l3",
     ("core1.chic.net.internet2.edu", "core2.kans.net.internet2.edu"), None),
    ("BACKBONE: CORE1.ALBA-CORE1.BOST2 | I2-ALBA-BOST2-400GE-194095", "backbone_phys",
     ("core1.alba.net.internet2.edu", "core1.bost2.net.internet2.edu"), None),
    ("PDP CHIC-CONN-MERIT-2 400G | I2-CHIC-CHIC-LAG-194268", "door", None, "MERIT"),
    ("PDP CHAR-CONN-MCNC-2 2X100G | I2-CHAR-CHAR-LAG-1", "door", None, "MCNC"),
    ("PDP DENV-CONN-NOAA-3-1", "door", None, "NOAA"),
    ("[BE80] PDP CHIC-CONN-MERIT-2 | I2-CHIC-CHIC-400GE-194204", "member", None, None),
    ("PDP ELPA-SMN-1 | I2-ELPA-ELPA-10GE-192824", "mgmt", None, "SMN"),
    ("[RE] LEARN ~ AS16905 | I2-S12523", "other", None, None),
])
def test_internet2_port_labels(desc, kind, ends, party):
    from green_routing.dataset import internet2 as i2
    assert i2.classify(desc) == kind
    assert i2.backbone_ends(desc) == ends
    assert i2.party_of(desc) == party
    assert i2.pop_of("core1.chic.net.internet2.edu") == "chic"


def test_us_grid_intensity_is_generation_weighted():
    from green_routing.carbon import us_grid
    idx = pd.MultiIndex.from_tuples([("X", pd.Timestamp("2026-01-01", tz="UTC")),
                                     ("X", pd.Timestamp("2026-01-01 01:00", tz="UTC"))], names=["ba", "time"])
    gen = pd.DataFrame({"coal": [100.0, 0.0], "gas": [100.0, 0.0], "wind": [200.0, 0.0]}, index=idx)
    f = pd.Series({"coal": 1000.0, "gas": 400.0, "wind": 0.0})
    I = us_grid.intensity(gen, f)
    assert I.iloc[0] == pytest.approx((100 * 1000 + 100 * 400) / 400)
    assert np.isnan(I.iloc[1])                       # no generation reported: unknown, not zero
