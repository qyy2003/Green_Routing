"""Hourly carbon bills per party: settled after the fact, and forecast a day ahead.

Settled bill of party k in hour h::

    C_k(h) = sum_d P^acc_{d,k} * I_{z(d)}(h)  +  B(h) * V_k(h) / sum_j V_j(h)
    B(h)   = sum_d P^bb_d * I_{z(d)}(h)  +  sum_e P_e * I_{z(e)}(h)

P^acc_{d,k}: access power of router d billed to k (access routers wholly to their site,
PoP routers by door speed); P^bb_d: its backbone share; P_e: optical shelf e; all at the
site, i.e. times PUE. I_z(h): hourly grid factor of the site's zone. V_k(h): in + out
volume through k's doors. Emissions, not energy, are pooled: each device's power is
multiplied by its own zone's factor before the split.

Forecast bill: the same formula with V-hat (the traffic models' day-ahead block
forecasts, issued 00:00 UTC) and I-hat (``grid.day_ahead``). Energy is static (router
power is < 1 % traffic-dependent), so its "forecast" is its measured/modelled value.

Scenarios separate the two error sources: each traffic model with the forecast grid
factor and with the actual one, and actual traffic with the forecast factor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import loaders
from . import allocation, assumptions as A, grid

MODELS = ("persistence", "seasonal_naive_day", "seasonal_naive_week", "dlinear", "gru", "patchtst", "tcn")
GB = A.GB_PER_MBPS_HOUR


# --- inputs over the window ---------------------------------------------------------

def window(run_dir: str | Path) -> pd.DatetimeIndex:
    """The hours of a run's dense test block (whole hours only)."""
    meta = json.load(open(Path(run_dir) / "dense" / "meta.json"))
    t0 = pd.Timestamp(meta["t0"], unit="s", tz="UTC")
    return pd.date_range(t0, periods=meta["n_targets"] // 12, freq="h")


def actual_volumes(doors: pd.DataFrame, H: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly in+out Mbit/s through every door over the hours ``H`` (hours with >= 6 of 12
    samples; gaps up to 6 h interpolated). Also returns ``exists``: False before a port's
    first sample. Pass ``H`` with a week of history when the result feeds the fallback
    forecast (same hour last week)."""
    uri = {f"{d}/{i}": u for d, i, u in loaders.iter_interface_files()}

    def one(link):
        try:
            s = loaders.interface_throughput(uri[link])["total_mbps"]
        except Exception:          # noqa: BLE001 -- a door without data bills 0 bytes
            return pd.Series(np.nan, index=H)
        s = s[(s.index >= H[0]) & (s.index < H[-1] + pd.Timedelta("1h"))]
        g = s.resample("1h")
        return g.mean().where(g.count() >= 6).reindex(H)
    VA = pd.DataFrame({link: one(link) for link in doors.link})
    exists = VA.notna().cumsum() > 0
    return VA.interpolate(limit=6, limit_area="inside").fillna(0), exists


def forecast_volumes(run_dir: str | Path, model: str, doors: pd.DataFrame, members: pd.Series,
                     H: pd.DatetimeIndex, VA: pd.DataFrame, exists: pd.DataFrame,
                     horizon: str = "day") -> tuple[pd.DataFrame, dict]:
    """Hourly forecast volumes of every door from a run's dense block log.

    A door in the panel uses its own forecast; a bundle whose members are all in the
    panel the members' sum; any other door "same hour last week" of its actual series
    (known at issue time). Returns the volumes and the source of each door."""
    run_dir = Path(run_dir)
    meta = json.load(open(run_dir / "dense" / "meta.json"))
    links = json.load(open(run_dir / "config.json"))["links"]
    li = {x: i for i, x in enumerate(links)}
    hi = {h["name"]: i for i, h in enumerate(meta["horizons"])}[horizon]
    arr = np.load(run_dir / "dense" / f"pred_block_{model}.npy", mmap_mode="r")
    NH = len(H)
    out, src = {}, {}
    for link in doors.link:
        mem = members.get(link, [])
        rows = [li[link]] if link in li else ([li[m] for m in mem] if mem and all(m in li for m in mem) else [])
        if rows:
            p = np.asarray(arr[rows, hi, :NH * 12], dtype=np.float64).sum(axis=0)
            out[link] = pd.Series(p.reshape(NH, 12).mean(axis=1), index=H)
            src[link] = "model"
        else:
            out[link] = VA[link].shift(168).reindex(H)
            src[link] = "fallback: same hour last week"
    VF = pd.DataFrame(out).clip(lower=0)
    last_week = VA.shift(168).reindex(H)          # VA may start before H (history)
    return VF.where(exists.reindex(H, fill_value=False), 0).fillna(last_week).fillna(0), src


def window_units(R: pd.DataFrame, D: pd.DataFrame, shelves: pd.DataFrame, sites: pd.DataFrame,
                 tiers: pd.Series, H: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Routers and optical shelves with their site power in the window (W, i.e. x PUE and
    x the share of the window in service), access share ``f_acc`` and grid zone.

    ``R``: routers with W, first, last (inventory.router_power); ``D``: device pools of the
    annual allocation (their access share f_acc is reused; routers missing there get their
    tier's median); ``shelves``: ECI table (mean W of the window's year, else the year
    before)."""
    ht = sites.drop_duplicates("code").set_index("code").host_type
    R = R.copy()
    R["d"] = R.dev.str.replace(r"^swi", "", regex=True)
    f = pd.to_datetime(R["first"]).dt.tz_localize("UTC")
    l = pd.to_datetime(R["last"]).dt.tz_localize("UTC")
    span = (H[-1] - H[0]).total_seconds()
    R["frac"] = ((l.clip(upper=H[-1]) - f.clip(lower=H[0])).dt.total_seconds() / span).clip(0, 1).fillna(0)
    R["W_site"] = R.W * R.site.map(ht).map(A.pue) * R.frac
    P = D.set_index("dev")
    fa = (P.access / (P.access + P.backbone)).replace([np.inf, np.nan], 0)
    med = fa.groupby(P.tier).median()
    R["tier"] = R.d.map(tiers).fillna("access")
    R["f_acc"] = [fa.get(d, 1.0 if t == "access" else med.get(t, 0.3)) for d, t in zip(R.d, R.tier)]
    R["zone"] = R.site.map(A.zone_of)
    y = H[0].year
    E = shelves[pd.to_datetime(shelves["last"]) >= H[0].tz_localize(None)].copy()
    E["W_site"] = E[f"W{y}"].fillna(E[f"W{y - 1}"]) * E.site.map(ht).map(A.pue)
    E["zone"] = E.site.map(A.zone_of)
    return R, E


def window_power(R: pd.DataFrame, D: pd.DataFrame, shelves: pd.DataFrame, sites: pd.DataFrame,
                 doors: pd.DataFrame, tiers: pd.Series, H: pd.DatetimeIndex):
    """Site power in the window: access power per (party, zone) and backbone power per zone (W).
    A PoP router's access share with no door on it joins the shared pool (``window_units``
    for the inputs)."""
    R, E = window_units(R, D, shelves, sites, tiers, H)
    zones = sorted(set(R.zone) | set(E.zone))
    Pbb = {z: (R.W_site * (1 - R.f_acc))[R.zone == z].sum() + E.W_site[E.zone == z].sum() for z in zones}
    acc = {}
    for z in zones:
        Rz = R[(R.zone == z) & (R.W_site > 0)]
        dev = pd.DataFrame({"dev": Rz.d, "tier": Rz.tier, "site": Rz.site, "access": Rz.W_site * Rz.f_acc})
        split = allocation.access_split(dev, doors)
        Pbb[z] += split.pop(None, 0.0)
        acc.update({(k, z): v for k, v in split.items()})
    info = dict(routers_kW=R.W_site.sum() / 1e3, optical_kW=E.W_site.sum() / 1e3,
                access_kW=sum(acc.values()) / 1e3, backbone_kW=sum(Pbb.values()) / 1e3)
    return pd.Series(acc), Pbb, info


# --- the bill ------------------------------------------------------------------------

def compute(V: pd.DataFrame, I: pd.DataFrame, ACC: pd.Series, Pbb: dict, doors: pd.DataFrame,
            parties: list[str]):
    """Hourly bills (g) for door volumes ``V`` (Mbit/s) and grid factors ``I`` (g/kWh by
    zone): access part, backbone part and party volumes."""
    H = V.index
    Vp = V.T.groupby(doors.set_index("link").party.reindex(V.columns).values).sum().T
    Vp = Vp.reindex(columns=parties, fill_value=0)
    acc_g = pd.DataFrame(0.0, index=H, columns=parties)
    for (k, z), w in ACC.items():
        acc_g[k] += w / 1000 * I[z]                       # W -> kW, x g/kWh = g per hour
    pool = sum(Pbb[z] / 1000 * I[z] for z in Pbb)
    bb_g = Vp.mul(pool / Vp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return acc_g, bb_g, Vp


def errors(res: dict, settled: str = "settled") -> pd.DataFrame:
    """Bill errors of every scenario against the settled bill: daily WAPE over parties
    (sum |error| / sum bill), median per-party APE, hourly WAPE, total bias."""
    a0, b0, _ = res[settled]
    C0 = a0 + b0
    C0d = C0.resample("D").sum()
    big = C0d.sum() > 0
    rows = []
    for k, (a, b, _) in res.items():
        if k == settled:
            continue
        Cd = (a + b).resample("D").sum()
        err = (Cd - C0d).abs()
        rows.append(dict(scenario=k, WAPE_daily=err.loc[:, big].values.sum() / C0d.loc[:, big].values.sum(),
                         median_party_APE=float(np.nanmedian((err / C0d.replace(0, np.nan)).loc[:, big].median())),
                         WAPE_hourly=((a + b) - C0).abs().values.sum() / C0.values.sum(),
                         total_bill_error=Cd.values.sum() / C0d.values.sum() - 1))
    return pd.DataFrame(rows).sort_values("WAPE_daily")


def scheduling_test(settled, forecast, min_gb_per_h: float = 1.0) -> pd.DataFrame:
    """Send 1 GB of flexible traffic per party per day in the hour the forecast bill says
    is cheapest in g/GB. Per party-day: day average, best hour in hindsight (oracle), the
    forecast's pick, and the worst hour (settled g/GB). Parties above ``min_gb_per_h``."""
    (a0, b0, V0), (fa, fb, fv) = settled, forecast
    gA = ((a0 + b0) / (V0 * GB)).replace([np.inf], np.nan)
    gF = ((fa + fb) / (fv * GB)).replace([np.inf], np.nan)
    sel = (V0.mean() * GB > min_gb_per_h) & ((a0 + b0).sum() > 0)
    out = []
    for day, g in gA.loc[:, sel].groupby(gA.index.date):
        pick = gF.loc[g.index, sel].idxmin()
        for k in g.columns:
            if g[k].notna().sum() < 20 or pd.isna(pick[k]):
                continue
            out.append(dict(day=day, party=k, avg=g[k].mean(), oracle=g[k].min(),
                            forecast_pick=g[k][pick[k]], worst=g[k].max()))
    return pd.DataFrame(out)


def marginal_g_per_gb(ci_mean: float) -> list[float]:
    """Physical (marginal) emissions of one more GB: energy per bit x routers on the path."""
    return [j * 8e9 * A.ROUTERS_PER_PATH / 3.6e6 * ci_mean for j in A.MARGINAL_J_PER_BIT]


# --- the whole step --------------------------------------------------------------------

def run(run_dir, inv_dir, alloc_dir, grid_dir, network_dir, sites: pd.DataFrame, out,
        models=MODELS, grid_rule: str = grid.DEFAULT_RULE, zones=("CH", "FR")) -> dict:
    """Settled and forecast bills over a run's test block; writes the tables into ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    H = window(run_dir)
    doors = pd.read_csv(Path(alloc_dir) / "doors.csv")
    I_net = pd.read_csv(Path(network_dir) / "interfaces.csv")
    members = I_net[I_net.bundle.notna()].groupby("bundle").link.apply(list)
    tiers = pd.read_csv(Path(network_dir) / "routers.csv").set_index("dev").tier
    print(f"[bills] window {H[0]} -> {H[-1]} ({len(H)} h); reading {len(doors)} doors", flush=True)
    Hx = pd.date_range(H[0] - pd.Timedelta(days=7), H[-1], freq="h")    # a week of history
    VAx, exists = actual_volumes(doors, Hx)
    VA = VAx.reindex(H)
    VF, src = {}, {}
    for m in models:
        VF[m], src = forecast_volumes(run_dir, m, doors, members, H, VAx, exists)

    IA = pd.DataFrame({z: grid.load(grid_dir, z).reindex(H).interpolate() for z in zones})
    IF = pd.DataFrame({z: grid.day_ahead(grid.load(grid_dir, z), grid_rule).reindex(H).interpolate() for z in zones})

    R = pd.read_csv(Path(inv_dir) / "routers.csv")
    D = pd.read_csv(Path(alloc_dir) / "device_pools.csv")
    E = pd.read_csv(Path(inv_dir) / "shelves.csv")
    ACC, Pbb, info = window_power(R, D, E, sites, doors, tiers, H)
    print(f"[bills] site power: routers {info['routers_kW']:.1f} kW, optical {info['optical_kW']:.1f} kW; "
          f"access {info['access_kW']:.1f} kW, backbone {info['backbone_kW']:.1f} kW")

    parties = sorted(set(doors.party) | {k for k, _ in ACC.index})
    S = {"settled": (VA, IA)}
    for m in models:
        S[f"{m} + grid fc"] = (VF[m], IF)
        S[f"{m} + grid actual"] = (VF[m], IA)
    S["traffic actual + grid fc"] = (VA, IF)
    res = {k: compute(V, I, ACC, Pbb, doors, parties) for k, (V, I) in S.items()}
    M = errors(res)
    M.to_csv(out / "bill_errors.csv", index=False)
    best = M[M.scenario.str.endswith("grid fc") & ~M.scenario.str.startswith("traffic")].iloc[0].scenario

    acc0, bb0, V0 = res["settled"]
    C0 = acc0 + bb0
    fa, fb, fv = res[best]
    CF = fa + fb
    day = lambda x: x.resample("D").sum()     # noqa: E731
    C0d, CFd = day(C0), day(CF)
    types = doors.groupby("party").ptype.first().reindex(parties).fillna("customer site")

    # the bills themselves: per party over the window, and per party and day
    PB = pd.DataFrame({"type": types, "settled_g": C0.sum(), "access_g": acc0.sum(), "backbone_g": bb0.sum(),
                       "forecast_g": CF.sum(), "GB": (V0 * GB).sum()})
    PB["g_per_GB"] = PB.settled_g / PB.GB.replace(0, np.nan)
    PB.sort_values("settled_g", ascending=False).rename_axis("party").to_csv(out / "party_bills.csv")
    pd.concat({"settled_g": C0d.stack(), "forecast_g": CFd.stack()}, axis=1).rename_axis(
        ["day", "party"]).to_csv(out / "daily_bills.csv")

    err = (CFd - C0d).abs()
    bt = pd.DataFrame({"settled_t": C0d.sum() / 1e6, "abs_err_t": err.sum() / 1e6, "type": types}).groupby("type").sum()
    bt["WAPE"] = bt.abs_err_t / bt.settled_t
    bt["share_of_bill"] = bt.settled_t / bt.settled_t.sum()
    bt.to_csv(out / "errors_by_type.csv")
    tt = pd.DataFrame({"access_t": acc0.sum() / 1e6, "backbone_t": bb0.sum() / 1e6, "type": types}).groupby("type").sum()
    tt["parties"] = types.value_counts()
    tt.to_csv(out / "settled_by_type.csv")

    SCH = scheduling_test(res["settled"], res[best])
    SCH.to_csv(out / "scheduling_test.csv", index=False)

    # aggregates for the report (no raw hourly grid series leave this directory)
    pd.DataFrame({"I_CH": IA.CH, "I_CH_fc": IF.CH, "V_total_Gbps": V0.sum(axis=1) / 1e3,
                  "V_total_fc_Gbps": fv.sum(axis=1) / 1e3,
                  "backbone_g_per_GB": bb0.sum(axis=1) / (V0.sum(axis=1) * GB)}).to_csv(out / "hourly_network.csv")
    rank = C0.sum().sort_values(ascending=False).index
    top = rank[:12]
    pd.DataFrame({"settled_t": C0d[top].sum() / 1e6, "forecast_t": CFd[top].sum() / 1e6, "type": types[top],
                  "access_share": acc0[top].sum() / C0[top].sum()}).to_csv(out / "top_parties.csv")
    pick = ["hidden:ix3", rank[3]] + [k for k in rank if k.startswith("site:")][:1]
    pd.concat({k: pd.DataFrame({"settled_g": C0d[k], "forecast_g": CFd[k]}) for k in pick if k in C0d},
              axis=1).to_csv(out / "daily_examples.csv")
    loc = H.tz_convert("Europe/Zurich").hour
    hn = pd.DataFrame({"I": IA.CH.values, "V": V0.sum(axis=1).values / 1e3,
                       "g": (bb0.sum(axis=1) / (V0.sum(axis=1) * GB)).values,
                       "Ierr": (IF.CH - IA.CH).abs().values}, index=loc)
    hn.groupby(level=0).median().to_csv(out / "hour_of_day_median.csv")

    med = SCH[["avg", "oracle", "forecast_pick", "worst"]].median()
    summary = {"best_scenario": best, "window": [str(H[0]), str(H[-1])], "hours": len(H), "grid_rule": grid_rule,
               "fallback_doors": sum(v != "model" for v in src.values()),
               "model_doors": sum(v == "model" for v in src.values()),
               "model_traffic_share": float(VA[[k for k, v in src.items() if v == "model"]].values.sum() / VA.values.sum()),
               **info,
               "settled_t_total": C0.values.sum() / 1e6, "parties": len(parties),
               "scheduling_median_g_per_GB": med.to_dict(),
               "scheduling_captured": float((SCH.avg - SCH.forecast_pick).sum() / (SCH.avg - SCH.oracle).sum()),
               "marginal_g_per_GB": marginal_g_per_gb(float(IA.CH.mean()))}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(M.round(4).to_string(index=False))
    print(f"[bills] settled {summary['settled_t_total']:.2f} t over {len(parties)} parties; best forecast: {best}")
    return summary
