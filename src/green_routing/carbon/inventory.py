"""Energy inventory: every router and optical shelf, its model, its power and its site.

Power is measured where SWITCH meters it and modelled where it does not:

- **PSU telemetry** (``router-power/``, four schemas, see ``config.PSU_SCHEMAS``) gives
  measured input power for ~100 routers, but stops on 2024-11-01; the 2024 median is
  used as-is (router power is < 1 % traffic-dependent, so it barely moves).
- **Router models** are not in the bucket. They come from matching the Network Power Zoo
  sample (May 2025, anonymised router ids, per-interface bytes/s) to our own interface
  means, then from the telemetry schema and port layout for the rest.
- Unmetered routers get their model peers' median, else a datasheet/literature value
  (``assumptions.MODEL_W``).
- **ECI Apollo shelves** (the DWDM layer, ``eci/apollo/``) report their own power.
- A router counts as in service while it exports traffic, and until the reverse-DNS
  snapshot when its interfaces are still named there (some stopped exporting in
  2024-25 but are alive).

Outputs (``$GR_ARTIFACTS_DIR/carbon/inventory``): psu_survey.csv, zoo_match.csv,
devices.csv, routers.csv (power and service span), routers_<year>.csv, shelves.csv,
site_energy_<year>.csv (routers and shelves with site, host type, site energy, zone).
"""
from __future__ import annotations

import collections
import io
import multiprocessing as mp
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import config, loaders, storage
from ..network.roles import site_of
from . import assumptions as A

ZOO_WEEK = (1746050400, 1746662400)     # the Zoo sample's week: 2025-04-30 22:00 -> 05-08 UTC


# --- PSU telemetry -------------------------------------------------------------

def psu_survey() -> pd.DataFrame:
    """One row per PSU file: schema, first/last sample, median and last-30-day power (W)."""
    rows = []
    for dev, psu, uri in loaders.iter_psu_files():
        p = loaders.psu_power(uri)
        ok = p[p > 0]
        last30 = p[p.index > p.index.max() - pd.Timedelta(days=30)]
        rows.append(dict(file=f"{dev}_{psu}.csv", dev=dev, kind=p.attrs["kind"],
                         first=ok.index.min().date() if len(ok) else None,
                         last=ok.index.max().date() if len(ok) else None,
                         W_med=ok.median() if len(ok) else np.nan, W_last30=last30.mean()))
    return pd.DataFrame(rows)


def _dependence(item):
    dev, psu_uris, if_uris, start, end = item
    parts = [loaders.psu_power(u) for u in psu_uris]
    W = pd.concat(parts, axis=1).dropna().sum(axis=1)     # all PSUs present
    tr = None
    for u in if_uris:
        try:
            t = loaders.interface_throughput(u)["total_mbps"]
        except Exception:        # noqa: BLE001 -- a broken interface file is skipped
            continue
        tr = t if tr is None else tr.add(t, fill_value=0)
    if tr is None or W.empty:
        return None
    df = pd.concat([W.rename("W"), tr.rename("mbps")], axis=1).loc[start:end].resample("1h").mean().dropna()
    if len(df) < 500:
        return None
    b = np.polyfit(df.mbps / 1e3, df.W, 1)[0]
    return dict(dev=dev, hours=len(df), W_mean=df.W.mean(), W_p5=df.W.quantile(.05), W_p95=df.W.quantile(.95),
                Gbps_mean=df.mbps.mean() / 1e3, Gbps_p95=df.mbps.quantile(.95) / 1e3,
                corr=df.corr().iloc[0, 1], W_per_Gbps=b, dyn_share=b * df.mbps.mean() / 1e3 / df.W.mean())


def traffic_dependence(start="2024-02-01", end="2024-10-31", workers: int = 16) -> pd.DataFrame:
    """How much of each metered router's power follows its traffic: hourly PSU power
    regressed on the router's summed interface traffic. ``dyn_share`` = slope x mean
    traffic / mean power. (Summing every interface counts some bytes twice; the share is
    invariant to that scale.)"""
    psu = collections.defaultdict(list)
    for dev, _, uri in loaders.iter_psu_files():
        if "lab" not in dev:
            psu[dev].append(uri)
    ifs = collections.defaultdict(list)
    for d, _, uri in loaders.iter_interface_files():
        ifs[full_name(d)].append(uri)
    items = [(d, psu[d], ifs[d], start, end) for d in sorted(psu) if ifs.get(d)]
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [r for r in pool.map(_dependence, items) if r]
    return pd.DataFrame(rows)


# --- What each router is --------------------------------------------------------

def _schema(header: str) -> str:
    cols = [x for x in header.split(",") if x != "timestamp" and not x.startswith("ds")]
    if any("920-" in x for x in cols):
        return "xe920"
    if any("tempIndexedModule" in x for x in cols):
        return "ios6500"
    if any("ap-core" in x for x in cols):
        return "xr-ap(N540?)"
    if any("x86-core" in x for x in cols):
        return "xr-x86"
    if any("pm0" in x for x in cols):
        return "xr-pm"
    if any("rsp-cpu" in x for x in cols) and any("mod0-in" in x for x in cols):
        return "xr-rsp-mod"
    if any("arista" in x for x in cols):
        return "arista"
    return "other:" + ",".join(cols[:4])


def router_schemas() -> dict[str, str]:
    """Telemetry schema of every router's ``routers/<dev>.csv`` (from the header only)."""
    objs = storage.walk_objects(config.SUBTREES["routers"], suffix=".csv")
    out = {}
    for o in objs:
        head = storage.read_head(o.uri, 3001).decode(errors="replace").splitlines()[0]
        out[storage.stem(o.uri)] = _schema(head)
    return out


def port_layout(links) -> dict[str, str]:
    """``<dev>`` -> compact description of its interface names, e.g.
    ``hundredgige[12]0_0_0_0..0_0_1_9 tengige[4]...`` (types, counts, range)."""
    ifs = collections.defaultdict(list)
    for link in links:
        d, i = link.split("/", 1)
        ifs[d].append(i)

    def desc(lst):
        g = collections.defaultdict(list)
        for x in lst:
            m = re.match(r"([a-z\-]+?)(\d[\d_]*)$", x)
            if m:
                g[m.group(1)].append(m.group(2))
        return " ".join(f"{k}[{len(v)}]{min(v)}..{max(v)}" for k, v in sorted(g.items()))
    return {d: desc(v) for d, v in ifs.items()}


def full_name(short: str) -> str:
    """Export path name -> device name: ``ag1`` -> ``swiag1`` (blu*/fga* keep theirs)."""
    return short if short.startswith(("blu", "fga")) else "swi" + short


def _week_means(item):
    dev, iface, uri = item
    try:
        df = loaders.read_series(uri)
    except Exception:            # noqa: BLE001 -- unreadable files simply do not match
        return None
    if "ifHCInOctets" not in df:
        return None
    t = df.index.astype("int64") // 10**9
    df = df[(t >= ZOO_WEEK[0]) & (t < ZOO_WEEK[1])]
    if len(df) < 100:
        return None
    return dict(dev=dev, iface=iface, bin=df.ifHCInOctets.mean(), bout=df.ifHCOutOctets.mean())


def zoo_match(example_zip: str | Path, workers: int = 16) -> pd.DataFrame:
    """Match the Network Power Zoo sample's anonymised routers to ours.

    For every Zoo router, count its interfaces whose mean bytes/s in and out agree with
    one of our interfaces of the same name within 2 % (both directions); the router with
    the most agreeing interfaces is the match. ``n`` and ``second`` (runner-up count) show
    how clear it is. The Zoo publishes bytes/s, like our export."""
    z = zipfile.ZipFile(example_zip)

    def read(name, **kw):
        member = next(m for m in z.namelist() if m.endswith(name))
        return pd.read_csv(io.BytesIO(z.read(member)), sep=";", **kw)
    zr = read("routers.csv").merge(read("router_models.csv")[["id", "model_uid"]],
                                   left_on="router_model_id", right_on="id")
    mi = read("measurements_interface.csv", usecols=["router_interface_id", "bytes_in", "bytes_out"])
    mi["router_id"] = mi.router_interface_id.str.split("_").str[0]
    mi["iface"] = mi.router_interface_id.str.split("_", n=1).str[1]
    fp = mi.groupby(["router_id", "iface"])[["bytes_in", "bytes_out"]].mean().reset_index()
    with mp.get_context("fork").Pool(workers) as pool:
        O = pd.DataFrame([r for r in pool.map(_week_means, list(loaders.iter_interface_files()),
                                              chunksize=8) if r])
    J = fp.merge(O, on="iface")

    def tol(a, b):
        return np.abs(a - b) <= 0.02 * np.maximum(a, b) + 1
    J["ok"] = tol(J.bytes_in, J.bin) & tol(J.bytes_out, J.bout) & ((J.bytes_in + J.bytes_out) > 10)
    score = J[J.ok].groupby(["router_id", "dev"]).size().reset_index(name="n")
    nif = fp.groupby("router_id").size()
    res = []
    for rid, g in score.groupby("router_id"):
        g = g.sort_values("n", ascending=False)
        res.append(dict(router_id=rid, dev=full_name(g.dev.iloc[0]), n=g.n.iloc[0],
                        second=g.n.iloc[1] if len(g) > 1 else 0, nif=nif[rid]))
    return zr[["router_id", "model_uid"]].merge(pd.DataFrame(res), on="router_id", how="left")


def model_rule(dev: str, schema, ifs, W_meas) -> tuple[str, str]:
    """Router model from its telemetry schema and port layout (Zoo matches win)."""
    s, i = str(schema), str(ifs)
    if s == "xe920":
        return "ASR-920 (variant ?)", "schema"
    if s == "xr-rsp-mod":
        return ("ASR-9000 chassis" if dev == "swiez3" else "ASR-9001"), "schema+ports"
    if s == "xr-pm":
        if "fourhundredgige" in i:
            return "8201-32FH", "ports"
        if "hundredgige" in i and "0_0_1_" in i:
            if W_meas == W_meas:      # metered: the 48-port box draws more
                return ("NCS-55A1-24Q6H-SS" if (W_meas or 0) < 315 else "NCS-55A1-48Q6H"), "ports"
            return "NCS-55A1-24Q6H-SS/48Q6H", "ports"
        return "NCS-55A1-24H", "ports"
    if s == "xr-x86":
        return "N540-28Z4C", "ports (lab twin)"
    if s == "xr-ap(N540?)":
        return "N540X (small)", "schema"
    if s == "ios6500":
        return "Catalyst 6500", "schema"
    if s.startswith("other:cpu1min"):
        return "ME3400-class switch", "schema"
    if s == "arista":
        return "Arista 7280", "schema"
    return "unknown", "-"


def devices(iface_stats: pd.DataFrame, schemas: dict, psu: pd.DataFrame, zoo: pd.DataFrame) -> pd.DataFrame:
    """One row per router: first/last day with traffic data, schema, port layout, Zoo
    model, measured PSU power (sum over its PSUs), model and how it was found."""
    s = iface_stats.assign(d=iface_stats.link.str.split("/").str[0])
    fd = pd.to_datetime(s.first_data.replace("", np.nan), utc=True)
    ld = pd.to_datetime(s.last_data.replace("", np.nan), utc=True)
    dv = pd.DataFrame({"d": s.d, "first": fd, "last": ld}).groupby("d").agg(
        n_if=("first", "size"), first=("first", "min"), last=("last", "max")).reset_index()
    dv["dev"] = dv.d.map(full_name)
    dv["first"], dv["last"] = dv["first"].dt.date, dv["last"].dt.date
    layout = port_layout(s.link)
    F = pd.DataFrame({"dev": list(schemas), "schema": list(schemas.values())})
    F["ifs"] = F.dev.map(lambda d: layout.get(d[3:] if d.startswith("swi") else d, ""))
    pm = psu.groupby("dev").agg(W_meas=("W_med", "sum"), kind=("kind", "first"))
    T = (dv.merge(F, on="dev", how="outer")
         .merge(zoo.dropna(subset=["dev"]).sort_values("n", ascending=False)
                .drop_duplicates("dev")[["dev", "model_uid"]], on="dev", how="left")
         .merge(pm, left_on="dev", right_index=True, how="left"))
    T = T[~T.dev.str.contains("lab|test|mgmt", na=False)].copy()
    T[["model", "model_src"]] = [
        (r.model_uid, "zoo-match") if isinstance(r.model_uid, str) else model_rule(r.dev, r.schema, r.ifs, r.W_meas)
        for r in T.itertuples()]
    return T


# --- Energy per year --------------------------------------------------------------

def router_power(dev: pd.DataFrame, live: set[str], live_until: str) -> pd.DataFrame:
    """Power (central / low / high W) and service span of every router.

    ``live``: short names of routers alive in the reverse-DNS snapshot taken on
    ``live_until``; they count as in service until then even after their export stopped,
    and those never exported at all join as ``unknown`` routers."""
    T = dev.copy()
    short = T.dev.str.replace("^swi", "", regex=True)
    T["dns_live"] = short.isin(live)
    T.loc[T.dns_live, "last"] = pd.Timestamp(live_until).date()
    new = [full_name(d) for d in sorted(live - set(short))]
    T = pd.concat([T, pd.DataFrame({"dev": new, "model": "unknown", "first": pd.Timestamp("2024-01-01").date(),
                                    "last": pd.Timestamp(live_until).date(), "dns_live": True})],
                  ignore_index=True)
    peer = T[T.W_meas.notna()].groupby("model").W_meas.median()

    def power(r):
        if r.W_meas == r.W_meas:
            if r.kind == "UI":   # DC output side of the PSU
                return r.W_meas, A.PSU_EFFICIENCY * r.W_meas, r.W_meas / A.PSU_EFFICIENCY, "measured 2024"
            return r.W_meas, r.W_meas - A.PIN_BAND_W[0], r.W_meas * A.PIN_BAND_W[1], "measured 2024"
        if r.model in peer:
            p = peer[r.model]
            return p, p * 0.9, p * 1.1, "model peers"
        c, lo, hi = A.MODEL_W.get(r.model, A.MODEL_W["unknown"])
        return c, lo, hi, "datasheet/literature"
    T[["W", "Wlo", "Whi", "W_src"]] = [power(r) for r in T.itertuples()]
    T["site"] = T.dev.str.replace("^swi", "", regex=True).map(site_of)
    return T


def service_days(T: pd.DataFrame, year: int) -> pd.Series:
    """Days each router was in service in ``year`` (first .. last day, inclusive)."""
    a, b = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
    f, l = pd.to_datetime(T["first"]), pd.to_datetime(T["last"])
    days = ((l.clip(upper=b) - f.clip(lower=a)).dt.days + 1).clip(lower=0).fillna(0)
    return days.clip(upper=(b - a).days)


def router_energy(T: pd.DataFrame, year: int) -> pd.DataFrame:
    """Routers in service in ``year`` with days in service and MWh (central/low/high)."""
    S = T.assign(days=service_days(T, year))
    S = S[S.days > 0].copy()
    for k in ("W", "Wlo", "Whi"):
        S["MWh_" + k] = S[k] * S.days * 24 / 1e6
    return S


def _shelf_stats(item):
    shelf, uri = item
    p = loaders.eci_shelf_power(uri).dropna()
    out = {"shelf": shelf}
    for y in range(2024, pd.Timestamp.now().year + 1):
        s = p[(p.index >= pd.Timestamp(f"{y}-01-01", tz="UTC")) & (p.index < pd.Timestamp(f"{y + 1}-01-01", tz="UTC"))]
        out[f"W{y}"] = s.mean()
        out[f"cov{y}"] = len(s) / 105120          # share of a year's 5-minute samples
        out[f"p5_{y}"], out[f"p95_{y}"] = s.quantile(.05), s.quantile(.95)
    out["first"], out["last"] = p.index.min().date(), p.index.max().date()
    return out


def shelves(workers: int = 8) -> pd.DataFrame:
    """ECI Apollo shelves: mean, coverage, p5/p95 power per calendar year; first/last day."""
    items = sorted(loaders.iter_eci_files())
    storage.prefetch([u for _, u in items])
    with mp.get_context("fork").Pool(workers) as pool:
        E = pd.DataFrame(pool.map(_shelf_stats, items))
    E = E[~E.shelf.str.startswith("olab")].copy()       # lab shelves
    E["site"] = E.shelf.str[1:].map(site_of)             # oag1 -> ag
    return E


def site_energy(R: pd.DataFrame, E: pd.DataFrame, sites: pd.DataFrame, year: int) -> pd.DataFrame:
    """Routers and shelves in ``year`` with host type, IT energy and site energy (x PUE,
    central/low/high) and grid zone."""
    ht = sites.drop_duplicates("code").set_index("code").host_type
    r = R[["dev", "site", "W", "MWh_W"]].rename(columns={"dev": "unit", "MWh_W": "MWh"}).assign(layer="router")
    e = E.assign(MWh=E[f"W{year}"].fillna(0) * E[f"cov{year}"].fillna(0) * 8.76 / 1e3, W=E[f"W{year}"])
    e = e[["shelf", "site", "W", "MWh"]].rename(columns={"shelf": "unit"}).assign(layer="optical")
    U = pd.concat([r, e], ignore_index=True)
    U["host_type"] = U.site.map(ht).fillna("unplaced")
    for i, k in enumerate(("site_MWh", "site_MWh_lo", "site_MWh_hi")):
        U[k] = U.MWh * U.host_type.map(lambda h: A.pue(h, i))
    U["zone"] = U.site.map(A.zone_of)
    return U


def scope2(U: pd.DataFrame, grid_means: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Location-based Scope 2 for the year: site energy x the annual mean grid factor of
    each unit's zone (flat load), central / low / high PUE, direct and lifecycle."""
    rows = []
    for k in ("site_MWh", "site_MWh_lo", "site_MWh_hi"):
        for basis in ("direct", "lifecycle"):
            t = sum(U.loc[U.zone == z, k].sum() * grid_means[z][basis] for z in U.zone.unique()) / 1e3
            rows.append(dict(pue={"site_MWh": "central", "site_MWh_lo": "low", "site_MWh_hi": "high"}[k],
                             basis=basis, site_MWh=U[k].sum(), tCO2e=t))
    return pd.DataFrame(rows)
