"""Total network traffic: the bytes SWITCHlan carries, each counted once, and its day-ahead forecast.

Bytes are counted only at the doors (``allocation.doors``), never on internal trunks: a
byte from party A to party B crosses any number of routers and trunks but shows up exactly
twice, as *in* on A's door (entering SWITCHlan) and *out* on B's door (leaving). So

    T(h) = 1/2 * sum_doors (in + out)          one-way bytes carried, no double counting

and the bill's backbone key (each party's in + out) is half a byte per end. Two checks:

- **Conservation**: if the doors form a closed cut, sum(in) = sum(out) every hour.
- **Same-router turnaround**: traffic that enters and leaves at doors of one router never
  touches a trunk, but is still billed backbone. Counters bound it per router:
  max(0, door_in - trunk_out, door_out - trunk_in) <= local <= min(door_in, door_out).

Two kinds of day-ahead forecast of T, issued 00:00 UTC for the 24 hours of the day:

- **top-down**: rules on T itself, and a GBDT trained on T (the last 7 days hourly,
  normalised by their mean, plus the target hour's calendar; one row per target hour);
- **bottom-up**: the sum of the per-link models' door forecasts (dense block logs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..prediction.calendar_features import HOLIDAYS
from . import assumptions as A, bills, grid

TRAIN_END = "2026-02-15"            # the year benchmark's training block ends here


# --- the series ----------------------------------------------------------------------

def door_matrix(traffic: dict, links, key: str) -> pd.DataFrame:
    """Hourly Mbit/s (``key`` = hin / hout) of the given interfaces from traffic.npz."""
    ix = {str(x): i for i, x in enumerate(traffic["links"])}
    H = pd.to_datetime(traffic["hours"], unit="s", utc=True)
    return pd.DataFrame({x: traffic[key][ix[x]] for x in links}, index=H).astype(float)


def _fill(X: pd.DataFrame) -> pd.DataFrame:
    return X.interpolate(limit=6, limit_area="inside").fillna(0)


def total_series(traffic: dict, doors: pd.DataFrame, min_coverage: float = 0.95):
    """T(h) in Gbit/s (one-way), door in and out (Mbit/s, gaps <= 6 h filled) and the
    traffic-weighted share of live doors reporting. Hours below ``min_coverage`` (a data
    outage, not a traffic dip) are NaN in T."""
    rin, rout = door_matrix(traffic, doors.link, "hin"), door_matrix(traffic, doors.link, "hout")
    raw = rin + rout
    w = raw.mean() / raw.mean().sum()
    first, last = raw.notna().idxmax(), raw[::-1].notna().idxmax()
    alive = pd.DataFrame({x: (raw.index >= first[x]) & (raw.index <= last[x]) for x in raw}, index=raw.index)
    cov = (raw.notna() * w).sum(axis=1) / (alive * w).sum(axis=1)
    din, dout = _fill(rin), _fill(rout)
    T = ((din + dout).sum(axis=1) / 2 / 1e3).where(cov >= min_coverage)
    return T, din, dout, cov


def conservation(din: pd.DataFrame, dout: pd.DataFrame, H) -> dict:
    si, so = din.sum(axis=1).reindex(H), dout.sum(axis=1).reindex(H)
    imb = (si - so).abs() / ((si + so) / 2)
    return dict(in_gbps=float(si.mean() / 1e3), out_gbps=float(so.mean() / 1e3), in_over_out=float(si.sum() / so.sum()),
                hourly_imbalance_median=float(imb.median()), hourly_imbalance_p90=float(imb.quantile(.9)))


def turnaround(traffic: dict, interfaces: pd.DataFrame, doors: pd.DataFrame, H) -> pd.DataFrame:
    """Per router: mean door traffic and the bounds on traffic that enters and leaves at
    its own customer/external doors without using a trunk (Gbit/s over ``H``). Trunks
    into hidden routers count as trunks here: those bytes do cross the backbone."""
    I = interfaces
    top = I[I.kind.isin(["phys", "bundle"]) & I.bundle.isna() & (I.role != "idle/absent")]
    real = set(doors.loc[doors.hidden_peer.isna(), "link"])
    known = {str(x) for x in traffic["links"]}
    rows = []
    for dev, g in top.groupby("dev"):
        dl = [x for x in g.link if x in real]
        tl = [x for x in g.link if x not in real and x in known]
        if not dl:
            continue
        di = _fill(door_matrix(traffic, dl, "hin")).sum(axis=1)
        do = _fill(door_matrix(traffic, dl, "hout")).sum(axis=1)
        ti = _fill(door_matrix(traffic, tl, "hin")).sum(axis=1) if tl else di * 0
        to = _fill(door_matrix(traffic, tl, "hout")).sum(axis=1) if tl else di * 0
        lo = np.maximum(np.maximum(di - to, do - ti), 0)
        hi = np.minimum(di, do)
        rows.append(dict(dev=dev, door_gbps=float(((di + do) / 2).reindex(H).mean() / 1e3),
                         local_lb_gbps=float(lo.reindex(H).mean() / 1e3), local_ub_gbps=float(hi.reindex(H).mean() / 1e3)))
    return pd.DataFrame(rows).sort_values("local_lb_gbps", ascending=False)


# --- top-down forecasts ----------------------------------------------------------------

def _week_mean(s, k):
    return sum(s.shift(168 * i) for i in range(1, k + 1)) / k


RULES = {
    "persistence (23:00, flat)": lambda s: s.groupby(s.index.normalize()).transform("last").shift(24),
    "same hour yesterday": lambda s: s.shift(24),
    "same hour last week": lambda s: s.shift(168),
    "mean same hour, last 4 weeks": lambda s: _week_mean(s, 4),
    "last week x yesterday's level": lambda s: s.shift(168) * (s.shift(24).rolling(24).mean()
                                                                 / s.shift(192).rolling(24).mean()),
}


def _calendar(t: pd.DatetimeIndex) -> np.ndarray:
    lt = t.tz_convert("Europe/Zurich")
    day = lt.strftime("%Y-%m-%d")
    hol = np.array([d in HOLIDAYS for d in day], dtype=float)
    return np.c_[lt.hour, lt.dayofweek, hol, ((lt.dayofweek < 5) & (hol == 0)).astype(float)]


def _design(s: pd.Series):
    """One row per (issue day, target hour): the 168 log values before 00:00 UTC minus the
    log of their mean, the target hour index, the target's local hour, weekday, holiday
    and workday flags. Target: log T - that same log mean."""
    v = s.to_numpy(float)
    H = s.index
    starts = np.where((H.hour == 0) & (np.arange(len(H)) >= 168) & (np.arange(len(H)) + 24 <= len(H)))[0]
    X, Y, base, t = [], [], [], []
    for i0 in starts:
        hist = v[i0 - 168:i0]
        b = np.log(hist.mean())
        lag = np.log(hist) - b
        tt = H[i0:i0 + 24]
        X.append(np.c_[np.tile(lag, (24, 1)), np.arange(24), _calendar(tt)])
        Y.append(np.log(v[i0:i0 + 24]) - b)
        base.append(np.full(24, b))
        t.append(tt)
    return np.vstack(X), np.concatenate(Y), np.concatenate(base), pd.DatetimeIndex(np.concatenate(t))


def gbdt_day_ahead(T: pd.Series, train_end: str = TRAIN_END, seed: int = 0) -> pd.Series:
    """Train on issue days before ``train_end`` (hours with real data only), forecast every
    later day. Histories are gap-filled by interpolation."""
    s = T.interpolate().bfill()
    X, Y, base, t = _design(s)
    ok = T.reindex(t).notna().to_numpy() & (t < pd.Timestamp(train_end, tz="UTC"))
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                      l2_regularization=1.0, random_state=seed).fit(X[ok], Y[ok])
    test = t >= pd.Timestamp(train_end, tz="UTC")
    return pd.Series(np.exp(m.predict(X[test]) + base[test]), index=t[test])


# --- bottom-up and evaluation ------------------------------------------------------------

def bottom_up(run_dir, model: str, doors: pd.DataFrame, members: pd.Series, H, VA: pd.DataFrame,
              exists: pd.DataFrame) -> pd.Series:
    """T-hat as the sum of the door forecasts of one traffic model (Gbit/s, one-way)."""
    VF, _ = bills.forecast_volumes(run_dir, model, doors, members, H, VA, exists)
    return VF.sum(axis=1) / 2 / 1e3


def errors(F: dict, y: pd.Series) -> pd.DataFrame:
    y = y.dropna()
    rows = []
    for k, f in F.items():
        f = f.reindex(y.index)
        e = f - y
        rows.append(dict(forecast=k, MAE_gbps=e.abs().mean(), hourly_WAPE=e.abs().sum() / y.sum(),
                         daily_WAPE=(f.resample("D").sum() - y.resample("D").sum()).abs().sum() / y.sum(),
                         peak_hour_APE=((f - y).abs() / y)[y >= y.quantile(.9)].mean(), bias=e.sum() / y.sum()))
    return pd.DataFrame(rows).sort_values("hourly_WAPE")


def run(run_dir, network_dir, alloc_dir, inv_dir, grid_dir, sites: pd.DataFrame, out,
        models=bills.MODELS, grid_rule: str = grid.DEFAULT_RULE) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    traffic = dict(np.load(Path(network_dir) / "traffic.npz"))
    doors = pd.read_csv(Path(alloc_dir) / "doors.csv")
    I = pd.read_csv(Path(network_dir) / "interfaces.csv")
    H = bills.window(run_dir)
    T, din, dout, cov = total_series(traffic, doors)
    cons = conservation(din, dout, H)
    TA = turnaround(traffic, I, doors, H)
    TA.to_csv(out / "turnaround.csv", index=False)
    y = T.reindex(H)

    F = {name: rule(T.interpolate())[H] for name, rule in RULES.items()}
    F["GBDT on the total (top-down)"] = gbdt_day_ahead(T).reindex(H)
    members = I[I.bundle.notna()].groupby("bundle").link.apply(list)
    VA = din.add(dout)
    exists = VA.ne(0).cumsum() > 0
    for m in models:
        F[f"sum of {m} door forecasts (bottom-up)"] = bottom_up(run_dir, m, doors, members, H, VA, exists)
    E = errors(F, y)
    E.to_csv(out / "forecast_errors.csv", index=False)
    pd.DataFrame({"T_gbps": y, **F}).to_csv(out / "total_hourly.csv")

    # the network-wide backbone signal B(h) / T(h): g per GB carried, actual and day-ahead
    tiers = pd.read_csv(Path(network_dir) / "routers.csv").set_index("dev").tier
    ACC, Pbb, _ = bills.window_power(pd.read_csv(Path(inv_dir) / "routers.csv"), pd.read_csv(Path(alloc_dir) / "device_pools.csv"),
                                     pd.read_csv(Path(inv_dir) / "shelves.csv"), sites, doors, tiers, H)
    zones = list(Pbb)
    IA = pd.DataFrame({z: grid.load(grid_dir, z).reindex(H).interpolate() for z in zones})
    IF = pd.DataFrame({z: grid.day_ahead(grid.load(grid_dir, z), grid_rule).reindex(H).interpolate() for z in zones})
    B = lambda I_: sum(Pbb[z] / 1000 * I_[z] for z in zones)          # noqa: E731  g per hour
    gb = lambda t: t * 1e3 * A.GB_PER_MBPS_HOUR                         # noqa: E731  GB per hour
    sig = B(IA) / gb(y)
    rows = []
    for k in ["same hour last week", "GBDT on the total (top-down)", "sum of patchtst door forecasts (bottom-up)"]:
        for gname, I_ in (("actual grid", IA), ("grid forecast", IF)):
            f = B(I_) / gb(F[k])
            rows.append(dict(traffic=k, grid=gname, signal_WAPE=float((f - sig).abs().sum() / sig.sum())))
    S = pd.DataFrame(rows)
    S.to_csv(out / "signal_errors.csv", index=False)
    summary = dict(window=[str(H[0]), str(H[-1])], mean_total_gbps=float(y.mean()), conservation=cons,
                   hidden_door_share=float(((din + dout)[doors.loc[doors.hidden_peer.notna(), "link"]].sum(axis=1)
                                            / (din + dout).sum(axis=1)).reindex(H).mean()),
                   turnaround_lb_share=float(TA.local_lb_gbps.sum() / y.mean()),
                   outage_hours=int((cov < 0.95).sum()), backbone_g_per_GB_carried=float(sig.mean()))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=1))
    print(E.round(4).to_string(index=False))
    print(S.round(4).to_string(index=False))
    return summary
