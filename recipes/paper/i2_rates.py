"""I2 Network, test block: realized ("settled") and day-ahead ("posted") per-resource usage rates
under the paper's idle-link cap, and what they give: city-pair rates, entity bills, the forecast
error of a resource's rate against its load.

Rates (``carbon.path_rate.capped_rates``): every hour a resource (core router, inter-PoP span)
charges E / U, its emissions over the traffic crossing it, at most the hour's cap; the emissions
it does not charge (above the cap, idle resources) are spread over all bytes. The cap is
median + 2 robust sigmas of the pooled log rates of the previous 28 days (``path_rate.idle_cap``;
the 28 days before the block come from i2_pre_setup.py).

- settled: actual emissions E, actual matrix (EM from the counters, as carbon.py).
- posted:  issued 00:00 UTC for the next 24 h (path_dayahead.forecast_inputs): link loads from
           Timestral's per-link block log at the day horizon (24 weeks of hourly input), door
           totals from Timestral's day-ahead total, EM; emissions with the grid of the same hour
           yesterday (I2's grid barely moves; see bill_periods.py for the network total).

usage: python i2_rates.py            writes $GR_ARTIFACTS_DIR/paper/i2/{rates.npz, citypairs.csv,
                                     entity_bills.csv, rate_error_by_load.csv, summary.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from green_routing.carbon import path_rate
from green_routing.dataset import internet2 as i2
from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
ROOT = ARTIFACTS_DIR / "internet2"
C, PRE, DATA = ROOT / "carbon", ROOT / "carbon_pre28", ROOT / "data"
OUT = ARTIFACTS_DIR / "paper" / "i2"
GB = 3600 / 8 / 1000                       # GB per Mbit/s-hour
LINK_MODEL, TOTAL = "timestral@1h_24w", "Timestral-209M, 24 weeks"


def log(*a):
    print(*a, flush=True)


def load_rates(refresh: bool = False) -> dict:
    """Settled and posted capped rates for every test hour (cached in OUT/rates.npz)."""
    f = OUT / "rates.npz"
    if f.exists() and not refresh:
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE.parent / "internet2"))
    import path_dayahead as PD
    F = PD.forecast_inputs(LINK_MODEL, TOTAL, ROOT / "links" / "merged" / "dense_fm", C / "timestral_total.csv")
    S, P = np.load(C / "test_setup.npz"), np.load(PRE / "test_setup.npz")
    A_res = F["A_res"]
    T = pd.to_datetime(np.r_[P["T"], S["T"]], utc=True)
    E_all = np.r_[P["E"], S["E"]]
    X_all = np.r_[P["X_em"], S["X_em"]]
    U_all = (A_res @ X_all.T).T
    mult = path_rate.idle_cap(E_all, U_all, T)[len(P["T"]):]       # causal: days before each test day
    E, X, U = S["E"], S["X_em"], U_all[len(P["T"]):]
    E_da = np.nan_to_num(F["E_da"])
    X_fc = F["X_fc"]
    U_fc = (A_res @ X_fc.T).T
    n_h, n_r = E.shape
    per_set, per_post = np.zeros((n_h, n_r)), np.zeros((n_h, n_r))
    add_set, add_post = np.zeros(n_h), np.zeros(n_h)
    for h in range(n_h):
        per_set[h], add_set[h] = path_rate.capped_rates(E[h], U[h], X[h].sum(), mult[h])
        per_post[h], add_post[h] = path_rate.capped_rates(E_da[h], U_fc[h], X_fc[h].sum(), mult[h])
    ok = F["ok"] & np.isfinite(F["E_da"]).all(axis=1)
    out = dict(T=S["T"], E=E, E_da=E_da, X=X, X_fc=X_fc, U=U, U_fc=U_fc, mult=mult, per_set=per_set,
               per_post=per_post, add_set=add_set, add_post=add_post, ok=ok, ends=S["ends"], res=S["res"],
               lks=S["lks"], In=S["In"], Eg=S["Eg"])
    np.savez_compressed(f, **out)
    sp.save_npz(OUT / "A_res.npz", A_res.tocsr())
    return out


def load_rates_k(k) -> dict:
    """``load_rates`` with another cap: k robust sigmas (k = 2 is the paper's), or no cap (k None:
    a multiple of 1e9, i.e. E / U with idle resources at an effectively infinite rate). Reuses the
    forecasts of the k = 2 run; cached in OUT/rates_k<k>.npz."""
    if k == 2:
        return load_rates()
    f = OUT / f"rates_k{k}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {kk: z[kk] for kk in z.files}
    R = dict(load_rates())
    A_res = sp.load_npz(OUT / "A_res.npz").tocsr()
    P = np.load(PRE / "test_setup.npz")
    T = pd.to_datetime(np.r_[P["T"], R["T"]], utc=True)
    if k is None:
        mult = np.full(len(R["T"]), 1e9)
    else:
        E_all = np.r_[P["E"], R["E"]]
        U_all = np.r_[(A_res @ P["X_em"].T).T, R["U"]]
        mult = path_rate.idle_cap(E_all, U_all, T, k=k)[len(P["T"]):]
    for h in range(len(R["T"])):
        R["per_set"][h], R["add_set"][h] = path_rate.capped_rates(R["E"][h], R["U"][h], R["X"][h].sum(), mult[h])
        R["per_post"][h], R["add_post"][h] = path_rate.capped_rates(R["E_da"][h], R["U_fc"][h], R["X_fc"][h].sum(), mult[h])
    R["mult"] = mult
    np.savez_compressed(f, **R)
    return R


def flat_rate(R) -> np.ndarray:
    """g per Mbit/s-h of every test hour: all emissions over all bytes (each byte once)."""
    return R["E"].sum(1) / R["X"].sum(1)


def res_capacity() -> np.ndarray:
    sys.path.insert(0, str(HERE.parent / "green_paths"))
    import simulate as SIM
    return SIM.Net().res_cap


def rate_error_by_load(R) -> pd.DataFrame:
    """Relative error of the posted rate of each resource-hour against its settled rate, by the
    resource's load (share of its capacity), with and without the cap."""
    ok = R["ok"]
    E, Ed, U, Uf = R["E"][ok], R["E_da"][ok], R["U"][ok], R["U_fc"][ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        un_set, un_post = E / U, Ed / Uf
    cap_set, cap_post = R["per_set"][ok], R["per_post"][ok]
    util = U / res_capacity()[None, :]
    bins = [0, 1e-4, 1e-3, 1e-2, 1e-1, np.inf]
    lab = ["<0.01%", "0.01-0.1%", "0.1-1%", "1-10%", ">10%"]
    rows = []
    live = E > 0
    for lo, hi, name in zip(bins[:-1], bins[1:], lab):
        m = live & (util >= lo) & (util < hi) & (U > 0)
        for kind, a, b in (("uncapped", un_post, un_set), ("capped", cap_post, cap_set)):
            err = np.abs(a[m] - b[m]) / b[m]
            err = np.where(np.isfinite(err), err, np.inf)          # posted load 0: infinite rate
            rows.append(dict(load=name, kind=kind, n=int(m.sum()), share=float(m.sum() / (live & (U > 0)).sum()),
                             median=float(np.median(err)), p25=float(np.quantile(err, .25)),
                             p75=float(np.quantile(err, .75)), p90=float(np.quantile(err, .9))))
    return pd.DataFrame(rows)


def od_costs(per, add, A_res):
    """g per Mbit/s-h of every OD and hour: the resource rates along its path plus the add-on."""
    return (A_res.T @ per.T).T + add[:, None]


def router_bills(X, cost, n):
    """Half of every flow's cost to its source router, half to its destination [hours, n]."""
    Cm = (X * cost).reshape(len(X), n, n)
    return (Cm.sum(2) + Cm.sum(1)) / 2


def citypairs(R, A_res) -> pd.DataFrame:
    """Settled path rate (g/GB, bytes-weighted over the block) per router pair and per PoP pair."""
    X, n = R["X"], len(R["ends"])
    c = od_costs(R["per_set"], R["add_set"], A_res)
    pop = [i2.pop_of(str(e)) for e in R["ends"]]
    s, t = np.divmod(np.arange(n * n), n)
    df = pd.DataFrame({"s": [pop[i] for i in s], "t": [pop[j] for j in t], "g": (X * c).sum(0), "x": X.sum(0)})
    df = df[(df.s != df.t) & (df.x > 0)]
    g = df.groupby(["s", "t"])[["g", "x"]].sum()
    g["GB"] = g.x * GB
    g["g_per_GB"] = g.g / g.GB
    return g.reset_index()[["s", "t", "GB", "g_per_GB"]]


def entity_bills(R, A_res) -> pd.DataFrame:
    """Bills of every party (door owner) over the block, flat and path (capped), half of every
    byte at its entry door and half at its exit door; and annualised (x 8760 / hours)."""
    T = pd.to_datetime(R["T"], utc=True)
    n, X = len(R["ends"]), R["X"]
    ends = [str(e) for e in R["ends"]]
    D = pd.read_csv(DATA / "doors.csv")
    DI = pd.read_csv(DATA / "doors_in_hourly.csv.gz", index_col=0, parse_dates=True).reindex(T).fillna(0)
    DO = pd.read_csv(DATA / "doors_out_hourly.csv.gz", index_col=0, parse_dates=True).reindex(T).fillna(0)
    node = D.set_index("door").node.reindex(DI.columns)
    party = D.set_index("door").party.reindex(DI.columns)
    keep = node.isin(ends).to_numpy()
    DI, DO, node, party = DI.loc[:, keep], DO.loc[:, keep], node[keep], party[keep]
    ri = np.array([ends.index(v) for v in node])
    din, dout = DI.to_numpy(), DO.to_numpy()
    # door shares of their router's ingress / egress, every hour
    rin = np.zeros((len(T), n))
    rout = np.zeros((len(T), n))
    for j in range(n):
        rin[:, j] = din[:, ri == j].sum(1)
        rout[:, j] = dout[:, ri == j].sum(1)
    s_in = np.divide(din, rin[:, ri], out=np.zeros_like(din), where=rin[:, ri] > 0)
    s_out = np.divide(dout, rout[:, ri], out=np.zeros_like(dout), where=rout[:, ri] > 0)
    c = od_costs(R["per_set"], R["add_set"], A_res)
    Cm = (X * c).reshape(len(T), n, n)
    src, dst = Cm.sum(2), Cm.sum(1)                                   # g per hour, per router
    path_door = (s_in * src[:, ri] + s_out * dst[:, ri]) / 2
    fr = flat_rate(R)
    xin, xout = X.reshape(-1, n, n).sum(2), X.reshape(-1, n, n).sum(1)  # router ingress / egress in the matrix
    flat_door = fr[:, None] * (s_in * xin[:, ri] + s_out * xout[:, ri]) / 2
    gb_door = (s_in * xin[:, ri] + s_out * xout[:, ri]) * GB          # GB in + out
    df = pd.DataFrame({"party": party.to_numpy(), "flat_g": flat_door.sum(0), "path_g": path_door.sum(0),
                       "GB_in_out": gb_door.sum(0)}).groupby("party").sum()
    scale = 8760 / len(T)
    df["flat_kg_per_year"] = df.flat_g * scale / 1e3
    df["path_kg_per_year"] = df.path_g * scale / 1e3
    df["path_over_flat"] = df.path_g / df.flat_g
    df = df[df.GB_in_out > 0].sort_values("flat_g", ascending=False)
    return df


def bill_errors(R, A_res) -> dict:
    """Router bills at the actual traffic, posted vs settled rates (path, capped) and flat, by period."""
    ok = R["ok"]
    n, X = len(R["ends"]), R["X"]
    T = pd.to_datetime(R["T"], utc=True)
    set_b = router_bills(X, od_costs(R["per_set"], R["add_set"], A_res), n)
    post_b = router_bills(X, od_costs(R["per_post"], R["add_post"], A_res), n)
    v = (X.reshape(-1, n, n).sum(2) + X.reshape(-1, n, n).sum(1)) / 2
    flat_set = v * flat_rate(R)[:, None]
    flat_post = v * (R["E_da"].sum(1) / R["X_fc"].sum(1))[:, None]
    out = {}
    for name, a, b in (("path", post_b, set_b), ("flat", flat_post, flat_set)):
        for per, key in (("hour", T.floor("h")), ("day", T.floor("D"))):
            A_ = pd.DataFrame(a[ok], index=key[ok]).groupby(level=0).sum()
            B_ = pd.DataFrame(b[ok], index=key[ok]).groupby(level=0).sum()
            out[f"{name}_router_{per}_wape"] = float((A_ - B_).abs().to_numpy().sum() / B_.abs().to_numpy().sum())
        day = T.floor("D")
        A_ = pd.DataFrame(a[ok], index=day[ok]).groupby(level=0).sum()
        B_ = pd.DataFrame(b[ok], index=day[ok]).groupby(level=0).sum()
        ra, rb = A_.rolling(30).sum().dropna(), B_.rolling(30).sum().dropna()
        out[f"{name}_router_30day_wape"] = float((ra - rb).abs().to_numpy().sum() / rb.abs().to_numpy().sum())
        # calendar months (complete months of the block only; no overlapping windows)
        mon = A_.index.tz_localize(None).to_period("M")
        full = [p for p in mon.unique() if (mon == p).sum() == p.days_in_month]
        am, bm = A_.groupby(mon).sum().loc[full], B_.groupby(mon).sum().loc[full]
        out[f"{name}_router_month_wape"] = float((am - bm).abs().to_numpy().sum() / bm.abs().to_numpy().sum())
        # 7-day blocks from the block's first day (complete blocks only)
        k = (A_.index - A_.index[0]).days // 7
        n = pd.Series(1, index=A_.index).groupby(k).size()
        kk = n[n == 7].index
        aw, bw = A_.groupby(k).sum().loc[kk], B_.groupby(k).sum().loc[kk]
        out[f"{name}_router_week_wape"] = float((aw - bw).abs().to_numpy().sum() / bw.abs().to_numpy().sum())
    return out


def main():
    R = load_rates(refresh="--refresh" in sys.argv)
    A_res = sp.load_npz(OUT / "A_res.npz").tocsr()
    fr = flat_rate(R)
    U, E = R["U"], R["E"]
    capped = (U > 0) & (R["per_set"] < E / np.where(U > 0, U, 1) * (1 - 1e-12))
    summ = {
        "hours": int(len(R["T"])), "hours_scored": int(R["ok"].sum()),
        "cap_multiple_of_sumE_over_sumU_p10_p50_p90": np.quantile(R["mult"], [.1, .5, .9]).tolist(),
        "cap_g_per_GB_median": float(np.median(R["mult"] * E.sum(1) / U.sum(1)) / GB),
        "cap_over_flat_rate_median": float(np.median(R["mult"] * (E.sum(1) / U.sum(1)) / fr)),
        "resource_hours_capped_share": float(capped[E > 0].mean()),
        "resource_hours_idle_share": float(((U <= 0) & (E > 0)).sum() / (E > 0).sum()),
        "emissions_spread_flat_share": float((R["add_set"] * R["X"].sum(1)).sum() / E.sum()),
        "flat_rate_g_per_GB": float(E.sum() / (R["X"].sum() * GB)),
    }
    summ |= bill_errors(R, A_res)
    log(json.dumps(summ, indent=1))
    RE = rate_error_by_load(R)
    RE.to_csv(OUT / "rate_error_by_load.csv", index=False)
    EF = error_factor_by_load(R)                                     # the paper's Fig. 3b
    EF.to_csv(OUT / "rate_error_factor_by_load.csv", index=False)
    log(EF.round(2).to_string(index=False))
    PE = path_rate_error(R, A_res)
    json.dump(PE, open(OUT / "path_rate_error.json", "w"), indent=1)
    summ |= PE
    CP = citypairs(R, A_res)
    CP.to_csv(OUT / "citypairs.csv", index=False)
    summ["citypair_g_per_GB_p10_p50_p90"] = CP.g_per_GB.quantile([.1, .5, .9]).tolist()
    summ["citypairs"] = int(len(CP))
    EB = entity_bills(R, A_res)
    EB.to_csv(OUT / "entity_bills.csv")
    summ["entities"] = int(len(EB))
    summ["entity_flat_kg_per_year_p10_p50_p90"] = EB.flat_kg_per_year.quantile([.1, .5, .9]).tolist()
    summ["entity_top10_share_flat"] = float(EB.flat_g.nlargest(10).sum() / EB.flat_g.sum())
    summ["entity_moved_share_path_vs_flat"] = float((EB.path_g - EB.flat_g).abs().sum() / EB.flat_g.sum() / 2)
    summ["entity_path_over_flat_p10_p50_p90"] = EB.path_over_flat.quantile([.1, .5, .9]).tolist()
    summ["entity_sum_check"] = [float(EB.flat_g.sum() / E.sum()), float(EB.path_g.sum() / E.sum())]
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    log(json.dumps({k: summ[k] for k in list(summ)[-9:]}, indent=1))
    log(EB.head(12).round(3).to_string())



def error_factor_by_load(R) -> pd.DataFrame:
    """Error factor max(posted / settled, settled / posted) of every resource-hour's rate, by the
    resource's load (share of its capacity), uncapped (E / max(U, 1 kbit/s)) and capped."""
    ok = R["ok"]
    E, Ed, U, Uf = R["E"][ok], R["E_da"][ok], R["U"][ok], R["U_fc"][ok]
    floor = 1e-3
    pairs = {"uncapped": (Ed / np.maximum(Uf, floor), E / np.maximum(U, floor)),
             "capped": (R["per_post"][ok], R["per_set"][ok])}
    util = U / res_capacity()[None, :]
    bins = [0, 1e-4, 1e-3, 1e-2, 1e-1, np.inf]
    lab = ["<0.01%", "0.01-0.1%", "0.1-1%", "1-10%", ">10%"]
    live = (E > 0) & np.isfinite(util)
    rows = []
    for lo, hi, name in zip(bins[:-1], bins[1:], lab):
        m = live & (util >= lo) & (util < hi)
        for kind, (a, b) in pairs.items():
            f = np.exp(np.abs(np.log(a[m] / b[m])))
            rows.append(dict(load=name, kind=kind, n=int(m.sum()), share=float(m.sum() / live.sum()),
                             median=float(np.median(f)), p75=float(np.quantile(f, .75)), p90=float(np.quantile(f, .9)),
                             p99=float(np.quantile(f, .99))))
    return pd.DataFrame(rows)


def path_rate_error(R, A_res) -> dict:
    """Bytes-weighted error of the posted path rate (what a host sees) against the settled one."""
    ok = R["ok"]
    X = R["X"][ok]
    floor = 1e-3
    out = {}
    for kind in ("uncapped", "capped"):
        if kind == "capped":
            cp = od_costs(R["per_post"][ok], R["add_post"][ok], A_res)
            cs = od_costs(R["per_set"][ok], R["add_set"][ok], A_res)
        else:
            cp = (A_res.T @ (R["E_da"][ok] / np.maximum(R["U_fc"][ok], floor)).T).T
            cs = (A_res.T @ (R["E"][ok] / np.maximum(R["U"][ok], floor)).T).T
        out[f"path_rate_wape_bytes_{kind}"] = float((X * np.abs(cp - cs)).sum() / (X * cs).sum())
        f = np.exp(np.abs(np.log(np.where(X > 0, cp, 1) / np.where(X > 0, cs, 1))))[X > 0]
        out[f"path_rate_error_factor_p50_p90_{kind}"] = np.quantile(f, [.5, .9]).tolist()
    out["flat_rate_hourly_wape"] = float(np.abs(R["E_da"][ok].sum(1) / R["X_fc"][ok].sum(1) - R["E"][ok].sum(1) / X.sum(1)).sum()
                                         / (R["E"][ok].sum(1) / X.sum(1)).sum())
    return out


if __name__ == "__main__":
    main()
