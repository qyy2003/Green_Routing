"""What per-path rates posted a day ahead let hosts do on the I2 Network (paper Fig. 4).

Rates are the paper's: usage-priced per resource (core router, inter-PoP span) with the idle-link cap
(i2_rates.py). A path's rate is the sum of its resources' rates plus the hour's flat add-on.

``unilateral`` (CIRo-style): every router pair, alone, takes its cheapest path under the *posted*
(day-ahead) rates if that saves >= 1 % over today's routing (ECMP, inverse-capacity weights); its
realized intensity is the *settled* rate of that path after its own bytes moved, against its
settled rate today. Oracle: the same choice made on the settled rates (known only afterwards).
Aggregated per PoP pair (router pairs at the same two PoPs, bytes-weighted over the block).

``everyone`` (closed loop, decentralized sleeping): every pair takes its cheapest posted path each
hour (staying on today's routing unless it saves >= 1 %); directed links above 80 % utilization
are penalized and the flows on them re-routed (rounds), so no link is pushed over 80 %. The posted
rates of day d come from the loads the network actually carried: each resource's load in the same
hour a week earlier (the steered loads; before the block, today's routing), the day-ahead
emissions (grid of the same hour yesterday) and Timestral's day-ahead total; the cap from the
previous 28 days of steered rates. Then the settled state: spans carrying nothing (their optics
could sleep), their emissions, utilization, hops, who pays more.

``central`` (reference, a centralized link-sleeping controller): one set of spans asleep for the
whole block, greedy from the least loaded span: a span goes to sleep if every router with doors
stays connected and ECMP routing (inverse-capacity) on the remaining links keeps every link at or
under 80 % (or under today's own maximum, where that is higher) in every hour of the block.

usage: python steering.py unilateral|everyone|central|all [--k=1|2|3|none]  (the cap's robust sigmas; default 2)
writes $GR_ARTIFACTS_DIR/paper/i2/{unilateral_pairs.csv, everyone_hourly.csv, central.json, ...}
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from green_routing.carbon import path_rate, tomography
from green_routing.dataset import internet2 as i2
from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "green_paths"))
import i2_rates as IR                     # noqa: E402
import simulate as SIM                    # noqa: E402

OUT = IR.OUT
GAIN = 0.01                               # a flow moves only if it saves at least 1 %
CAP_UTIL = 0.8
EMPTY = 1e-3                              # Mbit/s: a resource or link below this is empty
MAX_ROUNDS = 60
R = NET = None
K_CAP = 2.0                               # the cap's robust sigmas (None: no cap); --k on the command line


def _tag() -> str:
    return "" if K_CAP == 2.0 else f"_k{K_CAP}"


def log(*a):
    print(*a, flush=True)


def _init():
    global R, NET
    R = IR.load_rates_k(K_CAP)
    NET = SIM.Net()
    assert [str(r) for r in R["res"]] == NET.res and [str(e) for e in R["ends"]] == NET.ends


# --- one pair alone ----------------------------------------------------------------------------
def _after(B, k, x, U0, E, cap):
    """Settled cost (g per Mbit/s-h, resources only) of ODs k on paths B after their own bytes moved."""
    Ak, Bk = NET.A_res[:, k], B[:, k]
    Uk = U0[:, None] - Ak.multiply(x[k]).toarray() + Bk.multiply(x[k]).toarray()
    rate = np.where(Uk > EMPTY, np.minimum(E[:, None] / np.maximum(Uk, 1e-12), cap), cap)
    return np.asarray(Bk.multiply(rate).sum(axis=0)).ravel()


def uni_hour(h):
    x, E, U0 = R["X"][h], R["E"][h], R["U"][h]
    per_s, per_p = R["per_set"][h], R["per_post"][h]
    cap = R["mult"][h] * E.sum() / U0.sum()
    c0 = NET.A_res.T @ per_s
    k = np.nonzero((x > 0) & (NET.src != NET.dst))[0]
    out = {"k": k, "x": x[k], "c0": c0[k], "add": np.full(len(k), R["add_set"][h])}
    cp_best, Bp, _ = NET.shortest(per_p)
    cp_def = NET.A_res.T @ per_p
    mv = cp_best[k] < cp_def[k] * (1 - GAIN)
    c_fc = c0[k].copy()
    if mv.any():
        c_fc[mv] = _after(Bp, k[mv], x, U0, E, cap)
    out["c_fc"], out["moved_fc"] = c_fc, mv
    out["posted_saving"] = np.where(mv, 1 - cp_best[k] / cp_def[k], 0.0)
    co_best, Bo, _ = NET.shortest(per_s)
    mo = co_best[k] < c0[k] * (1 - GAIN)
    c_or = c0[k].copy()
    if mo.any():
        c_or[mo] = np.minimum(_after(Bo, k[mo], x, U0, E, cap), c0[k][mo])
    out["c_or"], out["moved_or"] = c_or, mo
    # does the day-ahead choice equal the choice made on the settled rates? (both stay, or both
    # move to the same path)
    diff = np.asarray(abs(Bp[:, k] - Bo[:, k]).sum(axis=0)).ravel() > 1e-9
    out["same"] = (~mv & ~mo) | (mv & mo & ~diff)
    return h, out


def unilateral(workers: int = 6):
    _init()
    hrs = [h for h in range(len(R["T"])) if R["ok"][h]]
    K = NET.n * NET.n
    acc = {k: np.zeros(K) for k in ("x", "b0", "bfc", "bor", "mfc", "mor", "post", "same")}
    with Pool(workers) as pool:
        for i, (h, o) in enumerate(pool.imap_unordered(uni_hour, hrs, chunksize=8)):
            k, x = o["k"], o["x"]
            acc["x"][k] += x
            acc["b0"][k] += x * (o["c0"] + o["add"])
            acc["bfc"][k] += x * (o["c_fc"] + o["add"])
            acc["bor"][k] += x * (o["c_or"] + o["add"])
            acc["mfc"][k] += x * o["moved_fc"]
            acc["mor"][k] += x * o["moved_or"]
            acc["post"][k] += x * o["posted_saving"]
            acc["same"][k] += x * o["same"]
            if i % 250 == 0:
                log(f"  unilateral {i}/{len(hrs)}")
    pop = [i2.pop_of(e) for e in NET.ends]
    df = pd.DataFrame({"s": [pop[i] for i in NET.src], "t": [pop[j] for j in NET.dst], **acc})
    df = df[(df.s != df.t) & (df.x > 0)]
    g = df.groupby(["s", "t"]).sum()
    P = pd.DataFrame({"GB": g.x * IR.GB, "reduction_forecast": 1 - g.bfc / g.b0, "reduction_oracle": 1 - g.bor / g.b0,
                      "moved_share_forecast": g.mfc / g.x, "moved_share_oracle": g.mor / g.x,
                      "posted_saving": g.post / g.x, "same_choice": g.same / g.x}).reset_index()
    P.to_csv(OUT / f"unilateral_pairs{_tag()}.csv", index=False)
    tot = {"pairs": int(len(P)), "bytes_weighted_reduction_forecast": float(1 - g.bfc.sum() / g.b0.sum()),
           "bytes_weighted_reduction_oracle": float(1 - g.bor.sum() / g.b0.sum()),
           "bytes_same_choice": float(g.same.sum() / g.x.sum()),
           "bytes_moved_forecast": float(g.mfc.sum() / g.x.sum()), "bytes_moved_oracle": float(g.mor.sum() / g.x.sum())}
    for col in ("reduction_forecast", "reduction_oracle"):
        for thr in (0.01, 0.1, 0.2, 0.3, 0.5):
            tot[f"pairs_{col}_ge_{int(thr * 100)}pct"] = float((P[col] >= thr).mean())
        tot[f"{col}_p50_p75_p90"] = P[col].quantile([.5, .75, .9]).tolist()
    tot["pairs_worse_forecast"] = float((P.reduction_forecast < -0.001).mean())
    json.dump(tot, open(OUT / f"unilateral{_tag()}.json", "w"), indent=1)
    log(json.dumps(tot, indent=1))


# --- everyone, closed loop ---------------------------------------------------------------------
def everyone(seed: int = 0):
    _init()
    rng = np.random.default_rng(seed)
    S = NET
    P = np.load(IR.PRE / "test_setup.npz")
    A = S.A_res.tocsr()
    T = pd.to_datetime(R["T"], utc=True)
    Tp = pd.to_datetime(P["T"], utc=True)
    hist_U = {t: u for t, u in zip(Tp, (A @ P["X_em"].T).T)}       # today's routing before the block
    hist_E = {t: e for t, e in zip(Tp, P["E"])}
    hist_X = {t: x.sum() for t, x in zip(Tp, P["X_em"])}
    K = S.n * S.n
    rows, span_dark = [], []
    mult, cur_day = None, None
    for h in range(len(T)):
        t = T[h]
        if t.floor("D") != cur_day:
            cur_day = t.floor("D")
            win = [u for u in hist_U if cur_day - pd.Timedelta(days=28) <= u < cur_day]
            mult = 1e9 if K_CAP is None else path_rate.cap_multiple(
                np.array([hist_E[u] for u in win]), np.array([hist_U[u] for u in win]), k=K_CAP)
        x, E = R["X"][h], R["E"][h]
        if not R["ok"][h]:
            continue
        Uhat = next((hist_U[t - pd.Timedelta(weeks=w)] for w in (1, 2, 3) if t - pd.Timedelta(weeks=w) in hist_U), R["U_fc"][h])
        per_p, _ = path_rate.capped_rates(R["E_da"][h], Uhat, R["X_fc"][h].sum(), mult)
        c_def = A.T @ per_p
        cost, B, BL = S.shortest(per_p)
        move = (x > 0) & (S.src != S.dst) & (cost < c_def * (1 - GAIN))
        stay = sp.diags((~move).astype(float))
        Cr = (S.A_res @ stay + B @ sp.diags(move.astype(float))).tocsc()
        Cl = (S.A_link @ stay + BL @ sp.diags(move.astype(float))).tocsc()
        pen = np.zeros(S.nl, bool)
        rounds = 0
        for rounds in range(MAX_ROUNDS):
            L = Cl @ x
            over = L / S.cap > CAP_UTIL
            if not over.any():
                break
            pen |= over
            cost2, B2, BL2 = S.shortest(per_p, np.where(pen, 1e3 * per_p.max(), 0.0))
            on = (np.asarray(Cl[over].sum(axis=0)).ravel() > 0) & (x > 0)
            mv = on & (rng.random(K) < 0.25)
            if not mv.any():
                mv = on & (np.arange(K) == np.nonzero(on)[0][rng.integers(on.sum())])
            keep, m = sp.diags((~mv).astype(float)), sp.diags(mv.astype(float))
            Cr, Cl = (Cr @ keep + B2 @ m).tocsc(), (Cl @ keep + BL2 @ m).tocsc()
        U, L = Cr @ x, Cl @ x
        per_s, add_s = path_rate.capped_rates(E, U, x.sum(), mult)
        c_now = Cr.T @ per_s + add_s
        c_today = A.T @ R["per_set"][h] + R["add_set"][h]
        dark = S.is_span & (U < EMPTY)
        util = L / S.cap
        hops = np.asarray(Cl.sum(axis=0)).ravel()
        rows.append(dict(hour=str(t), spans_dark=int(dark.sum()), dark_E_share=float(E[dark].sum() / E.sum()),
                         util_max=float(util.max()), links_over_80=int((util > 0.8 + 1e-9).sum()),
                         links_over_100=int((util > 1.0).sum()), rounds=rounds,
                         hops_per_byte=float((hops * x).sum() / x.sum()),
                         bytes_moved=float(x[move].sum() / x[(x > 0) & (S.src != S.dst)].sum()),
                         bytes_paying_more=float(x[c_now > c_today * 1.001].sum() / x.sum()),
                         bytes_paying_less=float(x[c_now < c_today * 0.999].sum() / x.sum()),
                         bill_check=float((x * c_now).sum() / E.sum()), cap_mult=mult,
                         today_spans_dark=int((S.is_span & (R["U"][h] < EMPTY)).sum()),
                         today_util_max=float(((S.A_link @ x) / S.cap).max()),
                         today_hops_per_byte=float((np.asarray(S.A_link.sum(axis=0)).ravel() * x).sum() / x.sum())))
        span_dark.append(dark[S.is_span])
        hist_U[t], hist_E[t] = U, E
        if h % 168 == 0:
            log(f"  {t} dark spans {dark.sum()} util max {util.max():.2f} rounds {rounds} moved {rows[-1]['bytes_moved']:.2f}")
    D = pd.DataFrame(rows)
    D.to_csv(OUT / f"everyone_hourly{_tag()}.csv", index=False)
    names = [r for r, s in zip(S.res, S.is_span) if s]
    sd = pd.DataFrame(np.array(span_dark), columns=names)
    pd.DataFrame({"share_hours_dark": sd.mean()}).to_csv(OUT / f"everyone_spans{_tag()}.csv")
    summ = D.drop(columns=["hour"]).describe(percentiles=[.1, .5, .9]).T[["mean", "10%", "50%", "90%", "max"]]
    log(summ.round(3).to_string())
    last = D.iloc[-24 * 28:]
    res = {"mean": D.drop(columns=["hour"]).mean().to_dict(), "last_4_weeks_mean": last.drop(columns=["hour"]).mean().to_dict(),
           "spans_dark_all_block": int((sd.mean() == 1).sum()), "spans_dark_last_4_weeks_all": int((sd.iloc[-24 * 28:].mean() == 1).sum()),
           "spans": int(len(names))}
    json.dump(res, open(OUT / f"everyone{_tag()}.json", "w"), indent=1)
    log(json.dumps(res, indent=1))


# --- centralized reference -----------------------------------------------------------------------
def central():
    _init()
    S = NET
    links = pd.read_csv(IR.DATA / "links.csv")
    cap = links.capacity_bps.fillna(links.capacity_bps.median())
    edges0 = {(u, v): 1e11 / c for u, v, c in zip(links.u, links.v, cap)}
    capd = {(u, v): c / 1e6 for u, v, c in zip(links.u, links.v, cap)}
    D = pd.read_csv(IR.DATA / "doors.csv")
    core = sorted(set(links.u) | set(links.v) | set(D.node))
    ends = S.ends
    ok = R["ok"]
    X = R["X"][ok]
    E = R["E"][ok]

    def evaluate(edges):
        lks, A_link, A_node, od = tomography.ecmp_routing(core, edges, ends)
        assert od == [(s, t) for s in ends for t in ends]
        L = A_link @ X.T                                          # [links, hours] Mbit/s
        c = np.array([capd[l] for l in lks])
        return lks, A_link, L / c[:, None]

    # routers with doors but no backbone link (core1.phoe3) are never connected: leave them out
    linked = [e for e in ends if any(e in k for k in edges0)]

    def connected(edges):
        nodes = sorted({u for u, _ in edges} | {v for _, v in edges} | set(linked))
        ix = {n: i for i, n in enumerate(nodes)}
        a, b = zip(*[(ix[u], ix[v]) for u, v in edges])
        _, lab = connected_components(sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes),) * 2), directed=False)
        return len({lab[ix[e]] for e in linked}) == 1

    lks0, Al0, util0 = evaluate(edges0)
    limit = np.maximum(CAP_UTIL, util0.max(axis=0))                 # per hour: 80 %, or today's max if higher
    spans = [r for r, s in zip(S.res, S.is_span) if s]
    sidx = [S.res.index(r) for r in spans]
    order = np.argsort(R["U"][ok][:, sidx].mean(0))
    asleep, edges = [], dict(edges0)
    for j in order:
        u, v = spans[j].split("|")
        trial = {e: w for e, w in edges.items() if e not in ((u, v), (v, u))}
        if len(trial) == len(edges) or not connected(trial):
            continue
        _, _, ut = evaluate(trial)
        if (ut.max(axis=0) <= limit + 1e-9).all():
            edges = trial
            asleep.append(spans[j])
            log(f"  asleep {len(asleep)}: {spans[j]}")
    lks, Al, ut = evaluate(edges)
    Uspan = {}
    for i, (u, v) in enumerate(lks):
        key = "|".join(sorted((u, v)))
        Uspan[key] = Uspan.get(key, 0) + ut[i].mean()
    dark = [s for s in spans if s in asleep or Uspan.get(s, 0) * 1 < 1e-12]
    di = [S.res.index(s) for s in dark]
    hops = np.asarray(Al.sum(axis=0)).ravel()
    res = {"spans": len(spans), "asleep": len(asleep), "dark": len(dark), "dark_E_share": float(E[:, di].sum() / E.sum()),
           "util_max": float(ut.max()), "util_p99_of_hourly_max": float(np.quantile(ut.max(axis=0), .99)),
           "today_util_max": float(util0.max()), "hours_today_over_80": int((util0.max(axis=0) > CAP_UTIL).sum()),
           "hops_per_byte": float((X @ hops).sum() / X.sum()),
           "today_hops_per_byte": float((X @ np.asarray(Al0.sum(axis=0)).ravel()).sum() / X.sum()),
           "asleep_spans": asleep}
    json.dump(res, open(OUT / "central.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in res.items() if k != "asleep_spans"}, indent=1))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for a in sys.argv[2:]:
        if a.startswith("--k="):
            v = a.split("=", 1)[1]
            K_CAP = None if v == "none" else float(v)
    for name, f in (("unilateral", unilateral), ("everyone", everyone), ("central", central)):
        if what in (name, "all"):
            log(f"== {name}")
            f()
