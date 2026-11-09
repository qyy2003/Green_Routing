"""Internet2 path rates with the 7-day correction (design §3.3) and a margin, on the test block.

Every day d, all posted resource rates and the flat add-on (i2_rates.py, k = 2) are scaled by
c_d = sum of measured emissions / sum of path bills over the previous 7 days, where a path bill is the posted
rates times the actual loads (every byte: resource rates x loads + add-on x bytes); then by (1 - m). Days at the
start of the block use the days of the block before them (day 1: c = 1); only hours with posted rates count.
The margin m: no per-link forecasts exist for the development period (Timestral runs per link on the test block
only), so m is the flat rate's m of Internet2 (flat_common.py, $PAPER_COMMON/flat/flat.json).

Reports the path own share (1 - bills / emissions) over the block and per calendar month, before and after, the
bytes-weighted error of the path rate a host sees, and checks that the choices of a pair acting alone do not
change (steering.py uni_hour on sampled hours with the scaled rates).

usage: GR_ARTIFACTS_DIR=$PAPER_COMMON/root python path_correction.py     -> $PAPER_COMMON/paths/correction.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "green_paths"))
import periods as P                                                  # noqa: E402
import i2_rates as IR                                                # noqa: E402

OUT = P.ROOT / "paths"
N_DAYS = 7


def hourly_bills(R, scale=None):
    s = np.ones(len(R["T"])) if scale is None else scale
    return s * ((R["per_post"] * R["U"]).sum(1) + R["add_post"] * R["X"].sum(1))


def day_factors(R, n=N_DAYS) -> np.ndarray:
    """c_d per hour of the block (the factor of the hour's day)."""
    ok = R["ok"]
    T = pd.to_datetime(R["T"], utc=True)
    day = T.floor("D")
    b = pd.Series(np.where(ok, hourly_bills(R), 0.0), index=T).groupby(day).sum()
    e = pd.Series(np.where(ok, R["E"].sum(1), 0.0), index=T).groupby(day).sum()
    c = (e.rolling(n, min_periods=1).sum() / b.rolling(n, min_periods=1).sum()).shift(1).fillna(1.0)
    return c.reindex(day).to_numpy()


def own_shares(R, bill_h) -> dict:
    ok = R["ok"]
    T = pd.to_datetime(R["T"], utc=True)[ok]
    E = R["E"][ok].sum(1)
    b = bill_h[ok]
    d = pd.DataFrame({"bill": pd.Series(b, index=T).groupby(T.floor("D")).sum(),
                      "E": pd.Series(E, index=T).groupby(T.floor("D")).sum()})
    mon = d.index.tz_localize(None).to_period("M")
    g = d.groupby(mon)
    om = 1 - g.bill.sum() / g.E.sum()
    k = (d.index - d.index[0]).days // 7
    wk = d.groupby(k).sum()[d.groupby(k).size() == 7]
    return {"own_share": float(1 - d.bill.sum() / d.E.sum()), "own_month": {str(a): float(v) for a, v in om.items()},
            "own_week_min": float((1 - wk.bill / wk.E).min()), "own_week_max": float((1 - wk.bill / wk.E).max()),
            "err_day": float((d.bill - d.E).abs().sum() / d.E.sum()),
            "err_week": float((wk.bill - wk.E).abs().sum() / wk.E.sum()),
            "err_month": float((g.bill.sum() - g.E.sum()).abs().sum() / g.E.sum().sum())}


def scaled(R, s):
    Q = dict(R)
    Q["per_post"] = R["per_post"] * s[:, None]
    Q["add_post"] = R["add_post"] * s
    Q["E_da"] = R["E_da"] * s[:, None]                                # capped_rates is linear in E
    return Q


def check_choices(s, n_hours=200) -> dict:
    """steering.py's one pair alone with the posted rates x s(h): same moves, paths and realized costs."""
    import steering as ST
    ST._init()
    hrs = [h for h in range(len(ST.R["T"])) if ST.R["ok"][h]]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(hrs, min(n_hours, len(hrs)), replace=False))
    base = {h: ST.uni_hour(h)[1] for h in pick}
    ST.R["per_post"] = ST.R["per_post"] * s[:, None]
    ST.R["add_post"] = ST.R["add_post"] * s
    diff = 0
    for h in pick:
        o, b = ST.uni_hour(h)[1], base[h]
        diff += not (np.array_equal(o["moved_fc"], b["moved_fc"]) and np.allclose(o["c_fc"], b["c_fc"], rtol=1e-12)
                     and np.array_equal(o["same"], b["same"]))
    return {"hours": len(pick), "hours_with_any_difference": int(diff),
            "scale_min_max": [float(s[pick].min()), float(s[pick].max())]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import scipy.sparse as sp
    R = IR.load_rates()
    A_res = sp.load_npz(IR.OUT / "A_res.npz").tocsr()
    m = float(json.load(open(P.ROOT / "flat" / "flat.json"))["i2"]["m"])
    c = day_factors(R)
    variants = {"posted": np.ones(len(c)), "margin_only": np.full(len(c), 1 - m), "corrected": c,
                "corrected_margin": c * (1 - m)}
    res = {"m": m, "m_source": "Internet2's flat-rate margin (flat_common.py): no per-link development forecasts",
           "factor_p10_p50_p90": np.quantile(c, [.1, .5, .9]).tolist(), "factor_first_days": c[::24][:8].tolist()}
    for name, s in variants.items():
        Q = scaled(R, s)
        r = own_shares(R, hourly_bills(R, s))
        r["host_rate_error"] = IR.path_rate_error(Q, A_res)
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "host_rate_error"}), flush=True)
        print("   host-seen rate error (capped, bytes-weighted):", round(r["host_rate_error"]["path_rate_wape_bytes_capped"], 4), flush=True)
    # where the over-billing comes from: resource part vs the flat add-on
    ok = R["ok"]
    res["resource_part_over_E"] = float((R["per_post"][ok] * R["U"][ok]).sum() / R["E"][ok].sum())
    res["addon_part_over_E"] = float((R["add_post"][ok] * R["X"][ok].sum(1)).sum() / R["E"][ok].sum())
    res["check_choices_corrected_margin"] = check_choices(c * (1 - m))
    print(json.dumps(res["check_choices_corrected_margin"]), flush=True)
    json.dump(res, open(OUT / "correction.json", "w"), indent=1)


if __name__ == "__main__":
    main()
