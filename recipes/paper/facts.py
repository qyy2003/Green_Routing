"""Numbers quoted in the paper's text that no figure shows (both networks).

- the hourly flat rate over a year: percentiles (per hour, and per byte), what drives its variance
  (log r = log E - log T: the grid's share, traffic's share), its daily cycle (median by hour of day),
  weekend vs weekday;
- kWh per GB of the shared backbone;
- how well tomorrow's power is known: I2's hourly DC power, "same hour yesterday";
usage: python facts.py      writes $GR_ARTIFACTS_DIR/paper/facts.json and hourly_year_{switch,i2}.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
OUT = ARTIFACTS_DIR / "paper"
I2C = ARTIFACTS_DIR / "internet2" / "carbon"


def switch_year() -> pd.DataFrame:
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    import grid_forecast as G
    import run as R
    from green_routing.carbon import assumptions as A
    E = G.switch_emissions()                                      # g/h, backbone, actual grid
    H = pd.date_range(R.TEST[-1] - pd.Timedelta(days=364), R.TEST[-1] + pd.Timedelta(hours=23), freq="h")
    T = R.T.reindex(H)                                            # Gbit/s, truth
    df = pd.DataFrame({"E": E.reindex(H), "T_gbps": T})
    df["r"] = df.E / (df.T_gbps * 1e3 * A.GB_PER_MBPS_HOUR)       # g/GB
    return df.dropna()


def i2_year() -> pd.DataFrame:
    fr = pd.read_csv(I2C / "flat_rate_hourly.csv", index_col=0, parse_dates=True)
    df = pd.DataFrame({"E": fr.E_g_per_h, "T_gbps": fr.V_mbps / 1e3, "r": fr.flat_g_per_GB})
    return df.dropna()


def describe(df: pd.DataFrame, tz: str) -> dict:
    r, w = df.r, df.T_gbps
    o = np.argsort(r.to_numpy())
    cw = np.cumsum(w.to_numpy()[o]) / w.sum()
    lr, le, lt = np.log(df.r), np.log(df.E), np.log(df.T_gbps)
    var = lr.var()
    loc = df.index.tz_convert(tz)
    by_h = r.groupby(loc.hour).median()
    wk = r.groupby(loc.dayofweek >= 5).median()
    day = r.groupby(loc.normalize()).mean()
    return {"hours": int(len(df)), "first": str(df.index[0]), "last": str(df.index[-1]),
            "rate_mean_g_per_GB": float(df.E.sum() / (df.T_gbps.sum() * 1e3 * 0.45)),
            "rate_hour_p10_p50_p90": r.quantile([.1, .5, .9]).tolist(),
            "rate_byte_p10_p50_p90": [float(np.interp(p, cw, r.to_numpy()[o])) for p in (.1, .5, .9)],
            "var_share_grid": float(le.var() / var), "var_share_traffic": float(lt.var() / var),
            "corr_logE_logT": float(np.corrcoef(le, lt)[0, 1]),
            "cv_daily_rate": float(day.std() / day.mean()),
            "cv_daily_E": float(df.E.groupby(loc.normalize()).mean().std() / df.E.groupby(loc.normalize()).mean().mean()),
            "cv_daily_T": float(df.T_gbps.groupby(loc.normalize()).mean().std() / df.T_gbps.groupby(loc.normalize()).mean().mean()),
            "hour_of_day_median_max_over_min": float(by_h.max() / by_h.min()),
            "hour_of_day_cheapest": int(by_h.idxmin()), "hour_of_day_dearest": int(by_h.idxmax()),
            "weekend_over_weekday_median": float(wk.get(True) / wk.get(False)),
            "hour_of_day_median": by_h.round(6).tolist()}


def switch_kwh_per_gb() -> dict:
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    import run as R
    from green_routing.carbon import assumptions as A, bills
    from green_routing.paths import reference_dir
    C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
    H = pd.date_range(R.TEST[0], R.TEST[-1] + pd.Timedelta(hours=23), freq="h")
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    tiers = pd.read_csv(NET / "routers.csv").set_index("dev").tier
    _, Pbb, _ = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"),
                                   pd.read_csv(C / "allocation" / "device_pools.csv"),
                                   pd.read_csv(C / "inventory" / "shelves.csv"),
                                   pd.read_csv(reference_dir() / "sites.csv"), doors, tiers, H)
    kw = float(sum(np.nanmean(np.asarray(Pbb[z], float)) for z in Pbb) / 1e3)
    T = R.T.reindex(H)
    gb = float(T.mean() * 1e3 * A.GB_PER_MBPS_HOUR * len(H))
    return {"backbone_kW_site": kw, "kWh_per_GB": kw * len(H) / gb, "window": [str(H[0]), str(H[-1])]}


def i2_kwh_and_power() -> dict:
    s = json.load(open(I2C / "summary.json"))
    kwh_gb = s["site_kW"] * s["year_hours"] / (s["PB_year"] * 1e6)
    W = pd.read_csv(ARTIFACTS_DIR / "internet2" / "data" / "power_hourly.csv.gz", parse_dates=["time"])
    W = W[W.dc & W["pop"].notna()]
    P = W.pivot_table(index="time", columns="pop", values="watts", aggfunc="sum")
    P = P.loc[(P.index >= s["test"][0]) & (P.index < s["test"][1])]
    P = P.loc[:, P.median() > 0]
    tot = P.sum(axis=1)
    pers = tot.shift(24)
    ok = pers.notna() & tot.notna()
    pop_err = ((P - P.shift(24)).abs().sum() / P.sum()).median()
    return {"site_kW": s["site_kW"], "kWh_per_GB": kwh_gb,
            "power_dayahead_persistence_wape_total": float((pers[ok] - tot[ok]).abs().sum() / tot[ok].sum()),
            "power_dayahead_persistence_wape_median_pop": float(pop_err),
            "power_cv_hourly_total": float(tot.std() / tot.mean()), "pops": int(P.shape[1])}


def main():
    sw, i2 = switch_year(), i2_year()
    sw.to_csv(OUT / "hourly_year_switch.csv")
    i2.to_csv(OUT / "hourly_year_i2.csv")
    facts = {"switch": describe(sw, "Europe/Zurich") | switch_kwh_per_gb(),
             "i2": describe(i2, "America/Chicago") | i2_kwh_and_power()}
    json.dump(facts, open(OUT / "facts.json", "w"), indent=1)
    print(json.dumps({k: {a: b for a, b in v.items() if a != "hour_of_day_median"} for k, v in facts.items()}, indent=1))


if __name__ == "__main__":
    main()
