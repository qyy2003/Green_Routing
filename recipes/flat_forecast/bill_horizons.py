"""Does the day-ahead flat-rate bill error cancel out over weeks and months?

The ISP publishes an hourly flat rate a day ahead, r_f(h) = E_f(h) / T_f(h): forecast backbone
emissions (power x day-ahead grid factor, "same hour yesterday") over forecast total traffic. Every
GB of hour h is billed at r_f(h). For a period P (day, week, month) the billed carbon is
  Bill_f(P) = sum_h r_f(h) T(h) = sum_h E_f(h) T(h) / T_f(h)
and what the network actually emitted is E(P) = sum_h E(h) (= the settled bill, r(h) = E(h)/T(h)).
Error of P: |Bill_f(P) - E(P)| / E(P); reported as WAPE over all periods of a length,
sum_P |Bill_f(P) - E(P)| / sum_P E(P). Same for the traffic volume, sum_P |T_f - T| / sum_P T.

Periods: UTC days; 7-day blocks from the test start; calendar months inside the test block; and
every 7-day / 30-day window (rolling, for a steadier estimate than three months). Variants: day-ahead
grid factor or the actual one ("perfect grid"), forecast or actual traffic ("perfect traffic").

usage: python bill_horizons.py           (both networks; writes <ARTIFACTS>/flat_forecast/bill_horizons.csv)
Inputs: flat_forecast (Switch LAN: run.py's truth, forecasts and carbon/ outputs) and
internet2/carbon (dayahead_hourly.csv, timestral_total.csv).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
OUT = ARTIFACTS_DIR / "flat_forecast"
SWITCH_FORECAST = "ensemble: 0.8 x Chronos-2 + 0.2 x GBDT, GBDT alone on holidays"   # final.json
SWITCH_TIMESTRAL = "timestral: 24 weeks + calendar covariates"     # the same model as on I2
I2_FORECAST = "Timestral-209M, 24 weeks"


def switch_hours(forecast: str = SWITCH_FORECAST) -> pd.DataFrame:
    """Hourly T (actual), T_f, E (actual grid), E_f (day-ahead grid) on the test block, as in run.py
    stage_rate."""
    sys.path.insert(0, str(HERE))
    import run as R
    from green_routing.carbon import bills, grid
    from green_routing.paths import reference_dir
    C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
    H = pd.date_range(R.TEST[0], R.TEST[-1] + pd.Timedelta(hours=23), freq="h")
    doors = pd.read_csv(C / "allocation" / "doors.csv")
    tiers = pd.read_csv(NET / "routers.csv").set_index("dev").tier
    _, Pbb, _ = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"),
                                   pd.read_csv(C / "allocation" / "device_pools.csv"),
                                   pd.read_csv(C / "inventory" / "shelves.csv"),
                                   pd.read_csv(reference_dir() / "sites.csv"), doors, tiers, H)
    IA = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in Pbb})
    IF = pd.DataFrame({z: grid.day_ahead(grid.load(C / "grid", z)).reindex(H).interpolate() for z in Pbb})
    B = lambda I_: sum(Pbb[z] / 1000 * I_[z] for z in Pbb)            # noqa: E731  g per hour
    f = R.cache()[forecast]
    df = pd.DataFrame({"T": R.T.reindex(H), "Tf": f.reindex(H), "E": B(IA), "Ef": B(IF)})
    return df.dropna()


def i2_hours() -> pd.DataFrame:
    C = ARTIFACTS_DIR / "internet2" / "carbon"
    tab = pd.read_csv(C / "dayahead_hourly.csv", index_col=0, parse_dates=True)
    fc = pd.read_csv(C / "timestral_total.csv", index_col=0, parse_dates=True)
    df = pd.DataFrame({"T": tab.actual, "Tf": fc[I2_FORECAST].reindex(tab.index),
                       "E": tab.E_g_per_h, "Ef": tab.E_dayahead_g_per_h})
    return df.dropna()


def period_errors(df: pd.DataFrame) -> list[dict]:
    """Errors of the billed carbon and of the traffic volume for every period length, plus the
    bias, spread and lag-1 autocorrelation of the daily relative bill error."""
    variants = {"forecast traffic, day-ahead grid": ("Tf", "Ef"),
                "forecast traffic, perfect grid": ("Tf", "E"),
                "perfect traffic, day-ahead grid": ("T", "Ef")}
    day = df.index.floor("D")
    days = pd.DatetimeIndex(sorted(set(day)))
    block = pd.Series(np.arange(len(days)) // 7, index=days)
    block = block[block.map(block.value_counts()) == 7]                 # complete 7-day blocks
    month = pd.Series(days.tz_localize(None).to_period("M"), index=days)
    full = [m for m in month.unique() if (month == m).sum() == m.days_in_month]
    month = month[month.isin(full)].astype(str)                          # complete calendar months
    rows = []
    for vname, (tf, ef) in variants.items():
        bill = df[ef] * df["T"] / df[tf]                                 # billed carbon per hour
        D = pd.DataFrame({"bill": bill.groupby(day).sum(), "E": df["E"].groupby(day).sum(),
                          "Tf": df[tf].groupby(day).sum(), "T": df["T"].groupby(day).sum()}).reindex(days)

        def grouped(keys, a, b):
            s = D.loc[keys.index, [a, b]].groupby(keys.to_numpy()).sum()
            return float((s[a] - s[b]).abs().sum() / s[b].sum()), len(s)

        def rolling(k, a, b):
            ra, rb = D[a].rolling(k).sum().dropna(), D[b].rolling(k).sum().dropna()
            return float((ra - rb).abs().sum() / rb.sum()), len(ra)

        per = {"day": lambda a, b: grouped(pd.Series(np.arange(len(days)), index=days), a, b),
               "week (7-day blocks)": lambda a, b: grouped(block, a, b),
               "week (every 7-day window)": lambda a, b: rolling(7, a, b),
               "month (calendar)": lambda a, b: grouped(month, a, b),
               "month (every 30-day window)": lambda a, b: rolling(30, a, b)}
        rel = (D.bill - D.E) / D.E
        for pname, fn in per.items():
            b_err, n = fn("bill", "E")
            t_err, _ = fn("Tf", "T")
            rows.append(dict(variant=vname, period=pname, bill_wape=b_err, traffic_wape=t_err, periods=n,
                             daily_bias=float(rel.mean()), daily_std=float(rel.std()),
                             daily_lag1_autocorr=float(rel.autocorr(1))))
    return rows


def main():
    out = []
    for name, f, fc in (("Switch LAN", switch_hours, SWITCH_FORECAST),
                        ("Switch LAN", lambda: switch_hours(SWITCH_TIMESTRAL), SWITCH_TIMESTRAL),
                        ("I2 Network", i2_hours, I2_FORECAST)):
        df = f()
        hourly_t = float((df.Tf - df["T"]).abs().sum() / df["T"].sum())
        print(f"== {name}: {len(df)} hours, {df.index[0].date()} -> {df.index[-1].date()}, traffic forecast "
              f"'{fc}', hourly traffic WAPE {100 * hourly_t:.1f} %", flush=True)
        R = pd.DataFrame(period_errors(df))
        R.insert(0, "network", name)
        pd.set_option("display.width", 220)
        show = R.assign(bill_wape=(100 * R.bill_wape).round(2), traffic_wape=(100 * R.traffic_wape).round(2),
                        daily_bias=(100 * R.daily_bias).round(2), daily_std=(100 * R.daily_std).round(2),
                        daily_lag1_autocorr=R.daily_lag1_autocorr.round(2))
        print(show.drop(columns="network").to_string(index=False))
        out.append(R.assign(hourly_traffic_wape=hourly_t, forecast=fc))
    pd.concat(out).to_csv(OUT / "bill_horizons.csv", index=False)


if __name__ == "__main__":
    main()
