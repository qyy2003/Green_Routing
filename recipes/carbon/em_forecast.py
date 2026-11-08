"""Electricity Maps' own day-ahead forecast of the grid factor, against our forecasts.

The API gives only the forecast made now (next ~24 h), so it is logged going forward and scored
once actual values exist:
  python recipes/carbon/em_forecast.py log      append the current forecast (CH, FR); run hourly
                                                (cron), the last fetch before 00:00 UTC is the
                                                day-ahead forecast of the next day
  python recipes/carbon/em_forecast.py score    refresh the actual history of CH and FR for the
                                                logged days and compare, on the same hours: EM's
                                                forecast, "same hour yesterday", Timestral median
                                                (24 weeks, zero-shot) and the same with the 7-day
                                                bias correction; per zone and for Switch LAN's
                                                backbone emissions (power-weighted CH + FR)
Log and history stay in $GR_ARTIFACTS_DIR/carbon/grid (licensed data: not in git, not published).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.carbon import grid
from green_routing.paths import ARTIFACTS_DIR

GRID = ARTIFACTS_DIR / "carbon" / "grid"
LOG = GRID / "forecast_log"
ZONES = ("CH", "FR")


def log():
    n = grid.log_forecast(LOG, ZONES)
    print(f"[em_forecast] {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC: {n} rows -> {LOG}", flush=True)


def score():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flat_forecast"))
    from green_routing.carbon import dayahead as DA
    import grid_forecast as G
    em = {z: grid.em_day_ahead(LOG, z) for z in ZONES}
    days = sorted(set(em["CH"].dropna().index.floor("D")) & set(em["FR"].dropna().index.floor("D")))
    days = [d for d in days if d + pd.Timedelta(days=1) <= pd.Timestamp.now(tz="UTC").floor("D")]
    if not days:
        print("no complete logged day yet")
        return
    first = days[0]
    grid.download(GRID / "raw", zones=ZONES, start=(first - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
    grid.process(GRID / "raw", GRID, zones=ZONES)
    D = pd.DatetimeIndex(days)
    hours = pd.DatetimeIndex(np.concatenate([pd.date_range(d, periods=24, freq="h") for d in D]))
    rows = []
    series = {z: grid.load(GRID, z).asfreq("h").interpolate() for z in ZONES}
    for name in ZONES + ("Switch LAN emissions",):
        if name == "Switch LAN emissions":
            w = _zone_power()
            a = sum(w[z] * series[z] for z in ZONES)
            fem = sum(w[z] * em[z] for z in ZONES)
        else:
            a, fem = series[name], em[name]
        pre = pd.DatetimeIndex([first - pd.Timedelta(days=k) for k in range(7, 0, -1)])
        ts = DA.foundation(a.dropna(), pre.append(D), model="timestral", context=4032)
        F = {"Electricity Maps": fem, "same hour yesterday": a.shift(24), "Timestral median": ts,
             "Timestral median + 7-day correction": G.bias_corrected(ts, a, 7)}
        act = a.reindex(hours)
        for k, f in F.items():
            f = f.reindex(hours)
            ok = f.notna() & act.notna()
            dd = hours[ok].floor("D")
            rows.append(dict(series=name, forecast=k, days=len(set(dd)),
                             bias=100 * (f[ok].sum() / act[ok].sum() - 1),
                             hourly=100 * (f[ok] - act[ok]).abs().sum() / act[ok].sum(),
                             daily=100 * (f[ok].groupby(dd).sum() - act[ok].groupby(dd).sum()).abs().sum() / act[ok].sum()))
    R = pd.DataFrame(rows)
    R.to_csv(LOG / "scores.csv", index=False)
    print(f"{len(days)} logged days, {days[0].date()} -> {days[-1].date()} (fewer than ~28: indicative only)")
    print(R.round(2).to_string(index=False))


def _zone_power() -> dict:
    """Switch LAN's static backbone power per grid zone (kW), as in flat_forecast/grid_forecast.py."""
    from green_routing.carbon import bills
    from green_routing.paths import reference_dir
    C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
    Hp = pd.date_range(pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2026-07-10 23:00", tz="UTC"), freq="h")
    _, Pbb, _ = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"),
                                   pd.read_csv(C / "allocation" / "device_pools.csv"),
                                   pd.read_csv(C / "inventory" / "shelves.csv"),
                                   pd.read_csv(reference_dir() / "sites.csv"),
                                   pd.read_csv(C / "allocation" / "doors.csv"),
                                   pd.read_csv(NET / "routers.csv").set_index("dev").tier, Hp)
    return {z: float(np.nanmean(np.asarray(Pbb[z], float))) / 1000 for z in Pbb}


if __name__ == "__main__":
    {"log": log, "score": score}[sys.argv[1]]()
