"""Flat carbon rate: one rate per hour for every byte, rate(h) = B(h) / T(h).

B(h): backbone emissions (static power x hourly grid factor, the flat pool of the bills);
T(h): bytes carried, each once (half the in + out of every door). Its day-ahead forecast
B-hat / T-hat needs a forecast of the total traffic, here from ONE model trained on the
total series, for every tier of the forecasting ladder (recipes/carbon_rates/run.sh train).

usage: python flat_rate.py
writes $GR_ARTIFACTS_DIR/carbon_rates/flat/: traffic_by_horizon.csv (every model, every
horizon, sampled + dense), rate_errors.csv (day-ahead rate), hourly.csv, summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from green_routing.carbon import assumptions as A, bills, grid
from green_routing.paths import ARTIFACTS_DIR, reference_dir

C, NET = ARTIFACTS_DIR / "carbon", ARTIFACTS_DIR / "network"
ROOT = ARTIFACTS_DIR / "carbon_rates"
RUN = ROOT / "flat" / "merged"
DENSE = RUN / "dense"
OUT = ROOT / "flat"
GB = A.GB_PER_MBPS_HOUR
H = bills.window(ARTIFACTS_DIR / "pred_year" / "merged")
LAB = {"persistence": "Persistence", "seasonal_naive_day": "Same time yesterday",
       "seasonal_naive_week": "Same time last week", "historical_average": "Historical average",
       "holt_winters": "Holt-Winters", "sarima": "SARIMA", "gbdt": "GBDT", "gru": "GRU", "tcn": "TCN",
       "dlinear": "DLinear", "patchtst": "PatchTST"}

# --- actual total, one-way Gbit/s, hourly
m = pd.read_csv(ROOT / "mirror" / "router-interfaces" / "total" / "all.csv")
tot = pd.Series((m.ifHCInOctets + m.ifHCOutOctets).to_numpy() * 8 / 1e6 / 2,
                index=pd.to_datetime(m.timestamp, unit="s", utc=True))       # Mbit/s one-way, 5 min
g = tot.resample("1h")
T = g.mean().where(g.count() >= 6).reindex(H)

# --- every model's forecast of the total (block = issued 00:00 UTC for the day; rolling = 24 h lead)
meta = json.load(open(DENSE / "meta.json"))
hi = {h["name"]: i for i, h in enumerate(meta["horizons"])}["day"]
t0 = pd.Timestamp(meta["t0"], unit="s", tz="UTC")
n = meta["n_targets"]
F = {}
for mode in ("block", "rolling"):
    for f in sorted(DENSE.glob(f"pred_{mode}_*.npy")):
        name = f.stem.split(f"pred_{mode}_")[1]
        if "@" in name or "+" in name:
            continue
        p = np.load(f)[0, hi, : (n // 12) * 12].astype(float) / 2                # one-way
        F[(mode, name)] = pd.Series(p.reshape(-1, 12).mean(1), index=pd.date_range(t0, periods=n // 12, freq="h")).reindex(H)
# for reference: the sum of the per-link models' door forecasts and the hourly GBDT (gr-carbon total)
TH = pd.read_csv(C / "total" / "total_hourly.csv", index_col=0, parse_dates=True)
for k in ("sum of tcn door forecasts (bottom-up)", "sum of patchtst door forecasts (bottom-up)", "GBDT on the total (top-down)"):
    F[("reference", k)] = TH[k].reindex(H) * 1e3

# --- backbone emissions B(h): actual grid and day-ahead grid forecast
doors = pd.read_csv(C / "allocation" / "doors.csv")
tiers = pd.read_csv(NET / "routers.csv").set_index("dev").tier
_, Pbb, _ = bills.window_power(pd.read_csv(C / "inventory" / "routers.csv"), pd.read_csv(C / "allocation" / "device_pools.csv"),
                               pd.read_csv(C / "inventory" / "shelves.csv"), pd.read_csv(reference_dir() / "sites.csv"),
                               doors, tiers, H)
IA = pd.DataFrame({z: grid.load(C / "grid", z).reindex(H).interpolate() for z in Pbb})
IF = pd.DataFrame({z: grid.day_ahead(grid.load(C / "grid", z)).reindex(H).interpolate() for z in Pbb})
B = lambda I_: sum(Pbb[z] / 1000 * I_[z] for z in Pbb)            # noqa: E731  g per hour
rate = B(IA) / (T * GB)                                          # g per GB carried
ok = rate.notna()


def score(f, y):
    f, y = f[ok], y[ok]
    e = f - y
    d = (f.resample("D").mean() - y.resample("D").mean()).abs().sum() / y.resample("D").mean().sum()
    return dict(hourly_WAPE=float(e.abs().sum() / y.sum()), daily_WAPE=float(d), bias=float(e.sum() / y.sum()))


rows = []
for (mode, name), f in F.items():
    tr = score(f, T)
    ra = score(B(IA) / (f * GB), rate)
    rf = score(B(IF) / (f * GB), rate)
    rows.append(dict(protocol=mode, model=name, label=LAB.get(name, name), traffic_hourly_WAPE=tr["hourly_WAPE"],
                     traffic_daily_WAPE=tr["daily_WAPE"], traffic_bias=tr["bias"],
                     rate_WAPE_actual_grid=ra["hourly_WAPE"], rate_WAPE_grid_forecast=rf["hourly_WAPE"],
                     rate_daily_WAPE_grid_forecast=rf["daily_WAPE"]))
rows.append(dict(protocol="reference", model="actual traffic", label="Actual traffic",
                 rate_WAPE_grid_forecast=score(B(IF) / (T * GB), rate)["hourly_WAPE"],
                 rate_daily_WAPE_grid_forecast=score(B(IF) / (T * GB), rate)["daily_WAPE"]))
R = pd.DataFrame(rows).sort_values(["protocol", "traffic_hourly_WAPE"])
R.to_csv(OUT / "rate_errors.csv", index=False)

# --- every horizon: the harness's sampled origins (all tiers) and the dense logs
agg = pd.read_csv(RUN / "aggregate.csv")
dense = []
for mode in ("rolling", "block"):
    f = DENSE / mode / "aggregate.csv"
    if f.exists():
        dense.append(pd.read_csv(f).assign(protocol=f"dense {mode}"))
TB = pd.concat([agg.assign(protocol="sampled 160 origins")] + dense, ignore_index=True)
TB.to_csv(OUT / "traffic_by_horizon.csv", index=False)

pd.DataFrame({"T_gbps": T / 1e3, "rate_g_per_GB": rate, "B_g": B(IA), "B_fc_g": B(IF),
              **{f"{mo}:{k}": v / 1e3 for (mo, k), v in F.items()}}).to_csv(OUT / "hourly.csv")
best = R[R.protocol == "block"].iloc[0]
summary = dict(window=[str(H[0]), str(H[-1])], mean_total_gbps=float(T.mean() / 1e3),
               mean_rate_g_per_GB=float(rate.mean()), rate_hour_p10_p90=[float(rate.quantile(.1)), float(rate.quantile(.9))],
               best_block_model=best.model, best_block_traffic_WAPE=float(best.traffic_hourly_WAPE))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
pd.set_option("display.width", 220)
print(json.dumps(summary, indent=1))
print(R.round(4).to_string(index=False))
