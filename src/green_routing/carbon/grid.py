"""Hourly grid carbon intensity: Electricity Maps history, and day-ahead forecasts of it.

Source: the Electricity Maps API (academic licence, key ``EM_API``): flow-traced
consumption intensity, ``direct`` (combustion only, the GHG Protocol Scope 2 basis)
and ``lifecycle``, hourly, per zone. Switzerland is CH; CERN draws from FR. The licence
forbids redistribution: keep the downloaded series out of the repository and out of
published pages (aggregates are fine).

Day-ahead forecast of the factor, issued at 00:00 UTC for the 24 hours of the day,
using only data up to 23:00 the day before. ``RULES`` are simple and reproducible;
the best of them on spring/summer 2026 was "same hour yesterday" (MAE 6.5 g/kWh on a
mean of 13). Electricity Maps' own forecast cannot be evaluated on the past (the API
returns only the current one): ``log_forecast`` keeps a log, ``em_day_ahead`` reads it back
(recipes/carbon/em_forecast.py).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.electricitymaps.com/v3"
ZONES = ("CH", "FR", "DE", "IT-NO", "AT")       # CH, CERN's FR, and the neighbours
CHUNK_DAYS = 10                                  # the API's limit per hourly request


def api_key() -> str:
    """``EM_API`` from the environment, else from ``export EM_API=...`` in ~/.bashrc.
    Never print or log it."""
    key = os.environ.get("EM_API")
    if not key:
        rc = Path("~/.bashrc").expanduser()
        lines = [x for x in rc.read_text().splitlines() if re.match(r"\s*(export\s+)?EM_API=", x)] if rc.exists() else []
        key = lines[-1].split("=", 1)[1].strip().strip("'\"") if lines else None
    if not key:
        raise RuntimeError("Electricity Maps key not found: set EM_API (environment or ~/.bashrc)")
    return key


def _get(url: str, key: str):
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"auth-token": key})
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"Electricity Maps {e.code}: {e.read()[:200]!r}") from None
        except OSError:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Electricity Maps: gave up on {url.split('?')[0]}")


def download(raw: str | Path, zones=ZONES, start="2023-01-01", end=None) -> int:
    """Fetch hourly history in 10-day chunks into ``raw`` (one JSON per chunk): direct and
    lifecycle intensity for every zone, plus the CH power breakdown. A chunk on disk is
    skipped when it was written at least a day after its window closed (EM revises
    recent hours). Returns the number of requests made."""
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    key = api_key()
    t_end = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").floor("D").tz_localize(None)
    jobs = [(z, "carbon-intensity", t) for z in zones for t in ("direct", "lifecycle")]
    if "CH" in zones:
        jobs.append(("CH", "power-breakdown", None))
    n = 0
    for zone, ep, eft in jobs:
        t = pd.Timestamp(start).to_pydatetime()
        while t < t_end:
            u = min(t + dt.timedelta(days=CHUNK_DAYS), t_end.to_pydatetime())
            f = raw / f"{zone}_{ep}_{eft or 'all'}_{t:%Y%m%d}.json"
            done = f.exists() and dt.datetime.fromtimestamp(f.stat().st_mtime) > u + dt.timedelta(days=1)
            if not done:
                url = (f"{API}/{ep}/past-range?zone={zone}"
                       f"&start={t:%Y-%m-%dT%H:%MZ}&end={u:%Y-%m-%dT%H:%MZ}")
                if eft:
                    url += f"&emissionFactorType={eft}"
                f.write_text(json.dumps(_get(url, key)))
                n += 1
                time.sleep(0.3)
            t = u
        print(f"[grid] {zone} {ep} {eft or ''}: up to date", flush=True)
    return n


def _load_raw(raw: Path, zone: str, ep: str, eft: str) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.glob(f"{zone}_{ep}_{eft}_*.json")):
        rows += json.load(open(f)).get("data", [])
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["datetime"] = pd.to_datetime(d.datetime, utc=True)
    return d.drop_duplicates("datetime").set_index("datetime").sort_index()


def process(raw: str | Path, out: str | Path, zones=ZONES) -> dict[str, pd.DataFrame]:
    """Raw chunks -> ``em_<zone>_hourly.csv`` (ci_direct, estimated, ci_lifecycle g/kWh)
    and ``em_CH_consumption_mix_hourly.csv`` (MW by source)."""
    raw, out = Path(raw), Path(out)
    ci = {}
    for z in zones:
        parts = []
        for t in ("direct", "lifecycle"):
            d = _load_raw(raw, z, "carbon-intensity", t)
            if d.empty:
                continue
            parts.append(d.carbonIntensity.rename(f"ci_{t}"))
            if t == "direct":
                parts.append(d.isEstimated.rename("estimated"))
        if parts:
            ci[z] = pd.concat(parts, axis=1).sort_index()
            ci[z].to_csv(out / f"em_{z}_hourly.csv")
    pb = _load_raw(raw, "CH", "power-breakdown", "all")
    if not pb.empty:
        mix = pd.DataFrame(list(pb.powerConsumptionBreakdown), index=pb.index).fillna(0)
        mix.to_csv(out / "em_CH_consumption_mix_hourly.csv")
    return ci


def load(grid_dir: str | Path, zone: str, basis: str = "direct") -> pd.Series:
    """Hourly intensity (g CO2e/kWh) of a zone, UTC-indexed."""
    s = pd.read_csv(Path(grid_dir) / f"em_{zone}_hourly.csv", index_col=0, parse_dates=True)[f"ci_{basis}"]
    return s


def annual_means(grid_dir: str | Path, year: int, zones=("CH", "FR")) -> dict[str, dict[str, float]]:
    """Time-mean direct and lifecycle intensity of each zone over a calendar year."""
    return {z: {b: float(load(grid_dir, z, b).loc[str(year)].mean()) for b in ("direct", "lifecycle")}
            for z in zones}


# --- Electricity Maps' own forecast (only the next ~24 h can be fetched: log it) ----

def log_forecast(out_dir: str | Path, zones=("CH", "FR"), basis: str = "direct") -> int:
    """Append Electricity Maps' current forecast of every zone to
    ``<out_dir>/em_forecast_<zone>.csv`` (fetched_at, updated_at, datetime, ci). The API
    returns only the forecast made now (a past ``datetime`` is ignored), so a backtest needs
    a log kept going forward. One request per zone; returns the number of rows written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = api_key()
    now = pd.Timestamp.now(tz="UTC").floor("s")
    n = 0
    for z in zones:
        b = _get(f"{API}/carbon-intensity/forecast?zone={z}&emissionFactorType={basis}", key)
        d = pd.DataFrame(b.get("forecast", []))
        if d.empty:
            continue
        d = pd.DataFrame({"fetched_at": now, "updated_at": pd.Timestamp(b.get("updatedAt")),
                          "datetime": pd.to_datetime(d.datetime, utc=True), f"ci_{basis}": d.carbonIntensity})
        f = out / f"em_forecast_{z}.csv"
        d.to_csv(f, mode="a", header=not f.exists(), index=False)
        n += len(d)
    return n


def em_day_ahead(out_dir: str | Path, zone: str, basis: str = "direct") -> pd.Series:
    """Day-ahead series from the log: for every hour of day d, the value of the latest
    forecast fetched at or before 00:00 UTC of d that covers it (NaN where none does)."""
    f = Path(out_dir) / f"em_forecast_{zone}.csv"
    d = pd.read_csv(f, parse_dates=["fetched_at", "updated_at", "datetime"])
    d["issue_day"] = d.datetime.dt.floor("D")
    d = d[d.fetched_at <= d.issue_day]
    d = d.sort_values("fetched_at").groupby("datetime").last()
    return d[f"ci_{basis}"].sort_index()


# --- Day-ahead forecasts --------------------------------------------------------
# Each rule maps the hourly history s (gap-free, hourly) to the forecast for every hour,
# made at 00:00 UTC of its day from data up to 23:00 of the day before.

def _same_hour_mean(s: pd.Series, k: int) -> pd.Series:
    return sum(s.shift(24 * i) for i in range(1, k + 1)) / k


RULES = {
    "yesterday same hour": lambda s: s.shift(24),
    "mean same hour last 3 d": lambda s: _same_hour_mean(s, 3),
    "mean same hour last 7 d": lambda s: _same_hour_mean(s, 7),
    "last observed hour, flat": lambda s: s.groupby(s.index.normalize()).transform("last").shift(24),
    "7-day profile rescaled to yesterday mean": lambda s: _same_hour_mean(s, 7) * (
        s.shift(24).rolling(24).mean() / _same_hour_mean(s, 7).rolling(24).mean()),
}
DEFAULT_RULE = "yesterday same hour"


def day_ahead(s: pd.Series, rule: str = DEFAULT_RULE) -> pd.Series:
    """Day-ahead forecast of an hourly intensity series by one of ``RULES``."""
    return RULES[rule](s.asfreq("h").interpolate())


def evaluate_rules(s: pd.Series, start, end) -> pd.DataFrame:
    """MAE, bias and daily-mean MAE (g/kWh) of every rule over [start, end)."""
    s = s.asfreq("h").interpolate()
    win = (s.index >= pd.Timestamp(start, tz="UTC")) & (s.index < pd.Timestamp(end, tz="UTC"))
    y = s[win]
    rows = []
    for name, rule in RULES.items():
        f = rule(s)[win]
        e = f - y
        rows.append(dict(rule=name, MAE=e.abs().mean(), bias=e.mean(),
                         daily_MAE=(f.resample("D").mean() - y.resample("D").mean()).abs().mean(),
                         mean_actual=y.mean()))
    return pd.DataFrame(rows).sort_values("MAE")

