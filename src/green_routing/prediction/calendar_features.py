"""Calendar covariates for the forecasting tiers.

Motivation (pred_547f, artifact 57b5f951): the calendar-blind deep tiers leave
weekends on the table — their weekend/weekday MAE ratio is 0.80-0.86 vs 0.645
for persistence and GBDT (which has day-of-week features). Day-of-week and
holidays are *known-future* covariates: at forecast time the calendar of the
whole target path is certain, so it can (and should) be fed to the model
rather than inferred from the window's shape.

``build_calendar(timestamps)`` returns a float32 matrix ``cal[T, K_CAL]``
aligned with the panel grid:

    0  sin(time-of-day)        2pi * step / 288
    1  cos(time-of-day)
    2..8  day-of-week one-hot  (Mon..Sun, Europe/Zurich local time)
    9  is_holiday              Swiss public holiday (see below)
    10 is_workday              Mon-Fri and not a holiday

Holidays: the ``holidays`` package is not installed in the green-pred env, so
the federal + widely-observed Swiss dates inside the panel range are
hardcoded (SWITCH is a national backbone; cantonal refinement via the
device->PoP mapping is future work). Extend HOLIDAYS when the panel range
grows — ``build_calendar`` raises if the range outgrows the table so a stale
list can never silently return all-zeros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_CAL = 11

# Swiss public holidays, federal + observed in (nearly) all cantons, for the
# current panel range. Easter 2026: Apr 5 => Good Friday Apr 3, Easter Monday
# Apr 6, Ascension May 14, Whit Monday May 25, Corpus Christi Jun 4 (catholic
# cantons; included — SWITCH sites in FR/LU/TI/VS observe it).
HOLIDAYS = {
    "2025-08-01",  # National Day 2025 (year-scale window)
    "2025-12-25",  # Christmas
    "2025-12-26",  # St. Stephen's Day
    "2026-01-01",  # New Year
    "2026-01-02",  # Berchtoldstag (most cantons)
    "2026-04-03",  # Good Friday
    "2026-04-06",  # Easter Monday
    "2026-05-01",  # Labour Day (several cantons incl. ZH, BS)
    "2026-05-14",  # Ascension
    "2026-05-25",  # Whit Monday
    "2026-06-04",  # Corpus Christi (catholic cantons)
    "2026-08-01",  # National Day
}
# Table validity range — build_calendar refuses timestamps beyond this.
_TABLE_SPAN = ("2025-06-01", "2026-08-31")


def build_calendar(timestamps: np.ndarray) -> np.ndarray:
    """cal[T, K_CAL] float32 for epoch-second ``timestamps`` (Europe/Zurich)."""
    ts = pd.to_datetime(np.asarray(timestamps), unit="s", utc=True)
    ts = ts.tz_convert("Europe/Zurich")
    lo, hi = pd.Timestamp(_TABLE_SPAN[0]), pd.Timestamp(_TABLE_SPAN[1])
    if ts.min().tz_localize(None) < lo or ts.max().tz_localize(None) > hi:
        raise ValueError(
            f"calendar table covers {_TABLE_SPAN}, panel spans "
            f"{ts.min().date()}..{ts.max().date()} — extend HOLIDAYS/_TABLE_SPAN")
    T = len(ts)
    cal = np.zeros((T, K_CAL), np.float32)
    step = (ts.hour * 12 + ts.minute // 5).to_numpy()
    cal[:, 0] = np.sin(2 * np.pi * step / 288.0)
    cal[:, 1] = np.cos(2 * np.pi * step / 288.0)
    dow = ts.dayofweek.to_numpy()
    cal[np.arange(T), 2 + dow] = 1.0
    dates = ts.strftime("%Y-%m-%d").to_numpy()
    hol = np.isin(dates, sorted(HOLIDAYS))
    cal[:, 9] = hol
    cal[:, 10] = (dow < 5) & ~hol
    return cal
