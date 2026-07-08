"""Tier 1 — the mandatory baselines (roadmap §1).

traffic = strong short-term autocorrelation + strong daily/weekly seasonality.
These two facts alone forecast most of the signal with *zero learning*, so every
fancier model must be judged as skill relative to them. Never skip this tier.

All four return a length-``ctx.horizon`` path from history strictly before the origin.
"""
from __future__ import annotations

import numpy as np

from dataset import config

from .harness import Context, Forecaster


class Persistence(Forecaster):
    """ŷ(t+k) = y(t): repeat the last observed value. The brutal 1-step baseline."""

    name = "persistence"
    tier = 1

    def predict(self, ctx: Context) -> np.ndarray:
        hist = ctx.history()
        last = _last_finite(hist)
        return np.full(ctx.horizon, last, dtype=float)


class SeasonalNaive(Forecaster):
    """ŷ(t+k) = y(t+k − season): same time one day (288) or one week (2016) ago."""

    def __init__(self, season: int = config.DAILY_STEPS, name: str | None = None):
        self.season = season
        self.name = name or (
            "seasonal_naive_day" if season == config.DAILY_STEPS
            else "seasonal_naive_week" if season == config.WEEKLY_STEPS
            else f"seasonal_naive_{season}"
        )
        self.tier = 1

    def predict(self, ctx: Context) -> np.ndarray:
        hist = ctx.history()
        o, s, H = ctx.origin, self.season, ctx.horizon
        out = np.empty(H, dtype=float)
        fallback = _last_finite(hist)
        for k in range(H):
            src = o - s + k          # index of "same time one period ago"
            v = hist[src] if 0 <= src < hist.size else np.nan
            out[k] = v if np.isfinite(v) else fallback
        return out


class HistoricalAverage(Forecaster):
    """ŷ(t+k) = mean of all past y sharing the same (time-of-day, day-of-week) bin.

    A smooth climatological profile — robust, captures the diurnal+weekly shape without
    the single-sample noise of seasonal-naive.
    """

    name = "historical_average"
    tier = 1

    def predict(self, ctx: Context) -> np.ndarray:
        hist = ctx.history()
        if hist.size == 0:
            return np.zeros(ctx.horizon)
        idx = ctx.hist_index()
        # bin key = day-of-week * 288 + step-of-day  (weekly climatology)
        step_of_day = (idx.hour * 3600 + idx.minute * 60 + idx.second) // config.STEP_SECONDS
        key_hist = idx.dayofweek.to_numpy() * config.DAILY_STEPS + step_of_day.to_numpy()
        finite = np.isfinite(hist)
        # mean per bin
        sums = np.zeros(config.WEEKLY_STEPS)
        cnts = np.zeros(config.WEEKLY_STEPS)
        np.add.at(sums, key_hist[finite], hist[finite])
        np.add.at(cnts, key_hist[finite], 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            profile = sums / cnts
        global_mean = float(np.nanmean(hist[finite])) if finite.any() else 0.0

        tgt = ctx.target_index()
        tstep = (tgt.hour * 3600 + tgt.minute * 60 + tgt.second) // config.STEP_SECONDS
        key_tgt = tgt.dayofweek.to_numpy() * config.DAILY_STEPS + tstep.to_numpy()
        out = profile[key_tgt]
        return np.where(np.isfinite(out), out, global_mean)


def _last_finite(hist: np.ndarray) -> float:
    finite = hist[np.isfinite(hist)]
    return float(finite[-1]) if finite.size else 0.0


def default_baselines() -> list[Forecaster]:
    return [
        Persistence(),
        SeasonalNaive(config.DAILY_STEPS),
        SeasonalNaive(config.WEEKLY_STEPS),
        HistoricalAverage(),
    ]
