"""Tier 2 — classical statistical models (roadmap §1, Tier 2).

Linear time-series models with explicit seasonality, fit **one series at a time** and
refit on history ≤ origin at each rolling origin. Per the roadmap these are a per-link
*sanity check*, not a production model: `s = 288` makes them heavy and often unstable,
they don't scale to thousands of links, and everything is linear.

Two practical concessions so the backtest stays tractable:
  * we fit in **log space** (`log1p`) — traffic is log-normal — and invert before scoring;
  * we refit on a bounded *recent window* of history, not the full multi-month span
    (statsmodels seasonal fits over tens of thousands of points at every origin×link are
    prohibitive, and recent history dominates the fit anyway).

Any fit failure falls back to a seasonal-naive path so the harness never crashes.
"""
from __future__ import annotations

import warnings

import numpy as np

from dataset import config

from .harness import Context, Forecaster

warnings.filterwarnings("ignore")   # statsmodels convergence chatter


def _log(x):
    return np.log1p(np.clip(x, 0, None))


def _inv(x):
    return np.expm1(x)


def _seasonal_naive_path(hist: np.ndarray, origin: int, season: int, H: int) -> np.ndarray:
    finite = hist[np.isfinite(hist)]
    fb = float(finite[-1]) if finite.size else 0.0
    out = np.empty(H)
    for k in range(H):
        src = origin - season + k
        v = hist[src] if 0 <= src < hist.size else np.nan
        out[k] = v if np.isfinite(v) else fb
    return out


class HoltWinters(Forecaster):
    """Triple exponential smoothing: level + trend + (daily) seasonality.

    Lightweight and a good diurnal baseline. Daily seasonality (288) by default.
    """

    name = "holt_winters"
    tier = 2
    is_global = False

    def __init__(self, seasonal_periods: int = config.DAILY_STEPS,
                 window: int = 4 * config.WEEKLY_STEPS, trend: str | None = "add"):
        self.seasonal_periods = seasonal_periods
        self.window = window
        self.trend = trend

    def predict(self, ctx: Context) -> np.ndarray:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        hist = ctx.history()
        H = ctx.horizon
        y = hist[-self.window:]
        y = np.where(np.isfinite(y), y, np.nan)
        # need >= 2 full seasons of mostly-present data
        if y.size < 2 * self.seasonal_periods or np.isnan(y).mean() > 0.2:
            return _seasonal_naive_path(hist, ctx.origin, self.seasonal_periods, H)
        y = _fill_interp(y)
        try:
            model = ExponentialSmoothing(
                _log(y), trend=self.trend, seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit()
            fc = _inv(np.asarray(fit.forecast(H)))
            return np.clip(fc, 0, None)
        except Exception:
            return _seasonal_naive_path(hist, ctx.origin, self.seasonal_periods, H)


class SARIMA(Forecaster):
    """(S)ARIMA(p,d,q)(P,D,Q)ₛ on the recent window, in log space.

    Default order is a light ARMA with no seasonal term (seasonal `s=288` fits are
    numerically heavy and unstable, exactly as the roadmap warns). Pass
    ``seasonal_order=(1,0,0,288)`` to exercise a true seasonal ARIMA on a small cohort.
    """

    name = "sarima"
    tier = 2
    is_global = False

    def __init__(self, order=(2, 0, 2), seasonal_order=(0, 0, 0, 0),
                 window: int = 2 * config.WEEKLY_STEPS):
        self.order = order
        self.seasonal_order = seasonal_order
        self.window = window
        s = seasonal_order[-1]
        if s:
            self.name = f"sarima_s{s}"

    def predict(self, ctx: Context) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        hist = ctx.history()
        H = ctx.horizon
        y = hist[-self.window:]
        if y.size < max(50, 3 * (self.seasonal_order[-1] or 1)) or \
                np.isnan(y).mean() > 0.2:
            return _seasonal_naive_path(hist, ctx.origin, config.DAILY_STEPS, H)
        y = _fill_interp(y)
        try:
            model = SARIMAX(
                _log(y), order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=50)
            fc = _inv(np.asarray(fit.forecast(H)))
            return np.clip(fc, 0, None)
        except Exception:
            return _seasonal_naive_path(hist, ctx.origin, config.DAILY_STEPS, H)


def _fill_interp(y: np.ndarray) -> np.ndarray:
    """Linear-interpolate interior NaNs; edge-fill ends. Keeps statsmodels happy."""
    y = y.astype(float).copy()
    n = y.size
    idx = np.arange(n)
    good = np.isfinite(y)
    if good.all():
        return y
    if not good.any():
        return np.zeros(n)
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def default_classical(seasonal_sarima: bool = False) -> list[Forecaster]:
    models: list[Forecaster] = [HoltWinters()]
    if seasonal_sarima:
        models.append(SARIMA(order=(2, 0, 1), seasonal_order=(1, 0, 0, config.DAILY_STEPS)))
    else:
        models.append(SARIMA())
    return models
