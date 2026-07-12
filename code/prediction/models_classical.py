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


class _quiet_fit(warnings.catch_warnings):
    """Silence statsmodels' per-fit chatter (ConvergenceWarning, RuntimeWarning from
    the optimizer, etc.) *locally*. A module-level ``filterwarnings`` is unreliable
    here: importing the sklearn/torch tiers afterwards resets the warnings registry
    and clobbers it, so we scope the suppression to each fit call instead."""

    def __enter__(self):
        super().__enter__()
        warnings.simplefilter("ignore")
        return self


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
            with _quiet_fit():
                model = ExponentialSmoothing(
                    _log(y), trend=self.trend, seasonal="add",
                    seasonal_periods=self.seasonal_periods,
                    initialization_method="estimated",
                )
                fit = model.fit()
                fc = _inv(np.asarray(fit.forecast(H)))
            sane = _sane_or_none(fc, y)
            if sane is not None:
                return sane
        except Exception:
            pass
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
            with _quiet_fit():
                model = SARIMAX(
                    _log(y), order=self.order, seasonal_order=self.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=50)
                fc = _inv(np.asarray(fit.forecast(H)))
            sane = _sane_or_none(fc, y)
            if sane is not None:
                return sane
        except Exception:
            pass
        return _seasonal_naive_path(hist, ctx.origin, config.DAILY_STEPS, H)


def _sane_or_none(fc: np.ndarray, y_raw: np.ndarray) -> np.ndarray | None:
    """Reject a diverged fit. Log-space (S)ARIMA/HW forecasts can blow up so that
    ``expm1`` returns astronomical Mbps (a single such link posts a ~1e214 MAE and
    wrecks every aggregate). Accept the path only if it is finite and stays within a
    generous multiple of the recent observed peak; otherwise signal a fallback."""
    finite = y_raw[np.isfinite(y_raw)]
    peak = float(finite.max()) if finite.size else 0.0
    cap = 50.0 * peak + 1e3                        # very loose: only catches blow-ups
    if not np.all(np.isfinite(fc)) or np.any(fc > cap):
        return None
    return np.clip(fc, 0, None)


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
