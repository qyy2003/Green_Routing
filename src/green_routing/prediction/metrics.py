"""Forecast error metrics, computed in the **original scale** (Mbps).

The roadmap is emphatic about two things:
  * MAPE explodes on small/idle links -> prefer sMAPE / WAPE, and MASE for a
    scale-free number comparable across heterogeneous links;
  * the number that actually means something is *skill vs. persistence and
    seasonal-naive*: ``skill = 1 - model_error / baseline_error``.

All functions take flat arrays of aligned (y_true, y_pred) with NaNs already handled
by the caller unless noted; ``_mask`` drops any pair where either side is NaN.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def _mask(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]


def mae(y_true, y_pred) -> float:
    yt, yp = _mask(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else np.nan


def rmse(y_true, y_pred) -> float:
    yt, yp = _mask(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else np.nan


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in [0, 200] percent. Robust to zeros (unlike MAPE)."""
    yt, yp = _mask(y_true, y_pred)
    if not yt.size:
        return np.nan
    denom = np.abs(yt) + np.abs(yp)
    return float(200.0 * np.mean(np.abs(yt - yp) / np.where(denom < EPS, np.nan, denom)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error = sum|e| / sum|y| (percent).

    Volume-weighted, so busy links dominate — the operationally relevant view.
    """
    yt, yp = _mask(y_true, y_pred)
    if not yt.size:
        return np.nan
    s = np.sum(np.abs(yt))
    return float(100.0 * np.sum(np.abs(yt - yp)) / s) if s > EPS else np.nan


def mase(y_true, y_pred, y_train, season: int = 1) -> float:
    """Mean absolute scaled error: MAE divided by the in-sample MAE of a
    (seasonal) naive forecast on the training series. season=1 -> naive-1.

    Scale-free: MASE<1 means better than the naive benchmark on train.
    """
    yt, yp = _mask(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float).ravel()
    y_train = y_train[np.isfinite(y_train)]
    if not yt.size or y_train.size <= season:
        return np.nan
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return float(np.mean(np.abs(yt - yp)) / scale) if scale > EPS else np.nan


def skill(model_error: float, baseline_error: float) -> float:
    """1 - model/baseline. >0 means the model beats the baseline; 0 ties; <0 worse."""
    if not np.isfinite(baseline_error) or baseline_error <= EPS:
        return np.nan
    return float(1.0 - model_error / baseline_error)


# Registry so run.py / harness can iterate a stable set of point metrics.
POINT_METRICS = {
    "mae": mae,
    "rmse": rmse,
    "smape": smape,
    "wape": wape,
}


def evaluate_all(y_true, y_pred) -> dict[str, float]:
    return {name: fn(y_true, y_pred) for name, fn in POINT_METRICS.items()}
