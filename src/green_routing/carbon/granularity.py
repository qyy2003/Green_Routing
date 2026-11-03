"""How fine to publish a carbon rate, how fine to forecast it, and whether to publish a
curve or a distribution instead of single values.

Everything works on 5-minute arrays over whole days starting at 00:00 UTC, so every grain
in ``GRAINS`` (a whole number of 5-minute steps per block, dividing a day) is a block mean
of the 5-minute series. The flat rate of a block is its emissions over its bytes:

    r_block = sum B_t / (sum T_t * GB5)        (B in g per 5 min, T in Mbit/s one-way)

A rate published at grain G is applied to real 5-minute traffic: party k pays
sum_t r_G(t) * V_kt / 2 * GB5 (V = in + out, half a byte at each end). Against the
5-minute rate this costs (i) representation error, the within-block variation it drops,
and (ii) forecast error, when the rate is published ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

GB5 = 300 / 8 / 1000                       # GB per (Mbit/s x 5 minutes)
GRAINS = {"5min": 1, "15min": 3, "1h": 12, "3h": 36, "6h": 72, "1d": 288}


def blocks(x: np.ndarray, k: int, how: str = "sum") -> np.ndarray:
    """[n, ...] -> [n/k, ...] block sums (or means) of consecutive k steps."""
    y = x.reshape(x.shape[0] // k, k, *x.shape[1:])
    return np.nansum(y, axis=1) if how == "sum" else np.nanmean(y, axis=1)


def expand(xb: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(xb, k, axis=0)


def block_rate(B5: np.ndarray, T5: np.ndarray, k: int) -> np.ndarray:
    """Rate of every block expanded back to 5 minutes (g/GB)."""
    return expand(blocks(B5, k) / (blocks(T5, k) * GB5), k)


def bills(rate5: np.ndarray, V5: np.ndarray) -> np.ndarray:
    """Party bills (g) for a 5-minute rate path and party volumes V5 [n, parties] (in + out)."""
    return (rate5[:, None] * V5 / 2 * GB5).sum(axis=0)


def wape(pred: np.ndarray, actual: np.ndarray, k: int = 1) -> float:
    """WAPE of the block means of k steps (traffic accuracy at grain k)."""
    pb, yb = blocks(pred, k, "mean"), blocks(actual, k, "mean")
    ok = np.isfinite(pb) & np.isfinite(yb)
    return float(np.abs(pb[ok] - yb[ok]).sum() / yb[ok].sum())


def within_block(x: np.ndarray, k: int = 12) -> np.ndarray:
    """The part of a 5-minute series that a block mean drops (x minus its block mean)."""
    return x - expand(blocks(x, k, "mean"), k)


def within_block_skill(pred: np.ndarray, actual: np.ndarray, k: int = 12) -> float:
    """How much of the actual within-block variation the forecast's own within-block shape
    explains: 1 - SSE / SS (1 = all, 0 = none, < 0 = the shape adds error)."""
    dp, dy = within_block(pred, k), within_block(actual, k)
    ok = np.isfinite(dp) & np.isfinite(dy)
    return float(1 - ((dy - dp)[ok] ** 2).sum() / (dy[ok] ** 2).sum())


def emission_error(rate_pub: np.ndarray, rate_true: np.ndarray, T5: np.ndarray) -> float:
    """Grams misbilled if all traffic paid the published rate, over the grams it should pay."""
    w = T5 * GB5
    return float(np.nansum(np.abs(rate_pub - rate_true) * w) / np.nansum(rate_true * w))


def bill_error(rate_pub: np.ndarray, rate_true: np.ndarray, V5: np.ndarray) -> tuple[float, float]:
    """Party-level error with real traffic: sum |bill_pub - bill_true| / sum bill_true, and
    the total bias (sum bill_pub / sum bill_true - 1)."""
    bp, bt = bills(rate_pub, V5), bills(rate_true, V5)
    return float(np.abs(bp - bt).sum() / bt.sum()), float(bp.sum() / bt.sum() - 1)


def daily_bill_error(rate_pub: np.ndarray, rate_true: np.ndarray, V5: np.ndarray) -> float:
    """Like ``bill_error`` but on daily statements: sum over days and parties of
    |bill_pub - bill_true| over the sum of true bills."""
    d = len(rate_true) // 288
    bp = (rate_pub[:, None] * V5 / 2 * GB5).reshape(d, 288, -1).sum(axis=1)
    bt = (rate_true[:, None] * V5 / 2 * GB5).reshape(d, 288, -1).sum(axis=1)
    return float(np.abs(bp - bt).sum() / bt.sum())


# --- direct forecasts at a grain -------------------------------------------------------

def _calendar(t: pd.DatetimeIndex, holidays, tz) -> np.ndarray:
    lt = t.tz_convert(tz)
    hol = np.array([d in holidays for d in lt.strftime("%Y-%m-%d")], dtype=float)
    return np.c_[lt.hour + lt.minute / 60, lt.dayofweek, hol, ((lt.dayofweek < 5) & (hol == 0)).astype(float)]


def gbdt_direct(T5: pd.Series, k: int, train_end: str, holidays=(), tz="UTC", seed: int = 0) -> pd.Series:
    """Day-ahead forecast of a 5-minute series made directly at grain k: block means of the
    last 7 days (normalised by their mean) and the target block's calendar -> the next day's
    blocks, one GBDT row per block; issued at 00:00 UTC for every day after ``train_end``.
    Returns block means expanded to 5 minutes."""
    spd = 288 // k
    s = T5.interpolate(limit_area="inside").bfill().ffill()
    s = s.iloc[: len(s) // 288 * 288]                       # whole days from 00:00 UTC
    v = blocks(s.to_numpy(float), k, "mean")
    t = s.index[::k]
    X, Y, base, tt = [], [], [], []
    for d0 in range(7 * spd, len(v) - spd + 1, spd):
        hist = v[d0 - 7 * spd:d0]
        b = np.log(hist.mean())
        X.append(np.c_[np.tile(np.log(hist) - b, (spd, 1)), np.arange(spd), _calendar(t[d0:d0 + spd], holidays, tz)])
        Y.append(np.log(v[d0:d0 + spd]) - b)
        base.append(np.full(spd, b))
        tt.append(t[d0:d0 + spd])
    X, Y, base, tt = np.vstack(X), np.concatenate(Y), np.concatenate(base), pd.DatetimeIndex(np.concatenate(tt))
    real = T5.reindex(tt).notna().to_numpy()
    tr = (tt < pd.Timestamp(train_end, tz="UTC")) & real
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=max(5, spd // 4),
                                      l2_regularization=1.0, random_state=seed).fit(X[tr], Y[tr])
    te = tt >= pd.Timestamp(train_end, tz="UTC")
    pred = pd.Series(np.exp(m.predict(X[te]) + base[te]), index=tt[te])
    return pd.Series(np.repeat(pred.to_numpy(), k), index=pd.date_range(pred.index[0], periods=len(pred) * k, freq="5min"))


# --- curves instead of values -------------------------------------------------------------

def fourier_fit(path: np.ndarray, K: int) -> np.ndarray:
    """Least-squares fit of each day's path (rows of 288 steps) with K daily harmonics."""
    n = path.shape[-1]
    x = np.arange(n) / n * 2 * np.pi
    basis = np.column_stack([np.ones(n)] + [f(j * x) for j in range(1, K + 1) for f in (np.cos, np.sin)])
    coef, *_ = np.linalg.lstsq(basis, path.T, rcond=None)
    return (basis @ coef).T


def knot_fit(path: np.ndarray, k: int) -> np.ndarray:
    """Piecewise-linear curve through the block means at every k steps (a curve with 288/k
    knots, continuous, instead of a staircase)."""
    n = path.shape[-1]
    centres = np.arange(k / 2 - 0.5, n, k)
    means = path.reshape(path.shape[0], n // k, k).mean(axis=2)
    return np.vstack([np.interp(np.arange(n), centres, m) for m in means])


# --- distributions ---------------------------------------------------------------------

LEVELS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def scenario_quantiles(point: np.ndarray, log_err: np.ndarray, d: int, window: int, spread: int = 1) -> np.ndarray:
    """Empirical quantiles of a day's hourly values from the forecast errors of the last
    ``window`` days: point [days, 24] forecasts, log_err [days, 24] log(actual / forecast)
    (NaN = unknown). Hours h-spread..h+spread of each past day are pooled. Returns
    [24, len(LEVELS)] quantiles for day d."""
    past = log_err[max(0, d - window):d]
    out = np.zeros((24, len(LEVELS)))
    for h in range(24):
        hs = [(h + j) % 24 for j in range(-spread, spread + 1)]
        e = past[:, hs].ravel()
        e = e[np.isfinite(e)]
        out[h] = point[d, h] * np.exp(np.quantile(e, LEVELS)) if len(e) else point[d, h]
    return out


def pinball(q: np.ndarray, y: np.ndarray) -> float:
    """Mean pinball loss over LEVELS x 2 ~ CRPS (q [..., levels], y [...])."""
    diff = y[..., None] - q
    return float(np.nanmean(np.maximum(LEVELS * diff, (LEVELS - 1) * diff)) * 2)


def coverage(q: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    i, j = int(np.argmin(abs(LEVELS - lo))), int(np.argmin(abs(LEVELS - hi)))
    ok = np.isfinite(y)
    return float(((y >= q[..., i]) & (y <= q[..., j]))[ok].mean())


# --- what a rate is good for: moving flexible traffic ---------------------------------------

def schedule(rate_fc: np.ndarray, rate_true: np.ndarray, k: int) -> dict:
    """Move one unit of flexible traffic per day into the block the forecast marks
    cheapest (spread evenly over its 5-minute steps); realised cost at the true 5-minute
    rate, against the day average and against the best 5-minute step in hindsight."""
    days = len(rate_true) // 288
    real, avg, best = [], [], []
    for d in range(days):
        f = rate_fc[d * 288:(d + 1) * 288].reshape(288 // k, k).mean(axis=1)
        tr = rate_true[d * 288:(d + 1) * 288]
        if not (np.isfinite(f).all() and np.isfinite(tr).all()):
            continue
        b = int(np.argmin(f))
        real.append(tr[b * k:(b + 1) * k].mean())
        avg.append(tr.mean())
        best.append(tr.min())
    real, avg, best = map(np.array, (real, avg, best))
    return dict(realised=float(real.mean()), day_average=float(avg.mean()), hindsight=float(best.mean()),
                captured=float((avg - real).sum() / (avg - best).sum()))
