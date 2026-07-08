"""Tier 3 — the pragmatic workhorse: global gradient boosting (roadmap §1, Tier 3).

Forecasting reframed as **tabular regression**. For each ``(link, anchor-time t)`` row we
build lag + rolling + calendar features and regress the log-traffic ``h`` steps ahead.
One **global** model is trained over *all* links' rows at once (more data, cross-link
transfer, generalises to new links) with a per-link categorical id — never one model per
link. Multi-step is handled **directly**: a separate regressor per horizon.

LightGBM/XGBoost aren't installed in this environment, so we use scikit-learn's
``HistGradientBoostingRegressor`` — a histogram-based gradient booster that is the direct
methodological equivalent (same "binned features + leaf-wise-ish boosted trees" idea) and
natively supports categorical features.
"""
from __future__ import annotations

import numpy as np

from dataset import config

from .harness import Context, Forecaster


class GlobalGBDT(Forecaster):
    name = "gbdt"
    tier = 3
    is_global = True

    def __init__(
        self,
        horizons=tuple(config.HORIZONS.values()),
        lags=(1, 2, 3, 4, 5, 6, 12, 24, config.DAILY_STEPS),
        roll_windows=(12, config.DAILY_STEPS),
        max_train_rows: int = 300_000,
        random_state: int = 0,
    ):
        self.horizons = tuple(sorted(set(horizons)))
        self.lags = tuple(lags)
        self.roll_windows = tuple(roll_windows)
        self.max_train_rows = max_train_rows
        self.random_state = random_state
        self.models: dict[int, object] = {}
        self._min_ctx = max(max(self.lags), max(self.roll_windows))

    # -- feature engineering ------------------------------------------------ #
    def _precompute(self, values_filled: np.ndarray, timestamps: np.ndarray) -> None:
        import pandas as pd

        self._log = np.log1p(np.clip(values_filled, 0, None)).astype(np.float32)
        T, L = self._log.shape
        # rolling mean/std (log) ending at each t, per column — only uses <= t
        self._roll_mean, self._roll_std = {}, {}
        for w in self.roll_windows:
            df = pd.DataFrame(self._log)
            self._roll_mean[w] = df.rolling(w, min_periods=1).mean().to_numpy(np.float32)
            self._roll_std[w] = (
                df.rolling(w, min_periods=2).std().fillna(0.0).to_numpy(np.float32)
            )
        # anchor-time calendar (cyclical)
        idx = pd.to_datetime(timestamps, unit="s", utc=True)
        tod = ((idx.hour * 3600 + idx.minute * 60 + idx.second)
               // config.STEP_SECONDS).to_numpy()
        dow = idx.dayofweek.to_numpy()
        self._cal = np.column_stack([
            np.sin(2 * np.pi * tod / config.DAILY_STEPS),
            np.cos(2 * np.pi * tod / config.DAILY_STEPS),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            (dow >= 5).astype(np.float32),
        ]).astype(np.float32)
        self._n_links = L

    def _feature_names(self) -> list[str]:
        names = [f"lag{k}" for k in self.lags]
        for w in self.roll_windows:
            names += [f"rmean{w}", f"rstd{w}"]
        names += ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "weekend", "link_id"]
        return names

    def _rows(self, anchors: np.ndarray, link: int) -> np.ndarray:
        """Feature matrix for the given anchor indices of one link (last col = link id)."""
        cols = [self._log[anchors - k, link] for k in self.lags]
        for w in self.roll_windows:
            cols.append(self._roll_mean[w][anchors, link])
            cols.append(self._roll_std[w][anchors, link])
        cal = self._cal[anchors]
        X = np.column_stack(cols + [cal, np.full(anchors.size, link, np.float32)])
        return X.astype(np.float32)

    # -- fit / predict ------------------------------------------------------ #
    def fit(self, values_filled, split, timestamps) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self._precompute(values_filled, timestamps)
        T, L = self._log.shape
        tr_start, tr_end = split.train
        h_max = max(self.horizons)
        lo = max(tr_start, self._min_ctx)
        hi = tr_end - h_max                         # target t+h must stay inside train
        if hi <= lo:
            raise ValueError("train slice too short for the requested lags/horizons")

        anchors_all = np.arange(lo, hi)
        # bound total rows: subsample anchors evenly, shared across links
        per_link_cap = max(1, self.max_train_rows // L)
        if anchors_all.size > per_link_cap:
            sel = np.linspace(0, anchors_all.size - 1, per_link_cap).round().astype(int)
            anchors = anchors_all[np.unique(sel)]
        else:
            anchors = anchors_all

        X_parts, y_parts = [], {h: [] for h in self.horizons}
        for j in range(L):
            X_parts.append(self._rows(anchors, j))
            for h in self.horizons:
                y_parts[h].append(self._log[anchors + h, j])
        X = np.vstack(X_parts)
        finite_rows = np.isfinite(X).all(axis=1)

        cat_mask = [False] * (X.shape[1] - 1) + [True]     # last col = link id
        for h in self.horizons:
            y = np.concatenate(y_parts[h])
            m = finite_rows & np.isfinite(y)
            reg = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=None,
                max_leaf_nodes=63, l2_regularization=1.0,
                categorical_features=cat_mask, random_state=self.random_state,
                early_stopping=True, validation_fraction=0.1,
            )
            reg.fit(X[m], y[m])
            self.models[h] = reg

    def predict(self, ctx: Context) -> np.ndarray:
        out = np.full(ctx.horizon, np.nan, dtype=float)
        anchor = np.array([ctx.origin - 1])         # last observed step
        if anchor[0] < self._min_ctx:
            return out
        X = self._rows(anchor, ctx.link)
        if not np.isfinite(X).all():
            return out
        for h in self.horizons:
            if h <= ctx.horizon and h in self.models:
                out[h - 1] = np.expm1(self.models[h].predict(X)[0])
        return np.clip(out, 0, None)
