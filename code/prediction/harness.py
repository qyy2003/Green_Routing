"""The testing loop: a rolling-origin backtest that scores every model on
*identical* targets, then reports skill vs. the naive baselines.

This is the "universal harness" of benchmark_design.md / the roadmap:

  * One chronological train/val/test split with a purge gap (from ``dataset.splits``).
  * Forecast **origins** are placed across the test block. At each origin `o` a model
    may use only data ``< o`` and emits a path of `max(horizons)` steps. We score the
    point at each lead time `h` against ``actual[o + h - 1]``.
  * Metrics are computed in the original scale (Mbps); models own their log/normalise
    transforms internally.
  * Every model sees the same origins, links and horizons, so a per-link SARIMA and a
    global booster land on the same targets — the only honest comparison.

A model is any :class:`Forecaster`. ``fit`` is called once (global models train on the
train slice; local models no-op). ``predict(ctx)`` returns a length-``ctx.horizon`` path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset import config, splits as splits_mod

from . import metrics as M


# --------------------------------------------------------------------------- #
# Model interface
# --------------------------------------------------------------------------- #
@dataclass
class Context:
    """Everything a model may look at to forecast one link at one origin.

    Hard rule: only use information with time index ``< origin`` (no leakage).
    """

    link: int
    origin: int                 # grid index of the first forecast step
    horizon: int                # path length to return = max(scored horizons)
    horizons: tuple[int, ...]   # the lead times actually scored (subset of 1..horizon)
    timestamps: np.ndarray      # full epoch-second grid [T]
    values_filled: np.ndarray   # [T, L] short-gap-filled, for building features
    train_end: int              # last train index (for train-only scaler stats)

    def history(self) -> np.ndarray:
        """This link's filled values strictly before the origin."""
        return self.values_filled[: self.origin, self.link]

    def hist_index(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps[: self.origin], unit="s", utc=True)

    def target_index(self) -> pd.DatetimeIndex:
        end = self.origin + self.horizon
        return pd.to_datetime(self.timestamps[self.origin : end], unit="s", utc=True)


class Forecaster:
    """Base class. Subclass and implement :meth:`predict` (and :meth:`fit` if global)."""

    name: str = "base"
    tier: int = 0
    is_global: bool = False

    # --- online (test-time) adaptation ------------------------------------- #
    # A global model that overrides ``online_update`` sets ``supports_online``.
    # When ``online_enabled`` is on, the harness periodically calls
    # ``online_update`` so the model may fine-tune on the most recent data — but
    # only on information *strictly before* the current forecast origin, so the
    # no-leakage guarantee is unchanged.
    supports_online: bool = False
    online_enabled: bool = False
    refit_every: int | None = None      # steps between refits (None => refit once)
    online_window: int | None = None    # sliding-window length (None => expanding)

    def fit(self, values_filled: np.ndarray, split: splits_mod.Split,
            timestamps: np.ndarray) -> None:
        """Train once. Local models leave this as a no-op."""

    def online_update(self, values_filled: np.ndarray, timestamps: np.ndarray,
                      upto: int) -> None:
        """Adapt to recent data. MUST use only indices ``< upto`` (no leakage).

        Default no-op: frozen models and local models (which already re-read the
        recent history inside ``predict``) ignore this.
        """

    def predict(self, ctx: Context) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Rolling-origin backtest
# --------------------------------------------------------------------------- #
def _fill(values: np.ndarray, limit: int) -> np.ndarray:
    """Forward/back-fill short gaps per link (for features/context only)."""
    df = pd.DataFrame(values)
    df = df.ffill(limit=limit).bfill(limit=limit)
    return df.to_numpy(dtype=np.float32)


def make_origins(split: splits_mod.Split, stride: int, max_horizon: int,
                 max_origins: int | None) -> list[int]:
    lo, hi = split.test
    cand = list(range(lo, hi - 1, stride))
    cand = [o for o in cand if o + 1 <= hi]          # need at least the 1-step target
    if max_origins and len(cand) > max_origins:
        # evenly subsample to bound cost of per-link models
        sel = np.linspace(0, len(cand) - 1, max_origins).round().astype(int)
        cand = [cand[i] for i in sorted(set(sel.tolist()))]
    return cand


def backtest(
    panel,
    models: list[Forecaster],
    split: splits_mod.Split,
    horizons: dict[str, int] | None = None,
    stride: int | None = None,
    max_origins: int | None = 120,
    fill_limit: int = config.DAILY_STEPS // 24,   # ~1 h
    verbose: bool = True,
) -> "BacktestResult":
    """Run every model over the test block and collect per-point predictions.

    horizons: name->steps (default config.HORIZONS). stride: origin spacing in steps
    (default = 6 h). max_origins caps the number of origins (bounds per-link model cost).
    """
    horizons = horizons or dict(config.HORIZONS)
    hsteps = tuple(sorted(set(horizons.values())))
    max_h = max(hsteps)
    stride = stride or (config.DAILY_STEPS // 4)   # every 6 h

    values_raw = panel.values
    values_filled = _fill(values_raw, fill_limit)
    T, Ln = values_raw.shape
    ts = panel.timestamps

    origins = make_origins(split, stride, max_h, max_origins)
    if verbose:
        print(f"[harness] {len(origins)} origins, {Ln} links, "
              f"horizons={hsteps}, stride={stride}")

    for m in models:
        if verbose:
            print(f"[harness] fit {m.name} (global={m.is_global}) ...")
        m.fit(values_filled, split, ts)

    records: list[dict] = []
    for m in models:
        if verbose:
            print(f"[harness] backtest {m.name} ...")
        do_online = getattr(m, "online_enabled", False) and getattr(m, "supports_online", False)
        last_refit: int | None = None
        for o in origins:                         # origins are ascending in time
            # Online adaptation: refit/fine-tune on data strictly < o (no leakage).
            # Refit at the first origin, then every ``refit_every`` steps.
            if do_online and (last_refit is None or
                              (m.refit_every and o - last_refit >= m.refit_every)):
                try:
                    m.online_update(values_filled, ts, upto=o)
                except Exception as exc:          # keep the run going on a bad refit
                    if verbose:
                        print(f"[harness]   {m.name} online_update @ {o} failed: {exc}")
                last_refit = o                    # avoid retrying every origin
            path_len = min(max_h, T - o)
            if path_len < 1:
                continue
            hs_here = tuple(h for h in hsteps if h <= path_len)
            for j in range(Ln):
                ctx = Context(
                    link=j, origin=o, horizon=path_len, horizons=hs_here,
                    timestamps=ts, values_filled=values_filled,
                    train_end=split.train[1],
                )
                try:
                    path = np.asarray(m.predict(ctx), dtype=float).ravel()
                except Exception:
                    continue
                for h in hs_here:
                    if h - 1 >= path.size:
                        continue
                    yhat = path[h - 1]
                    ytrue = values_raw[o + h - 1, j]
                    if not (np.isfinite(yhat) and np.isfinite(ytrue)):
                        continue
                    records.append(
                        {"model": m.name, "tier": m.tier, "link": panel.links[j],
                         "link_idx": j, "horizon": h, "origin": o,
                         "y_true": ytrue, "y_pred": yhat}
                    )

    return BacktestResult(
        pd.DataFrame.from_records(records),
        panel=panel, split=split, horizons=horizons,
    )


# --------------------------------------------------------------------------- #
# Result aggregation + skill
# --------------------------------------------------------------------------- #
class BacktestResult:
    def __init__(self, records: pd.DataFrame, panel, split, horizons):
        self.records = records
        self.panel = panel
        self.split = split
        self.horizons = horizons
        self._h_name = {v: k for k, v in horizons.items()}

    def _train_scale(self, link_idx: int) -> float:
        """In-sample naive-1 MAE on the train slice, for MASE."""
        tr = self.panel.values[self.split.train[0]:self.split.train[1], link_idx]
        tr = tr[np.isfinite(tr)]
        if tr.size <= 1:
            return np.nan
        return float(np.mean(np.abs(np.diff(tr))))

    def per_link(self) -> pd.DataFrame:
        """Point metrics per (model, horizon, link)."""
        rows = []
        for (model, tier, h, link, j), g in self.records.groupby(
            ["model", "tier", "horizon", "link", "link_idx"]
        ):
            r = {"model": model, "tier": tier, "horizon": h,
                 "horizon_name": self._h_name.get(h, str(h)), "link": link}
            r.update(M.evaluate_all(g["y_true"].to_numpy(), g["y_pred"].to_numpy()))
            r["mase"] = M.mase(g["y_true"].to_numpy(), g["y_pred"].to_numpy(),
                               self.panel.values[:self.split.train[1], j])
            r["n"] = len(g)
            rows.append(r)
        return pd.DataFrame(rows)

    def aggregate(self) -> pd.DataFrame:
        """Point metrics pooled over all links, per (model, horizon)."""
        rows = []
        for (model, tier, h), g in self.records.groupby(["model", "tier", "horizon"]):
            r = {"model": model, "tier": tier, "horizon": h,
                 "horizon_name": self._h_name.get(h, str(h))}
            r.update(M.evaluate_all(g["y_true"].to_numpy(), g["y_pred"].to_numpy()))
            r["n"] = len(g)
            rows.append(r)
        return pd.DataFrame(rows).sort_values(["horizon", "mae"]).reset_index(drop=True)

    def skill_table(self, metric: str = "mae",
                    baselines=("persistence", "seasonal_naive_day")) -> pd.DataFrame:
        """Wide table: `metric` per model x horizon plus skill vs each baseline.

        skill = 1 - model/baseline (positive => the model beats that baseline).
        """
        agg = self.aggregate()
        wide = agg.pivot_table(index=["model", "tier"], columns="horizon_name",
                               values=metric)
        # order horizon columns by their step count
        order = [self._h_name[v] for v in sorted(self.horizons.values())
                 if self._h_name[v] in wide.columns]
        wide = wide[order]
        base_vals = {b: agg[agg.model == b].set_index("horizon_name")[metric]
                     for b in baselines if (agg.model == b).any()}
        for b, bv in base_vals.items():
            for col in order:
                if col in bv.index:
                    wide[(f"skill_vs_{b}", col)] = wide[col].apply(
                        lambda x, base=bv[col]: M.skill(x, base))
        return wide.sort_index()
