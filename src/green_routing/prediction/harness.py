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

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import config, splits as splits_mod

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

    # Set to an already-fitted twin (same class + config) to make the harness copy
    # its fitted state instead of training a second time — used by the +online
    # variants, whose initial fit is by construction identical to the frozen one.
    warm_start_from: "Forecaster | None" = None

    # Optional checkpoint protocol (duck-typed; local models simply omit it):
    #   save(dir_path) -> Path | None   persist fitted state under dir_path
    #   load(path, values_filled, split, timestamps) -> bool   restore it
    # Implemented by torch_common.TorchTrainer (tiers 4-6) and GlobalGBDT.

    def warm_start(self, source: "Forecaster") -> None:
        """Adopt the fitted state of ``source`` instead of fitting from scratch.

        Deep-copies the twin's entire instance state (nets, boosters, scaler
        stats). Attributes set only on ``self`` — the +online configuration and
        the ``+online`` name — survive, since they are absent from the twin's
        ``__dict__``. The copy keeps the twins independent: online fine-tuning
        never bleeds back into the frozen model.
        """
        if type(source) is not type(self):
            raise TypeError(f"warm_start: {type(source).__name__} is not "
                            f"a {type(self).__name__}")
        self.__dict__.update(copy.deepcopy(source.__dict__))

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

    # Optional: ``predict_batch(origin, horizon, n_links) -> array [n_links, horizon]``
    # — every link's path at one origin in a single call, identical to calling
    # predict() per link. The harness uses it when a model defines it (GBDT does).


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
    out_dir: Path | str | None = None,
    resume: bool = False,
) -> "BacktestResult":
    """Run every model over the test block and collect per-point predictions.

    horizons: name->steps (default config.HORIZONS). stride: origin spacing in steps
    (default = 6 h). max_origins caps the number of origins (bounds per-link model cost).

    out_dir: when given, every fitted model that supports it is checkpointed to
    ``out_dir/checkpoints/`` right after training, and each model's raw records are
    flushed to ``out_dir/records/<name>.csv`` as soon as its backtest finishes — so a
    Ctrl-C loses at most the model currently running and still returns the completed
    models' results. resume: reuse those files from an earlier run — models with cached
    records are skipped outright, and pending models reload their checkpoint (skipping
    training) when one exists.
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

    ckpt_dir = rec_dir = None
    if out_dir is not None:
        ckpt_dir = Path(out_dir) / "checkpoints"
        rec_dir = Path(out_dir) / "records"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        rec_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint(m):
        """Final-weights checkpoint for m — NOT the transient mid-training
        ``<name>.train.pt`` state, which is a full optimiser bundle that the
        model's own fit resumes from (torch_common._fit_torch)."""
        hits = [p for p in sorted(ckpt_dir.glob(f"{m.name}.*"))
                if not p.name.endswith(".train.pt")] if ckpt_dir is not None else []
        return hits[0] if hits else None

    done: set[str] = set()
    if resume and rec_dir is not None:
        done = {m.name for m in models if (rec_dir / f"{m.name}.csv").exists()}
        if done and verbose:
            print(f"[harness] resume: cached records for {sorted(done)}")

    # models with cached records still need their fit if a pending twin warm-starts
    # from them (and no checkpoint of their own can stand in)
    needed_srcs = {id(m.warm_start_from) for m in models
                   if m.name not in done and m.warm_start_from is not None}

    interrupted = False
    try:
        for m in models:
            # Arm per-epoch training-state checkpointing (torch tiers write
            # <name>.train.pt after every epoch and resume from it, so a crash
            # mid-training costs at most one epoch instead of the whole fit).
            if ckpt_dir is not None:
                m.train_ckpt_dir = ckpt_dir
            # A checkpoint reload is cheap, so on resume prefer it even for models
            # whose records are cached — they stay usable (overlay plots, twins).
            ck = _checkpoint(m) if resume else None
            loader = getattr(m, "load", None)
            if ck is not None and callable(loader):
                try:
                    if loader(ck, values_filled, split, ts):
                        if verbose:
                            print(f"[harness] load {m.name} <- {ck}")
                        continue
                except Exception as exc:
                    if verbose:
                        print(f"[harness] load {m.name} failed ({exc}); fitting")
            if m.name in done and id(m) not in needed_srcs:
                continue                # cached records and nobody needs its fit
            if m.warm_start_from is not None:
                if verbose:
                    print(f"[harness] warm-start {m.name} from fitted "
                          f"{m.warm_start_from.name} (no second training)")
                m.warm_start(m.warm_start_from)
            else:
                if verbose:
                    print(f"[harness] fit {m.name} (global={m.is_global}) ...")
                m.fit(values_filled, split, ts)
            saver = getattr(m, "save", None)
            if ckpt_dir is not None and callable(saver):
                p = saver(ckpt_dir)
                if p is not None and verbose:
                    print(f"[harness] checkpoint -> {p}")
    except KeyboardInterrupt:
        interrupted = True
        print("[harness] interrupted during fit — finished checkpoints are on disk; "
              "rerun with --resume to continue")

    rec_cols = ["model", "tier", "link", "link_idx", "horizon", "origin",
                "y_true", "y_pred"]
    frames: list[pd.DataFrame] = []
    for m in models:
        rec_path = rec_dir / f"{m.name}.csv" if rec_dir is not None else None
        if m.name in done:
            frames.append(pd.read_csv(rec_path))
            continue
        if interrupted:
            continue
        if verbose:
            print(f"[harness] backtest {m.name} ...")
        do_online = getattr(m, "online_enabled", False) and getattr(m, "supports_online", False)
        last_refit: int | None = None
        mrecs: list[dict] = []
        try:
            for o in origins:                     # origins are ascending in time
                # Online adaptation: refit/fine-tune on data strictly < o (no leakage).
                # Refit at the first origin, then every ``refit_every`` steps.
                if do_online and (last_refit is None or
                                  (m.refit_every and o - last_refit >= m.refit_every)):
                    try:
                        m.online_update(values_filled, ts, upto=o)
                    except Exception as exc:      # keep the run going on a bad refit
                        if verbose:
                            print(f"[harness]   {m.name} online_update @ {o} failed: {exc}")
                    last_refit = o                # avoid retrying every origin
                path_len = min(max_h, T - o)
                if path_len < 1:
                    continue
                hs_here = tuple(h for h in hsteps if h <= path_len)
                # Optional batch hook: every link at this origin in one call (same
                # forecasts as predict(), much faster for row-wise models like GBDT)
                paths = None
                if callable(getattr(m, "predict_batch", None)):
                    try:
                        paths = np.asarray(m.predict_batch(o, path_len, Ln), dtype=float)
                    except Exception:
                        paths = None
                for j in range(Ln):
                    if paths is not None:
                        path = paths[j]
                    else:
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
                        mrecs.append(
                            {"model": m.name, "tier": m.tier, "link": panel.links[j],
                             "link_idx": j, "horizon": h, "origin": o,
                             "y_true": ytrue, "y_pred": yhat}
                        )
        except KeyboardInterrupt:
            interrupted = True
            print(f"[harness] interrupted — dropping partial {m.name}; finished "
                  f"models are kept (rerun with --resume to continue)")
            continue
        df = pd.DataFrame(mrecs, columns=rec_cols)
        if rec_path is not None:
            df.to_csv(rec_path, index=False)      # flush: this model now survives Ctrl-C
        frames.append(df)

    return BacktestResult(
        pd.concat(frames, ignore_index=True) if frames
        else pd.DataFrame(columns=rec_cols),
        panel=panel, split=split, horizons=horizons, interrupted=interrupted,
    )


# --------------------------------------------------------------------------- #
# Result aggregation + skill
# --------------------------------------------------------------------------- #
class BacktestResult:
    def __init__(self, records: pd.DataFrame, panel, split, horizons,
                 interrupted: bool = False):
        self.records = records
        self.panel = panel
        self.split = split
        self.horizons = horizons
        self.interrupted = interrupted    # Ctrl-C mid-run: records are partial
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

    def ratio_table(self, metric: str = "mae", baseline: str = "persistence",
                    per_link: pd.DataFrame | None = None) -> pd.DataFrame:
        """Normalise-then-weight evaluation: per-link ``metric`` ratio vs ``baseline``.

        Pooled raw-Mbps metrics are a traffic-weighted vote (per-link MAE scales
        with link traffic across ~4 decades, so the giants own the aggregate; on
        the 547-link cohort the top 50 links hold ~70% of pooled MAE). Here every
        link is first normalised by the baseline's error on that same link, then
        aggregated three ways:

          median_ratio    one link one vote, robust to outlier links;
          geomean_ratio   one link one vote, uses every link (log-average);
          logv_ratio      weight = log10(test-mean Mbps) — gentle size preference
                          (top-50 share drops to ~20% on the 547 cohort).

        Plus two absolute companions: ``median_wape`` (median over links of the
        per-link WAPE = sum|e|/sum|y|, in %) and ``share_beats_base`` (fraction of
        links where the model's error is below the baseline's).

        Ratios < 1 beat the baseline. Two robustness guards: links whose baseline
        error is tiny in absolute terms (< 1e-3 Mbps — near-constant series where
        the ratio is pure noise amplification) are excluded, and the ratios
        entering the *arithmetic* logv mean are winsorised at 20 (ratios are
        unbounded above, so one catastrophic link otherwise owns the mean — the
        pred_547c run showed values of 33 and 454 without this). The median and
        geometric mean are computed on unclipped ratios; ``n_links`` reports the
        count used.
        """
        pl = self.per_link() if per_link is None else per_link
        vol = self.records.groupby("link")["y_true"].apply(lambda s: s.abs().mean())
        w_all = np.log10(np.maximum(vol, 1.0)) + 0.01
        base = pl[pl.model == baseline].set_index(["horizon", "link"])[metric]
        rows = []
        for (model, tier, h), g in pl.groupby(["model", "tier", "horizon"]):
            if h not in base.index.get_level_values(0):
                continue
            b = base.loc[h]
            g = g.set_index("link")[metric]
            common = [l for l in g.index
                      if l in b.index and np.isfinite(b[l]) and b[l] > 1e-3
                      and np.isfinite(g[l])]
            if not common:
                continue
            r = (g.loc[common] / b.loc[common]).astype(float)
            w = w_all.reindex(common).fillna(0.01)
            wl = pl[(pl.model == model) & (pl.horizon == h)].set_index("link")["wape"]
            wl = wl.reindex(common)
            wl = wl[np.isfinite(wl)]
            rows.append({
                "model": model, "tier": tier, "horizon": h,
                "horizon_name": self._h_name.get(h, str(h)),
                "median_ratio": float(r.median()),
                "geomean_ratio": float(np.exp(np.log(r.clip(lower=1e-9)).mean())),
                "logv_ratio": float((r.clip(upper=20.0) * w).sum() / w.sum()),
                # per-link WAPE (sum|e|/sum|y|, %), one link one vote: the typical
                # link's error as a share of its own traffic
                "median_wape": float(wl.median()) if len(wl) else np.nan,
                "share_beats_base": float((r < 1.0).mean()),
                "n_links": len(common),
            })
        if not rows:            # e.g. the baseline model wasn't in this run
            return pd.DataFrame(columns=["model", "tier", "horizon", "horizon_name",
                                         "median_ratio", "geomean_ratio",
                                         "logv_ratio", "median_wape",
                                         "share_beats_base", "n_links"])
        return (pd.DataFrame(rows)
                .sort_values(["horizon", "logv_ratio"]).reset_index(drop=True))

    def wape_table(self, per_link: pd.DataFrame | None = None) -> pd.DataFrame:
        """WAPE = sum|e| / sum|y| (%) per (model, horizon), two ways:

          pooled_wape   over every scored point of every link — volume-weighted,
                        so the big links dominate (the carbon-bill view: the
                        bill's error is in bytes);
          median_wape   median of the per-link WAPEs — one link one vote, the
                        typical link's error as a share of its own traffic;
          p90_wape      90th percentile of the per-link WAPEs — the hard tail.

        On identical targets, pooled WAPE ranks models exactly like pooled MAE
        (same denominator), so its skill vs a baseline equals the MAE skill; the
        per-link median is the genuinely new number.
        """
        pl = self.per_link() if per_link is None else per_link
        agg = self.aggregate().set_index(["model", "horizon"])["wape"]
        rows = []
        for (model, tier, h), g in pl.groupby(["model", "tier", "horizon"]):
            w = g["wape"].to_numpy(dtype=float)
            w = w[np.isfinite(w)]
            rows.append({
                "model": model, "tier": tier, "horizon": h,
                "horizon_name": self._h_name.get(h, str(h)),
                "pooled_wape": float(agg.get((model, h), np.nan)),
                "median_wape": float(np.median(w)) if w.size else np.nan,
                "p90_wape": float(np.quantile(w, 0.9)) if w.size else np.nan,
                "n_links": int(w.size),
            })
        cols = ["model", "tier", "horizon", "horizon_name", "pooled_wape",
                "median_wape", "p90_wape", "n_links"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return (pd.DataFrame(rows, columns=cols)
                .sort_values(["horizon", "median_wape"]).reset_index(drop=True))

    def peak_table(self, q: float = 0.90) -> pd.DataFrame:
        """Peak behaviour per (model, horizon): how forecasts treat each link's
        top-decile actuals — the traffic that matters for capacity decisions.

        For every (link, horizon) the threshold is the q-quantile of y_true (the
        targets are identical across models, so thresholds are too). Reported per
        (model, horizon), pooled over links:

          peak_bias_pct        signed error on peak points, % of peak volume
                               (negative = under-prediction / peak shaving)
          peak_underpred_rate  fraction of peak points predicted below actual
          peak_mae             MAE on peak points (Mbps)
          peak_wape            WAPE on peak points, sum|e|/sum|y| (%)
          off_bias_pct         signed error off-peak — the price paid for any
                               peak correction shows up here as over-prediction

        Note an irreducible part: conditioning on a high actual guarantees any
        no-lookahead forecaster under-shoots somewhat (persistence sits at
        -16..-34% on pred_547d); judge models against the naive rows, not zero.
        """
        r = self.records
        if r.empty:
            return pd.DataFrame(columns=["model", "tier", "horizon", "horizon_name",
                                         "peak_bias_pct", "peak_underpred_rate",
                                         "peak_mae", "peak_wape", "off_bias_pct",
                                         "n_peak"])
        thr = (r.groupby(["link", "horizon"])["y_true"].quantile(q)).rename("thr")
        d = r.join(thr, on=["link", "horizon"])
        d["is_peak"] = d.y_true >= d.thr
        rows = []
        for (model, tier, h), g in d.groupby(["model", "tier", "horizon"]):
            pk, off = g[g.is_peak], g[~g.is_peak]
            if not len(pk):
                continue
            err = pk.y_pred - pk.y_true
            rows.append({
                "model": model, "tier": tier, "horizon": h,
                "horizon_name": self._h_name.get(h, str(h)),
                "peak_bias_pct": float(100 * err.sum() / pk.y_true.sum()),
                "peak_underpred_rate": float((err < 0).mean()),
                "peak_mae": float(err.abs().mean()),
                "peak_wape": float(100 * err.abs().sum() / max(pk.y_true.abs().sum(), 1e-9)),
                "off_bias_pct": float(100 * (off.y_pred - off.y_true).sum() /
                                      max(off.y_true.sum(), 1e-9)),
                "n_peak": int(len(pk)),
            })
        return (pd.DataFrame(rows)
                .sort_values(["horizon", "peak_bias_pct"], ascending=[True, False])
                .reset_index(drop=True))

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
