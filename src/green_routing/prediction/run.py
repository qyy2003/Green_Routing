"""Run the method ladder end-to-end and report skill vs. the naive baselines.

    gr-benchmark --start 2024-06-01 --end 2024-10-01 \
        --train-end 2024-08-15 --val-start 2024-08-22 \
        --val-end 2024-09-05 --test-start 2024-09-12 \
        --tiers 1 3 --horizons 5min 1h day \
        --max-links 30 --out artifacts/pred_demo

(``gr-benchmark`` is ``python -m green_routing.prediction.run``.)

Writes to --out: aggregate.csv, per_link.csv, skill_<metric>.csv, and PNG plots
(predicted-vs-actual overlays + a skill bar), plus checkpoints/ (fitted model weights,
saved right after each training), records/ (per-model raw predictions, flushed as each
model finishes), history/ (per-epoch learning curves) and run.log (the full console
output of this run). Ctrl-C keeps everything already flushed; rerun with --resume to
skip the finished models and reload saved weights. Everything is scored on the identical
rolling origins, so "model X beats persistence by Y%" is an honest number.

A run directory is working state — it mixes every model together and gets pruned and
overwritten. When a run turns out well, promote the models worth keeping into the
model zoo, which gives each one its own immutable, self-describing entry:

    gr-zoo add <out> --models tcn,gbdt+online --note "why"

or pass --zoo here to promote automatically once the run completes. See prediction/zoo.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import config, splits as splits_mod

from . import baselines, data as data_mod, harness, zoo as zoo_mod
from .models_classical import default_classical
from .models_gbdt import GlobalGBDT


class _Tee:
    """Mirror a stream into a file, so the console output of a run is also on disk.

    The training curve, the fit/backtest ordering and every warning are printed, not
    logged — which means they only ever lived in terminal scrollback. Teeing to
    ``<out>/run.log`` makes a run's log an artifact the zoo can carry.
    """

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.fh = open(path, "a", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return self.stream.isatty()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


def save_histories(models, out: Path) -> None:
    """Persist each trainer's per-epoch curve to ``<out>/history/``.

    Only the torch tiers record one (see torch_common._fit_torch); models without a
    history simply contribute nothing.
    """
    hist_dir = out / "history"
    for m in models:
        rows = getattr(m, "train_history", None)
        summary = getattr(m, "train_summary", None)
        if not rows and not summary:
            continue
        hist_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            pd.DataFrame(rows).to_csv(hist_dir / f"{m.name}.csv", index=False)
        if summary:
            with open(hist_dir / f"{m.name}.summary.json", "w") as fh:
                json.dump(summary, fh, indent=2)


def _scale_weight_arg(v: str):
    """--scale-weight accepts a float exponent or the string 'log'."""
    if v.strip().lower() == "log":
        return "log"
    return float(v)


def _new_global_models(tiers, hsteps, panel, lookback, epochs, max_train_samples=None,
                       scale_weight=0.0):
    """The global (fit-once) tiers only — instantiable twice (frozen + online)."""
    models: list[harness.Forecaster] = []
    if 3 in tiers:
        models.append(GlobalGBDT(horizons=hsteps))
    kw = {} if max_train_samples is None else {"max_train_samples": max_train_samples}
    if scale_weight:
        kw["scale_weight"] = scale_weight
    if 4 in tiers:
        from .models_deep import GRUForecaster, TCNForecaster
        models += [GRUForecaster(lookback=lookback, horizons=hsteps, epochs=epochs, **kw),
                   TCNForecaster(lookback=lookback, horizons=hsteps, epochs=epochs, **kw)]
    if 5 in tiers:
        from .models_deep import DLinearForecaster, PatchTSTForecaster
        models += [DLinearForecaster(lookback=lookback, horizons=hsteps, epochs=epochs, **kw),
                   PatchTSTForecaster(lookback=lookback, horizons=hsteps, epochs=epochs, **kw)]
    if 6 in tiers:
        from .models_gnn import STGNNForecaster
        # STGNN samples are whole-panel (all L links per anchor), so its budget is
        # per-anchor: match the per-link coverage of the deep tiers, capped for RAM.
        skw = {} if max_train_samples is None else \
            {"max_train_samples": min(3000, max(1000, max_train_samples // max(1, panel.L)))}
        if scale_weight:
            skw["scale_weight"] = scale_weight
        models.append(STGNNForecaster(panel.devices, lookback=lookback,
                                      horizons=hsteps, epochs=epochs, **skw))
    return models


def build_models(tiers, hsteps, seasonal_sarima, panel, lookback, epochs, online_cfg=None,
                 max_train_samples=None, scale_weight=0.0, under_weight=1.0,
                 calendar=False):
    """Assemble the requested tiers. Tiers 4-6 import torch lazily (only if requested).

    If ``online_cfg`` is given, an ``<name>+online`` variant of every global tier is
    added alongside its frozen twin. The online variant re-fits/fine-tunes on data
    strictly before each forecast origin (see harness.Forecaster.online_update), so
    the two land on identical origins — the honest "how much is recent data worth?".
    Its initial fit is identical to the frozen twin's, so it warm-starts from the
    twin's fitted state instead of training a second time.
    """
    models: list[harness.Forecaster] = []
    if 1 in tiers:
        models += baselines.default_baselines()
    if 2 in tiers:
        models += default_classical(seasonal_sarima=seasonal_sarima)

    frozen_globals = _new_global_models(tiers, hsteps, panel, lookback, epochs,
                                        max_train_samples, scale_weight)
    models += frozen_globals

    if online_cfg is not None:
        for base, m in zip(frozen_globals,
                           _new_global_models(tiers, hsteps, panel, lookback, epochs,
                                              max_train_samples, scale_weight)):
            m.warm_start_from = base
            m.online_enabled = True
            m.refit_every = online_cfg["refit_every"]
            m.online_window = online_cfg["window"]
            if online_cfg.get("epochs") and hasattr(m, "online_epochs"):
                m.online_epochs = online_cfg["epochs"]
            if online_cfg.get("lr") and hasattr(m, "online_lr"):
                m.online_lr = online_cfg["lr"]
            m.name = f"{m.name}+online"
            models.append(m)

    if under_weight and under_weight != 1.0:
        # asymmetric Huber on every torch tier (see torch_common.under_weight)
        for m in models:
            if hasattr(m, "under_weight"):
                m.under_weight = under_weight
    if calendar:
        # day-of-week + Swiss-holiday conditioning (designs A+C, see
        # calendar_features / models_deep.CalendarCond). Rides the link-embedding
        # plumbing, so DLinear (the linear control) is excluded by the same rule;
        # GBDT gets is_holiday/is_workday feature columns instead.
        for m in models:
            if (getattr(m, "use_link_embedding", False)
                    and not getattr(m, "calendar_opt_out", False)):
                m.use_calendar = True
            if hasattr(m, "extra_calendar"):
                m.extra_calendar = True
    return models


# GBDT context presets: "day" is the original feature set (lags to 1 day, rolling
# 1 h / 1 day); "month" adds weekly lags and 1-week / 1-month rolling windows — the
# booster's analogue of a 1-month input window.
GBDT_CONTEXT = {
    "day": dict(lags=(1, 2, 3, 4, 5, 6, 12, 24, config.DAILY_STEPS),
                roll_windows=(12, config.DAILY_STEPS)),
    "month": dict(lags=(1, 2, 3, 4, 5, 6, 12, 24, config.DAILY_STEPS,
                        config.WEEKLY_STEPS, 2 * config.WEEKLY_STEPS,
                        3 * config.WEEKLY_STEPS, 4 * config.WEEKLY_STEPS),
                  roll_windows=(12, config.DAILY_STEPS, config.WEEKLY_STEPS,
                                30 * config.DAILY_STEPS)),
}


def apply_overrides(models, cfg: dict) -> list[str]:
    """Architecture/training knobs that are not constructor arguments, applied
    after build_models. ``cfg`` is the argparse namespace as a dict (or a run's
    config.json), so prediction.dense rebuilds exactly the trained models.
    Unset keys leave the defaults — earlier runs stay bit-identical."""
    notes = []
    bs = cfg.get("batch_size")
    for m in models:
        if bs and hasattr(m, "batch_size") and hasattr(m, "under_weight"):
            m.batch_size = int(bs)
        if cfg.get("tcn_max_levels") and m.name.startswith("tcn"):
            m.max_levels = int(cfg["tcn_max_levels"])
        if m.name.startswith("patchtst"):
            if cfg.get("patchtst_patch"):
                m.patch = int(cfg["patchtst_patch"])
            if cfg.get("patchtst_stride"):
                m.stride = int(cfg["patchtst_stride"])
        if cfg.get("target_smooth") and hasattr(m, "target_smooth"):
            m.target_smooth = int(cfg["target_smooth"])
        ctx = cfg.get("gbdt_context")
        if ctx and ctx != "day" and hasattr(m, "lags") and hasattr(m, "roll_windows"):
            m.lags = tuple(GBDT_CONTEXT[ctx]["lags"])
            m.roll_windows = tuple(GBDT_CONTEXT[ctx]["roll_windows"])
            m._min_ctx = max(max(m.lags), max(m.roll_windows))
    for k in ("batch_size", "tcn_max_levels", "patchtst_patch", "patchtst_stride",
              "target_smooth", "gbdt_context"):
        if cfg.get(k):
            notes.append(f"{k}={cfg[k]}")
    return notes


def prune_checkpoints(out: Path) -> None:
    """After a *complete* run (results on disk), drop the ``+online`` checkpoints.

    They hold the warm-start state — a byte-identical copy of the frozen twin's
    trained weights, never the fine-tuned state (a resumed backtest replays the
    online adaptation from the trained fit anyway) — so once the results are
    recorded they are pure duplicates. What happened is appended to
    ``<out>/cleanup.log``.
    """
    ckpt_dir = out / "checkpoints"
    if not ckpt_dir.is_dir():
        return
    removed, kept = [], []
    for p in sorted(ckpt_dir.iterdir()):
        (removed if "+online" in p.name else kept).append(p)
    for p in removed:
        p.unlink()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log = out / "cleanup.log"
    with open(log, "a") as fh:
        fh.write(f"[{stamp}] run complete, results recorded — pruning intermediate "
                 f"checkpoints\n")
        for p in removed:
            fh.write(f"  deleted {p.relative_to(out)}  "
                     f"(+online warm-start duplicate of its frozen twin)\n")
        for p in kept:
            fh.write(f"  kept    {p.relative_to(out)}  (trained model weights)\n")
    if removed:
        print(f"[cleanup] removed {len(removed)} +online checkpoint(s), kept "
              f"{len(kept)} trained one(s) — see {log}")


def make_split(panel, args) -> splits_mod.Split:
    if all([args.train_end, args.val_start, args.val_end, args.test_start]):
        return splits_mod.chronological_split_by_date(
            panel.timestamps, args.train_end, args.val_start,
            args.val_end, args.test_start,
        )
    return splits_mod.chronological_split(panel.T)


def _representative_links(panel, split, k: int = 3) -> list[int]:
    """Pick a busy, a quiet, and a bursty link from the train slice."""
    tr = panel.values[split.train[0]:split.train[1]]
    mean = np.nanmean(tr, axis=0)
    std = np.nanstd(tr, axis=0)
    burst = np.where(mean > 0, std / (mean + 1e-9), 0)
    picks = {int(np.nanargmax(mean)), int(np.nanargmax(burst))}
    valid = np.where(np.isfinite(mean) & (mean > 0))[0]
    if valid.size:
        picks.add(int(valid[np.argmin(mean[valid])]))
    return sorted(picks)[:k]


def plot_overlays(result, models, panel, split, out: Path, horizon_steps: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    links = _representative_links(panel, split)
    lo, hi = split.test
    o = lo + (hi - lo) // 2                       # a mid-test origin
    o = min(o, panel.T - horizon_steps - 1)
    idx = panel.index
    vf = harness._fill(panel.values, config.DAILY_STEPS // 24)

    fig, axes = plt.subplots(len(links), 1, figsize=(11, 3 * len(links)), squeeze=False)
    for ax, j in zip(axes[:, 0], links):
        window = slice(max(0, o - horizon_steps), o + horizon_steps)
        ax.plot(idx[window], panel.values[window, j], color="black", lw=1.4,
                label="actual", zorder=5)
        ax.axvline(idx[o], color="gray", ls=":", lw=1)
        tgt_idx = idx[o:o + horizon_steps]
        for m in models:
            ctx = harness.Context(
                link=j, origin=o, horizon=horizon_steps,
                horizons=(horizon_steps,), timestamps=panel.timestamps,
                values_filled=vf, train_end=split.train[1],
            )
            try:
                path = np.asarray(m.predict(ctx), dtype=float).ravel()[:horizon_steps]
            except Exception:
                continue
            if np.isfinite(path).any():
                ax.plot(tgt_idx, path, lw=1.0, alpha=0.85, label=m.name)
        ax.set_title(f"{panel.links[j]}  (mid-test origin, {horizon_steps}-step path)")
        ax.set_ylabel("Mbps")
        ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    p = out / "overlay_forecasts.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"[plot] {p}")


def render_skill_bar(agg, order, out: Path, metric: str, baseline: str,
                     clip=(-1.0, 1.0)):
    """Skill (1 - metric/baseline) as a grouped bar chart, robust to blow-ups.

    A numerically unstable model (e.g. a diverging SARIMA) can post an astronomical
    metric, making its skill ~-1e214 and flattening every other bar. We clip the
    *displayed* skill to ``clip`` and list any off-scale bars in a caption, so the
    chart stays legible while staying honest about what was clipped.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (agg.model == baseline).sum() == 0:
        return
    base = agg[agg.model == baseline].set_index("horizon_name")[metric]
    piv = agg.pivot_table(index="model", columns="horizon_name", values=metric)
    order = [h for h in order if h in piv.columns]
    piv = piv[order]
    sk = 1.0 - piv.div(base[order], axis=1)
    sk = sk.drop(index=[b for b in (baseline,) if b in sk.index])
    if sk.empty:                                  # only the baseline was run
        return

    lo, hi = clip
    off = []                                      # (model, horizon, true skill) off-scale
    for mdl in sk.index:
        for col in order:
            v = sk.loc[mdl, col]
            if np.isfinite(v) and (v < lo or v > hi):
                off.append((mdl, col, float(v)))
    sk_disp = sk.clip(lower=lo, upper=hi)

    ax = sk_disp.plot(kind="bar", figsize=(11, 5.5))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(lo * 1.05, hi * 1.08)
    ax.set_ylabel(f"skill vs {baseline}  (1 - {metric}/base)")
    ax.set_title(f"Skill over {baseline} by horizon ({metric})")
    ax.legend(title="horizon", fontsize=8)
    if off:
        worst = min(off, key=lambda t: t[2])
        note = (f"skill clipped to [{lo:g}, {hi:g}] — {len(off)} bar(s) off-scale; "
                f"worst: {worst[0]}/{worst[1]} skill={worst[2]:.2g}")
        ax.text(0.01, 0.98, note, transform=ax.transAxes, ha="left",
                va="top", fontsize=7, color="crimson")   # use the empty top space
    fig = ax.figure
    fig.tight_layout()
    p = out / f"skill_vs_{baseline}_{metric}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"[plot] {p}")


def plot_skill(result, out: Path, metric: str, baseline: str):
    order = [result._h_name[v] for v in sorted(result.horizons.values())]
    render_skill_bar(result.aggregate(), order, out, metric, baseline)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--train-end")
    ap.add_argument("--val-start")
    ap.add_argument("--val-end")
    ap.add_argument("--test-start")
    ap.add_argument("--value-col", default="total_mbps")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--max-links", type=int, default=None,
                    help="cap the panel at the first N dense links (quick "
                         "experiments only); default: keep every usable link "
                         "with coverage >= --min-coverage")
    ap.add_argument("--devices", nargs="*", default=None,
                    help="short device names to restrict the cohort (e.g. ag1 be1)")
    ap.add_argument("--tiers", nargs="+", type=int, default=[1, 3],
                    help="1=baselines 2=classical 3=gbdt 4=gru/tcn 5=dlinear/patchtst 6=stgnn")
    ap.add_argument("--seasonal-sarima", action="store_true")
    ap.add_argument("--lookback", type=int, default=config.DAILY_STEPS,
                    help="input window for deep/GNN tiers (steps)")
    ap.add_argument("--epochs", type=int, default=12, help="deep/GNN training epochs")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="training-window budget for the deep tiers (default: "
                         "100k windows; STGNN gets a per-anchor share)")
    ap.add_argument("--min-mean-mbps", type=float, default=0.0,
                    help="drop links whose mean throughput over the window is below "
                         "this (0 = keep all). Removes idle ports that otherwise "
                         "subsidise scale-free metrics.")
    ap.add_argument("--connected-links", type=int, default=None,
                    help="keep only ~N links whose host routers form a CONNECTED "
                         "sub-network (BFS over the topology from the busiest router) "
                         "— a real slice of the graph for a quick STGNN-meaningful run. "
                         "Applied after --min-mean-mbps.")
    ap.add_argument("--scale-weight", type=_scale_weight_arg, default=0.0,
                    help="deep-tier training-loss weight from per-link mean traffic: "
                         "a float exponent (0 = uniform/RevIN; 1 = weight ~ Mbps, "
                         "aligns with pooled MAE) or 'log' (= weight ~ log10(Mbps), "
                         "the training-side mirror of the log-volume-weighted ratio "
                         "metric). Applies to GRU/TCN/DLinear/PatchTST and, as "
                         "per-node weights, to the STGNN.")
    ap.add_argument("--under-weight", type=float, default=1.0,
                    help="asymmetric-Huber factor on UNDER-predicted steps for the "
                         "torch tiers (1 = symmetric). ~2 counteracts peak shaving "
                         "by moving the point forecast above the conditional "
                         "median; judge the effect in peaks.csv.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed offset for the torch tiers (per-model base seed + "
                         "this) -- for multi-seed error bars; 0 = canonical.")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated model names to keep (applied after tier "
                         "assembly), e.g. 'persistence,seasonal_naive_day,tcn' — "
                         "for cheap targeted ablations. Skill/ratio tables need "
                         "'persistence' in the list.")
    ap.add_argument("--calendar", action="store_true",
                    help="feed day-of-week + Swiss public holidays to the models: "
                         "anchor covariates + future-known path code on the torch "
                         "tiers' heads (GRU/TCN/PatchTST; DLinear stays the plain "
                         "control), is_holiday/is_workday features on GBDT.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="torch-tier batch size (default: shrinks with --lookback, "
                         "256*672//lookback — pin it to compare lookbacks fairly)")
    ap.add_argument("--tcn-max-levels", type=int, default=None,
                    help="TCN depth cap (default 9 = receptive field 2045, one "
                         "week; 12 covers a 1-month lookback)")
    ap.add_argument("--patchtst-patch", type=int, default=None,
                    help="PatchTST patch length (default 16)")
    ap.add_argument("--patchtst-stride", type=int, default=None,
                    help="PatchTST patch stride (default 8)")
    ap.add_argument("--target-smooth", type=int, default=0,
                    help="train the torch tiers on a centred moving average of the "
                         "target over this many steps (0 = raw targets)")
    ap.add_argument("--gbdt-context", choices=list(GBDT_CONTEXT), default="day",
                    help="GBDT lag/rolling feature set: 'day' (default) or 'month' "
                         "(+ weekly lags, 1-week and 1-month rolling windows)")
    ap.add_argument("--horizons", nargs="+", default=["5min", "1h", "day"],
                    choices=list(config.HORIZONS))
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max-origins", type=int, default=120)
    ap.add_argument("--online", action="store_true",
                    help="add an <name>+online variant of each global tier that "
                         "re-fits/fine-tunes on data before each origin (no leakage)")
    ap.add_argument("--refit-every", type=int, default=config.DAILY_STEPS,
                    help="steps between online refits (default = 1 day / 288 steps)")
    ap.add_argument("--online-window", type=int, default=None,
                    help="sliding online train window in steps (default: expanding)")
    ap.add_argument("--online-epochs", type=int, default=None,
                    help="warm-start fine-tune epochs per refit (torch tiers)")
    ap.add_argument("--online-lr", type=float, default=None,
                    help="warm-start fine-tune learning rate (torch tiers)")
    ap.add_argument("--metric", default="mae",
                    choices=["mae", "rmse", "smape", "wape"])
    ap.add_argument("--resume", action="store_true",
                    help="reuse <out>/records/*.csv and <out>/checkpoints/* from an "
                         "earlier (e.g. interrupted) run: finished models are not "
                         "re-run, fitted torch/GBDT models reload their weights")
    ap.add_argument("--zoo", action="store_true",
                    help="on a complete run, promote the trained models into the "
                         "model zoo (artifacts/zoo/) — weights + learning curve + "
                         "log slice + metrics per model. See prediction/zoo.py")
    ap.add_argument("--zoo-models", default=None,
                    help="comma-separated names to promote (default: every model "
                         "that has saved weights)")
    ap.add_argument("--zoo-tag", default=None,
                    help="zoo entry name (default: the --out directory's name)")
    ap.add_argument("--zoo-note", default="",
                    help="why this run is worth keeping — shown in the zoo index")
    ap.add_argument("--zoo-records", action="store_true",
                    help="also store the raw per-point predictions in the entry")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Tee the console into <out>/run.log before anything else prints, so the run's
    # log is a file the zoo can carry rather than terminal scrollback.
    tee = _Tee(sys.stdout, out / "run.log")
    sys.stdout = tee
    try:
        print(f"# {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
              f"$ {' '.join(sys.argv)}")
        _execute(args, out)
    finally:
        sys.stdout = tee.stream
        tee.close()


def _execute(args, out: Path):
    panel = data_mod.build_panel(
        args.start, args.end, value_col=args.value_col,
        min_coverage=args.min_coverage, max_links=args.max_links,
        devices=args.devices,
    )
    if args.min_mean_mbps > 0:
        keep = panel.mean_mbps() >= args.min_mean_mbps
        print(f"[cohort] {int(keep.sum())}/{panel.L} links have mean >= "
              f"{args.min_mean_mbps:g} Mbps (dropped {int((~keep).sum())} idle/low)")
        panel = panel.subset(keep)
        if panel.L == 0:
            raise SystemExit("no links pass --min-mean-mbps; lower the threshold")
    if args.connected_links:
        panel = data_mod.select_connected(panel, args.connected_links)
    split = make_split(panel, args)
    print(f"[split] train={split.train} val={split.val} test={split.test} "
          f"purge={split.purge_steps} of n={panel.T}")

    horizons = {h: config.HORIZONS[h] for h in args.horizons}
    hsteps = tuple(sorted(horizons.values()))
    online_cfg = None
    if args.online:
        online_cfg = {"refit_every": args.refit_every, "window": args.online_window,
                      "epochs": args.online_epochs, "lr": args.online_lr}
        print(f"[online] refit_every={args.refit_every} window={args.online_window} "
              f"(expanding if None) epochs={args.online_epochs} lr={args.online_lr}")
    models = build_models(args.tiers, hsteps, args.seasonal_sarima, panel,
                          args.lookback, args.epochs, online_cfg=online_cfg,
                          max_train_samples=args.max_train_samples,
                          scale_weight=args.scale_weight,
                          under_weight=args.under_weight,
                          calendar=args.calendar)
    notes = apply_overrides(models, vars(args))
    if notes:
        print(f"[overrides] {', '.join(notes)}")
    if args.seed:
        for m in models:
            if hasattr(m, "under_weight"):   # torch tiers
                m.seed_offset = args.seed
        print(f"[seed] torch-tier seed offset {args.seed}")
    if args.only:
        keep = {n.strip() for n in args.only.split(",") if n.strip()}
        models = [m for m in models if m.name in keep]
        missing = keep - {m.name for m in models}
        if missing:
            print(f"[models] warning: --only names not found: {sorted(missing)}")
    if args.calendar:
        print("[features] calendar conditioning ON: dow one-hot + sin/cos tod + "
              "Swiss holidays; anchor vec + future-path code (torch tiers), "
              "holiday/workday columns (GBDT)")
    if args.under_weight != 1.0:
        print(f"[loss] asymmetric Huber: under-prediction weighted "
              f"{args.under_weight:g}x (torch tiers, train + val criterion)")
    if args.scale_weight:
        how = "w ~ log10(Mbps)" if args.scale_weight == "log" \
            else f"w ~ Mbps**{args.scale_weight:g}"
        print(f"[loss] scale-weighted training on ({how}) "
              f"for GRU/TCN/DLinear/PatchTST + STGNN (per-node)")
    if any(t in args.tiers for t in (4, 5, 6)):
        from . import torch_common as tc
        print(f"[device] torch tiers on: {tc.get_device()}")
    print(f"[models] {[m.name for m in models]}")

    result = harness.backtest(
        panel, models, split, horizons=horizons,
        stride=args.stride, max_origins=args.max_origins,
        out_dir=out, resume=args.resume,
    )
    if result.records.empty:
        print(f"\nno completed model results — partial state (checkpoints/records) "
              f"is in {out}; rerun with --resume to continue")
        return

    agg = result.aggregate()
    per_link = result.per_link()
    skill = result.skill_table(metric=args.metric)
    ratio = result.ratio_table(metric=args.metric, per_link=per_link)
    wape = result.wape_table(per_link)
    peaks = result.peak_table()

    agg.to_csv(out / "aggregate.csv", index=False)
    per_link.to_csv(out / "per_link.csv", index=False)
    skill.to_csv(out / f"skill_{args.metric}.csv")
    ratio.to_csv(out / f"ratio_vs_persistence_{args.metric}.csv", index=False)
    wape.to_csv(out / "wape.csv", index=False)
    peaks.to_csv(out / "peaks.csv", index=False)
    with open(out / "config.json", "w") as fh:
        json.dump(vars(args) | {"n_links": panel.L, "n_steps": panel.T,
                                "links": panel.links,
                                "data_source": config.describe_source()}, fh, indent=2)
    save_histories(models, out)

    pd.set_option("display.width", 140, "display.max_columns", 30)
    print("\n===== aggregate (pooled over links) =====")
    print(agg.to_string(index=False,
          columns=["model", "tier", "horizon_name", "mae", "rmse", "smape", "wape", "n"]))
    print(f"\n===== skill table ({args.metric}) =====")
    print(skill.to_string())
    if not ratio.empty:
        print(f"\n===== normalise-then-weight: per-link {args.metric} ratio vs "
              f"persistence (<1 beats it) =====")
        rw = ratio.pivot_table(index=["model", "tier"], columns="horizon_name",
                               values=["logv_ratio", "median_ratio", "geomean_ratio"])
        horder = [h for h in config.HORIZONS          # step order, incl. 8h
                  if h in ratio["horizon_name"].unique()]
        rw = rw.reindex(columns=pd.MultiIndex.from_product(
            [["logv_ratio", "median_ratio", "geomean_ratio"], horder]))
        print(rw.round(3).to_string())
    if not wape.empty:
        print("\n===== WAPE = sum|e|/sum|y| (%): pooled over links / median per-link "
              "=====")
        horder = [h for h in config.HORIZONS if h in wape["horizon_name"].unique()]
        ww = wape.pivot_table(index=["model", "tier"], columns="horizon_name",
                              values=["pooled_wape", "median_wape"])
        ww = ww.reindex(columns=pd.MultiIndex.from_product(
            [["pooled_wape", "median_wape"], horder]))
        print(ww.round(1).to_string())
    if not peaks.empty:
        print("\n===== peaks: top-decile actuals per link "
              "(bias% / underpred rate / WAPE%) =====")
        pw = peaks.assign(cell=peaks.peak_bias_pct.round(1).astype(str) + "% / " +
                          (peaks.peak_underpred_rate * 100).round(0).astype(int).astype(str) + "% / " +
                          peaks.peak_wape.round(1).astype(str) + "%")
        print(pw.pivot_table(index=["model", "tier"], columns="horizon_name",
                             values="cell", aggfunc="first").to_string())

    plot_skill(result, out, args.metric, "persistence")
    plot_overlays(result, models, panel, split, out,
                  horizon_steps=min(config.DAILY_STEPS, max(hsteps)))
    if result.interrupted:
        print(f"\npartial results (interrupted) -> {out} — checkpoints kept for --resume")
        return
    if args.zoo:
        # Before prune_checkpoints: the +online weights are dropped there as
        # warm-start duplicates, but an entry should still carry its own file.
        names = ([m.strip() for m in args.zoo_models.split(",")]
                 if args.zoo_models else None)
        print(f"\n[zoo] promoting -> {zoo_mod.ZOO_DIR}")
        try:
            zoo_mod.add(out, models=names, tag=args.zoo_tag, note=args.zoo_note,
                        with_records=args.zoo_records, force=True,
                        metric=args.metric)
        except SystemExit as exc:      # a zoo problem must not lose a good run
            print(f"[zoo] skipped: {exc}")
    prune_checkpoints(out)
    print(f"\ndone -> {out}")


if __name__ == "__main__":
    main()
