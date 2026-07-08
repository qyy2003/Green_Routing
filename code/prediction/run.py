"""Run the method ladder end-to-end and report skill vs. the naive baselines.

    cd /home/yuyqin/ETH_Master_Study/Green_Routing/Green_Routing/code
    /home/yuyqin/anaconda3/bin/python -m prediction.run \
        --start 2024-06-01 --end 2024-10-01 \
        --train-end 2024-08-15 --val-start 2024-08-22 \
        --val-end 2024-09-05 --test-start 2024-09-12 \
        --tiers 1 3 --horizons 5min 1h day \
        --max-links 30 --out ../artifacts/pred_demo

Writes to --out: aggregate.csv, per_link.csv, skill_<metric>.csv, and PNG plots
(predicted-vs-actual overlays + a skill bar). Everything is scored on the identical
rolling origins, so "model X beats persistence by Y%" is an honest number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import config, splits as splits_mod

from . import baselines, data as data_mod, harness
from .models_classical import default_classical
from .models_gbdt import GlobalGBDT


def build_models(tiers, hsteps, seasonal_sarima, panel, lookback, epochs):
    """Assemble the requested tiers. Tiers 4-6 import torch lazily (only if requested)."""
    models: list[harness.Forecaster] = []
    if 1 in tiers:
        models += baselines.default_baselines()
    if 2 in tiers:
        models += default_classical(seasonal_sarima=seasonal_sarima)
    if 3 in tiers:
        models.append(GlobalGBDT(horizons=hsteps))
    if 4 in tiers:
        from .models_deep import GRUForecaster, TCNForecaster
        models += [GRUForecaster(lookback=lookback, horizons=hsteps, epochs=epochs),
                   TCNForecaster(lookback=lookback, horizons=hsteps, epochs=epochs)]
    if 5 in tiers:
        from .models_deep import DLinearForecaster, PatchTSTForecaster
        models += [DLinearForecaster(lookback=lookback, horizons=hsteps, epochs=epochs),
                   PatchTSTForecaster(lookback=lookback, horizons=hsteps, epochs=epochs)]
    if 6 in tiers:
        from .models_gnn import STGNNForecaster
        models.append(STGNNForecaster(panel.devices, lookback=lookback,
                                      horizons=hsteps, epochs=epochs))
    return models


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


def plot_skill(result, out: Path, metric: str, baseline: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = result.aggregate()
    if (agg.model == baseline).sum() == 0:
        return
    base = agg[agg.model == baseline].set_index("horizon_name")[metric]
    piv = agg.pivot_table(index="model", columns="horizon_name", values=metric)
    order = [result._h_name[v] for v in sorted(result.horizons.values())
             if result._h_name[v] in piv.columns]
    piv = piv[order]
    sk = 1.0 - piv.div(base[order], axis=1)
    sk = sk.drop(index=[b for b in (baseline,) if b in sk.index])
    ax = sk.plot(kind="bar", figsize=(10, 5))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"skill vs {baseline}  (1 - {metric}/base)")
    ax.set_title(f"Skill over {baseline} by horizon ({metric})")
    ax.legend(title="horizon", fontsize=8)
    ax.figure.tight_layout()
    p = out / f"skill_vs_{baseline}_{metric}.png"
    ax.figure.savefig(p, dpi=110)
    import matplotlib.pyplot as _plt
    _plt.close(ax.figure)
    print(f"[plot] {p}")


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
    ap.add_argument("--max-links", type=int, default=30)
    ap.add_argument("--devices", nargs="*", default=None,
                    help="short device names to restrict the cohort (e.g. ag1 be1)")
    ap.add_argument("--tiers", nargs="+", type=int, default=[1, 3],
                    help="1=baselines 2=classical 3=gbdt 4=gru/tcn 5=dlinear/patchtst 6=stgnn")
    ap.add_argument("--seasonal-sarima", action="store_true")
    ap.add_argument("--lookback", type=int, default=config.DAILY_STEPS,
                    help="input window for deep/GNN tiers (steps)")
    ap.add_argument("--epochs", type=int, default=12, help="deep/GNN training epochs")
    ap.add_argument("--horizons", nargs="+", default=["5min", "1h", "day"],
                    choices=list(config.HORIZONS))
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max-origins", type=int, default=120)
    ap.add_argument("--metric", default="mae",
                    choices=["mae", "rmse", "smape", "wape"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = data_mod.build_panel(
        args.start, args.end, value_col=args.value_col,
        min_coverage=args.min_coverage, max_links=args.max_links,
        devices=args.devices,
    )
    split = make_split(panel, args)
    print(f"[split] train={split.train} val={split.val} test={split.test} "
          f"purge={split.purge_steps} of n={panel.T}")

    horizons = {h: config.HORIZONS[h] for h in args.horizons}
    hsteps = tuple(sorted(horizons.values()))
    models = build_models(args.tiers, hsteps, args.seasonal_sarima, panel,
                          args.lookback, args.epochs)
    print(f"[models] {[m.name for m in models]}")

    result = harness.backtest(
        panel, models, split, horizons=horizons,
        stride=args.stride, max_origins=args.max_origins,
    )

    agg = result.aggregate()
    per_link = result.per_link()
    skill = result.skill_table(metric=args.metric)

    agg.to_csv(out / "aggregate.csv", index=False)
    per_link.to_csv(out / "per_link.csv", index=False)
    skill.to_csv(out / f"skill_{args.metric}.csv")
    with open(out / "config.json", "w") as fh:
        json.dump(vars(args) | {"n_links": panel.L, "n_steps": panel.T,
                                "links": panel.links}, fh, indent=2)

    pd.set_option("display.width", 140, "display.max_columns", 30)
    print("\n===== aggregate (pooled over links) =====")
    print(agg.to_string(index=False,
          columns=["model", "tier", "horizon_name", "mae", "rmse", "smape", "wape", "n"]))
    print(f"\n===== skill table ({args.metric}) =====")
    print(skill.to_string())

    plot_skill(result, out, args.metric, "persistence")
    plot_overlays(result, models, panel, split, out,
                  horizon_steps=min(config.DAILY_STEPS, max(hsteps)))
    print(f"\ndone -> {out}")


if __name__ == "__main__":
    main()
