# `prediction/` — per-link ISP traffic forecasting

Implements the method ladder in
[per-link-traffic-prediction-roadmap.md](per-link-traffic-prediction-roadmap.md):
predict per-link traffic at 5-minute granularity, start with the dumbest baseline, and
escalate one tier at a time — **keeping a more complex model only when it clearly beats the
simpler one on the *same* chronological split and metric.**

Data comes from the [`dataset/`](../dataset/) package (the SWITCH Cricket CSVs at
`~/switch`, see `dataset/config.py:SWITCH_ROOT`); this package never re-implements
loading/coalescing.

## Environment

Use the dedicated **`green-pred`** conda env (Python 3.11 + pandas/numpy/sklearn/
statsmodels/matplotlib/scipy/torch-CUDA):

```bash
/home/qyy/anaconda3/envs/green-pred/bin/python    # or: conda run -n green-pred python
```

> **Clear `PYTHONPATH`** when invoking: this box sources ROS Noetic in `~/.bashrc`,
> which injects `/opt/ros/.../dist-packages` into every env's `sys.path` and can shadow
> the env's own numpy. Prefix commands with `env -u PYTHONPATH`.

> LightGBM/XGBoost are **not** installed — Tier 3 uses scikit-learn's
> `HistGradientBoostingRegressor` (the histogram-booster equivalent). Torch is the
> **CUDA** build (RTX 3060 Laptop, CUDA 12.1); the deep/GNN tiers (4–6) run on the GPU
> automatically. Override with `PRED_DEVICE=cpu` (or `cuda`). Tiers 1–3 are CPU-only by
> nature (naive/statsmodels/sklearn).

## Modules

| Module | Role |
|---|---|
| `data.py` | Build/cache a per-link **panel** `values[T, L]` on a common 5-min grid (uses `dataset.loaders`/`cohort`). Cached to `artifacts/panel_<hash>.npz`. |
| `metrics.py` | MAE, RMSE, sMAPE, WAPE, MASE — all in **original scale** — plus `skill = 1 − model/baseline`. |
| `harness.py` | **The testing loop**: rolling-origin backtest. `Forecaster` interface + `backtest()` + `BacktestResult` (per-link / aggregate / skill table). |
| `baselines.py` | **Tier 1** — persistence, seasonal-naive (day/week), historical average. |
| `models_classical.py` | **Tier 2** — Holt-Winters, SARIMA (per-link, log space, recent-window refit). |
| `models_gbdt.py` | **Tier 3** — global histogram gradient boosting; lag + rolling + calendar features, per-link id, direct multi-horizon. |
| `torch_common.py` | Shared trainer for torch tiers: global model, log1p, RevIN per window, Huber loss, early-stop. |
| `models_deep.py` | **Tier 4** GRU, TCN; **Tier 5** DLinear (linear control) + PatchTST-lite (patch transformer). |
| `models_gnn.py` | **Tier 6** STGNN (Graph-WaveNet-lite) over a link-level graph (shared/adjacent routers from `dataset.topology`) + learned adaptive adjacency. |
| `run.py` | CLI: build panel → run tiers → tabulate skill → plot. `--tiers 1 2 3 4 5 6`, `--lookback`, `--epochs`. |

## The testing loop (harness)

Every model — a `Forecaster` — is scored on **identical rolling origins**:

1. One chronological `train/val/test` split with a purge gap (`dataset.splits`).
2. Forecast origins across the test block (spaced by `--stride`, capped by `--max-origins`).
   At each origin `o`, a model may use only data `< o` and returns a path of `max(horizon)`
   steps. Point `path[h-1]` is scored against `actual[o+h-1]`.
3. Global models (`is_global=True`, e.g. GBDT) call `fit()` once on the train slice;
   local models (baselines, ARIMA) refit inside `predict`.
4. Metrics are computed in Mbps; **models own their `log1p`/normalise transforms**.

Result: `aggregate()` (pooled over links, per horizon), `per_link()`, and
`skill_table()` (skill vs persistence and seasonal-naive per horizon).

To add a model, subclass `Forecaster`, implement `predict(ctx)` (and `fit` if global),
and add it in `run.build_models`. It is automatically held to the same origins/metrics.

## Online adaptation (test-time fine-tuning)

Global models fit **once** on the train block and then carry frozen weights through the
whole test period — so at a September origin their parameters have never seen August,
even though the actual August traffic sits right there in the past. Local models
(persistence, SARIMA) don't have this handicap: they re-read the recent history at every
origin. **`--online`** removes the handicap for the global tiers too.

With `--online`, each global tier gets a second **`<name>+online`** variant that
periodically **re-fits / warm-start fine-tunes on the most recent data** as the test
block advances — GBDT re-fits on the extended window, torch tiers continue training the
existing net a few epochs at a low LR. Both variants run on the **identical** rolling
origins, so the table shows exactly *how much accuracy recent data buys*.

Leakage guarantee (unchanged): at origin `o`, `online_update` may train only on anchors
whose entire target path stays `< o` (`anchor + max_horizon ≤ o − 1`) — never on data at
or after the forecast origin. The harness enforces the cadence; the model enforces the
cutoff.

```bash
… --tiers 3 4 --online --refit-every 288 --online-window 4032 --online-epochs 3
```

Flags: `--online` (add the variants), `--refit-every` (steps between refits, default =
1 day / 288), `--online-window` (sliding train window in steps; omit for an *expanding*
window from train-start), `--online-epochs` / `--online-lr` (torch warm-start knobs).
Cost scales with the number of refits (`test_length / refit_every`), so a coarser
`--refit-every` trades adaptation speed for runtime.

## Run it

```bash
cd /home/qyy/Green_Routing/Green_Routing/code
env -u PYTHONPATH /home/qyy/anaconda3/envs/green-pred/bin/python -m prediction.run \
    --start 2024-06-01 --end 2024-10-01 \
    --train-end 2024-08-15 --val-start 2024-08-22 \
    --val-end 2024-09-05 --test-start 2024-09-12 \
    --tiers 1 3 --horizons 5min 1h day \
    --max-links 30 --max-origins 60 \
    --out ../artifacts/pred_demo
```

Key flags: `--tiers 1 2 3`, `--horizons {5min,1h,day,week}`, `--devices ag1 be1 …`
(restrict the cohort), `--min-coverage`, `--max-links`, `--stride`, `--max-origins`,
`--metric {mae,rmse,smape,wape}`, `--seasonal-sarima` (enable a true `s=288` SARIMA),
`--online` (add test-time-fine-tuned `+online` variants — see *Online adaptation*).

Writes to `--out`: `aggregate.csv`, `per_link.csv`, `skill_<metric>.csv`, `config.json`,
`skill_vs_persistence_<metric>.png`, `overlay_forecasts.png` (busy / bursty / quiet link
paths — the roadmap's "did it just learn to output the last value one step late?" check).

## What the demo run shows (2024-06 → 2024-10, 30 links)

| Horizon | Winner | Skill of GBDT vs persistence |
|---|---|---|
| 5 min (1-step) | **persistence** | −0.40 (can't beat it — as the roadmap warns) |
| 1 h (12-step) | **GBDT** | +0.28 |
| day (288-step) | **GBDT** | +0.32 |

Persistence is unbeatable at 1-step; the learned model earns its cost only as the horizon
grows — the roadmap's central thesis, reproduced on real SWITCH data.

## Deep/GNN results (2024-06→2024-10, 30 links, lookback 96, 8 epochs, CPU)

Skill vs persistence (MAE), short horizons:

| model | tier | 15 min | 30 min | 1 h |
|---|---|---|---|---|
| **patchtst** | 5 | **+0.16** | **+0.22** | +0.27 |
| gbdt | 3 | +0.06 | +0.11 | **+0.27** |
| stgnn | 6 | +0.08 | +0.14 | +0.22 |
| gru | 4 | +0.01 | +0.10 | +0.14 |
| tcn | 4 | +0.05 | +0.08 | +0.06 |
| dlinear | 5 | −0.03 | −0.04 | −0.04 |

Read through the roadmap's "keep complexity only if it *clearly* beats the simpler tier"
rule: **PatchTST earns its cost at 15/30 min** (clearly > GBDT), but at **1 h it only ties
the far-cheaper GBDT** → ship GBDT there. **STGNN** beats persistence and GRU/TCN but not
GBDT on this mostly-*access*-link cohort at short horizons — the spatial/reroute signal a
graph exploits needs core links and/or longer horizons to pay off. DLinear (the linear
control) underperforming confirms PatchTST's gain is real, not an artifact.

These are small CPU nets at short horizons (where the roadmap expects deep models to be
*least* advantaged). Widen the lead by: longer horizons (`--horizons day week`), longer
`--lookback`, more `--epochs`, a core-link cohort, or bigger nets.

## Frontier (not built)

Tier-5 iTransformer/Informer, Tier-6 DCRNN/MTGNN, foundation models (TimesFM/Chronos
zero-shot), diffusion (CRPS). Each must clear the same bar: beat the cheapest tier that
already wins at that horizon, on these identical origins.
```


```
env -u PYTHONPATH PRED_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/qyy/anaconda3/envs/green-pred/bin/python -m prediction.run \
    --start 2025-12-01 --end 2026-07-11 \
    --train-end 2026-05-01 --val-start 2026-05-08 \
    --val-end 2026-05-29 --test-start 2026-06-05 \
    --tiers 1 3 4 5 6 \
    --horizons 15min 1h day week \
    --max-links 20 --max-origins 48 \
    --lookback 336 --epochs 12 \
    --out ../artifacts/pred_real 2>&1 | tee ../artifacts/pred_real/run.log
```