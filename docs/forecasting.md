# Forecasting: models, harness, logs, metrics

`green_routing.prediction` forecasts each link's traffic (Mbit/s, 5-minute steps) from
15 minutes to a week ahead, with a ladder of models from "repeat the last value" to
transformers and a graph network. One rule runs through it ([roadmap.md](roadmap.md)):
**a more complex model stays only if it clearly beats the simpler one on the same split,
the same origins and the same metric.**

- `gr-benchmark` trains the chosen tiers and backtests them on identical rolling origins.
- `gr-dense` re-runs the fitted models at every 5-minute origin and scores the full log.
- `gr-viewer` shows any run link by link ([viewers.md](viewers.md)).
- `gr-zoo` keeps good models with their provenance.

## A first run

Baselines and GBDT on a small cohort — minutes on a CPU once the data cache is warm:

```bash
gr-benchmark --start 2026-01-01 --end 2026-04-01 \
    --train-end 2026-02-20 --val-start 2026-02-27 --val-end 2026-03-06 \
    --test-start 2026-03-13 --tiers 1 3 --horizons 1h 8h day \
    --max-links 30 --max-origins 60 --out artifacts/demo
```

Add `--tiers 1 3 4 5` for the neural models (GPU), `--tiers ... 6` for Graph WaveNet.
The full year-scale configuration is in [`recipes/year/flags.sh`](../recipes/year/flags.sh)
and [reproduce.md](reproduce.md).

## The models

| Tier | Model | Design | Module |
|:--:|---|---|---|
| 1 | persistence, seasonal-naive (day, week), historical average | no learning; the bar everything else is scored against | `baselines.py` |
| 2 | Holt–Winters, SARIMA | per link, log space, recent-window refit (diverges at s = 288 on this data) | `models_classical.py` |
| 3 | GBDT | one global `HistGradientBoostingRegressor` per horizon (direct); lags 5 min–1 day, 1-h / 1-day rolling mean & std, time of day, day of week, weekend (+ holiday/workday with `--calendar`), the link's train-window level as identity | `models_gbdt.py` |
| 4 | GRU | 2 × 128, last state + link embedding → linear head over the whole path | `models_deep.py` |
| 4 | TCN | Bai et al. residual blocks, 64 channels, kernel 3, depth sized so the receptive field covers the lookback, capped at `--tcn-max-levels` (9 = 2,045 steps) | `models_deep.py` |
| 5 | DLinear | trend/seasonal split + two linear maps; no embedding, no calendar — the linear control | `models_deep.py` |
| 5 | PatchTST | patches 16/8, d = 128, 16 heads, 3 layers, low-rank flatten head (→ 384 → path, ~13 M params); skips `--calendar` (it hurt in a seed-clean ablation) | `models_deep.py` |
| 6 | Graph WaveNet | gated dilated convs, order-2 diffusion over the router graph + adaptive adjacency | `models_gnn.py` |

All neural models are **global** (one network for every link, with a learned 16-d link
embedding except DLinear) and **direct multi-horizon** (one forward pass emits the whole
path at 5-minute resolution). Each input window is `log1p`-transformed and normalised by
its own mean and spread (RevIN; spread floored at 0.05, targets clipped at ±20), so a
100 Gbit/s core port and a 1 Mbit/s access port share one model.

### Training (`torch_common.py`)

- **Data**: up to `--max-train-samples` windows per epoch (default 100,000, spread evenly
  over links), **re-drawn every epoch** from the train block; validation windows are
  anchored in the val block (targets strictly after every training target).
- **Loss**: Huber on the normalised path, weighted 1/√lead (near leads matter more).
  `--under-weight W` multiplies the loss of under-forecast steps by W (peak-aware
  training; the year runs use 2).
- **Model selection**: validation loss at the scored horizons only, equally weighted (with
  1/√lead weighting, two TCNs whose 1-day error differed by 13 % looked identical).
- **Schedule**: Adam (lr 1e-3; PatchTST 1e-4), ReduceLROnPlateau (×0.3, patience 3), early
  stop after 10 bad epochs but only once the LR was cut twice; gradient clip 5; best
  checkpoint restored.
- **Batch**: 256 × 672 / lookback (85 for a 1-week window), floored at 32; pin it with
  `--batch-size` when comparing lookbacks.
- **Seeds**: per model name, so results don't depend on training order; `--seed N`
  offsets them for multi-seed runs.
- **Calendar** (`--calendar`): day-of-week, time-of-day and Swiss public holidays as
  known-future covariates — at the origin and along the forecast path (hourly).
  The holiday table covers 2025-06 → 2026-08; outside it `build_calendar` refuses
  (extend `HOLIDAYS` in `calendar_features.py`).
- **Safety**: an optimiser step with a non-finite gradient is skipped and counted; any
  skipped step prints a warning. (TCN weight-norm direction vectors are excluded from
  weight decay: on dead ReLU channels decay collapsed them until every gradient was NaN
  and training silently stopped — see the knowledge base.)
- **Resumable**: full training state is saved after every epoch
  (`checkpoints/<model>.train.pt`); rerunning the same command continues from it.

## The harness (`harness.py`)

Every model is a `Forecaster`; `backtest()` scores them all on **identical rolling
origins** in the test block:

1. One chronological train / val / test split from the CLI dates (gaps = purge).
2. Origins every `--stride` steps (default 72 = 6 h), evenly subsampled to
   `--max-origins`. At origin `o` a model sees only data `< o` and returns a path;
   `path[h-1]` is scored against `actual[o+h-1]`.
3. Global models `fit()` once on the train block; local ones work inside `predict`.
4. Metrics in Mbit/s; models own their transforms.

A model may implement `predict_batch(origin, horizon, n_links)` to forecast every link at
an origin in one call (GBDT does: identical numbers, ~200× faster than one row per call).

**Online adaptation** (`--online`): each global model gets a `<name>+online` twin that
re-fits (GBDT) or fine-tunes at a low LR (networks) on data strictly before each origin,
every `--refit-every` steps (default 1 day), on the same origins — the value of recent
data, measured. The twin starts from a copy of its frozen model instead of retraining.

**Adding a model**: subclass `Forecaster`, implement `predict(ctx)` (+ `fit` if global),
add it in `run.build_models`. It is then held to the same origins and metrics.

## `gr-benchmark` options

| Group | Options |
|---|---|
| window & cohort | `--start --end` (panel window), `--min-coverage 0.9`, `--min-mean-mbps` (drop idle links; 1 in the year runs), `--max-links N` (quick tests only — truncates alphabetically), `--devices ag1 be1 …`, `--connected-links N` (a connected sub-network, for the graph model), `--value-col total_mbps` |
| split | `--train-end --val-start --val-end --test-start` |
| evaluation | `--horizons` (`5min 15min 30min 1h 8h day week`), `--max-origins`, `--stride`, `--metric mae` |
| models | `--tiers 1 2 3 4 5 6`, `--only name,name` (keep `persistence` for the ratio tables), `--seasonal-sarima` |
| training | `--lookback` (steps; 2016 = 1 week), `--epochs`, `--max-train-samples`, `--batch-size`, `--under-weight`, `--calendar`, `--scale-weight {exp,log}`, `--seed` |
| architecture knobs | `--tcn-max-levels`, `--patchtst-patch`, `--patchtst-stride`, `--gbdt-context {day,month}` (month adds 1–4-week lags and 1-week/30-day rolling features), `--target-smooth N` (train on an N-step centred moving average of the target) |
| online | `--online --refit-every --online-window --online-epochs --online-lr` |
| run management | `--out`, `--resume`, `--zoo` (+ `--zoo-models --zoo-tag --zoo-note --zoo-records`) |

### What a run writes (`--out`)

| File | Content |
|---|---|
| `config.json` | every flag + the link list + the data source actually read |
| `run.log` | the full console output (training curves included) |
| `records/<model>.csv` | every scored forecast: link, horizon, origin, y_true, y_pred (flushed per model) |
| `checkpoints/` | fitted weights (`.pt` state dicts, `gbdt.pkl`) |
| `history/` | per-epoch curves and training summaries |
| `aggregate.csv` | pooled MAE, RMSE, sMAPE, WAPE per model × horizon |
| `per_link.csv` | the same per link, + MASE |
| `ratio_vs_persistence_mae.csv` | per-link MAE ÷ persistence MAE, summarised (see metrics) |
| `wape.csv`, `peaks.csv`, `skill_mae.csv` | WAPE views, top-decile behaviour, skill tables |
| `overlay_forecasts.png`, `skill_vs_persistence_mae.png` | sanity plots |

Ctrl-C is safe: finished models keep their records and weights; `--resume` skips them,
reloads checkpoints, and only rebuilds the tables when everything is done.

## Metrics

Everything is in Mbit/s on the original scale. Raw pooled errors are dominated by the
busiest links (traffic spans four orders of magnitude), so the headline is per link:

| Metric | Definition | Reads as |
|---|---|---|
| **median ratio** | median over links of MAE(model) / MAE(persistence) | the typical link; < 1 beats persistence; the report's headline |
| geomean / logv ratio | geometric mean; mean weighted by log₁₀(mean Mbit/s) | robustness checks |
| share beating | fraction of links with ratio < 1 | how broad a gain is |
| pooled MAE | mean absolute error over all points of all links | volume-weighted |
| WAPE | Σ\|e\| / Σ\|y\| in % — pooled, or per link then median/p90 | error as a share of traffic |
| hourly / daily-volume WAPE | WAPE of hourly / daily *totals* (dense logs) | what a carbon bill sums |
| roughness | mean \|Δforecast\| / mean \|Δactual\| between consecutive steps (dense logs) | 1 = as jagged as the traffic |
| peak bias / peak WAPE | signed error / WAPE on each link's top-decile actuals | peak shaving |
| skill | 1 − error / baseline error | pooled improvement |

Per-link comparisons between two models are paired on identical targets: the median of
the per-link ratio, the share of links that improve, and a Wilcoxon signed-rank test
(`recipes/window_smoothing/analyze.py`).

## Dense prediction logs (`gr-dense`)

The harness scores a sample of origins — and on the year run its 160 origins (a 6-h grid
subsampled to 18-h spacing) fall at only **four times of day** (≈ 02, 08, 14, 20 h local).
`gr-dense` re-runs a finished run's checkpoints at **every** 5-minute origin of the test
block, batched on the GPU with exactly each model's `predict` logic, and logs two layouts
per model, indexed by target time:

- **rolling** — the forecast for *t* made exactly *h* earlier (the harness's horizon
  definition, at every origin);
- **block** — window by window: for horizon *H* the test block is cut into consecutive
  *H*-step windows, each forecast in full from its start, so every 5-minute step has a
  forecast; metrics average over leads 1…*H*. GBDT predicts horizons directly and has no
  path, hence no block log.

```bash
gr-dense predict --run RUN [--models tcn,gru] [--batch 1024]   # -> RUN/dense/pred_<mode>_<model>.npy
gr-dense verify  --run RUN                                      # vs RUN/records: max/p99 relative diff
gr-dense metrics --run RUN --common                             # -> RUN/dense/<mode>/*.csv
```

| Option | Meaning |
|---|---|
| `--tag @month` | log a *variant* run into another run's dense dir (`--out RUN/dense`) under the name `<model>@month` |
| `--smooth 3,12,36 [--skip-raw]` | also log the paths smoothed by centred moving averages (`<model>+ma15m`, `+ma1h`, `+ma3h`); every value from the same origin |
| `--origin-stride 3` | forecast from every 3rd origin: block logs stay exact (every horizon is a multiple of 3), rolling logs score every 15 min |
| `--tf32` | allow TF32 (faster; ~3–7 × 10⁻⁴ relative drift). Default is full fp32, which matches the harness to ~10⁻⁶ |
| `metrics --common` | score every model of a mode on the targets all of them forecast |
| `metrics --models untagged` / list | score a subset (`untagged` = the run's own models, no variants) |
| `metrics --tables-dir DIR` | write the tables elsewhere (the viewer reads `RUN/dense/<mode>/`) |

Each log is float32 `[links, horizons, targets]`, ~340 MB for 534 links × 111 days.
On the L40S: baselines and DLinear take seconds, GRU and PatchTST ~14 min, TCN ~40 min,
GBDT a few minutes (CPU). `verify` on the year run: baselines and GBDT exact, DLinear
~3 × 10⁻⁶, GRU/TCN/PatchTST p99 ~2 × 10⁻⁴ (cuDNN picks different kernels at batch 1 and
batch 1024).

## Model zoo (`gr-zoo`)

A run directory is working state. A model worth keeping goes to
`$GR_ARTIFACTS_DIR/zoo/<model>/<tag>/` (`GR_ZOO_DIR` moves it) with its weights (+
sha256), learning curve, log slice, metrics, the run's flags and a manifest: data window,
data source, git commit and dirty flag, environment, training summary.

```bash
gr-zoo add artifacts/pred_year/merged --models tcn,gbdt --tag year-2026-10 --note "reference run"
gr-zoo list
gr-zoo show tcn/year-2026-10
gr-zoo pack tcn/year-2026-10        # -> .tar.gz and the upload command (it never uploads)
```

## Evaluation pitfalls learned the hard way

- **Sampled origins are time-of-day biased** (above); trust the dense rolling numbers.
- **Validation losses are not comparable across lookbacks** — each window is normalised by
  its own statistics. Compare test error in Mbit/s on identical targets.
- **Pooled MAE is a vote of the biggest links**; lead with per-link ratios.
- **A run that changes the window changes the cohort** (coverage and the 1 Mbit/s floor
  are computed over the window). Compare runs on the same window.
- **Month-window models need a gap-free 30-day history**: they forecast ~0.06 % fewer
  points; `--common` scoring handles it.

## Not built yet

iTransformer / Informer, DCRNN / MTGNN, foundation models (TimesFM, Chronos) zero-shot,
diffusion forecasts scored by CRPS, per-link fine-tuning at scale. Each has to beat the
cheapest tier that already wins at its horizon, on these origins.
