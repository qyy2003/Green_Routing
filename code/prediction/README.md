# `prediction/` — per-link ISP traffic forecasting

Implements the method ladder in
[per-link-traffic-prediction-roadmap.md](per-link-traffic-prediction-roadmap.md):
predict per-link traffic at 5-minute granularity, start with the dumbest baseline, and
escalate one tier at a time — **keeping a more complex model only when it clearly beats the
simpler one on the *same* chronological split and metric.**

Data comes from the [`dataset/`](../dataset/) package, which reads the SWITCH Cricket
CSVs **straight out of SWITCHengines object storage** (`s3://switchlan-load-timeseries`,
see [`dataset/README.md`](../dataset/README.md#data-source) for the env vars and
`python -m dataset check`); a local mirror is opt-in via `GR_DATA_SOURCE=local`. This
package never re-implements loading/coalescing. `build_panel` warms the object cache in
parallel before its scan, so the first run pays the download and later runs read from
local disk.

**Link census** (scan 2026-07-13, details + CSVs in `artifacts/link_census/`): the raw
tree has 1 923 interface series on 208 devices; over the working window
2025-12-01 → 2026-07-11, **886 links on 141 devices are usable** (coverage ≥ 0.9 on the
5-min grid; 830 are ≥ 0.99). The panel therefore keeps **all usable links by default** —
`--max-links` is only a quick-experiment cap (it truncates in alphabetical device
order, so a capped cohort is small *and* biased). For comparison, the router-level
topology has 395 router-pair links over 85 nodes.

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

### Model sizes (paper-scale, GPU)

The torch tiers default to their published configurations: GRU 2×128; TCN with
Bai-et-al. residual blocks (64 ch, depth auto-sized so the receptive field covers the
lookback); PatchTST at the paper's supervised encoder config (patch 16 / stride 8,
d_model 128, 16 heads, 3 layers, d_ff 256, dropout 0.2, lr 1e-4) but with a
**low-rank flatten head** (flatten → 384-d bottleneck → h, ~13 M params): the paper's
flatten head (n_patch·d→h, ~65 M at week horizons) memorised the train regime by
epoch 2 on the 547-link cohort, and its attention-pooled replacement (d→h, ~0.3 M)
over-corrected — squeezing 251 patch tokens through one 128-d vector made it the
worst learned tier on pred_547d (lost to GRU on 94% of links); Graph WaveNet at its paper design
(gated dilated convs 32 ch, skip 256 / end 512, order-2 diffusion over fixed + adaptive
adjacency, dropout 0.3 — input truncated to the receptive field, as GWN is a
short-window model). DLinear is unchanged: it is linear by design (the control).
Training-window budget defaults to 100k (`--max-train-samples`); batches shrink
automatically as `--lookback` grows (and, for the STGNN, as the node count grows) so a
week-long window over hundreds of links fits a 6 GB GPU — the STGNN also
gradient-checkpoints its layers to bound activation memory. LR is cosine-annealed over
the run, which stabilises the GRU (its constant-LR curve had late-training blow-ups).

**Peak-aware training (`--under-weight`)**: every learned tier under-predicted each
link's top-decile traffic on pred_547d (−36…−46% signed error at day/week vs −28% for
seasonal-naive-week — the irreducible regression-to-the-mean part). `--under-weight W`
multiplies the Huber loss of *under-predicted* steps by `W` for the torch tiers, in
the train **and** val criterion, moving the point forecast above the conditional
median. `W≈2` is a sensible start; the cost is mild off-peak over-prediction. Judge
the trade in `peaks.csv` (peak bias %, underprediction rate, peak MAE, off-peak bias —
per model × horizon; thresholds are per-link q90 of the identically-scored targets,
so the naive rows give the irreducible floor).

## Modules

| Module | Role |
|---|---|
| `data.py` | Build/cache a per-link **panel** `values[T, L]` on a common 5-min grid (uses `dataset.loaders`/`cohort`). Cached to `artifacts/panel_<hash>.npz`. |
| `metrics.py` | MAE, RMSE, sMAPE, WAPE, MASE — all in **original scale** — plus `skill = 1 − model/baseline`. |
| `harness.py` | **The testing loop**: rolling-origin backtest. `Forecaster` interface + `backtest()` + `BacktestResult` (per-link / aggregate / skill table). |
| `baselines.py` | **Tier 1** — persistence, seasonal-naive (day/week), historical average. |
| `models_classical.py` | **Tier 2** — Holt-Winters, SARIMA (per-link, log space, recent-window refit). |
| `models_gbdt.py` | **Tier 3** — global histogram gradient boosting; lag + rolling + calendar features, per-link id, direct multi-horizon. |
| `torch_common.py` | Shared trainer for torch tiers: global model, log1p, RevIN per window (std floored at 0.05, targets clipped ±20), Huber loss **horizon-weighted 1/√h**, **per-epoch anchor resampling** (fresh windows every epoch), **ReduceLROnPlateau** on the chronological val loss with LR-gated early stopping, grad clip, CPU→GPU batch streaming (OOM-safe at week horizons). GRU/TCN/PatchTST condition their heads on a **learned link embedding** (DLinear stays the pure-linear control). |
| `models_deep.py` | **Tier 4** GRU (2×128), residual TCN (Bai et al., 64ch, RF-covering depth); **Tier 5** DLinear (linear control) + PatchTST at the paper's supervised config (patch 16/stride 8, d128, 16 heads, 3 layers). |
| `models_gnn.py` | **Tier 6** Graph WaveNet at its paper design (gated dilated convs, skip 256/end 512, order-2 diffusion, adaptive adjacency) over a link-level graph (shared/adjacent routers from `dataset.topology`). |
| `run.py` | CLI: build panel → run tiers → tabulate skill → plot. `--tiers 1 2 3 4 5 6`, `--lookback`, `--epochs`. Tees the console to `<out>/run.log` and writes per-epoch curves to `<out>/history/`. |
| `zoo.py` | **Model zoo**: promote the models worth keeping out of a run directory into `artifacts/zoo/<model>/<tag>/` — weights + learning curve + that model's log slice + metrics + a provenance manifest. CLI: `add`, `list`, `show`, `pack`, `remove`, `index`. |

## The testing loop (harness)

Every model — a `Forecaster` — is scored on **identical rolling origins**:

1. One chronological `train/val/test` split (`dataset.splits`). The gap between
   blocks is whatever the CLI dates encode — passing `--val-start` == `--train-end`
   and `--test-start` == `--val-end` gives a standard contiguous split (purge 0).
   The deep tiers early-stop and select checkpoints on windows anchored **in the
   val block** (context may reach back into train — real history; targets stay
   inside val), so model selection measures forward generalisation.
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

The `+online` variant's *initial* fit is by construction identical to its frozen twin's,
so it **warm-starts from the twin's fitted state** (a deep copy — fine-tuning never
bleeds back into the frozen model) instead of training a second time. `--online`
therefore adds only the refit cost, not a duplicate training run.

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
(restrict the cohort), `--min-coverage` (default 0.9), `--max-links` (**default: none —
the panel keeps every usable link**, i.e. all series with coverage ≥ `--min-coverage`
over the window; set a number only for quick experiments — it truncates at the first N
links in alphabetical device order), `--stride`, `--max-origins`,
`--max-train-samples` (deep-tier data budget), `--lookback`, `--epochs`,
`--min-mean-mbps` (**cohort traffic floor** — drop links whose mean throughput over the
window is below this; `--min-mean-mbps 1` removes idle ports that otherwise subsidise
scale-free metrics, leaving 547 of 886 links), `--connected-links N` (keep ~N links whose
host routers form a **connected** sub-network — BFS over the topology from the busiest
router, spread across ~6 routers — for a quick run where the STGNN graph is a real slice
of the network; applied after `--min-mean-mbps`), `--scale-weight` (deep-tier loss
weight from per-link train-mean traffic: a float exponent — 0 = uniform/RevIN,
1 = weight ~ Mbps to align the fit with pooled MAE — or `log` = weight ~ log10(Mbps),
the training-side mirror of the log-volume-weighted ratio metric; applies to
GRU/TCN/DLinear/PatchTST and, per-node, to the STGNN),
`--metric {mae,rmse,smape,wape}`, `--seasonal-sarima` (enable a true `s=288` SARIMA),
`--zoo` (+ `--zoo-models/--zoo-tag/--zoo-note/--zoo-records`, see [Model zoo](#model-zoo--keeping-the-good-runs)),
`--online` (add test-time-fine-tuned `+online` variants — see *Online adaptation*),
`--resume` (continue an interrupted run — see *Checkpoints*).

Writes to `--out`: `aggregate.csv`, `per_link.csv`, `skill_<metric>.csv`,
`ratio_vs_persistence_<metric>.csv` (normalise-then-weight table: per-link error
ratio vs persistence aggregated as median / geometric mean / log10(volume)-weighted
mean — the scale-domination-free leaderboard; <1 beats persistence — plus the
per-link median WAPE and the share of links beating persistence), `wape.csv`
(WAPE = Σ|e|/Σ|y| in %, per model × horizon: pooled over all links, median and p90
of the per-link WAPEs — pooled WAPE ranks models exactly like pooled MAE on these
identical targets, so the per-link median is the new information), `peaks.csv`
(peak bias / underprediction rate / peak MAE / peak WAPE per model × horizon — see
*Peak-aware training*), `config.json`,
`skill_vs_persistence_<metric>.png`, `overlay_forecasts.png` (busy / bursty / quiet link
paths — the roadmap's "did it just learn to output the last value one step late?" check).

Deep-tier epoch lines log both losses (`[gru] epoch 3/100 train=0.171 val=0.254 …`),
where `val` is the **chronological** val-block loss (fit-time model selection),
evaluated **at the scored horizons only, equally weighted** — training keeps dense
1/√h path supervision, but selection must see day/week quality (with the 1/√h val,
two TCNs whose day-horizon test MAE differed by 13% were indistinguishable).
Torch tiers are seeded **per model name** (`torch_common.seed_model`), so results
are bit-reproducible and independent of training order; `--seed N` offsets all
torch-tier seeds for multi-seed error bars. `--only name,name` runs a subset of the
assembled models (keep `persistence` for the skill/ratio tables). `--calendar`
skips PatchTST by design (seed-clean ablation showed net harm — see
`PatchTSTForecaster.calendar_opt_out`);
with the RevIN variance floor (std floored at 0.05 in log1p space, targets clipped to
±20) healthy epoch-1 val is ~0.3–0.7 — a val stuck at 10+ means degenerate windows are
back. Online fine-tunes (`--online`) have no future block to validate on and fall back
to a random train-window split as a stability guard.

### Checkpoints, incremental results, and `--resume`

Long runs are safe to interrupt:

* **`<out>/checkpoints/`** — every fitted global model is saved right after training
  (torch tiers as CPU `state_dict` `.pt`, GBDT as pickled boosters `.pkl`), so trained
  weights survive a Ctrl-C even mid-backtest. Once a run **completes** and the results
  are recorded, the `+online` checkpoints are pruned automatically — they are
  warm-start duplicates of the frozen twin's trained weights (never the fine-tuned
  state), so only one trained checkpoint per model is kept; every delete/keep is
  appended to `<out>/cleanup.log`. Interrupted runs prune nothing.
* **`<out>/records/<model>.csv`** — each model's raw predictions are flushed as soon as
  its backtest finishes. A Ctrl-C drops only the model currently running and still
  reports/plots the finished ones.
* **`--resume`** — rerun the same command with this flag to continue: models with cached
  records are not re-run, and pending models reload their checkpoint instead of
  retraining (a changed `--lookback`/cohort invalidates the checkpoint shape, which
  falls back to a fresh fit). Use a fresh `--out` (or drop the flag) when the config
  changes in ways that alter what the cached records mean.

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

## Dense prediction log (`dense.py`) — every 5-min step

The harness scores a *sample* of origins (`--max-origins`), and the sample is
time-of-day biased: on pred_year the 160 origins (6-h grid, subsampled to 18 h
spacing) fall at only four local times (~02/08/14/20 h). `prediction.dense` re-runs
a finished run's checkpoints at **every** 5-min origin of the test block (batched
on the GPU, mirroring each model's `predict`; full fp32 — TF32 drifted ~3e-4) and
logs two layouts per model, indexed by target time:

* **rolling** — fixed lead: the forecast for t made exactly h steps earlier
  (the harness's horizon definition, at all ~32k origins);
* **block** — window by window: consecutive h-step windows, each forecast in full
  from its start (a prediction at every 5-min step; metrics average leads 1..h).
  GBDT is direct-per-horizon, so it has no block log.

```bash
python -m prediction.dense predict --run ../artifacts/pred_year/merged --models tcn --batch 1024
python -m prediction.dense verify  --run ../artifacts/pred_year/merged   # vs records/
python -m prediction.dense metrics --run ../artifacts/pred_year/merged   # -> dense/<mode>/*.csv
```

Output in `<run>/dense/`: `pred_<mode>_<model>.npy` (float32 [L, H, n_targets],
~340 MB each), `meta.json`, and per mode the harness's tables (`aggregate`,
`per_link`, `ratio_vs_persistence_mae`, `wape`, `peaks`). `web_pred/` serves all
three views. Cost on an L40S: baselines/DLinear seconds, GRU/PatchTST ~14 min,
TCN ~40 min, GBDT ~48 min (CPU). `verify` on pred_year: baselines/GBDT exact,
DLinear ~3e-6, GRU/TCN/PatchTST p99 ~2e-4 relative (cuDNN picks different kernels
at batch 1 vs 1024).

## Model zoo — keeping the good runs

A run directory is *working state*: it mixes every model together, is named after an
experiment rather than a result, and gets pruned (`prune_checkpoints`) and overwritten.
The zoo is where a run goes when it turns out well. Each promoted model gets its own
immutable directory:

```
artifacts/zoo/<model>/<tag>/
    weights.pt | weights.pkl     the fitted model (sha256 in the manifest)
    history.csv                  per-epoch train/val loss, lr, seconds (torch tiers)
    train.log                    this model's slice of the run's console log
    aggregate.csv per_link.csv   its metrics, sliced out of the run tables
    ratio_vs_persistence.csv peaks.csv
    run_config.json              the exact flags the run was invoked with
    records.csv.gz               raw per-point predictions (only with --with-records)
    manifest.json                everything above plus provenance (below)
```

`manifest.json` is what makes an entry worth uploading on its own: the data window and
**the source the run actually read** (S3 vs local mirror — recorded by the run, not
guessed at promotion time), the split dates, link count, the git commit and whether the
tree was dirty, the environment (python/torch/numpy/pandas, GPU), the training summary
(best val loss, epochs run, why it stopped), and the weights' sha256.

```bash
# promote — explicit, nothing is saved unless you ask
python -m prediction.zoo add ../artifacts/pred_online \
    --models gbdt+online,tcn --note "best day-horizon so far"

python -m prediction.zoo list                    # table of every entry + skill
python -m prediction.zoo show tcn/pred_online    # one manifest
python -m prediction.zoo pack tcn/pred_online    # -> .tar.gz + the upload command
python -m prediction.zoo remove tcn/pred_online
```

Or promote straight from a run with `--zoo` (plus `--zoo-models`, `--zoo-tag`,
`--zoo-note`, `--zoo-records`). That runs **before** `prune_checkpoints`, so an
`+online` variant keeps its own weights file even though the run then drops it as a
warm-start duplicate.

**Uploading is yours to do.** `pack` builds the archive and prints an `aws s3 cp`
(and `rsync`) line — it never uploads. Publish to a bucket you own, not the read-only
data bucket. `artifacts/` is gitignored, so weights never land in git by accident; if
you want an entry in version control, copy it out deliberately.

Entries are keyed `<model>/<tag>`, tag defaulting to the run directory's name. Re-adding
the same pair needs `--force`; use `--tag` to keep several versions of one model side by
side. `index.json` and a browsable `README.md` are regenerated on every change, and
`GR_ZOO_DIR` moves the whole zoo elsewhere.

## Frontier (not built)

Tier-5 iTransformer/Informer, Tier-6 DCRNN/MTGNN, foundation models (TimesFM/Chronos
zero-shot), diffusion (CRPS). Each must clear the same bar: beat the cheapest tier that
already wins at that horizon, on these identical origins.

## Full-cohort real-data run (all usable links)

No `--max-links`: the panel keeps every usable link — 886 links / 141 devices over this
window (see *Link census* above). The first cold build reads all ~19 GB once
(~10–15 min), then the panel is cached to `artifacts/panel_<hash>.npz`.

```bash
env -u PYTHONPATH PRED_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/yuyqin/anaconda3/envs/green-pred/bin/python -m prediction.run \
    --start 2025-12-01 --end 2026-07-11 \
    --train-end 2026-05-01 --val-start 2026-05-08 \
    --val-end 2026-05-29 --test-start 2026-06-05 \
    --tiers 1 3 4 5 6 \
    --horizons 15min 1h day week \
    --max-origins 48 \
    --lookback 336 --epochs 12 \
    --out ../artifacts/pred_full
```

> Do **not** append `| tee ../artifacts/pred_full/run.log` any more: `run.py` writes
> that file itself, and a shell `tee` on the same path interleaves with it. The run
> creates `--out` too, so the old `mkdir -p` is unnecessary.

(The earlier `pred_real`/`pred_paper` results above were produced with
`--max-links 20`, i.e. ~2 % of the usable cohort — alphabetically biased. Full-cohort
numbers supersede them once this run lands.)