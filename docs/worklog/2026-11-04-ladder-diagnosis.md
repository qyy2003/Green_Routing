# 2026-11-04: why the trained ladder trails Timestral, and what closes the gap

Owner's question: why are the trained models (TCN, PatchTST, ...) so weak per link, why do they all
score about the same, and how can zero-shot Timestral beat them? Data problem, or a pipeline
issue (pre/post-processing, a bottleneck)? Try I2 too; learn from Timestral. Times UTC (from file
modification times); nothing committed.

Report: https://claude.ai/artifact/1EuQg8mZQbBZAQhm6MYx2W (copy `docs/reports/ladder-diagnosis.html`).
Code: `recipes/ladder_diagnosis/` (`run.sh` lists the steps). Output:
`artifacts/ladder_diagnosis/{i2,switch}/runs/<tag>/` (hourly test forecasts, summary, history),
`all_runs.csv` (every run). Metric throughout: day-ahead (and week) **block** forecasts issued at
00:00 UTC, median over links of per-link **hourly WAPE** over the test block, exactly the
`wape_1h` of `gr-dense metrics` (the evaluator reproduces the dense_fm tables to 3 decimals:
I2 day PatchTST 19.146, Timestral 16.224, seasonal naive week 28.210). Test blocks as before
(Switch LAN 2026-03-22 -> 07-11, 534 links; I2 2026-07-19 -> 11-01, 113 links).

## Answer in one paragraph

Nothing is broken in the sense of a bug, and the data is not the main problem. The ladder was
set up to forecast 5-minute values from **one week** of 5-minute history, and from that input
every architecture learns the same function: training loss flat after 2-3 epochs, DLinear's
validation loss within 1 % of PatchTST's, PatchTST's errors correlated 0.91-0.95 with the other three's.
**Timestral given the same one week is no better** (I2 19.73 vs ladder PatchTST 19.15; Switch
35.04 vs 35.41, both n.s.), and a trained model with the input fixed beats it on equal input.
Timestral's whole lead comes from reading **24 weeks of hourly history** (I2 19.73 -> 16.22,
Switch 35.04 -> 28.05 as its context grows from 1 to 24 weeks), which the ladder never saw.
With hourly targets, mean scaling + L1 and all history, trained models close 57 % (I2) and 74 %
(Switch) of the gap, but they gain little from long context themselves (at most 0.7 points from 1
to 4 weeks): what a pretrained model brings is the prior for reading long, noisy context, which
113-534 links x 1-2 years cannot teach (training on both networks at once did not help either).

## Findings

**1. Same input, same result.** Day-ahead, median link hourly WAPE, paired per link on common
hours, 95 % intervals from a bootstrap over test days (`compare.py`):

| | I2 | Switch LAN |
|---|---|---|
| ladder PatchTST (5 min, 1 week, log1p+RevIN+Huber) | 19.15 | 35.41 |
| Timestral zero-shot, 1 week hourly | 19.73 | 35.04 |
| ladder - Timestral(1 week) | -0.58 [-2.00, +0.14] | +0.37 [-1.56, +2.35] |
| hourly PatchTST, 1 week, mean scaling + L1 (panel history) | 18.58 | 31.36 |
| same, - Timestral(1 week) | | -3.68 [-5.35, -2.09] |
| hourly PatchTST, 1 week, + extended history | 17.80 (3 seeds 17.80/18.80/17.93; ensemble 17.67) | 31.15 |
| same (seed 0), - Timestral(1 week) | -1.92 [-3.16, -1.12] | |

PatchTST's errors correlate 0.91-0.95 with DLinear's, GRU's and TCN's (I2, per link, hourly); val losses on the
year run: TCN 0.2115, PatchTST 0.2089, GRU 0.2171, DLinear 0.2105; train loss flat from epoch
2-3 on both networks (no over- or under-fitting signature: an information limit).

**2. Timestral's edge is context length.** Timestral zero-shot, hourly input:

| context | 1 w | 2 w | 4 w | 12 w | 24 w |
|---|---|---|---|---|---|
| I2 day | 19.73 | 17.90 | 16.97 | 16.71 | 16.22 |
| I2 week | 24.48 | 22.49 | 22.13 | 21.44 | 20.55 |
| Switch day | 35.04 | 31.11 | 29.49 | 28.27 | 28.05 |
| Switch week | 37.95 | 35.34 | 33.46 | 32.61 | 32.54 |

24 w - 1 w: I2 -3.50 [-4.65, -2.43], Switch -6.99 [-8.99, -5.32]. The gain is uniform: on I2 it is
~3.4 points whether the last week was typical or unusual (bins of |log(last week / median of
weeks 2-8)|): long context is a better estimate of level and shape everywhere, not only a
guard against an odd week. Timestral 24 w vs ladder PatchTST (I2): daily-volume error 13.6 -> 11.5,
shape error at the true daily volume 15.6 -> 12.9 (both parts improve). Plain multi-week profiles (no learning) do not get there:
best 22.8 (I2, 4-week median rescaled to the last day), 38.0 (Switch).

**3. Pipeline choices, ranked by effect** (hourly harness `hourly_train.py`, which reproduces the
ladder: linear 1 w log+Huber 22.65 vs ladder DLinear 22.53 on I2; MLP 19.87 vs TCN 19.41):

- *One-week lookback* (2016 five-minute steps). The largest. At 5 minutes, 24 weeks are 48k steps
  (the 1-month experiment already cost x9 for the TCN); hourly means make it 4,032 steps.
- *5-minute resolution, on Switch only.* Same model, settings and data, 5-min vs hourly input and
  target: 34.18 vs 31.55 (day), 39.20 vs 38.09 (week). Ladder (35.41) vs hourly with its own
  preprocessing: -3.86 [-4.67, -2.25], better on 80 % of links. Within-hour variance is 12.5 % of
  the log 5-minute variance on Switch (21.5 % linear, p75 39 %), 6.6 % on I2; on I2 resolution does
  not matter (ladder 19.15 vs hourly 19.59). Split on Switch: 5-minute input with hourly targets
  (`--agg 12`) 33.50 / 39.20, so ~2.0 of the 2.6 points are the noisy input, ~0.7 the 5-minute targets. The remaining ~1.2 points between the ladder and the
  harness's 5-min PatchTST are not isolated: 1/sqrt(h) weights + link embedding cost nothing at 5
  minutes (34.12 vs 34.18) and +0.22 at hourly (31.77 vs 31.55), 1/sqrt(h) alone +0.56 [-0.18, +1.05] at
  hourly; the rest is in the ladder's own code (overlapping 16/8 patches, post-norm encoder, plateau
  schedule on a 5-point val loss, filled targets).
- *log1p + per-window RevIN + Huber.* Biases forecasts low (median link bias at a day: Switch -14
  to -16 %, I2 -4.6 to -9.2 %; seasonal naive +0.4 % / -2.6 %): the hourly arithmetic mean is above the
  within-hour geometric mean (Switch: median link 6.4 %, p10 23.5 %; I2 0.8 %). Removing the bias
  does not lower WAPE (even an oracle per-link rescale is worse: absolute error wants the median),
  so the bias is a symptom. Mean scaling + L1 vs log+Huber, same hourly PatchTST: I2 -1.01
  [-1.89, -0.38] (significant), Switch -0.19 [-1.30, +0.15] (n.s.); at a week larger (I2 MLP
  26.12 -> 23.69, PatchTST 25.54 -> 23.27).
- *Training history limited to the panel* (Switch 2025-07 -> 2026-02, I2 2025-11 -> 2026-06) while
  10-18 more months exist (I2 hourly file from 2024-11 matches the panel to +-0.3 %; Switch
  hourly 2024-01 -> 2025-07 from the frozen S3 cache). Ext vs panel, I2 PatchTST 1 w: -0.78
  [-0.94, -0.03]. At long context it decides whether the model works at all: I2 24 weeks, linear
  26.05 -> 20.10, ProfMLP 20.01 -> 18.24; Switch PatchTST 1 w 31.36 -> 31.15, MLP 4 w 31.85 -> 30.91.
- *Model selection on 5 path points* (val loss at steps 3, 12, 96, 288, 2016 of each 2016-step
  path) and filled targets (ffill/bfill 1 h, which can also put a few future values into a
  context next to a long gap): small, not isolated.

**4. Trained models with long context.** A dense map from 4,032 inputs over-fits (MLP 24 w: I2
24.64, Switch 32.46). Robust summaries work better: `ProfMLP` = last 2 weeks + per-hour-of-week
median over K weeks and over the last 4 + weekly means. Best trained (median link hourly WAPE):

| | I2 day | I2 week | Switch day | Switch week |
|---|---|---|---|---|
| ladder PatchTST | 19.15 | 25.01 | 35.41 | 40.81 |
| PatchTST 1 w, ext, 3-seed ensemble | 17.67 | 22.32 | | |
| ProfMLP 12 w, ext, 3-seed ensemble | 17.94 | 21.89 | | |
| PatchTST 4 w, ext | 17.84 | 22.39 | 30.42 | 34.97 |
| ProfMLP 4 w, ext | 18.31 | 22.53 | 30.65 | 34.41 |
| ProfMLP 12 w, ext | 17.87 (seeds 17.96, 18.42) | 21.62 | 30.78 | 34.47 |
| ensemble (I2: the 6 seeds above; Switch: PatchTST 4 w + ProfMLP 4 w + 12 w) | 17.47 | 21.74 | 29.98 | 34.30 |
| Timestral 24 w zero-shot | 16.22 | 20.55 | 28.05 | 32.54 |

vs the ladder: I2 PatchTST ensemble -1.48 [-1.97, -0.31] (day), ProfMLP ensemble -3.13 [-4.11,
-1.52] (week); vs Timestral: 6-model ensemble +1.24 [+0.47, +1.74] (day), +1.19 [+0.36, +1.80]
(week); Switch ensemble +1.93 [+1.32, +2.93] (day), +1.76 [+1.04, +2.45] (week); Switch PatchTST 4 w vs
the ladder -4.99 [-6.65, -3.61] (day), -5.84 [-7.48, -4.52] (week), vs 1 w -0.74 [-1.55, +0.02]. The fixes close
57 % (I2) and 74 % (Switch) of the ladder-to-Timestral gap at a day. Blending trained models into Timestral does not
help (I2 0.7/0.3: 16.30, Switch 28.45).

**5. More of what Timestral has?** One model trained on both networks (each cut at the evaluated
network's training end): I2 18.15 (single-network 17.87-18.42), Switch 4 w 30.88 (30.65), 12 w
32.07 (30.78), PatchTST 1 w I2 18.08 (17.80), Switch 31.26 (31.15): no gain. Fine-tuning Timestral-209M on I2's history (pinball over its
21 quantiles, one 32-hour patch, 24-week context, bs 16, 300 steps per epoch, per-window loss capped): LR 3e-6
never beat zero-shot on validation (20.29); LR 1e-5 best at epoch 4 (val 20.08), test 16.06 / 20.52 vs 16.22 /
20.55 zero-shot, paired -0.16 [-0.37, +0.14] (day, better on 82 % of links), -0.03 [-0.44, +0.27] (week): not
significant. A week-stacked shared-MLP (`wstack`) and longer PatchTST contexts were worse
(I2 PatchTST 12 w 19.32 vs 1 w 17.80).

## Recommendations

- Per-link day/week forecasts: Timestral zero-shot with 24 weeks of hourly context stays the
  best model on both networks; the ladder should not be the reference.
- If a trained ladder is kept: forecast hourly means for day/week horizons (keep 5 minutes for
  15 min / 1 h, where the ladder ties Timestral: I2 1 h TCN 9.45 vs 9.36), mean scaling with an
  L1/pinball loss, train on all history (not the panel window), long context through robust
  profile features. Measured: I2 19.15 -> 17.5, Switch 35.4 -> 30.0 at a day (ensembles).
- Compare models on equal information (context length and resolution) before reading model
  rankings: the ladder vs Timestral comparison mixed "1 week vs 24 weeks" with "trained vs
  pretrained".

## Not done / caveats

- Not tried: Timestral fine-tuning with LoRA or on Switch; quantile outputs; calendar covariates;
  per-link fine-tuning; longer training of the long-context models (most stopped at the 30-epoch
  schedule end); the original ladder code with hourly input (all ablations ran in the harness,
  which reproduces the ladder's numbers but is not the same code).
- I2's test block is public and reaches Timestral's release date (2026-10-15): overlap with its
  training data cannot be excluded. Switch LAN is private and shows the same pattern.
- Seeds: only for the best I2 configurations (spread ~0.4-1.0 points at a day for a single run);
  single-seed differences below ~0.5 points are noise.
- Switch history before 2025-02 is a different door set (pitfall 28); per link it was used as is.

## Log

(times UTC, from file modification times)
- 08:49 Start. Read the pipeline (data, torch_common, models_deep, dense, metrics) and the
  training curves of both year runs.
- ~08:55 Evaluator (`common.py`) calibrated against the dense_fm tables; multi-week profiles
  (`baselines.py`): best 22.8 / 38.0, all ladder models biased low.
- 08:56-09:43 Grids on the GPU (`grids/*.txt`, `sweep.py`); I2 hourly file found to match the panel
  (in+out of both directions); Switch history 2024-01 -> 2025-07 extracted (`switch_history.py`,
  1 min, 16 processes on the frozen cache).
- 09:01-09:28 Timestral context sweep (`timestral_ctx.py`): the 1-week result changed the question.
- 09:15 Resolution ablation on Switch (`--res 5m`, `--agg`, `--hweight`, `--lemb`).
- 09:27 First fine-tune: loss exploded (windows with a ~0 context mean and large targets dominate
  a context-mean-scaled pinball loss); val did not improve (20.29 -> 20.24-20.43). Rerun with a
  per-window cap and a 1 Mbit/s scale floor.
- 09:35 Joint training (`--joint i2,switch`).
- One OOM (fine-tune smoke run next to six grids) killed two ProfMLP runs; rerun in `e2`.
- 09:43-10:19 Robust fine-tunes (LR 3e-6, then 1e-5). 10:19 Timestral 24 w on Switch reproduces the
  dense_fm log (28.05 / 32.54). Last 5-minute ablation done ~10:20. All runs in `all_runs.csv`.
- Stopped to free the GPU: I2 PatchTST 24 w (panel and ext), the remaining wstack runs, Switch PatchTST
  12 / 24 w (annotated in `grids/*.txt`).
