# 2026-11-04 (afternoon): trained models with long context at 5-minute resolution

Owner's question, after the ladder diagnosis (`2026-11-04-ladder-diagnosis.md`): none of the trained
methods takes 24 weeks of context; keep the 5-minute interval, find ways to extend the context, try
several methods, train them, report the results. Times CET; nothing committed.

Report: `docs/reports/long-context-5m.html` (published as a claude.ai artifact, link in AGENTS.md).
Code: `recipes/ladder_diagnosis/long5m.py` (data, models, training, scoring), `refs5m.py` (references and
untrained 5-minute profiles), `compare.py` (now also `day5`/`week5` = 5-minute WAPE),
`switch_history.py --res 5m`, grids `grids/l5_*.txt` (first grid), `l5b_*` (bounded inputs), `l5c_*`
(seeds, variants), `l5d_i2` (20-epoch full-resolution rerun). Output:
`artifacts/ladder_diagnosis/{i2,switch}/runs/L5*/` (hourly `pred_{day,week}.npy` for `compare.py`,
5-minute `pred5_{day,week}.npy`, summary, history); every run in `long5m_runs.csv`.

Setup: 5-minute input, 5-minute output path of one week (2,016 steps), block forecasts at 00:00 UTC,
scored as median link hourly WAPE (as in the diagnosis) and 5-minute WAPE. Scaling: x / mean of the
last week of input; L1 loss on 5-minute targets; 100k windows per epoch, <= 20 epochs, OneCycle, early
stop on validation hourly WAPE at a day; hourly origins; history from 2024-01 (Switch, 5-minute from the
frozen S3 cache) or 2024-11 (I2: hourly file repeated over 12 steps before the panel, flagged with
mask 0.5; TSDS keeps no older 5-minute data).

## Answer

- **Yes, trained models can take 24 weeks of 5-minute values** (48,384 steps). The ladder's models
  cannot as built (DLinear's weights and PatchTST's flatten head grow with input length, the GRU is
  sequential, the TCN needs a level per doubling). Patching with a per-token head, multi-resolution
  tokens, folding by week, and weekly-profile summaries all train at 24 weeks: 4-370 ms per batch of 128,
  <= 7.1 GB.
- **1 -> 4 weeks helps; 4 -> 24 weeks does not.** 3-seed ensembles, day / week, paired, day bootstrap:
  4 vs 1 w I2 -1.34 [-2.03, -0.75] / -1.93; Switch -1.61 [-3.15, -1.10] / -2.64. 24 vs 4 w I2 +0.43
  (n.s.) / +0.86 [+0.30, +2.09]; Switch +0.18 (n.s.) / +0.08 (n.s.). Same for every method (table below).
- **A pipeline problem that grows with context:** recent-mean scaling leaves old weeks unbounded (p99 of a
  window's largest input 46 at 1 w, 739,439 at 24 w on I2; max 6.5e7). The profile MLP diverged (loss in
  the thousands, I2 27.48 at 24 w). `--squash 10` (identity to 10, then 10 + log1p) fixes it (18.63);
  transformers move by about half a point or less.
- Best at 5 minutes: ensemble of 11 runs (4 and 24 w) I2 17.50 / 22.37, Switch 30.72 / 34.83, from the
  ladder's 19.15 / 25.01 and 35.41 / 40.81. Timestral with 16 weeks of 5-minute input: I2 16.48 / 21.36
  (ahead by 1.02 [+0.44, +1.93] at a day), Switch 29.07 / 34.53 (ahead by 1.65 [+0.76, +2.68] at a day;
  week +0.31 [-0.19, +1.68], n.s.).

## Results

Day-ahead median link hourly WAPE, single seeds; † = without bounded inputs (no bounded run).

| model | I2 1 w | 4 w | 12 w | 24 w | Switch 1 w | 4 w | 12 w | 24 w |
|---|---|---|---|---|---|---|---|---|
| full-resolution transformer (hourly tokens) | 19.16 | 18.06 | | 19.56 (20 ep budget; 10 ep: 20.31) | 32.88 | 31.44 | | 33.10 (10 ep) |
| coarse patches (168 tokens) | = full | 18.46† | | 18.65 | = full | 31.90† | | 31.64 |
| multi-resolution (last week hourly, older daily) | = full | 17.99 | 19.18 | 18.66 | = full | 31.50 | 31.26 | 31.57 |
| multi-resolution, old days as 24 hourly means | | | | 18.54 | | | | 31.94 |
| multi-resolution, context truncation 0.5 | | | | 19.72 | | | | 31.73 |
| last week + profile query tokens | | 18.55† | | 18.98 | | 32.84† | | 31.91 |
| fold (week-folded shared MLP) | 20.33† | 19.55† | | 19.22 | 35.98† | 33.76† | | 33.35 |
| profile MLP | 21.21† | 18.91 | | 18.63 | 33.98† | 32.30 | | 32.42 |
| ladder PatchTST | 19.15 | | | | 35.41 | | | |

Seeds (multi-resolution; 1 w = full): I2 1 w 19.16 / 19.72 / 19.62, 4 w 17.99 / 17.96 / 18.51, 24 w
18.66 / 18.73 / 18.42; Switch 1 w 32.88 / 33.22 / 32.52, 4 w 31.50 / 31.13 / 31.33, 24 w 31.57 / 31.46 /
31.31. Ensembles: I2 19.20 / 17.86 / 18.29, Switch 32.50 / 30.90 / 31.08. 4 + 24 w averaged vs 4 w: n.s.
everywhere. 1-week ensemble vs ladder: I2 +0.06 (n.s.), Switch -2.91 [-3.65, -1.43] (the gain is mean
scaling + L1 alone; the input is the ladder's).

References (`refs5m.csv`, day hourly / day 5-minute WAPE): ladder PatchTST I2 19.15 / 21.62, Switch 35.41 /
43.49; Timestral 5 m 4 w 16.77 / 19.28, 29.79 / 37.83; Timestral 5 m 16 w (I2) 16.48 / 18.91; Timestral 1 h
24 w 16.22 / 19.19, 28.05 / 41.14; Timestral 5 m 16 w (Switch) 29.07 / 37.27 (week 34.53). Untrained per-5-minute-of-week median over K weeks, level-rescaled:
I2 K = 1/4/12/24: 28.21 (= seasonal naive) / 22.43 / 22.41 / 22.59; Switch 45.66 / 37.47 / 39.18 / 41.65.

## Why no gain beyond 4 weeks (evidence, not proof)

- Plain profiles stop improving at 4 weeks (I2 flat, Switch worse): little in the extra weeks that simple
  statistics can use.
- Timestral at 5-minute input gains -0.29 [-0.71, +0.11] (n.s.) from 4 to 16 weeks on I2 and -0.72 [-1.29,
  -0.25] at a day on Switch (week +0.26, n.s.); with hourly input
  0.75 (I2) / 1.44 (Switch) from 4 to 24. What it gets out of long context is a pretrained prior.
- Validation favours 24 w slightly (I2 val 24.03-24.72 vs 24.48-24.80), test does not: what the extra weeks
  teach is specific to the training period.
- Hourly summaries of old weeks (removes I2's hourly-filled vs real 5-minute mismatch) and random context
  truncation do not help.
- Full attention over 4,200 tokens costs 20x the multi-resolution model per step and trains slowest.
  With a 20-epoch budget on I2 validation was best at epoch 10 (25.0 vs 24.0-24.7 for the cheaper models),
  early stop at 15; test 19.56, +1.50 [+0.84, +2.35] vs the same model at 4 w, +0.40 (n.s.) vs 1 w.

## Log

- 12:20 Switch 5-minute history 2024-01 -> 2025-07 (`switch_history.py --res 5m`; first launch stopped
  within seconds: a variable clash would have overwritten `history_hourly.npz`; it had not written).
  Checks: contiguous with the panel, hourly means equal the hourly file (max diff 0.02 Mbit/s).
- 12:26 Grid `l5_*` (7 methods x 1/4/24 w, both networks in parallel); `refs5m.py`.
- 12:30 Timestral 5 m 16 w on I2 via `gr-dense predict --tag @5m_16w --fm-context 32256` (first attempt
  failed: needs the I2 environment of `recipes/internet2/run.sh`); 47 min next to two trainings.
- 14:40 Profile MLP diverged; input magnitudes measured; `--squash` added. First grid's full-resolution
  24 w runs (unbounded, ~2 h each) and its 12 w runs stopped by PID.
- 14:45 Grid `l5b_*` (bounded inputs). 16:05 grid `l5c_*` (hourly-summary multires `--seg mrh`,
  `--drop-ctx`, seeds 1-2).
- 19:00 Full-resolution 24 w (10 epochs, 3 h each with four jobs on the GPU) done; 20-epoch rerun on I2
  (`l5d_i2`) and Timestral 5 m 16 w on Switch started.
- 21:45 Both done (2.6 h and 2.7 h). Report republished; `long5m_runs.csv` and `refs5m.csv` refreshed.

## Not done / caveats

- Most runs are single seeds; seeds only for the multi-resolution curve.
- I2 before 2025-11-15 is hourly only, so 64 % of its 24-week training contexts are hourly-filled.
- Not tried: quantile loss, calendar features, joint training on both networks at 5 minutes, LoRA or
  fine-tuning Timestral at 5 minutes, state-space / long-convolution models over the raw sequence.
