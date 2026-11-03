# 2026-11-03: why links are hard, and Timestral on both networks

Owner's questions: why is per-link traffic so hard to forecast, and how does Mistral's Timestral
(https://mistral.ai/news/timestral/, released 2026-10-15) do on Switch LAN and the I2 Network?
Nothing committed. Times UTC.

## What was run

| Step | Code | Output |
|---|---|---|
| Difficulty diagnostics (day-ahead block log, PatchTST, hourly) | `recipes/link_difficulty/analyze.py` | `artifacts/link_difficulty/{i2,switch_sym}_patchtst.json` |
| Timestral installed (`pip install --no-deps git+https://github.com/mistralai/timestral.git@a534bdb`; deps already in `green`) | | weights in `~/.cache/huggingface` (209M 837 MB, 28M 111 MB) |
| Pretrained models in the day-ahead total | `carbon.dayahead.foundation(model="timestral" / "timestral-28M")` | |
| Switch LAN total: 6 Timestral variants | `recipes/flat_forecast/run.py` (FOUNDATION) `foundation score` | `artifacts/flat_forecast/scores_*.csv`; previous outputs in `before_timestral/` |
| I2 total: 4 Timestral variants + Chronos-2 24 weeks | `recipes/internet2/timestral_total.py` | `artifacts/internet2/carbon/{timestral_total,rate_errors_timestral}.csv` |
| Per link, zero-shot, block logs | `gr-dense predict --models timestral --tag @1h_24w --fm-step 1h --fm-context 4032` (new, block only) | `<run>/dense_fm/` (links to the ladder's block logs + Timestral's), tables in `dense_fm/tables_*` |
| Comparison tables | `recipes/link_difficulty/fm_tables.py` | |

`dense_fm/` keeps the original `dense/` (meta, tables, viewer) untouched.

## Results

Why links are hard (day ahead, median link hourly WAPE; I2 / Switch LAN):
- PatchTST 19.1 / 35.4 %. With each day rescaled to the true daily volume 15.6 / 28.4 %; daily-volume
  WAPE 13.6 / 25.6 %. In-sample weekly profile of the test block + true daily volume (a bound for any
  periodic forecast): 13.4 / 22.8 %.
- Size: Spearman rho -0.73 / -0.40; quartiles I2 26.1 / 20.3 / 19.4 / 11.9 % (2.3 / 11 / 31 / 74 Gbit/s),
  Switch LAN 43.0 / 48.5 / 31.1 / 24.4 % (5 / 30 / 403 / 7,041 Mbit/s).
- Hours at < 1/3 or > 3x the usual level (median of the same hour, 4 weeks before): 4.3 / 9.0 % of
  hours, 8.5 / 13.8 % of the error (pooled). Worst 5 % of hours: 24 / 33 % of a link's error.
- Errors of links on the same router correlate +0.23 / +0.22 (random pairs +0.09 / +0.11); network
  sum of link forecasts 5.9 / 13.1 % vs 3.3 / 8.3 % with each link's errors shifted by random days.
  So: shared surges + independent noise, not traffic moving between neighbouring links.

Total, day ahead, hourly WAPE (test blocks as before):
- I2: Timestral-209M 24 weeks 4.29 %, 48 weeks 4.27, 12 weeks 4.52; Timestral-28M 4.67; Chronos-2 24
  weeks 4.79, 12 weeks (carbon.py) 4.88; GBDT 7.14. Paired vs Chronos-2 24 weeks: -0.50 points
  [-0.66, -0.36], 69 % of days better. Rate WAPE 5.0 % actual grid / 6.3 % grid rule (Chronos-2 5.5 / 6.8).
  No development period on I2: variants compared on test (all beat Chronos-2).
- Switch LAN: Timestral 24 weeks + calendar covariates 10.24 % (DEV 10.34, best on DEV), 12 weeks 10.58,
  48 weeks 10.59, 5-minute input 4 / 16 weeks 11.01 / 10.69, 28M 10.86; Chronos-2 24 weeks + calendar
  10.56; old final 9.96. Blend 0.9 Timestral + 0.1 GBDT, GBDT alone on holidays (weight on DEV): 9.83 %,
  paired vs old final -0.12 [-0.39, +0.13] (n.s.). The score stage's rule (lowest DEV) now names
  Timestral alone (10.24 % on test, worse than the old final): `final.json` restored to the old final
  pending the owner's decision. Old rows of scores_dev/test identical (max diff 0).

Per link, block logs, median link hourly WAPE (zero-shot, hourly input, 24 weeks of context):

| | I2 1 h | I2 8 h | I2 day | I2 week | SW 8 h | SW day | SW week |
|---|---|---|---|---|---|---|---|
| persistence | 11.3 | 27.2 | 32.1 | 41.3 | 52.7 | 70.1 | 74.1 |
| best ladder | 9.5 TCN | 16.3 PatchTST | 19.1 PatchTST | 25.0 PatchTST | 31.5 PatchTST | 35.4 PatchTST (34.9 peak-aware) | 40.8 PatchTST |
| Timestral@1h_24w | 9.4 | 14.5 | 16.2 | 20.5 | 24.7 | 28.1 | 32.5 |
| links where it beats PatchTST | 93 % | 96 % | 98 % | 97 % | 97 % | 97 % (98 % peak-aware) | 97 % |

- 5-minute input, 4 weeks (`timestral@5m_4w`): I2 14.6 / 16.8 / 22.0 % (8 h / day / week), Switch LAN
  29.8 / 34.3 % (day / week): hourly input is better on hourly WAPE. On 5-minute MAE (ratio vs
  persistence at a day) the 5-minute input wins on Switch LAN (0.551 vs 0.620 hourly input, PatchTST
  0.643), not on I2 (0.642 vs 0.629).
- Peaks (day, per-link q90 of 5-min values): Timestral peak WAPE 17.8 % I2 (PatchTST 20.1), 33.0 % SW
  (symmetric PatchTST 37.1, peak-aware 32.3); off-peak bias +4.9 % SW (peak-aware ladder +12.3).
- Caveats: Timestral sees 24 weeks, the ladder 1 week (lookback 2016; the 1-month window gained 3-5 %
  at a day). I2's test block (2026-07-19 -> 11-01) is public and reaches the release date: overlap with
  Timestral's training data cannot be excluded. Switch LAN is private, so it is the clean test.

## Log

(times from file modification times)
- 11:44 Timestral installed; speed: 209M, 64 contexts of 32,256 steps -> 288 steps in 3.8 s (RTX PRO
  4500, 32 GB); 28M about 5x faster.
- by 11:58 difficulty diagnostics on both runs (CPU).
- by 12:01 totals: Switch LAN (flat_forecast foundation + score), I2 (timestral_total.py).
- 11:47-12:27 I2 per link, hourly input (1 h, 8 h, day, week): 41 min.
- 12:28-12:40 I2 per link, 5-minute input, 4 weeks.
- 12:40-13:09 Switch LAN per link, hourly input.
- 13:10-13:29 Switch LAN per link, 5-minute input, 4 weeks (day, week); the 16-week 5-minute runs
  dropped (5-minute input lost to hourly on I2 and on the totals).
- Fix: `gr-dense metrics` summary print crashed (KeyError) when a horizon had no common target;
  the tables were already written. Now reindexed.
- Another session worked in the repository at the same time (edited `recipes/internet2/path_dayahead.py`,
  added `path_dayahead_models.py`, wrote to the flat_forecast cache). Not checked here.

## Open points for the owner

- Use Timestral for the flat-rate total? On I2 it is significantly better than Chronos-2; on Switch
  LAN only inside the combination, not significantly.
- Per-link forecasts for the bills and path rates: Timestral would replace the ladder (-15 to -21 %
  hourly error at a day). Not yet propagated to bills or `path_dayahead`.
- Not tried: Timestral with calendar covariates per link, multivariate (links of a router jointly),
  quantiles (it outputs 21), fine-tuning, the ladder with a longer window for a matched comparison.
