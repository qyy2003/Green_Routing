# Results

Everything here comes from runs on the reference machine (NVIDIA L40S) in October 2026,
on the year-scale configuration in [`recipes/year/flags.sh`](../recipes/year/flags.sh):
534 SWITCH backbone links with mean traffic ≥ 1 Mbit/s, data 2025-07-01 → 2026-07-11,
test period 2026-03-22 → 2026-07-11 (111 days). The two HTML reports in
[`reports/`](reports/) present them for readers; this page is the reference.

Metric throughout: **per-link median MAE ratio vs persistence** (lower is better, 1.000 =
as good as repeating the last value), with the share of links beating persistence in
brackets. Bold = best learned model per horizon.

## 1. The carbon-accounting report reproduces

The report ([`reports/carbon-accounting-report.html`](reports/carbon-accounting-report.html))
was made on a laptop RTX 3060 before the data moved to S3; its command was never saved.
The configuration was reconstructed and verified on this machine:

| Horizon | Report: best model, ratio | Reproduction: best model, ratio | Pooled MAE, report / reproduction (Mbit/s) | Persistence MAE, report / reproduction |
|---|---|---|---|---|
| 1 h | GRU 0.801 | TCN **0.787** | 585 / 569 | 712 / 711.9 |
| 8 h | GBDT ≈ TCN ≈ PatchTST 0.545 | TCN **0.538** | 836 / 828 | 1541 / 1540.6 |
| 1 day | TCN 0.743 | GBDT 0.748 | 850 / 843 | 1061 / 1060.5 |
| 1 week | GBDT 0.842 | GBDT 0.849 | 973 / 962 | 1145 / 1144.7 |

Persistence matches to the Mbit/s, which pins the data, the cohort, the test window and
the 160 origins exactly; the learned models land within about 1 % of the report. Two
things had made the code look worse before:

- the run that was being compared (`pred_full`) was a different, smaller experiment
  (a 7-month window, 886 links including idle ports, 28-h lookback, 12 epochs, no 8 h);
- **a real bug**: at a 1-week lookback the TCN's weight-norm direction vectors collapsed
  under Adam's weight decay, every gradient became NaN from epoch 4, and the trainer
  skipped every step without saying so. Fixed (`torch_common.py`); the trainer now
  reports skipped steps.

## 2. Year benchmark, full tables

**Sampled origins (160, the harness)** — per-link median MAE ratio vs persistence (share of links beating persistence)

| Model | 15 min | 1 h | 8 h | 1 day | 1 week |
|---|---:|---:|---:|---:|---:|
| persistence | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seasonal-naive day | 1.712 (11%) | 1.432 (9%) | 0.751 (81%) | 1.000 (0%) | 1.000 (0%) |
| seasonal-naive week | 1.644 (10%) | 1.319 (14%) | 0.712 (77%) | 0.954 (55%) | 1.000 (0%) |
| historical average | 1.559 (14%) | 1.220 (27%) | 0.674 (74%) | 0.822 (67%) | 0.872 (62%) |
| GBDT | 0.965 (61%) | 0.848 (79%) | 0.539 (88%) | **0.748 (84%)** | **0.849 (77%)** |
| GRU | 0.882 (78%) | 0.838 (80%) | 0.553 (92%) | 0.776 (86%) | 0.877 (81%) |
| TCN | **0.853 (80%)** | **0.787 (86%)** | **0.538 (91%)** | 0.754 (87%) | 0.875 (83%) |
| DLinear | 0.898 (71%) | 0.821 (83%) | 0.559 (90%) | 0.782 (87%) | 0.861 (87%) |
| PatchTST | 0.916 (67%) | 0.805 (81%) | 0.548 (92%) | 0.753 (90%) | 0.860 (84%) |

**Dense, rolling (every origin, fixed lead)** — per-link median MAE ratio vs persistence (share of links beating persistence)

| Model | 15 min | 1 h | 8 h | 1 day | 1 week |
|---|---:|---:|---:|---:|---:|
| persistence | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seasonal-naive day | 1.747 (11%) | 1.340 (15%) | 0.661 (87%) | 1.000 (0%) | 1.000 (0%) |
| seasonal-naive week | 1.693 (10%) | 1.203 (19%) | 0.629 (82%) | 0.973 (55%) | 1.000 (0%) |
| historical average | 1.577 (13%) | 1.113 (37%) | 0.605 (78%) | 0.896 (61%) | 0.883 (63%) |
| GBDT | 0.927 (72%) | 0.813 (86%) | 0.514 (91%) | 0.753 (84%) | **0.859 (76%)** |
| GRU | 0.931 (76%) | 0.802 (91%) | 0.511 (96%) | 0.782 (86%) | 0.868 (83%) |
| TCN | **0.888 (88%)** | 0.780 (92%) | 0.496 (95%) | 0.759 (87%) | 0.896 (81%) |
| DLinear | 0.897 (75%) | 0.790 (89%) | 0.501 (94%) | 0.809 (90%) | 0.876 (85%) |
| PatchTST | 0.923 (69%) | **0.780 (89%)** | **0.489 (96%)** | **0.751 (88%)** | 0.878 (81%) |

**Dense, block (window by window, every 5-min step)** — per-link median MAE ratio vs persistence (share of links beating persistence)

| Model | 15 min | 1 h | 8 h | 1 day | 1 week |
|---|---:|---:|---:|---:|---:|
| persistence | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seasonal-naive day | 2.032 (8%) | 1.559 (13%) | 0.997 (52%) | 0.898 (67%) | 0.998 (56%) |
| seasonal-naive week | 1.964 (7%) | 1.449 (10%) | 0.943 (61%) | 0.872 (65%) | 0.825 (72%) |
| historical average | 1.814 (7%) | 1.359 (16%) | 0.896 (60%) | 0.820 (68%) | 0.760 (75%) |
| GRU | 0.972 (58%) | 0.861 (89%) | 0.705 (94%) | 0.720 (91%) | 0.719 (90%) |
| TCN | **0.927 (76%)** | **0.841 (90%)** | **0.659 (92%)** | **0.671 (92%)** | **0.674 (92%)** |
| DLinear | 0.932 (68%) | 0.863 (86%) | 0.686 (92%) | 0.688 (91%) | 0.692 (92%) |
| PatchTST | 1.008 (48%) | 0.865 (85%) | 0.670 (93%) | 0.676 (94%) | 0.689 (93%) |

**Sampled vs dense.** The harness's 160 origins are a 6-h grid subsampled to 18-h
spacing, so every sampled forecast is issued at one of four local times (≈ 02, 08, 14,
20 h). The dense logs issue a forecast at every 5-minute step (rolling: ~31,500 targets
per link and horizon), so they are the fairer estimate. They tell the same story with an
even stronger 8-hour result (0.489, PatchTST: about half the typical link's error of
persistence).

**Block mode** forecasts each window in full from its start and averages the error over
leads 1…H — what a scheduler consuming one forecast per 8-hour shift would see. There
the fully trained TCN is the best model at every horizon.

**Why 8 hours is the sweet spot.** Persistence ("8 h from now = now") lands on a different
part of the daily cycle; its error peaks at 8 h (pooled WAPE 57.5 % sampled, 60.6 %
dense) and drops again at 1 day (43.8 / 41.4 %). The learned models' relative advantage
is largest exactly there.

Pooled WAPE (Σ|error| / Σ|traffic|, all links) and volume-aggregated errors are in
`aggregate.csv` / `wape.csv` of each table set; hourly-volume and daily-volume WAPE are
much lower than 5-minute WAPE (TCN, 8 h, block: 27.5 % → 22.7 % → 11.1 %), because most
5-minute error cancels within the hour.

## 3. Experiments: month window and smoothing

Report: [`reports/month-window-smoothing.html`](reports/month-window-smoothing.html).
Each variant changes one thing against the year models and is compared on identical
targets, paired per link (median change, share of links improved, Wilcoxon signed-rank
test; ° = not significant at p < 0.01). Recipe: [`recipes/window_smoothing/`](../recipes/window_smoothing/).

### E1 — a 30-day input window

**E1 — 1-month vs 1-week window, rolling: median per-link MAE change (links improved)**

| Comparison | 15 min | 1 h | 8 h | 1 day | 1 week |
|---|---:|---:|---:|---:|---:|
| GBDT | +0.5% (38%) | +0.6% (36%) | −0.1% (51%)° | −2.6% (78%) | −3.0% (80%) |
| GRU | +0.4% (43%) | +0.1% (49%)° | −1.1% (60%) | −0.1% (51%)° | +0.9% (37%) |
| TCN (both 20 epochs) | +4.8% (4%) | +3.1% (7%) | −2.1% (78%) | −4.7% (90%) | −3.2% (86%) |
| DLinear | −0.2% (55%)° | −1.0% (67%) | −3.1% (81%) | −3.4% (82%) | −1.2% (71%) |
| PatchTST | −0.5% (55%)° | −0.8% (59%)° | −1.7% (72%) | −2.8% (81%) | −2.8% (90%) |

- **Helps 1-day and 1-week forecasts, not short ones.** At 1 day: −2.6 % (GBDT, monthly
  lag features), −3.4 % (DLinear), −2.8 % (PatchTST), −4.7 % (TCN at a matched 20-epoch
  budget), each on 78–90 % of links. At 15 min / 1 h: within ±1 %, TCN clearly worse.
- **GRU does not use it** (−0.1 % at 1 day, n.s.; +0.9 % at 1 week).
- **The gain is a weekend effect** (PatchTST, 1 day: −2.5 / −2.9 points of pooled WAPE
  on Saturdays / Sundays, ±1 on weekdays) and grows with how weekly a link is
  (DLinear: −1.6 % on the least weekly third of links, −5.3 % on the most weekly).
- **Cost** per training step: ×1.3 (PatchTST, constant token budget), ×4.1 (DLinear),
  ×4.2 (GRU), ×9.0 (TCN, 12 levels).
- **Best ratios** (dense rolling on the experiments' common targets — every 15 min, so
  slightly different from §2's every-5-min tables): 1 day 0.749 → 0.733 (GBDT with month
  features), 8 h 0.487 → 0.477 (PatchTST), 1 week 0.856 → 0.830 (GBDT).

### E2 — smooth instead of spiky

The learned forecasts are already far smoother than the traffic (roughness = mean
5-minute change of the forecast ÷ that of the traffic):

**Roughness and volume error, block mode, 8 h ahead**

| Forecast | roughness | WAPE % (5 min) | WAPE % (hourly volume) | WAPE % (daily volume) | peak bias % |
|---|---:|---:|---:|---:|---:|
| TCN raw | 0.18 | 27.5 | 22.7 | 11.1 | −23.4 |
| TCN + 1 h MA | 0.09 | 27.6 | 22.8 | 11.1 | −23.9 |
| TCN 1 h target | 0.09 | 27.8 | 22.8 | 10.9 | −24.4 |
| PatchTST raw | 0.26 | 27.8 | 23.3 | 11.6 | −21.6 |
| PatchTST + 1 h MA | 0.10 | 28.1 | 23.3 | 11.6 | −22.4 |
| PatchTST 1 h target | 0.10 | 28.2 | 23.3 | 11.7 | −23.1 |
| GRU raw | 0.16 | 28.5 | 23.7 | 11.3 | −24.1 |
| GRU + 1 h MA | 0.08 | 28.5 | 23.7 | 11.4 | −24.3 |
| GRU 1 h target | 0.10 | 28.2 | 23.2 | 11.0 | −24.4 |
| DLinear raw | 0.45 | 29.0 | 24.8 | 12.7 | −22.4 |
| DLinear + 1 h MA | 0.12 | 29.5 | 24.9 | 12.8 | −23.8 |
| DLinear 1 h target | 0.16 | 31.4 | 26.8 | 14.8 | −19.0 |

**E2a — post-smoothing, block: median per-link MAE change vs the raw forecast**

| Comparison | 15 min | 1 h | 8 h | 1 day | 1 week |
|---|---:|---:|---:|---:|---:|
| TCN + 15m MA | +0.3% (23%) | 0.0% (42%) | 0.0% (67%) | 0.0% (70%) | 0.0% (66%) |
| TCN + 1h MA | +5.4% (8%) | +1.2% (10%) | 0.0% (46%) | 0.0% (51%) | 0.0% (53%) |
| TCN + 3h MA | +17.1% (4%) | +7.3% (3%) | +0.9% (23%) | +0.9% (21%) | +0.7% (19%) |
| GRU + 15m MA | +0.1% (34%) | −0.1% (75%) | −0.1% (87%) | 0.0% (82%) | 0.0% (82%) |
| GRU + 1h MA | +2.3% (14%) | +0.2% (35%) | −0.1% (76%) | −0.2% (77%) | −0.1% (73%) |
| GRU + 3h MA | +11.8% (5%) | +3.9% (6%) | 0.0% (50%)° | 0.0% (52%)° | +0.1% (45%) |
| PatchTST + 15m MA | +0.4% (14%) | +0.1% (27%) | 0.0% (47%) | 0.0% (52%) | 0.0% (44%) |
| PatchTST + 1h MA | +4.5% (1%) | +1.5% (4%) | +0.3% (30%) | +0.1% (36%) | +0.1% (37%) |
| PatchTST + 3h MA | +15.5% (1%) | +7.1% (2%) | +1.9% (12%) | +1.7% (13%) | +1.2% (12%) |
| DLinear + 15m MA | +0.7% (31%) | 0.0% (49%) | −0.1% (55%)° | −0.1% (53%)° | 0.0% (53%)° |
| DLinear + 1h MA | +7.9% (8%) | +1.3% (32%) | +0.3% (43%) | +0.1% (43%) | 0.0% (46%) |
| DLinear + 3h MA | +21.8% (1%) | +6.4% (5%) | +1.9% (22%) | +1.9% (20%) | +1.8% (19%) |

**E2b — trained on a 1-h smoothed target, block: median per-link MAE change**

| Comparison | 15 min | 1 h | 8 h | 1 day | 1 week |
|---|---:|---:|---:|---:|---:|
| TCN | +7.1% (10%) | +3.9% (13%) | +2.5% (18%) | +3.7% (6%) | +3.2% (10%) |
| GRU | +3.5% (28%) | +2.1% (30%) | +1.5% (34%) | −1.5% (63%) | +4.4% (21%) |
| PatchTST | +4.0% (20%) | +3.0% (23%) | +3.2% (16%) | +3.6% (13%) | +3.7% (15%) |
| DLinear | +12.2% (3%) | +8.6% (3%) | +8.0% (2%) | +3.6% (1%) | +3.4% (2%) |

- **Post-smoothing up to 1 h is nearly free at 8 h and longer** (−0.2 … +0.3 %) and cuts
  roughness 2–4×; hourly- and daily-volume errors don't move. It hurts at 15 min / 1 h,
  and a 3-h window costs up to ~2 %.
- **Retraining on a smoothed target** gives the same smoothness at a higher price
  (+1.5 … +8 % at 8 h); the one exception is GRU at 1 day (−1.5 %, 63 % of links).

## 4. Carbon bills from the forecasts

The traffic forecasts feed per-party carbon bills ([carbon.md](carbon.md), `recipes/carbon`):
over the same test period, 2.66 t CO₂ (location-based, combustion only) split among 306
parties (customers and external networks), settled per hour and forecast a day ahead from
the block logs (`day` horizon). Daily bill error, WAPE across parties:

| Traffic forecast | Actual grid factor | Day-ahead grid factor (same hour yesterday) |
|---|---|---|
| PatchTST | **11.5 %** | **31.1 %** |
| TCN | 11.7 % | 31.2 % |
| GRU | 11.7 % | 31.3 % |
| DLinear | 12.6 % | 31.5 % |
| seasonal naive, day / week | 14.8 % / 20.8 % | 32.9 % / 37.5 % |
| persistence | 24.0 % | 38.9 % |
| actual traffic | 0 | 28.0 % |

The ranking of the traffic models carries over from the link-level metrics, but the bill
error is dominated by the grid-factor forecast: better grid forecasts are worth more than
better traffic models. GBDT has no block log (it forecasts one point per horizon), so it
is not in the table.

The total traffic (bytes counted once at the doors, mean 261 Gb/s) forecast a day ahead:
a GBDT trained on the total alone reaches 11.1 % hourly WAPE, the same as the sum of the
TCN door forecasts (11.1 %; daily 7.9 % vs 7.2 %), against 16.7 % for the same hour last
week and 26.5 % for persistence. It does not replace the door forecasts in the bill,
which depends only on each party's share ([carbon.md](carbon.md#total-network-traffic)).

## 5. Carbon rates: one model for the total traffic

For the flat carbon rate (backbone emissions / bytes carried) only the total traffic needs
forecasting. Trained on that single series with the year flags, day-ahead hourly WAPE:
TCN 12.0 %, GRU 13.0 %, PatchTST 13.9 %, DLinear 13.9 %, historical average 14.3 %, same
time last week 16.7 %, persistence 29.2 % (a GBDT on the hourly total 11.1 %; the sum of
the per-door forecasts 11.1 %). Every-origin WAPE by horizon is in
`artifacts/carbon_rates/flat/traffic_by_horizon.csv`; the path-based alternative and the
flow-vs-counters check are in [carbon.md](carbon.md#carbon-rates-flat-or-by-path).

## 6. Second dataset: Abilene (NetFlow, 2004)

The same ladder on Abilene (`recipes/abilene`; 27 link and router series, and the network
total; train to 2004-07-24, validation 2 weeks, test 2004-08-14 → 09-10, no calendar
features). Per-link median MAE ratio vs persistence, 112 sampled origins:

| Horizon | 15 min | 1 h | 8 h | 1 day | 1 week |
|---|---|---|---|---|---|
| best model | Holt-Winters **0.921** | DLinear **0.931** | PatchTST **0.555** | PatchTST **0.677** | GRU **0.798** |

The shape matches SWITCHlan's (0.787 / 0.538 / 0.748 / 0.849 at 1 h / 8 h / 1 day / 1 week):
little to gain at 15 minutes, the most at 8 hours. Holt-Winters and SARIMA, which refit per
link and origin, help only at 15 minutes and fall far behind persistence at a week
(1.52 and 1.74). One model on the network total,
day-ahead hourly WAPE: PatchTST 11.9 %, DLinear 13.2 %, TCN 13.7 %, GRU 16.2 %,
historical average 18.7 %, same time last week 20.6 %, persistence 21.4 %. Carbon rates
on Abilene: [carbon.md](carbon.md#second-dataset-abilene-netflow-2004).

## 7. How fine to publish and forecast the rate

From `recipes/carbon_granularity` ([carbon.md](carbon.md#granularity-and-uncertainty-of-the-rate)):
publish hourly (98.8 % of the 5-minute rate's variance kept, daily statements within 0.4 %),
forecast at 5 minutes and sum or directly at hours (the same hourly forecast; the model
itself is more accurate on hours, 13.6 % → 12.0 % traffic WAPE, because the within-hour
wiggle is unforecastable a day ahead), update hourly (52 % → 38 %
error on SWITCHlan), publish calibrated quantiles (80 % band covers 78.5 %, CRPS −18 %);
fitted curves add nothing over hourly values.

## 8. Forecasting the total for the flat rate

From `recipes/flat_forecast` ([carbon.md](carbon.md#forecasting-the-total-for-the-flat-rate)),
day-ahead hourly WAPE of the total on the test block (choices made on Nov 2025 – Mar 2026):
GBDT as used so far 11.0 %, TCN 12.0 %; Chronos-2 zero-shot 10.7 % (10.7 % on the
development period, where the best GBDT is 11.6 %); final forecast 0.8 × Chronos-2 +
0.2 × GBDT with the GBDT alone on holidays: **10.0 %** (−1.1 points, 95 % interval
0.5–1.8). Chronos-2 on 5-minute input 15.4 %; fine-tuned 11.7 %; TCN with a symmetric loss
11.3 % (12.1 % with the peak-aware loss, three seeds each); with the TCN as a third member 9.8 %
(test only). Abilene: Chronos-2 12.2 % vs 15.4–15.6 % for trained models.

## 9. Caveats

- Single seed per model; the measured run-to-run noise is about ±1 % of the median ratio.
- The month TCN comparison is at a matched 20-epoch budget, not to convergence.
- The year runs' train/val dates are reconstructed (the data and test window are exact).
- Experiment logs use TF32 and every 3rd origin (both far below the effects reported).

Earlier, smaller runs (2024 → mid-2026): [archive/early-runs.md](archive/early-runs.md).
