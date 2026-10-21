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

## 4. Caveats

- Single seed per model; the measured run-to-run noise is about ±1 % of the median ratio.
- The month TCN comparison is at a matched 20-epoch budget, not to convergence.
- The year runs' train/val dates are reconstructed (the data and test window are exact).
- Experiment logs use TF32 and every 3rd origin (both far below the effects reported).

Earlier, smaller runs (2024 → mid-2026): [archive/early-runs.md](archive/early-runs.md).
