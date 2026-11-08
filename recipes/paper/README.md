# recipes/paper — the IMC paper's evaluation

Produces every number and plot of the paper's Section 6 (`sections/evaluation.tex` in the Overleaf
checkout `6aea077f0929c9a3a1ac5660/`, gitignored). `run.sh` runs the scripts in order; outputs go
to `$GR_ARTIFACTS_DIR/paper/`, PDFs to the checkout's `figures/`.

| Script | What |
|---|---|
| `i2_pre_setup.py` | I2 test setup for the 28 days before the test block (runs `internet2/carbon.py` with a moved window) |
| `i2_rates.py` | I2 settled and day-ahead per-resource rates under the idle-link cap; city pairs, entity bills, rate errors |
| `switch_rates.py` | SWITCHlan path rates under the cap; city pairs (towns), entity bills (+ access equipment) |
| `bill_periods.py` | network bill error by accounting period, both networks; perfect-traffic / perfect-grid variants |
| `facts.py` | per-GB rate over a year, variance split grid vs traffic, daily cycle, kWh/GB, I2 power persistence |
| `steering.py` | `unilateral` (CIRo-style), `everyone` (closed loop on day-ahead rates), `central` (Hypnos-like reference) |
| `power_traffic.py` | Fig. 1: hourly power and traffic of SWITCHlan's 116 metered routers (2024) and Internet2's PoPs, checked against the published summaries |
| `appendix.py` | the appendix: every forecaster on totals and links with training cost, context / resolution / horizon, Level 1 vs 2, grid candidates, decision quality, the cap's k (`steering.py --k=1/3/none`); PDFs to `figure_appendix/` |
| `total_neural.py` | the four trained architectures (hourly versions, `ladder_diagnosis/hourly_train.py` classes) on each network's hourly total, day ahead, three seeds; fills Internet2's total in the appendix model table |
| `tradeoff.py` | `sweep` (Timestral / Chronos-2 / TimesFM-2.5 at 1-48 weeks on the totals), `timing` (inference time of one day per configuration, one GPU, run alone), `figures` (Pareto: error vs inference time; learning curve: error vs training history) |
| `text_numbers.py` | numbers combining the above |
| `margin.py` | the margin on the posted flat rate (2026-11-08): `forecasts` (Timestral on each network's development period; SWITCHlan traffic after the test block from the S3 cache), `score` (choice of m, sensitivity, months, path-rate own share), `check_paths` / `check_everyone` (path choices unchanged under the margin); outputs `paper/margin/` |
| `factcheck.py` | numbers for the fact check (REPORT.md in the paper folder): energy and carbon per byte on common windows, window definitions of the bill error, dark-span statistics, the non-robust cap, timings; `paper/factcheck.json` |
| `figures.py`, `paper.mplstyle` | the seven PDFs (Fig. 1 + six of Sec. 6) (`--png=DIR` for previews) |

The idle-link cap (`carbon.path_rate.idle_cap` / `capped_rates`, constants in
`carbon/assumptions.py`): no resource charges more than median + 2 robust sigmas (1.4826 x MAD)
of the pooled log rates of all resources over the previous 28 days; the rest is spread flat.
