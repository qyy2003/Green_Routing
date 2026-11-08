# Recipes

Scripted, resumable reproductions of the published results. Run from the repository root
with the environment active (`conda activate green`; `pip install -e ".[dev]"` done).

| Recipe | What | Guide |
|---|---|---|
| [`year/`](year/) | the year-scale benchmark behind the carbon-accounting report, with a cross-check against the reference machine (`expected.json`) | [docs/reproduce.md](../docs/reproduce.md#the-year-benchmark) |
| [`window_smoothing/`](window_smoothing/) | the 1-month-window and smoothing experiments, their analysis and HTML report | [docs/reproduce.md](../docs/reproduce.md#the-experiments-month-window-smoothing) |
| [`carbon_rates/`](carbon_rates/) | flat vs path-based carbon rate: every forecasting tier on the total traffic, path allocation from counters, flow-vs-counters check on Abilene, report | [docs/carbon.md](../docs/carbon.md#carbon-rates-flat-or-by-path) |
| [`abilene/`](abilene/) | the second dataset (Abilene, NetFlow 2004): the forecasting ladder on its links and total, its energy and 2004 grid factors, flat and path rates from measured flows | [docs/data.md](../docs/data.md#second-dataset-abilene-netflow-2004), [docs/carbon.md](../docs/carbon.md#second-dataset-abilene-netflow-2004) |
| [`carbon_granularity/`](carbon_granularity/) | at what grain to publish and forecast the carbon rate, how often to update it, curves vs values, calibrated distributions; report | [docs/carbon.md](../docs/carbon.md#granularity-and-uncertainty-of-the-rate) |
| [`flat_forecast/`](flat_forecast/) | how best to forecast the total traffic a day ahead for the flat rate: models (incl. pretrained Chronos-2 / TimesFM), 5-minute vs hourly, holidays and semesters, training process, ensembles; Abilene check; report | [docs/carbon.md](../docs/carbon.md#forecasting-the-total-for-the-flat-rate) |
| [`internet2/`](internet2/) | the second current network from public data: Internet2's port counters, port labels and PoP power (GlobalNOC TSDS), hourly US grid factors (EIA-930), the 2021 MPLS tunnel mesh as a measured matrix; forecasting ladder on its backbone links, carbon rates and bills | [docs/worklog/2026-11-02-internet2.md](../docs/worklog/2026-11-02-internet2.md) |
| [`green_paths/`](green_paths/) | exploratory: greenest-path routing on Internet2, one party alone vs everyone (empty links, lock-in, overloads, resilience) | `simulate.py` docstring, AGENTS.md §6 |
| [`paper/`](paper/) | the IMC paper's evaluation: idle-link cap, day-ahead bill by accounting period, entity bills and city-pair rates, CIRo-style path choice and decentralized link sleeping on day-ahead rates, all six plots | [paper/README.md](paper/README.md) |
| [`carbon/`](carbon/) | link roles, energy inventory, grid factors, allocation, settled and day-ahead forecast carbon bills, their report and a check against `expected.json` | [docs/carbon.md](../docs/carbon.md), [docs/reproduce.md](../docs/reproduce.md#the-carbon-bills) |

```bash
recipes/year/run.sh all                  # check, train, merge, dense, verify  (~4.5 h on an L40S)
recipes/window_smoothing/run.sh all      # needs the year run                  (~13 h)
recipes/carbon/run.sh all                # needs the year run's dense logs     (~3 min warm)
recipes/abilene/run.sh all               # the second dataset; downloads 170 MB  (~1 h)
recipes/carbon_rates/run.sh all          # needs recipes/year, carbon, abilene  (~30 min)
recipes/flat_forecast/run.sh all         # needs carbon, carbon_rates; pip install -e ".[fm]"  (~3 h)
recipes/internet2/run.sh all             # public data only; ~3 h download cold, then ~2.5 h
recipes/paper/run.sh                     # needs carbon, carbon_rates, flat_forecast, internet2  (~30 min)
```

Every step can be run on its own and re-run: finished parts are skipped, interrupted
training resumes from its last epoch.

| Variable | Default | Meaning |
|---|---|---|
| `GR_ARTIFACTS_DIR` | `<repo>/artifacts` | where runs go |
| `PARALLEL` | 5 (year), 3 (experiments) | training jobs at once — lower it on smaller GPUs |
| `THREADS_PER_JOB` | 4 | CPU threads per job (unbounded threads overload the CPU) |
| `DENSE_BATCH` | per model | origins per GPU pass in the dense logs (lower on small GPUs) |
| `LOSS` | `peak` | year recipe: `peak` = `--under-weight 2` into `pred_year` (the reference); `symmetric` = `--under-weight 1` into `pred_year_sym` (baselines and GBDT copied from `pred_year`) |
| `WORKERS`, `YEAR`, `DOWNLOAD` | 16, 2025, 0 | carbon recipe: worker processes, inventory year, `1` = fetch Electricity Maps again |

`lib.sh` holds the shared helpers (`pool` runs named jobs with bounded parallelism and a
log per job).
