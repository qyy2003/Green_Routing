# Recipes

Scripted, resumable reproductions of the published results. Run from the repository root
with the environment active (`conda activate green`; `pip install -e ".[dev]"` done).

| Recipe | What | Guide |
|---|---|---|
| [`year/`](year/) | the year-scale benchmark behind the carbon-accounting report, with a cross-check against the reference machine (`expected.json`) | [docs/reproduce.md](../docs/reproduce.md#the-year-benchmark) |
| [`window_smoothing/`](window_smoothing/) | the 1-month-window and smoothing experiments, their analysis and HTML report | [docs/reproduce.md](../docs/reproduce.md#the-experiments-month-window-smoothing) |
| [`carbon/`](carbon/) | link roles, energy inventory, grid factors, allocation, settled and day-ahead forecast carbon bills, their report and a check against `expected.json` | [docs/carbon.md](../docs/carbon.md), [docs/reproduce.md](../docs/reproduce.md#the-carbon-bills) |

```bash
recipes/year/run.sh all                  # check, train, merge, dense, verify  (~4.5 h on an L40S)
recipes/window_smoothing/run.sh all      # needs the year run                  (~13 h)
recipes/carbon/run.sh all                # needs the year run's dense logs     (~3 min warm)
```

Every step can be run on its own and re-run: finished parts are skipped, interrupted
training resumes from its last epoch.

| Variable | Default | Meaning |
|---|---|---|
| `GR_ARTIFACTS_DIR` | `<repo>/artifacts` | where runs go |
| `PARALLEL` | 5 (year), 3 (experiments) | training jobs at once — lower it on smaller GPUs |
| `THREADS_PER_JOB` | 4 | CPU threads per job (unbounded threads overload the CPU) |
| `DENSE_BATCH` | per model | origins per GPU pass in the dense logs (lower on small GPUs) |
| `WORKERS`, `YEAR`, `DOWNLOAD` | 16, 2025, 0 | carbon recipe: worker processes, inventory year, `1` = fetch Electricity Maps again |

`lib.sh` holds the shared helpers (`pool` runs named jobs with bounded parallelism and a
log per job).
