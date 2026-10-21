# Recipes

Scripted, resumable reproductions of the published results. Run from the repository root
with the environment active (`conda activate green`; `pip install -e ".[dev]"` done).

| Recipe | What | Guide |
|---|---|---|
| [`year/`](year/) | the year-scale benchmark behind the carbon-accounting report, with a cross-check against the reference machine (`expected.json`) | [docs/reproduce.md](../docs/reproduce.md#the-year-benchmark) |
| [`window_smoothing/`](window_smoothing/) | the 1-month-window and smoothing experiments, their analysis and HTML report | [docs/reproduce.md](../docs/reproduce.md#the-experiments-month-window-smoothing) |

```bash
recipes/year/run.sh all                  # check, train, merge, dense, verify  (~4.5 h on an L40S)
recipes/window_smoothing/run.sh all      # needs the year run                  (~13 h)
```

Every step can be run on its own and re-run: finished parts are skipped, interrupted
training resumes from its last epoch.

| Variable | Default | Meaning |
|---|---|---|
| `GR_ARTIFACTS_DIR` | `<repo>/artifacts` | where runs go |
| `PARALLEL` | 5 (year), 3 (experiments) | training jobs at once — lower it on smaller GPUs |
| `THREADS_PER_JOB` | 4 | CPU threads per job (unbounded threads overload the CPU) |
| `DENSE_BATCH` | per model | origins per GPU pass in the dense logs (lower on small GPUs) |

`lib.sh` holds the shared helpers (`pool` runs named jobs with bounded parallelism and a
log per job).
