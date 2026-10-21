# Reproducing the results on a new machine

Two studies are scripted end to end in [`recipes/`](../recipes/):

| Recipe | Reproduces | Time on an L40S | Disk |
|---|---|---|---|
| [`recipes/year/`](../recipes/year/) | the year-scale benchmark behind the carbon-accounting report ([`docs/reports/carbon-accounting-report.html`](reports/carbon-accounting-report.html)) | ~4.5 h | ~8 GB + 20 GB data cache |
| [`recipes/window_smoothing/`](../recipes/window_smoothing/) | the month-window and smoothing experiments ([`docs/reports/month-window-smoothing.html`](reports/month-window-smoothing.html)) | ~13 h (needs the year run) | ~15 GB |

Both write under `$GR_ARTIFACTS_DIR` (default `artifacts/`), skip finished steps and
resume interrupted ones, so a step can simply be run again.

## Before you start

Follow [setup.md](setup.md) to the end: environment, credentials, `gr-data check` OK,
`pytest -m e2e` passing. Then, from the repository root with the environment active:

```bash
recipes/year/run.sh check
```

This builds the year panel from the bucket (the first time it downloads and caches the
interface CSVs, ~20 GB) and compares its fingerprint with the reference machine's:

```text
== data
  PASS    n_links                       534
  PASS    links_sha1                    ad22d2bc5bf52333553161a8e1d735e48019abb8
  PASS    finite_count                  57246264
  PASS    finite_sum_mbps               1.498167e+11 (relative Δ 0.0e+00)
```

**If the fingerprint fails, stop**: the panel built from today's bucket is not the one
the published results were computed on, and every later number would differ for that
reason alone. Which line fails says what changed:

| Fails | Meaning |
|---|---|
| `n_steps`, `t_first`, `t_last` | the window itself: check `flags.sh` and the code version |
| `n_links`, `links_sha1` | the cohort: links crossed the 90 % coverage or 1 Mbit/s line (backfilled gaps, a re-export, renamed interfaces) |
| only `finite_count` / `finite_sum_mbps` | same links, different values (backfill or a re-export) |

To see which links, dump this machine's cohort and diff it with the reference run's
(`links` in `artifacts/pred_year/merged/config.json` on the reference machine):

```bash
python recipes/year/verify.py --data-only --dump-links links_new.txt
python -c "import json; print('\n'.join(json.load(open('ref_config.json'))['links']))" > links_ref.txt
diff links_ref.txt links_new.txt
```

**Pinning the data instead.** The panel cache is keyed by window, column and coverage
only, so copying the reference machine's `artifacts/panel_19ea24bef019.npz` (330 MB) into
`artifacts/` makes every step use exactly the reference data, whatever the bucket holds
now, and the year recipe then needs no S3 credentials at all (`run.sh check` only warns
about the source). That isolates the hardware/software part of the cross-check; building
the panel from the bucket is the stronger test, because it also covers the data path.

## The year benchmark

```bash
recipes/year/run.sh train     # ~2 h: five parts in parallel (PARALLEL=5)
recipes/year/run.sh merge     # ~10 min: one run directory, all tables
recipes/year/run.sh dense     # ~1.5 h: every-origin logs, verified, scored
recipes/year/run.sh verify    # ~2 min: the cross-check
# or all of it: recipes/year/run.sh all
```

What each step does:

| Step | Detail |
|---|---|
| `train` | `gr-benchmark` with [`flags.sh`](../recipes/year/flags.sh) for five parts — baselines, GBDT, GRU, TCN, DLinear + PatchTST — each with persistence, into `pred_year/part_*`. On the L40S the parts train concurrently in ~1 h 40 min (epochs: TCN 71, PatchTST 71, DLinear 49, GRU 46; GPU memory ≤ 3 GB each). Logs: `pred_year/logs/part_*.out`. On a GPU under 16 GB set `PARALLEL=2`. |
| `merge` | copies records/checkpoints into `pred_year/merged` and rebuilds the tables with `--resume` (no training). |
| `dense` | `gr-dense predict` for all nine models (fp32, every origin), `verify` against the sampled records, `metrics --models untagged --common` into `merged/dense_year_tables/` (and for the viewer). `DENSE_BATCH` lowers the GPU batch on small cards. |
| `verify` | [`verify.py`](../recipes/year/verify.py) against [`expected.json`](../recipes/year/expected.json) and the report. |

### Reading the verification

Each check prints the new value, the reference value and their difference. On the
reference machine itself every Δ is 0:

```text
== sampled origins (harness, 160 origins)
  PASS    persistence MAE 1h (Mbps)     711.89 vs 711.89 (Δ 0.0000)
  PASS    gbdt day                      0.7482 vs 0.7482 (Δ 0.0000)
  PASS    tcn day                       0.7536 vs 0.7536 (Δ 0.0000)
== dense logs, rolling mode
  PASS    persistence MAE 1h (Mbps)     751.61 vs 751.61 (Δ 0.0000)
...
137 pass, 0 warn, 0 fail, 0 missing
== against the report (sampled origins, best learned model per horizon)
  1h                     gru 0.801                tcn 0.787   585 / 569 (persistence 712 / 712)
```

On another machine the baselines should still show Δ 0 and the trained models small
non-zero differences:

| Layer | Expectation |
|---|---|
| data fingerprint | exact |
| baselines (persistence, seasonal-naive, historical average) | exact — deterministic numpy on identical data |
| GBDT | near-exact (CPU, fixed seeds; sklearn version may matter — the reference used 1.9.0) |
| neural models | per-link median ratio within ±0.02 = PASS, ±0.04 = WARN. GPUs differ (cuDNN kernels, TF32), so runs are not bit-identical across machines; the reference run's own noise floor was about ±1 % of the ratio. |

A few WARNs among ~140 checks is normal GPU-to-GPU variation. A FAIL on a learned model,
or many WARNs in one direction, deserves a look: compare `history/<model>.csv` (did it
stop much earlier or later?), check `run.log` for `steps skipped`, and confirm the
package version (`git log -1`).

The headline numbers to expect (per-link median MAE ratio vs persistence, lower is better):

| Horizon | Report | Reference run, sampled origins | Reference run, dense rolling (every origin) |
|---|---|---|---|
| 1 h | 0.801 (GRU) | 0.787 (TCN) | 0.780 (TCN, PatchTST) |
| 8 h | 0.545 | 0.538 (TCN) | 0.489 (PatchTST) |
| 1 day | 0.743 (TCN) | 0.748 (GBDT) | 0.751 (PatchTST) |
| 1 week | 0.842 (GBDT) | 0.849 (GBDT) | 0.859 (GBDT) |

Persistence pooled MAE (sampled): 711.9 / 1540.6 / 1060.5 / 1144.7 Mbit/s at 1 h / 8 h /
1 day / 1 week — equal to the report's 712 / 1541 / 1061 / 1145. Full tables:
[results.md](results.md).

### Looking at it

```bash
recipes/year/run.sh viewer        # gr-viewer on artifacts/pred_year/merged, port 8138
```

## The experiments (month window, smoothing)

Needs the year run (its models are the baselines of every comparison, and the new logs
are written next to theirs).

```bash
recipes/window_smoothing/run.sh train     # ~10 h on an L40S with PARALLEL=3
recipes/window_smoothing/run.sh dense     # ~2-3 h
recipes/window_smoothing/run.sh analyze   # ~20 min (metrics, paired tests, cost benchmark)
recipes/window_smoothing/run.sh report    # -> artifacts/exp/report/month-window-smoothing.html
```

| Run | Change against the year flags |
|---|---|
| `month_{gru,patchtst,dlinear,gbdt}` | 30-day window (`--lookback 8640`), batch pinned at 85, PatchTST patches 64/32, GBDT month lag features |
| `month_tcn` + `wk20_tcn` | matched pair, both 20 epochs: a month TCN costs ~9× a week TCN per step |
| `starget_{gru,tcn,patchtst,dlinear}` | trained on a 1-h smoothed target |
| `+ma15m/+ma1h/+ma3h` logs | the year models' paths, post-smoothed (no training) |

Memory: month GRU ~12–20 GB, month TCN ~11 GB, the rest ≤ 3 GB — `PARALLEL=3` on 46 GB,
1–2 on 24 GB. The dense logs use TF32 and every 3rd origin, as in the original study.

Results to expect (median per-link MAE change, dense rolling; full tables in
[results.md](results.md)): month window at 1 day −2.6 % (GBDT), −3.4 % (DLinear),
−2.8 % (PatchTST), −4.7 % (TCN, 20 epochs), GRU ≈ 0; at 15 min / 1 h ≈ 0 or worse.
Post-smoothing ≤ 1 h at 8 h+ horizons: −0.2 … +0.3 %. Smoothed-target training at 8 h:
+1.5 … +8 %. These came from single runs; expect the same signs and similar sizes, not
identical decimals.

## Moving results between machines

Everything a run produced is in its directory; copy it whole:

```bash
rsync -a --info=progress2 artifacts/pred_year/merged/ newbox:Green_Routing/artifacts/pred_year/merged/
rsync -a artifacts/panel_19ea24bef019.npz newbox:Green_Routing/artifacts/   # the viewer needs the panel
```

`dense/` is most of the size (~6 GB for the year models, ~15 GB more with the
experiments). To ship a single model with its provenance, use the zoo
(`gr-zoo add` / `gr-zoo pack`, [forecasting.md](forecasting.md#model-zoo-gr-zoo)).
GBDT checkpoints are scikit-learn pickles and load only with the same scikit-learn version.

A stronger cross-check than retraining: copy this machine's `merged/checkpoints/` to the
new box and run `gr-dense predict` + `gr-dense verify` there — the logs should match the
sampled records to ~10⁻⁴ even on different hardware.

## Rebuilding expected.json

Only from a run you trust as the new reference:

```bash
python recipes/year/make_expected.py --run artifacts/pred_year/merged \
    --dense-tables artifacts/pred_year/merged/dense_year_tables --note "why"
```
