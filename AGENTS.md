# Green Routing — knowledge base

For whoever picks this project up next, human or agent. It records what is not obvious
from the code: the state of the work, the facts the numbers rest on, the decisions taken,
and the traps already fallen into. The user-facing docs are in `docs/`; this file links to
them rather than repeating them.

Last updated: 2026-10-30.

## 1. State of the project

- **Goal**: forecast the carbon bill of every user/stakeholder of the SWITCH research
  backbone: traffic on every link (5-minute SNMP data, 15 minutes to a week ahead) and the
  grid's carbon intensity, combined into hourly per-party bills, settled and a day ahead;
  later carbon-aware ("green") routing. The forecasting question is how much each tier of
  model complexity buys over strong naive baselines, measured honestly.
- **Done**:
  - Year-scale benchmark (534 links, 2025-07 → 2026-07, 111-day test) reproduced and
    verified against the original report; persistence matches it to the Mbit/s.
  - Dense prediction logs: every model's forecast at every 5-minute origin, in two
    layouts (rolling, block); viewer on top.
  - Experiments: 1-month input window and forecast smoothing (results below; report
    in `docs/reports/month-window-smoothing.html`).
  - Repository reorganised into an installable package (`src/green_routing`), command-line
    tools, scripted recipes with verification, docs, and tests (unit + end-to-end on
    synthetic data). All tests pass.
  - Carbon accounting (2026-10-29/30, [docs/carbon.md](docs/carbon.md)): link roles from
    traffic matching + SWITCH reverse DNS (`green_routing.network`, `gr-network`); energy
    inventory, Electricity Maps grid factors, access/backbone allocation to doors and
    parties, settled and day-ahead forecast bills over the year test block
    (`green_routing.carbon`, `gr-carbon`); `recipes/carbon` with report and `verify`
    (23/23 PASS). Moved from scripts in `artifacts/carbon_screening_2025` and
    `artifacts/link_roles`, which stay as the frozen reference it was checked against.
- **Next expected tasks**: improve the day-ahead grid-factor forecast (it causes most of
  the bill error: log Electricity Maps' own forecast daily and score it); get the 31
  routers missing from the export back from SWITCH. The year benchmark was cross-verified
  on a second machine (135 PASS / 2 WARN).

## 2. Working agreements with the owner

- Python dependencies go into a **dedicated conda environment** (`green`, Anaconda — not
  Miniforge); never `sudo apt` or system `pip` for Python packages.
- Don't commit or push unless asked. `artifacts/` is gitignored (runs, logs, caches).
- Claims about results must be checked against the data before they are reported; say
  what was not verified. Negative results get reported like positive ones.
- Reports for the owner have been published as HTML pages (claude.ai artifacts); copies
  of the two forecasting reports are in `docs/reports/`. The carbon-bills report
  (https://claude.ai/artifact/7QE4r5LaNaHAnvXNrzz6bk) is deliberately **not** copied there:
  the repository is public, and it carries operator detail and Electricity-Maps-derived
  aggregates. Keep licensed grid data, the API key and `artifacts/reference/` out of git.

## 3. Where things are

| Path | What |
|---|---|
| `src/green_routing/dataset/` | data access: `config` (every env setting), `storage` (S3 + cache), `loaders` (era coalescing), `cohort`, `tensor`, `splits`, `topology` |
| `src/green_routing/prediction/` | `run.py` (`gr-benchmark`, model assembly, `apply_overrides`), `harness.py` (backtest, tables), `models_*.py`, `torch_common.py` (trainer), `dense.py` (`gr-dense`), `metrics.py`, `data.py` (panels), `zoo.py`, `calendar_features.py` |
| `src/green_routing/viewer/` | `gr-viewer` (server.py + static/) |
| `src/green_routing/network/` | `gr-network`: `traffic` (one pass over every interface CSV), `matching` (link ends, bundles), `roles` (tiers, roles), `dns` (reverse-DNS sweep and far ends) |
| `src/green_routing/carbon/` | `gr-carbon`: `assumptions` (every assumed number), `inventory`, `grid` (Electricity Maps, day-ahead rules), `allocation` (pools, doors, parties), `bills` |
| `recipes/carbon/` | `run.sh`, `build_report.py` + `report_template.html`, `verify.py` + `expected.json` |
| `artifacts/reference/` | curated inputs not in the bucket (sites, external links, DNS snapshots, Zoo sample); gitignored, copy between machines |
| `src/green_routing/paths.py` | repo root, `GR_ARTIFACTS_DIR` |
| `recipes/year/` | flags of the year run, `run.sh`, `verify.py`, `expected.json`, `make_expected.py` |
| `recipes/window_smoothing/` | the experiments: `run.sh`, `analyze.py`, `cost.py`, `build_report.py`, `report_template.html` |
| `web/` | topology viewer (stdlib server; `_source.py` reads through the package) |
| `tests/` | `test_units.py`, `test_pipeline.py` (e2e, synthetic mirror in `conftest.py`), `test_live.py` |
| `docs/` | setup, data, forecasting, reproduce, viewers, results, roadmap, reports/, archive/ |
| `tools/` | topology map scripts; reference copies of SWITCH's nightly jobs |

## 4. Environment facts (reference machine)

| | |
|---|---|
| Machine | SWITCHengines VM, 40 cores, 339 GB RAM, NVIDIA L40S 46 GB, Ubuntu, kernel 6.8 |
| GPU driver | `nvidia-driver-580-server-open` 580.173.02 (CUDA 13.0) — the fresh VM had *no* driver; `nvidia-smi` missing ≠ no GPU (`lspci | grep -i nvidia`) |
| Conda | Anaconda at `~/anaconda3`, env `green` = `environment.yml` (Python 3.11.15, pandas 2.2.3, numpy 2.1.3, scipy 1.17.1, scikit-learn 1.9.0, statsmodels 0.14.6, matplotlib 3.11.0, boto3 1.43) |
| PyTorch | 2.13.0+cu130, pip-installed (the conda channel for torch is deprecated) |
| Package | `pip install -e ".[dev]"` → `gr-data`, `gr-benchmark`, `gr-dense`, `gr-zoo`, `gr-viewer`, `gr-build-tensor` |
| Network | public IP (kept out of this public repo); prediction viewer on :8138 (Basic auth, password in `~/.config/green-routing/viewer_password`); topology viewer :8137 |
| Electricity Maps | academic account; key as `export EM_API=...` in `~/.bashrc` (never print it, never commit it); hourly history max 10 days per request |

Anaconda's base env ships pandas 3; the pipeline was never validated on it, hence the pin.

## 5. Data facts

| | |
|---|---|
| Bucket | `network-energy-efficiency-research`, endpoint `https://zhw-b.s3.cloud.switch.ch`, region `eu-central-1`, path-style. The upstream uploader writes `switchlan-load-timeseries`, which 404s for the project's credentials (it was the stale default until 2026-10-21). |
| Credentials | `~/.aws/credentials` `[default]` (EC2-style keys of the SWITCHengines project) |
| Contents | `router-interfaces/` 1,986 CSVs / 215 devices / ~20 GB; also `routers/`, `router-power/`, `eci/`, `transceiver-monitoring/`, `cpu-usage/`. Rewritten daily (cumulative files). |
| Cache | `~/.cache/green-routing/s3`, ETag-validated; first full scan downloads ~20 GB (~10–15 min) |
| CSV quirk | first 1–3 days of every series in raw `ds0..dsN` columns, the rest in named columns; `loaders.read_series` coalesces. Values are bytes/s → ×8/1e6 = Mbit/s. UTC epoch seconds, 5-min grid. |
| Year panel | `artifacts/panel_19ea24bef019.npz`: 835 links × 108,001 steps (2025-07-01 → 2026-07-11) at coverage ≥ 0.9; **534** after `--min-mean-mbps 1`. Fingerprint (links sha1 `ad22d2bc…`, finite count 57,246,264, finite sum 1.498167e11) in `recipes/year/expected.json`. Rebuilt from the bucket as it was on 2026-10-21: identical (Δ 0), so a later mismatch means the bucket changed after that date. |
| Year split | train → 2026-02-15, val 2026-02-22 → 03-15, test 2026-03-22 → 07-11 (index 76,032, 31,969 targets). Train/val dates are reconstructed; the test part is verified. |
| Sampled origins | 160, stride 72 subsampled to ~18 h spacing → only **4 times of day** (≈ 02/08/14/20 h local). Dense logs fix this. |
| Holidays | `calendar_features.HOLIDAYS` covers 2025-06-01 → 2026-08-31; `--calendar` refuses dates outside. Extend it before using newer windows. |
| Topology | not in the bucket; the committed `switch-network-topology.txt` is a device-level reconstruction (fine for the graph model, no interface names). The real one is in the research checkout (`GR_RESEARCH_ROOT`). |
| Power | `router-power/` (PSU) **ends 2024-11-01**, four schemas (`config.PSU_SCHEMAS`: Pin W, Pin_mW, U×I at 12 V, Arista U×I ×1e-4); `eci/apollo/` shelves live; `routers/` headers identify the router family. Router models are not in the bucket: matched from the Network Power Zoo sample (94/114). |
| Missing routers | 31 routers alive in reverse DNS are absent from the SNMP export (ix3 since 2024-06, ce1 2025-01, ba3 2025-05, ez31, si2, ge3, ez11, bs1, …); ix3/ce1/ba3 hold most external links. SWITCH was asked to re-export them. |
| Reverse DNS | SWITCH names router ports (`swiEZ2-400GE-0-0-0-3` = `ez2/fourhundredgige0_0_0_3`); link ends share a /31, a named /30 or a /64. Snapshot 2026-10-30 in `artifacts/reference/dns/`. |

## 6. Numbers to expect

Per-link median MAE ratio vs persistence (lower = better). Full tables:
[docs/results.md](docs/results.md).

| Horizon | Report (RTX 3060, 2026-07) | Reference run, sampled | Dense rolling | Dense block |
|---|---|---|---|---|
| 15 min | — | 0.853 TCN | 0.888 TCN | 0.927 TCN |
| 1 h | 0.801 GRU | 0.787 TCN | 0.780 PatchTST/TCN | 0.841 TCN |
| 8 h | 0.545 | 0.538 TCN | 0.489 PatchTST | 0.659 TCN |
| 1 day | 0.743 TCN | 0.748 GBDT | 0.751 PatchTST | 0.671 TCN |
| 1 week | 0.842 GBDT | 0.849 GBDT | 0.859 GBDT | 0.674 TCN |

- Persistence pooled MAE, sampled: 589.75 / 711.89 / 1540.58 / 1060.54 / 1144.70 Mbit/s
  (15 min … 1 week) — deterministic; any difference means different data.
- `recipes/year/verify.py` against the reference: 137 checks, all PASS on the reference
  machine. Learned models: PASS within ±0.02 of the ratio, WARN within ±0.04.
- Training on the L40S, five parts in parallel: ~1 h 40 min wall clock; TCN 71 epochs
  (92 min), PatchTST 71 (69 min), GRU 46 (40 min), DLinear 49 (18 min).
- Dense logs (fp32): TCN ~40 min, GRU ~14, PatchTST ~14, DLinear < 1, GBDT minutes (CPU).

Carbon bills (`recipes/carbon/expected.json`, [docs/carbon.md](docs/carbon.md#results-reference-run-2026-10-30)):
2025 inventory 192 routers 254.6 MWh, optical 212.9 MWh, site 738.7 MWh, Scope 2 direct
15.7 t; pools access 210 / backbone 529 MWh; 414 doors, 306 parties. Test block: 77.9 kW
at the sites, settled 2.66 t; daily bill WAPE PatchTST 11.7 % (actual grid) / 31.4 %
(grid forecast), persistence 24.5 % / 39.4 %, grid forecast alone 28.0 %.

Experiments (dense, paired per link, 534 links; [report](docs/reports/month-window-smoothing.html)):

- 1-month window at 1 day: GBDT −2.6 %, DLinear −3.4 %, PatchTST −2.8 %, TCN −4.7 %
  (20-epoch matched pair), GRU −0.1 % (n.s.); 15 min / 1 h: ±1 % or worse. Gain is a
  weekend effect and grows with weekly autocorrelation. Step cost ×1.3 (PatchTST) … ×9 (TCN).
- Post-smoothing (centred MA on the forecast path) ≤ 1 h at 8 h–1 week: −0.2 … +0.3 %
  MAE, roughness ÷2–4, hourly/daily-volume WAPE unchanged; hurts at 15 min / 1 h.
- Smoothed-target training (1 h): +1.5 … +8 % at 8 h (GRU at 1 day the exception, −1.5 %).

## 7. History and decisions

| Date | What |
|---|---|
| 2026-07 | Year runs (`pred_year*`) on a laptop RTX 3060 → the carbon-accounting report. Command never recorded. |
| 2026-08-20 | Data moved to S3 (commit `3e36d79`); new cloud box (L40S) set up; a quick 7-month run (`pred_full`, different config) made the code *look* worse than the report. |
| 2026-10-01 | Reconstructed the year flags; persistence matched the report exactly (data fine). Found and fixed the **TCN weight-norm NaN stall** (§8.1). Built dense logs, the viewer's dense views, WAPE everywhere. GBDT's sampled records were derived from its dense log because the per-row harness backtest took > 2 h (since fixed, §8.4). |
| 2026-10-02 | Experiments: month window (with controls: batch pinned at 85, TCN depth uncapped to 12, PatchTST patches 64/32, GBDT month lags) and smoothing (post-hoc MA, smoothed-target training). Month TCN at a matched 20-epoch budget (9× step cost). |
| 2026-10-21 | Reorganised for other machines: package `src/green_routing` (was `code/`), CLI tools, `recipes/` (were scripts in gitignored `artifacts/`), `expected.json` + `verify.py`, docs, tests. Default bucket fixed. GBDT `predict_batch`. `web/` builders read S3. Viewer password moved to `~/.config/green-routing/`. |

| 2026-10-29 | Carbon screening for 2025 (scripts in `artifacts/carbon_screening_2025`): PSU survey, Zoo model match, sites from DNS LOC, Electricity Maps grid. Link roles from traffic matching (`artifacts/link_roles`). |
| 2026-10-30 | Reverse-DNS link map, hidden routers as doors, allocation rule agreed with the owner, forecast bills + report. Then moved into `green_routing.network` / `green_routing.carbon` + `recipes/carbon`; verified against the scripts (exact for roles, DNS, grid; three bugs fixed, +1.4 % on the bill). |

Design decisions worth keeping:

- **Headline metric = per-link median ratio vs persistence** (pooled errors are dominated
  by the biggest links).
- **Dense rolling is the fairer estimate** than the sampled harness (time-of-day bias).
- New flags must **default to the old behaviour** (bit-identical runs) and be applied in
  `run.apply_overrides`, so `gr-dense` can rebuild any run's models from `config.json`.
- Comparisons between variants are **paired per link on common targets**
  (`gr-dense metrics --common`).
- **Carbon allocation** (agreed with the owner, 2026-10-30): access pool by allocated
  bandwidth (port speed until contracted rates exist), backbone pool by bytes; customers
  *and* external networks are billed, each byte half at its entry and half at its exit
  door; trunks into routers missing from the export are virtual doors. Emissions (power ×
  zone factor), not energy, are pooled. Energy is static: router power is < 1 %
  traffic-dependent on 113 of 116 metered routers.
- **Day-ahead bill** = block logs at the `day` horizon + a grid rule, issued 00:00 UTC; the
  rolling logs would flatter it.

## 8. Pitfalls already hit

1. **TCN weight-norm collapse.** At lookback 2016 (9 levels), Adam's weight decay drove
   dead channels' weight-norm direction vectors to ‖v‖ ≈ 1e-13; the norm's backward
   overflowed, every gradient was NaN, every step was skipped silently from epoch 4
   (validation loss frozen to 4 decimals). Fix: no weight decay on
   `parametrizations.weight.original1` (`torch_common._fit_torch`) + a warning per epoch
   with skipped steps. If you see `steps skipped`, stop and investigate.
2. **CPU oversubscription.** Eight torch processes with default threads each → load ~300
   on 40 cores, everything several times slower. Set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
   per job (the recipes do: `THREADS_PER_JOB=4`).
3. **GPU memory when sharing.** Week models ≤ 3 GB each; month GRU 12 GB allocated
   (~20 GB with the caching allocator), month TCN ~11 GB; GRU dense inference at batch
   1024 needs a ~16 GB cuDNN workspace. Concurrent jobs OOM'd; lower `PARALLEL` /
   `DENSE_BATCH`.
4. **GBDT harness backtest was per row** (sklearn call per link × origin × horizon, with
   OpenMP start-up each time): > 2 h. Fixed with `predict_batch` (bit-identical, ~200×).
5. **Dense numerics.** fp32 dense logs match the harness to ~1e-6 for linear models;
   GRU/TCN/PatchTST differ by p99 ~2e-4 even in fp32 (cuDNN picks different kernels at
   batch 1 vs 1024). TF32 adds ~3e-4. Default is fp32; experiments used `--tf32`.
6. **Validation losses are not comparable across lookbacks** (per-window normalisation).
7. **`pkill -f <pattern>` kills the calling shell** when the pattern appears in its own
   command line; kill by PID (`ps -eo pid,args | awk '$2 ~ /python$/ && /…/'`).
8. **Long jobs** must be detached (`setsid nohup … &`) — tool-session background tasks die
   with the session, and the VM has rebooted mid-project. Training resumes from
   `checkpoints/<model>.train.pt`; recipes skip finished parts.
9. **Moving a run** without its panel breaks the viewer: copy `artifacts/panel_<hash>.npz`
   too. GBDT checkpoints need the same scikit-learn version.
10. **`build_topology.py` without the research checkout** would wipe the link map (the
    fallback topology has no interface names); it now refuses.
11. **A run's window defines its cohort** (coverage and the 1 Mbit/s floor are computed over
    the window); compare runs on the same window only.
12. **Month-window models need a gap-free 30-day history** → ~0.06 % fewer forecasts;
    `--common` scoring handles it.
13. **A warm cache hides broken credentials.** Bucket listings are cached for 12 h
    (`GR_S3_LIST_TTL`) and CSVs by ETag, so a warm machine can read everything with no
    network at all. `gr-data check` used to pass that way with no credentials; it now
    makes one live request. Likewise a cached `panel_<hash>.npz` short-circuits the
    bucket entirely (which is what pinning the data relies on).
14. **PSU schemas.** `router-power/` has four column schemas; the old loader assumed U×I
    only, which gives nonsense for the others (and the Arista scale is 1e-4). Use
    `loaders.psu_power`. The topology viewer still reads U×I only.
15. **Units in the inventory.** The Zoo sample is bytes/s like our export (a ×300 slip made
    matching fail); W × 8.76 is kWh per year, not MWh (÷1e3).
16. **Shell scans of CSV tails lose data**: a `tail -c | grep` for the last data row
    misses files that end in empty rows (it dropped `swislf2` from the 2025 inventory).
    Use `read_series` (the network extract records `first_data`/`last_data`).
17. **Ad-hoc post-filters get lost.** The first link-role run filtered bundles with
    `rel_err < 0.05` in a one-liner that never made it into the script; rerunning the
    script gave 44 bundles instead of 32. Now `matching.MAX_BUNDLE_ERR`. Put every filter
    in code.
18. **The listing re-validates the cache.** `GR_S3_VALIDATE=0` only skips HEAD requests;
    objects covered by a fresh listing are still re-downloaded when their ETag changed. To
    rerun on a frozen cache, point a local source at it: `GR_DATA_SOURCE=local
    GR_LOCAL_ROOT=~/.cache/green-routing/s3/network-energy-efficiency-research`.
19. **Licensed and private data.** Electricity Maps (academic) forbids redistribution:
    `artifacts/carbon/grid/`, `bills/hourly_network.csv` and any hourly grid series stay
    out of git and out of published pages. The Zoo match de-anonymises a public sample.

## 9. Conventions for changes

- New model: subclass `Forecaster`, register in `run.build_models`; it is automatically
  scored on the same origins. Implement `predict_batch` if it predicts row-wise.
- New option: argparse in `run.py`, applied in `apply_overrides`, default = old behaviour,
  plus an e2e test in `tests/test_pipeline.py`.
- Before claiming an improvement: dense logs, `metrics --common`, paired per-link test
  (`recipes/window_smoothing/analyze.py` shows how), and say what was *not* controlled.
- Carbon: every assumed number goes into `carbon/assumptions.py`; after a change, rerun
  `recipes/carbon/run.sh all` and read `verify` (re-snapshot only from a trusted run).
- Keep `docs/` and this file current: results → `docs/results.md`; facts and pitfalls →
  here.
- Run `pytest -m "not e2e"` (seconds) after any change, `pytest -m e2e` (~4 min) before
  committing.

## 10. Open questions and next steps

- **Cross-verify on the new machine** (the owner's plan): `recipes/year/run.sh all`.
  Expect exact baselines and ±0.02 on learned models; investigate a data fingerprint
  mismatch before anything else (`verify.py --data-only --dump-links`). If the bucket
  has drifted, copying the reference `panel_19ea24bef019.npz` pins the data
  ([docs/reproduce.md](docs/reproduce.md)).
- **Month-window TCN to convergence** (only a 20-epoch matched pair exists; ~9× step cost).
- **Multi-seed error bars** (`--seed N`); currently single runs.
- **Graph WaveNet at year scale** — blocked on the 6 GB laptop, feasible on 46 GB.
- **Per-link fine-tuning** for the ~11 % of links the global model loses on
  (`--online` machinery exists; compute-bound).
- **Foundation models** (TimesFM, Chronos) zero-shot on the same origins.
- **Grid-factor forecast**: the bill's main error (28 % of 31 %). Log Electricity Maps'
  forecast endpoint daily and score it against `carbon.grid.RULES`; consider weather and
  imports.
- **Data from SWITCH**: the 31 missing routers (back-filled from 2025-01), PSU export,
  `ifAlias` and contracted bandwidths, measured site PUE.
- **Winter window** for the bills (needs a newer forecasting test block).
- The actual green-routing optimiser (the marginal signal inside SWITCHlan is ~1e-6
  g/GB); **privacy** of operator data ([docs/ideas.md](docs/ideas.md)).
- **Newer windows**: extend the holiday table; re-snapshot `expected.json` only from a
  trusted run.
