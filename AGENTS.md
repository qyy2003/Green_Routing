# Green Routing — knowledge base

For whoever picks this project up next, human or agent. It records what is not obvious
from the code: the state of the work, the facts the numbers rest on, the decisions taken,
and the traps already fallen into. The user-facing docs are in `docs/`; this file links to
them rather than repeating them.

Last updated: 2026-11-03.

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
  - Carbon rates (2026-10-31, `recipes/carbon_rates`): flat rate with every tier trained on
    the total traffic (one series), path-based rate from a counter-estimated traffic matrix
    (`carbon.tomography`, `carbon.path_rate`), flow-vs-counters check on Abilene NetFlow data.
  - Second dataset (2026-10-31): Abilene, NetFlow-derived traffic matrices (2004), via
    `green_routing.dataset.abilene` and `recipes/abilene` (forecasting ladder on links and
    total, energy from Juniper/Cisco documents, EPA eGRID 2004 grid factors, flat and path
    rates from measured flows). Added to the carbon-rates report.
  - Rate granularity (2026-10-31, `recipes/carbon_granularity`, `carbon.granularity`): publish
    hourly, forecast at 5 min or hours (equal), update hourly, publish calibrated quantiles
    (rolling 28-day joint traffic+grid errors), no gain from fitted curves.
  - Day-ahead forecast of the total for the flat rate (2026-10-31, `recipes/flat_forecast`,
    `carbon.dayahead`): choices on a development period, test block reported once; Chronos-2
    zero-shot ties or beats every trained model; final 0.8 Chronos-2 + 0.2 GBDT with the GBDT
    alone on holidays, 10.0 % hourly WAPE (GBDT so far 11.0 %, TCN 12.0 %); symmetric loss
    fixes the ladder's upward bias. Report https://claude.ai/artifact/1EBDAKgqujEv3PmdyvVSEM.
  - Internet2 as the second current network (2026-11-02, `recipes/internet2`,
    `dataset.internet2`, `carbon.us_grid`; log `docs/worklog/2026-11-02-internet2.md`): port
    counters, port labels and PoP DC power from Internet2's public GlobalNOC TSDS, hourly US grid
    factors from EIA-930, and the old network's MPLS tunnel mesh (March 2021) as a measured
    traffic matrix. Replaces Abilene (2004) in the IMC paper draft.
  - Timestral (2026-11-03, log `docs/worklog/2026-11-03-timestral.md`): Mistral's pretrained
    model (2026-10-15), zero-shot on hourly means with 24 weeks of context, beats the trained
    ladder per link a day ahead on both networks (hourly WAPE -15 % I2, -21 % Switch LAN, on 97-98 %
    of links) and Chronos-2 on I2's total. `gr-dense predict --models timestral` (block logs in
    `<run>/dense_fm/`), `carbon.dayahead.foundation(model="timestral")`. Why links are hard:
    `recipes/link_difficulty`.
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
| `src/green_routing/carbon/dayahead.py` | day-ahead forecasts of the hourly total: weighted holidays, lecture weeks, features, refitted backtests, pretrained models, paired scoring |
| `recipes/flat_forecast/` | how to forecast the total for the flat rate: `run.sh` (history, ladder seeds), `run.py` (staged, cached), report |
| `src/green_routing/dataset/internet2.py` | Internet2 TSDS client (cached, rate-limited) and port-label parser (`classify`, `backbone_ends`, `party_of`) |
| `src/green_routing/carbon/us_grid.py` | hourly production-based g/kWh per US balancing authority from EIA-930 |
| `recipes/internet2/` | `fetch.py` (download stages), `make_data.py` (ports, links, doors, power, 5-min mirror), `carbon.py`, `tunnels.py` (2021 matrix check), `run.sh`, `pops.csv` (PoP -> BA), `fuel_factors.csv`, `timestral_total.py` |
| `recipes/link_difficulty/` | why per-link traffic is hard (`analyze.py`: level vs shape, oracle profile, size, events, cross-link correlation); `fm_tables.py` (zero-shot vs ladder from `dense_fm` tables) |
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
| Pretrained models | `pip install -e ".[fm]"` (chronos-forecasting 2.3.2, timesfm 3.0.2, transformers 5); weights (Chronos-2, Chronos-Bolt base, TimesFM 2.5) in `~/.cache/huggingface`, ~1.5 GB, downloaded on first use. Timestral 1.0.0: `pip install --no-deps git+https://github.com/mistralai/timestral.git@a534bdb` (needs only torch, numpy, safetensors, huggingface_hub); weights 209M 837 MB, 28M 111 MB (Modified MIT) |
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
| Internet2 | GlobalNOC TSDS `https://services.tsds.bldc.net.internet2.edu/i2/services/` (no login). 73 core routers (`core1/core2.<pop>`) at 51 PoPs, 4,265 ports (Oct 2026); doors 543 over 2024-11 -> 2026-11 (453 connector ports, 90 links to aggregation routers), 181 parties, 254 directed links. Power: 106 DC plants report (amps, A+B feeds), 6 core PoPs none. Old network `rtsw.<pop>` 2016-2022 with a full LSP mesh (1-minute, 2018-03 -> 2021-09). Raw in `artifacts/internet2/raw`, cache `artifacts/internet2/cache` (~1 GB). EIA-930 in `$GR_REFERENCE_DIR/eia930`. |
| Abilene | second dataset (until 2026-11-02): `$GR_REFERENCE_DIR/abilene` (public archive), weeks X06–X24 = 2004-05-01 → 09-10; X21 repeats X20's last day, 08-20 missing, one corrupt hour in X24; NetFlow 1:100 scaled; energy and grid inputs in `recipes/abilene/*.csv` |
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
at the sites, settled 2.66 t; daily bill WAPE PatchTST 11.5 % (actual grid) / 31.1 %
(grid forecast), persistence 24.0 % / 38.9 %, grid forecast alone 28.0 %. Total traffic
(doors, each byte once) 261 Gb/s; doors in/out 1.003; day-ahead hourly WAPE GBDT on the
total 11.1 % = sum of TCN doors 11.1 %.

Symmetric-loss year run (`artifacts/pred_year_sym`, `--under-weight 1`; dense, 523 links, per-link
median MAE ratio vs persistence, peak-aware → symmetric): rolling 1 day TCN 0.757 → 0.703, PatchTST
0.760 → 0.714, GRU 0.788 → 0.714, DLinear 0.809 → 0.749 (GBDT unchanged 0.752, so it no longer
wins at a day; at a week DLinear 0.825 beats GBDT 0.858); block 1 day TCN 0.669 → 0.651, PatchTST
0.670 → 0.643. Off-peak bias +6…+20 % → −1…+6 %, but peaks under-forecast more (peak WAPE
+1…+5 points). Bills on it (`artifacts/carbon_sym`, inventory/grid/allocation symlinked to
`carbon/`): daily WAPE PatchTST 12.0 % (actual grid) / 31.4 % (grid forecast), vs 11.5 / 31.1 %
peak-aware; the scheduling test captures 51 % (vs 41 %).

Carbon rates (`artifacts/carbon_rates`): flat 0.0060 g/GB average (1.87 t / 313 PB);
total-traffic day-ahead hourly WAPE TCN 12.0 % … persistence 29.2 %; routing by capacity
predicts hidden link loads within 34.9 % (hops 43.6 %); usage-priced path bills move 35 %
of emissions between parties, capacity-priced 1 %; Abilene counters vs flows 5.4 %.

Flat-rate traffic forecast (`artifacts/flat_forecast`, test block, hourly WAPE of the total): GBDT
as used so far 11.0 %, TCN 12.0 % (peak-aware) / 11.3 % (symmetric, 3 seeds); Chronos-2 zero-shot
10.7 % (dev 10.7 %); final ensemble with the holiday rule 10.0 % (dev 10.6 %); with the TCN 9.8 %;
Chronos-2 on 5-minute input 15.4 %, fine-tuned 11.7 %; holidays: Chronos-2 27 %, GBDT 7 %. Abilene:
Chronos-2 12.2 % vs 15.4-15.6 % trained. Rate error with the real grid 11.5 → 10.6 %; with the
day-ahead grid rule ~52 % whatever the traffic forecast (49.2 % with perfect traffic).

Internet2 (`artifacts/internet2/carbon`, `tunnels`; year 2025-11-15 -> 2026-11-01, test from
2026-07-19): 1.39 Tbit/s mean (doors in/out 0.986); IT 123 kW at core PoPs (11 imputed) + 12.6 kW
other sites, 204 kW site (PUE 1.5); 328 g/kWh power-weighted (production-based, 25 BAs); 561 t;
flat 0.107 g/GB (test 0.0996); path rates p10-p90 0.0083-0.20 g/GB; M 25.9 %; hold-out link WAPE
29.6 % (capacity weights) vs 41.4 % (hops); PoP power vs traffic median R^2 0.013; day-ahead total
Chronos-2 4.9 % hourly WAPE (GBDT 7.1, same hour last week 8.5, persistence 13.8), rate 5.5 %
(actual grid) / 6.8 % (grid rule), perfect traffic + grid rule 3.4 %. Tunnel mesh 2021-03:
counters -> EM path bills within 12.8 % of the measured matrix's (gravity 20.2 %, measured
counters + shortest paths 17.2 %), M 24 %. Ladder on 113 I2 backbone links (symmetric loss,
test 2026-07-19 -> 11-01): dense rolling median ratio at a day TCN 0.742 / PatchTST 0.744, block
0.698 / 0.697; median link hourly WAPE at a day (block) PatchTST 19.1 %. Day-ahead path rates
(`path_dayahead.py`): uncapped x670 (idle resources), capped at 10x / 100x the flat rate 17.9 % /
46.5 % vs flat 4.9 % (bills at actual traffic per router-hour).

Timestral, zero-shot (2026-11-03; `<run>/dense_fm/tables_*`, `flat_forecast/scores_test.csv`,
`internet2/carbon/rate_errors_timestral.csv`). Per link, block, median link hourly WAPE, hourly input
with 24 weeks of context (`timestral@1h_24w`) vs the best ladder model: I2 1 h 9.4 vs 9.5 (TCN), 8 h
14.5 vs 16.3, day 16.2 vs 19.1, week 20.5 vs 25.0 (PatchTST); Switch LAN (532 common links) 8 h 24.7
vs 31.5, day 28.1 vs 35.4 (peak-aware 34.9), week 32.5 vs 40.8; better than PatchTST on 93-98 % of
links. 5-minute input with 4 weeks: worse on hourly WAPE (day: I2 16.8, Switch LAN 29.8), but on
Switch LAN best on 5-minute MAE (ratio vs persistence at a day 0.551; hourly input 0.620, PatchTST 0.643). Total day ahead: I2 4.3 %
(Chronos-2 4.8-4.9, paired -0.50 points [-0.66, -0.36]), rate 5.0 / 6.3 %; Switch LAN 10.2 % (24
weeks + calendar covariates; Chronos-2 10.6, old final 10.0, blend with GBDT 9.8, n.s.).
Per-link difficulty (`artifacts/link_difficulty`, PatchTST day, I2 / Switch LAN): true daily volume
19.1 -> 15.6 / 35.4 -> 28.4 %; in-sample weekly profile + true daily volume 13.4 / 22.8 %; hours at
< 1/3 or > 3x the usual level 4 / 9 % of hours, 8 / 14 % of the error; same-router error correlation
+0.23 / +0.22 (no sign of traffic moving between neighbours).

Abilene (`artifacts/abilene`): 19 weeks, 3.2 Gbit/s; 44 kW at the sites, 611 g/kWh, flat
rate 18.4 g/GB; per-link median ratio vs persistence 0.555 at 8 h (PatchTST), 0.677 at a
day; total day-ahead PatchTST 11.9 %; counters vs flows on path bills 4.0 %.

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
| 2026-10-30 | Reverse-DNS link map, hidden routers as doors, allocation rule agreed with the owner, forecast bills + report. Then moved into `green_routing.network` / `green_routing.carbon` + `recipes/carbon`; verified against the scripts (exact for roles, DNS, grid; four bugs fixed, +1.4 % on the bill). Byte counting checked (doors form a closed cut); total-traffic forecasts (`gr-carbon total`). |

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
19. **Seasonal fallbacks need history before the window.** The bill's fallback ("same hour
    last week") was computed on a frame that started at the window, so 220 small doors
    were forecast as zero for the first week. Load the history (bills does now).
20. **The panel cache ignores the data source.** `panel_<hash>.npz` is keyed by window and
    selection only, so a run on another source (e.g. the one-series total mirror) with the
    same window loads the 534-link panel. Give such runs their own `GR_ARTIFACTS_DIR`.
21. **Tomography with an inferred routing.** Plain tomogravity fit SWITCH's link loads only
    with 32 % negative flows; clipping them destroyed the fit. Use the non-negative EM
    estimate, then IPF to the measured totals (without it, small routers' rows inflate and
    their doors pay for phantom traffic).
22. **Usage-priced path rates explode on idle links** (a backup span at 0.3 Mbit/s with
    474 W of optics: ~1,200 g/GB). Price by capacity, or report usage pricing with that
    caveat. Price a span's power in the grid zone it draws from (CERN's shelves are French).
23. **Abilene's topology file** has a router section and a link section with four fields per
    line each; parsing it as one table overwrote router coordinates with link rows (wrong
    fibre lengths). Parse by section (`dataset.abilene.load`).
24. **`--calendar` refuses dates outside its holiday table** (Swiss, 2025-06 → 2026-08): run
    other datasets without it, or extend the table with that country's holidays.
25. **Quantile indices.** With levels 0.05 … 0.95 in steps of 0.05, `Q[..., -3]` is P85, not
    P90; look levels up by value (`granularity.LEVELS`). This once made a calibrated P90
    look 16 % exceeded instead of 10.8 %.
26. **Licensed and private data.** Electricity Maps (academic) forbids redistribution:
    `artifacts/carbon/grid/`, `bills/hourly_network.csv` and any hourly grid series stay
    out of git and out of published pages. The Zoo match de-anonymises a public sample.
27. **The ladder's peak-aware loss biases day-ahead forecasts up.** `--under-weight 2` (in the
    year flags, copied into carbon_rates) makes under-forecasts cost twice, so the models aim
    at about the 2/3 quantile: the TCN on the total ran +3.8 % high. For a symmetric metric
    (WAPE, the flat rate) train with `--under-weight 1`. The per-link year run has both:
    `LOSS=symmetric recipes/year/run.sh …` writes `pred_year_sym` (run 2026-11-02 on the
    second machine, train + merge + dense, no NaN); the default `LOSS=peak` is still
    `pred_year`, untouched. Neither is better everywhere (§6): symmetric wins on MAE/WAPE, peak-aware
    on peaks and, slightly, on the bills (the allocation cancels a bias common to all doors).
28. **History before February 2025 is not the same network.** With the current doors the total
    is 170-200 Gbit/s in 2024 (vs ~260 now) and collapses in Dec 2024 - Jan 2025 (an export
    gap); from 2025-02 it joins the year window smoothly. `recipes/flat_forecast` starts there.
29. **Holidays are not all alike.** Holidays during lectures cut the total by 10-30 %, the
    Christmas closure by 35-50 %, Corpus Christi not at all (it is only catholic cantons).
    `calendar_features.HOLIDAYS` (binary, includes Corpus Christi) is kept for bit-identical
    year runs; `carbon.dayahead.HOLIDAYS` weights each day by the population that has it off.
30. **Pretrained models want hourly input.** Chronos-2 on 5-minute values (context 8,192 steps =
    4 weeks, 288-step horizon) is far worse than on hourly values. TimesFM's covariates need
    `timesfm[xreg]` (JAX with CUDA 12 libraries next to torch's CUDA 13: not installed); LoRA
    fine-tuning needs `peft` (not installed).
31. **Internet2 TSDS keeps 1-minute data for a rolling year** (from ~2025-11-10 on 2026-11-02),
    hourly before; a 5-minute port-year costs ~14 s alone and ~45 s with other queries running,
    and hourly aggregation over the last year is slow too (the server averages the 1-minute
    records). Unfiltered `interface` queries time out: name a node. Regex in the query text takes
    one backslash (`description like "^\["`). `get_meta_field_values` answers an error, not an
    empty list, when nothing matches.
32. **Internet2 doors change over time.** Ports are relabelled and moved: a door list from one
    month covers 47 % of the doors two years earlier. Classify every row by the label it had at
    that time (`make_data.label_rows`). Doors = connector ports (`PDP <PoP>-CONN-<name>`: every
    service of a member, layer 2 included; `[RE]`/`[i2pxcust]` subinterfaces alone miss the
    layer-2 traffic) + backbone links to routers outside the core (aggregation routers). With
    both, in/out balance to 1-2 % in every month.
33. **Export outages read as zeros, not gaps.** One Internet2 hour has every door at 0 (infinite
    rate); hours under 20 % of the week's median count as missing. EM leaves tiny flows on every
    OD pair, which on a nearly idle router give 1e18 g/GB rates: zero flows under 1 kbit/s.
34. **Pretrained per-link logs go in `<run>/dense_fm/`, not `dense/`.** They cover only some
    horizons and no rolling layout; in `dense/`, `gr-dense metrics --common` would empty the other
    horizons for every model. `dense_fm/` links the ladder's block logs; score with `--models ...
    --tables-dir dense_fm/tables_<x>`. A `meta.json` copied from `dense/` still lists the rolling
    logs (rewrite its `modes`, or let `gr-dense predict` rewrite it). Timestral at 5-minute input
    with 32k steps of context costs ~60 ms per forecast (209M); hourly input is better and cheaper.

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
