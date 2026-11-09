# Green Routing — knowledge base

For whoever picks this project up next, human or agent. It records what is not obvious
from the code: the state of the work, the facts the numbers rest on, the decisions taken,
and the traps already fallen into. The user-facing docs are in `docs/`; this file links to
them rather than repeating them.

Last updated: 2026-11-09.

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
  - Ladder diagnosis (2026-11-04, `recipes/ladder_diagnosis`, log
    `docs/worklog/2026-11-04-ladder-diagnosis.md`): why the trained ladder trails Timestral. No bug;
    the ladder sees one week of 5-minute history, and on that input every architecture (and
    Timestral itself) lands at the same error. Timestral's lead is its 24 weeks of hourly context.
    Fixes measured in an hourly harness that reproduces the ladder: hourly targets (Switch -2.6
    points), mean scaling + L1 (I2 -1.0), all history instead of the panel window (I2 -0.8, and
    what makes long context learnable), long context via weekly-profile features. Day-ahead median
    link hourly WAPE: I2 19.15 -> 17.5, Switch 35.4 -> 30.0 (ensembles); Timestral 16.2 / 28.1 stays ahead.
    Report https://claude.ai/artifact/1EuQg8mZQbBZAQhm6MYx2W (copy in `docs/reports/ladder-diagnosis.html`).
  - Long context at 5 minutes (2026-11-04, `recipes/ladder_diagnosis/long5m.py`, log
    `docs/worklog/2026-11-04-long-context-5m.md`): trained models with 5-minute input and output and up
    to 24 weeks of context (48,384 steps): full-resolution and coarse patch transformers,
    multi-resolution tokens, profile query tokens, week folding, profile MLP. All train at 24 weeks
    (4-370 ms per batch of 128). 1 -> 4 weeks helps (3-seed ensembles, day: I2 -1.3, Switch -1.6
    points; week -1.9 / -2.6); 4 -> 24 weeks adds nothing for any method. Best 5-minute ensembles
    17.5 / 30.7 (ladder 19.15 / 35.41; Timestral, 16 weeks of 5-minute input, 16.5 / 29.1).
    Report https://claude.ai/artifact/7vASEr8hQGbhCQv7BAqV7Z (copy in `docs/reports/long-context-5m.html`).
  - Why long context helps Timestral and not trained models (2026-11-05/06, log
    `docs/worklog/2026-11-05-why-long-context.md`): old weeks improve the level and daily-shape estimate
    (blanking Timestral's weeks 5-24: +0.75 I2 / +1.44 Switch); trained 24-week models read them and fit
    the training months better (I2 -0.64) but lose later (+0.30): over-fitting. I2 hourly history
    downloaded back to 2021-05 (5.1 years of training data instead of 1.6, `i2_history.py`) plus 567
    door series: the 24-week penalty falls from +2.9 to +0.2, no gain appears, best trained 18.00 vs
    Timestral 16.22. Preprocessing is not the gap; fine-tuning Timestral on 5 years of I2 or on Switch
    changes nothing (+-0.15, n.s.). Report https://claude.ai/artifact/GadZfe1LnWiXk6JRjBfH8h (copy in
    `docs/reports/why-long-context.html`).
  - Greenest-path routing on I2 (2026-11-04, `recipes/green_paths`, exploratory, owner's idea): one
    router pair alone on its lowest usage-rate path vs today's routing, and everyone at once
    (best-response rounds, rates re-posted from loads), on the counter-estimated matrix of the test
    block. One pair: bill -19 % (bytes-weighted), physical change 0 (a transfer). Everyone: about
    half the spans go dark, the lit network becomes nearly a tree, and dark spans lock in (§6).
    Repeated on the measured 2021 tunnel matrix (`setup_2021.py`, hourly tunnel paths as today's
    routing): same picture (a third of the spans dark, nearly a tree), and the counter estimate
    gives the same network-level outcome as the measured matrix.
  - IMC paper evaluation (2026-11-06, `recipes/paper`, owner's three points): what the bill looks like
    (per GB, when vs where, entity bills), why forecasting carbon is not forecasting traffic (grid,
    bias, division), what day-ahead per-path rates let hosts do (CIRo-style pair CDF, decentralized
    sleeping vs a centralized controller). Idle-link cap chosen by the owner: median + 2 robust sigmas
    of pooled log rates, previous 28 days (`carbon.path_rate.idle_cap`, `capped_rates`). Written into
    the Overleaf checkout `6aea077f0929c9a3a1ac5660/sections/evaluation.tex` + `figures/eval_*.pdf`.
  - Margin and fact check for the paper, now "CarbonQuote" (2026-11-08, `recipes/paper/margin.py`, `factcheck.py`;
    report `6aea077f0929c9a3a1ac5660/REPORT.md`): posted flat rate x (1 - m) after the 7-day grid correction, m = smallest
    on a 0.5 % grid with own share >= 0 in >= 90 % of the development period's 30-day windows: SWITCHlan 6.0 %, I2 0.5 %;
    every main-text number checked; new Fig. 2 (`figures/power_vs_traffic.pdf`). Paper not compiled (no TeX on the box).
  - Paper round 2, one set of periods for both networks (2026-11-09, branch `paper-common-periods`; config
    `recipes/paper/periods.py`, driver `recipes/paper/run_common.sh`, outputs `artifacts/paper_common/`, report
    `6aea077f…/REPORT2.md` with a replacement table for every number): test 2026-04-01 -> 06-30, development 2025-04-01 ->
    2026-03-31, annual 2025-07-01 -> 2026-06-30, later 2026-07-01 -> 09-30; per-link models retrained from 2025-11-15 (555
    SWITCHlan links, 113 I2), hourly models and totals from 2025-02-01. The paper scripts run unchanged on an overlay
    (`GR_ARTIFACTS_DIR=artifacts/paper_common/root`, `GR_LD_*` overrides for `ladder_diagnosis/common.py`).
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
| `recipes/carbon/` | `run.sh`, `build_report.py` + `report_template.html`, `verify.py` + `expected.json`; `em_forecast.py` (`log`: Electricity Maps' current forecast, CH + FR, into `carbon/grid/forecast_log/`; `score`: against the rule and Timestral once days are logged) |
| `src/green_routing/carbon/dayahead.py` | day-ahead forecasts of the hourly total: weighted holidays, lecture weeks, features, refitted backtests, pretrained models, paired scoring |
| `recipes/flat_forecast/` | how to forecast the total for the flat rate: `run.sh` (history, ladder seeds), `run.py` (staged, cached), report; `bill_horizons.py` (day-ahead flat-rate bill error by day / week / month, both networks); `rate_timestral.py` (Timestral zero-shot on the grid, the traffic or the rate itself; median vs mean; capping); `bias.py` (bias and quantile calibration); `grid_forecast.py` (grid-forecast candidates, bias correction, DEV/TEST) |
| `src/green_routing/dataset/internet2.py` | Internet2 TSDS client (cached, rate-limited) and port-label parser (`classify`, `backbone_ends`, `party_of`) |
| `src/green_routing/carbon/us_grid.py` | hourly production-based g/kWh per US balancing authority from EIA-930 |
| `recipes/internet2/` | `fetch.py` (download stages), `make_data.py` (ports, links, doors, power, 5-min mirror), `carbon.py`, `tunnels.py` (2021 matrix check), `run.sh`, `pops.csv` (PoP -> BA), `fuel_factors.csv`, `timestral_total.py` |
| `recipes/ladder_diagnosis/` | ladder vs Timestral: `common.py` (block hourly-WAPE evaluator, matches `gr-dense metrics`), `hourly_train.py` (ablation harness: resolution, context, normalisation, loss, history, joint networks), `timestral_ctx.py`, `ft_timestral.py`, `compare.py` (paired, day bootstrap; `day5`/`week5` = 5-minute WAPE), `grids/*.txt` + `sweep.py`; `long5m.py` (5-minute input, up to 24 weeks: `--model seg --seg full/coarse/multires/mrh/recent --qprof`, `fold`, `profmlp`; `--squash`, `--drop-ctx`), `refs5m.py`, `switch_history.py [--res 5m]`; `i2_history.py` (I2 hourly 2021 -> 2024), `uses_context.py` (blank old weeks at test time; training / validation / test periods), `decompose.py` (daily volume vs shape error); `hourly_train.py --hist ext21 --train-months N --extra i2doors`, models `dlinear` `gru` `tcn` |
| `recipes/green_paths/` | `simulate.py {unilateral,static,everyone} [--carry] [--cap-util 0.8] [--setup tm2021 --matrix meas/est --baseline own/ecmp]`: greenest-path routing on I2 from `internet2/carbon/test_setup.npz` (`run.sh`, ~40 min) or the 2021 tunnel matrix (`setup_2021.py`, `run_2021.sh`, ~25 min); `tables.py [tm2021]` |
| `recipes/paper/` | the paper's Section 6: `run.sh` (`i2_pre_setup`, `i2_rates`, `switch_rates`, `bill_periods`, `facts`, `steering {unilateral,everyone,central}`, `text_numbers`, `figures [--png=DIR]`); outputs `artifacts/paper/`, PDFs into the Overleaf checkout (`6aea077f…/`, gitignored, owner's paper repo). Round 2 (common periods): `periods.py` (every period), `run_common.sh` (stage order), `ladder_common.sh` (per-link and total ladders), `timestral_links.py`, `i2_carbon_common.py`, `total_mirror.py`, `flat_common.py` (flat rate, margin, grid candidates, billing periods, dedicated equipment), `path_correction.py`, `switch_routing_check.py`, `appendix_runs.sh` + `grids/` (appendix experiments), `report2_numbers.py` (every number with its source) |
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
| Holidays | `calendar_features.HOLIDAYS` covers 2025-01-01 → 2026-08-31 (extended from 2025-06 in round 2); `--calendar` refuses dates outside. Extend it before using newer windows. |
| Topology | not in the bucket; the committed `switch-network-topology.txt` is a device-level reconstruction (fine for the graph model, no interface names). The real one is in the research checkout (`GR_RESEARCH_ROOT`). |
| Power | `router-power/` (PSU) **ends 2024-11-01**, four schemas (`config.PSU_SCHEMAS`: Pin W, Pin_mW, U×I at 12 V, Arista U×I ×1e-4); `eci/apollo/` shelves live; `routers/` headers identify the router family. Router models are not in the bucket: matched from the Network Power Zoo sample (94/114). |
| Missing routers | 31 routers alive in reverse DNS are absent from the SNMP export (ix3 since 2024-06, ce1 2025-01, ba3 2025-05, ez31, si2, ge3, ez11, bs1, …); ix3/ce1/ba3 hold most external links. SWITCH was asked to re-export them. |
| Internet2 | GlobalNOC TSDS `https://services.tsds.bldc.net.internet2.edu/i2/services/` (no login). 73 core routers (`core1/core2.<pop>`) at 51 PoPs, 4,265 ports (Oct 2026); doors 543 over 2024-11 -> 2026-11 (453 connector ports, 90 links to aggregation routers; hourly history of the current core network from 2021-05: `recipes/ladder_diagnosis/i2_history.py`, nodes queried by FQDN `core1.chic.net.internet2.edu`), 181 parties, 254 directed links. Power: 106 DC plants report (amps, A+B feeds), 6 core PoPs none. Old network `rtsw.<pop>` 2016-2022 with a full LSP mesh (1-minute, 2018-03 -> 2021-09). Raw in `artifacts/internet2/raw`, cache `artifacts/internet2/cache` (~1 GB). EIA-930 in `$GR_REFERENCE_DIR/eia930`. |
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
day-ahead grid rule ~52 % whatever the traffic forecast (49.2 % with perfect traffic). Bill error by horizon
(`bill_horizons.py`, `flat_forecast/bill_horizons.csv`; hourly rate published a day ahead = forecast emissions /
forecast traffic, billed carbon = rate x actual traffic, WAPE against actual emissions): Switch LAN (final ensemble,
grid "same hour yesterday") daily 30.0 %, weekly 10.7 % (7-day blocks) / 9.8 % (every window), monthly 7.5 % (3
calendar months: +15.5 / +1.4 / +2.7 %) / 4.9 % (every 30-day window); block bias +6.1 % (grid rule +4.0 %, traffic
-1.5 %); daily emissions vary with CV 94 %, so yesterday's grid is a poor forecast. Perfect grid: 6.8 / 2.5 / 1.7-3.1 %.
I2 Network (Timestral 24 w): daily 4.1 %, weekly 1.3 %, monthly 0.5-0.7 % (every month within +-1.2 %), bias ~0,
emissions CV 5.4 %. Errors shrink about as 1/sqrt(days) on both networks; a bias does not. Switch with Timestral
(24 w + calendar, 10.2 % hourly, the same model as on I2): daily 29.0, weekly 9.1-10.1, monthly 4.1-5.9 %. Why
Switch is worse: 94 % of its backbone power is on the Swiss grid (mean 13 g/kWh, hourly p10-p90 4-29, daily CV 97 %,
"same hour yesterday" 50 % hourly / 28 % daily error); perfect traffic + day-ahead grid 27.5 % vs I2 3.0 %. In
absolute terms Switch's daily error is 5 kg of 17 kg/day, I2's 71 kg of 1.7 t/day. Timestral zero-shot
(`rate_timestral.py`, 24 w, every-window WAPE day / week / month, bias): grid (emissions) forecast hourly / daily
Switch 48 / 27 % (rule "same hour yesterday") vs 36 / 22 % (Timestral), I2 3.6 / 3.0 vs 2.6 / 2.1 %. Bills, Switch: rule
grid / Timestral traffic 28.8 / 8.9 / 4.0 %, +4.8 %; Timestral grid 24.6 / 12.8 / 12.2 %, -4.0 % (median below the mean
of the skewed Swiss grid); Timestral on the rate directly 23.7 / 11.7 / 10.8 %, -4.3 %; Timestral mean instead of
median: bias +7.8 % (rate directly +10.6 %). I2: rule 4.1 / 1.3 / 0.5 %, Timestral grid 3.5 / 1.3 / 0.5 %, rate directly
3.65 / 1.4 / 0.5 %. Division safe for network totals: actual / forecast traffic 0.58-1.79 (Switch), 0.32-1.25 (I2);
capping the rate history at 3x its 28-day median hurts Switch (bias -6.6 %, month 13.5 %): 10 % of its hours are real
grid spikes. `dayahead.foundation(point="mean")` (Timestral) integrates the 21 quantiles. Bias (`bias.py`,
test block, 95 % day bootstrap): traffic unbiased (Timestral Switch 0.0 % [-1.8, +1.8], I2 +0.2 % [-0.6, +1.1]; recipe
ensemble -1.5 % [-3.0, +0.2]; months within +-2.5 %; busiest 10 % of days -5.4 % / -1.1 %), quantiles calibrated
(Switch 9.8 / 50.9 / 89.8 / 99.2 % at q10 / q50 / q90 / q99). I2 grid unbiased (all within +-0.1 %, months +-0.9 %).
Switch grid: rule +3.9 % [-3.6, +11.4] (months +6.9 / +8.2 / +0.6 / 0.0 / -9.1 %), Timestral median -4.6 % [-10.6, +1.0]
(May / June -16 %), mean +7.7 % [+2.0, +14.0] (April +19.6 %). Timestral's Swiss-grid quantiles are calibrated in
coverage (52 % below the median, 99.4 % below q99) but very right-skewed (q99 = 4.7x the actual on average; I2 1.1x),
so the median is low for a sum and the mean is pulled up by the tail. Unbiased grid (`grid_forecast.py`, chosen on
DEV 2025-03-22 -> 2026-03-21, TEST once): every model alone is biased on the Swiss grid (pretrained medians and
log-space GBDT / ridge -2 to -9 %, Timestral mean +1 / +8 % DEV / TEST, worst months 12-23 %). A causal 7-day
correction (x actual / forecast of the previous 7 days) fixes most: Timestral median + correction DEV bias +1.1 %,
12 of 13 months within +-4 % (worst +8.1 %), daily 17.7 % vs the rule's 21.8 %; TEST months +2.7 / +2.7 / +3.7 %,
bills (Timestral traffic) day / week / month 25.5 / 9.4 / 3.7 % vs rule 28.8 / 8.9 / 4.0 % and Timestral alone
24.6 / 12.8 / 12.2 %. 28-day correction is worse. I2: corrections change nothing (bills 3.6 / 1.4 / 0.5 %).
Where the bias comes from (`bias_sources.py`, `extremes.py`): model forecasts are right half the time (actual
above the Timestral median in 49 % of hours) but under-forecasts are larger (16.4 vs 12.3 % of the mean); log
error spread 0.46-0.51 on Switch vs 0.06 on I2. The rule's period bias telescopes exactly to (E day before - E last
day) / E period. Extremes are real, not glitches: CH factor > 3x its median in 23 % of hours, in episodes (median
6 h, longest 164 h), only 4 isolated 1-hour spikes in 3.5 years; winter median day 2-3x, summer 0.5x; without the
top 10 % days the biases barely change (Timestral median -4.1 -> -3.9 %). Losses for the same GBDT, DEV / TEST bias:
log squared -8.8 / -6.1 %, peak-aware (quantile 2/3) +6.3 / +12.7 % (worst test month +27.5 %), linear squared
(mean) -0.5 / +7.3 %, log-normal fitted mean (max likelihood) +2.3 / +7.7 %; with the 7-day correction all end at
DEV +0.6-0.9 %, TEST 30-day 4.0-4.4 %. Second held-out period 2026-07-11 -> 10-28 (`grid_forecast.py period`, grid
only, no Switch traffic after July): Timestral median -12.0 % (months -6 to -15 %), + 7-day correction +2.5 % (months
+0.9 to +3.9 %, daily 29.4 %), rule -0.5 % (months within +-2.7 %, daily 32.0 %, 30-day 1.7 % vs 2.6 %). The corrected
forecast runs slightly high in all three periods (+1.1 / +4.3 / +2.5 %): the ratio of 7-day sums is a little inflated.
The rule is unbiased by construction (a shifted copy: its period sum differs only by the two end days) but worst per
day; averaging it 50/50 with corrected Timestral is best or near-best everywhere (DEV / TEST / Jul-Oct: day 18.9 /
23.8 / 29.6 %, 7-day 4.8 / 8.1 / 5.8 %, 30-day 1.4 / 3.0 / 1.5 %, bias +0.3 / +4.1 / +1.0 %); DEV alone would choose it
(30-day 1.4 vs 1.9 rule, 2.1 corrected Timestral). Not yet used for bills.

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

Ladder diagnosis (2026-11-04, `artifacts/ladder_diagnosis/all_runs.csv`; day-ahead block, median
link hourly WAPE, I2 / Switch LAN). Same 1-week input: ladder PatchTST 19.15 / 35.41, Timestral 19.73 /
35.04 (n.s.), hourly PatchTST with mean scaling + L1 18.58 / 31.36. Timestral by context (1, 2, 4, 12,
24 weeks): I2 19.73, 17.90, 16.97, 16.71, 16.22; Switch 35.04, 31.11, 29.49, 28.27, 28.05. Same model 5-min
vs hourly (Switch): 34.18 vs 31.55. Extended history (I2 hourly file from 2024-11, Switch from
2024-01): I2 PatchTST 1 w 18.58 -> 17.80 (3 seeds 17.80 / 18.80 / 17.93, ensemble 17.67); ProfMLP
(weekly-profile features) 12 w 17.87 (ensemble 17.94, week 21.89); 24 w 20.01 (panel) -> 18.24 (ext);
Switch PatchTST 4 w 30.42 / week 34.97, ensemble with ProfMLP 4 + 12 w 29.98 / 34.30. Ensembles vs
Timestral 24 w: I2 +1.24 [+0.47, +1.74], Switch +1.93 [+1.32, +2.93]. Timestral fine-tuned on I2's
history (LR 1e-5): 16.06 / 20.52 vs 16.22 / 20.55, paired -0.16 [-0.37, +0.14] (n.s.). Joint training on both networks: no gain. Ladder bias at a day: median
link -14..-16 % Switch, -4.6..-9.2 % I2 (within-hour geometric vs arithmetic mean: Switch 6.4 %, I2
0.8 %); removing it does not lower WAPE.

Long context at 5 minutes (2026-11-04, `artifacts/ladder_diagnosis/long5m_runs.csv`, `{i2,switch}/refs5m.csv`;
5-minute input and output path, day-ahead block, median link hourly WAPE, I2 / Switch LAN). Multi-resolution
transformer, bounded inputs, 3 seeds each: 1 w 19.16 / 19.72 / 19.62 and 32.88 / 33.22 / 32.52; 4 w 17.99 /
17.96 / 18.51 and 31.50 / 31.13 / 31.33; 24 w 18.66 / 18.73 / 18.42 and 31.57 / 31.46 / 31.31; ensembles
19.20 / 17.86 / 18.29 and 32.50 / 30.90 / 31.08. Paired 4 vs 1 w: day -1.34 [-2.03, -0.75] / -1.61 [-3.15,
-1.10], week -1.93 / -2.64; 24 vs 4 w: day +0.43 / +0.18 (n.s.), week +0.86 [+0.30, +2.09] / +0.08 (n.s.).
Full-resolution transformer (4,032 hourly tokens) at 24 w: I2 19.56 (20-epoch budget, early stop 15; 10
epochs 20.31), Switch 33.10 (10 epochs); 4 w 18.06 / 31.44.
Ensemble of 11 runs (4 and 24 w): 17.50 / 22.37 (week) and 30.72 / 34.83; 5-minute WAPE at a day 19.93 /
40.54 (ladder 21.62 / 43.49). Timestral 5-minute input: I2 4 w 16.77, 16 w 16.48 (n.s. apart), Switch 4 w
29.79, 16 w 29.07 (-0.72 [-1.29, -0.25]; 5-minute WAPE 37.83 / 37.27). Untrained level-rescaled 5-minute profiles over 4 / 12 / 24 weeks: I2 22.43 /
22.41 / 22.59, Switch 37.47 / 39.18 / 41.65.

Why long context (2026-11-05/06, `artifacts/ladder_diagnosis/i2/data_size_runs.csv`, `*/uses_context_*.csv`;
day-ahead median link hourly WAPE). Old weeks blanked at test time: Timestral 24 w 16.22 -> 16.97 (I2), 28.05 ->
29.49 (Switch); trained 5-minute multires 24 w 18.30 -> 20.29 / 31.25 -> 31.35. Last 13 training weeks vs test,
trained 4 w / 24 w: I2 18.44 / 17.80 and 18.00 / 18.30; Switch 31.23 / 30.89 and 31.46 / 31.25. I2 hourly with history
from 2021 (ext21), 2 seeds, PatchTST 4 w / 24 w by training months 6 / 12 / 24 / 61 / 61 + doors: 18.73 / 22.00,
18.45 / 20.10, 18.27 / 19.04, 18.43 / 19.02, 18.36 / 18.66 (paired 24-4 w +2.94 [1.85, 3.23] ... +0.19 [-0.21, 0.87]);
profile MLP 24 w + doors 18.00 (Timestral +1.78 [0.88, 2.31] better). Volume / shape error, I2, Timestral 4 -> 24 w
12.09 -> 11.47 / 13.53 -> 12.91, PatchTST 12.57 -> 12.47 / 14.10 -> 14.75. Preprocessing in the same 5-minute
transformer (1 / 4 / 24 w): ladder I2 22.96 / 18.44 / 19.76, Timestral's scaler 20.72 / 19.48 / 22.48, ours 19.16 /
17.99 / 18.66 (Switch 38.25 / 32.90 / 33.19, 33.08 / 31.73 / 32.27, 32.88 / 31.50 / 31.57). Timestral fine-tuned: I2
ext21 16.07-16.13 (n.s.), Switch 28.05-28.08 (n.s.). Hourly ladder architectures 1 / 4 / 24 w (I2): DLinear 21.95 /
20.68 / 20.45, PatchTST 18.85 / 18.35 / 19.51, GRU 19.49 / 19.23 / 18.80, TCN 19.99 / 19.48 / 18.84 (Switch: DLinear
37.40 / 35.93 / 36.11, PatchTST 32.73 / 32.49 / 33.25, GRU 36.21 / 35.79 / 35.47, TCN 35.40 / 36.28 / 36.49): the small
models are not hurt by 24 weeks, PatchTST's flatten head over-fits (+1.16 / +0.76 vs 4 w).

Greenest paths on I2 (`artifacts/internet2/green_paths`, test block, 2,512 h, EM matrix, usage rates,
today = ECMP inverse-capacity). One router pair moving alone: 28 % of bytes have a path >= 1 % greener;
bill -19 % bytes-weighted (median byte 0, p90 -33 %; per source router p10/50/90 -7.5/-18/-34 %); movers'
hops x1.30; physically nothing (static power), paid by the flows that stay. Everyone (each hour from
today's routing, ~32 rounds, all converge): directed links empty 6.5 -> 111 of 254, spans dark 3 -> 54
of 107, dark spans' emissions 25 % of the network's (the only physical gain, if their optics can be
switched off; upper bound under the 45 % optical share), lit graph nearly a tree (bridges 1.5 -> 55),
hops per byte 2.54 -> 3.08, 73 % of bytes pay more (idle emissions spread flat), a link > 100 % in 5 %
of hours (80 % cap: same 54 dark spans, no link > 100 % (0.5 % of hours with `--carry`), > 80 % in
2-3 % of hours, where the rounds stop at 200 without settling: 15 % of hours, 5 % with `--carry`). Continuous
(`--carry`): 54 spans dark for the whole block (E / 0 = never chosen again: lock-in), lit set changes 72
times in 2,511 h (vs ~every hour when restarted). Static rate E / capacity: no concentration (3.6 spans
dark, hops +12 %) but routes change in 2,095 of 2,511 hours. Which spans go dark follows today's load
(Spearman with the share of hours dark: load -0.44, grid g/kWh +0.03): usage-priced "greenest" = busiest.

Greenest paths on the measured 2021 matrix (`artifacts/internet2/green_paths/tm2021`, old network
2021-03-01 -> 03-28, 672 h, 43 PoPs, 134 directed links, 67 spans, 719 Gbit/s of tunnel traffic; power
of the current PoPs, EIA-930 March 2021 grid; today = each tunnel's path in each hour, 72.6 % of bytes
traced, ECMP for the rest, link-load WAPE 20 % vs ECMP inverse-capacity 30 %, hops 41 %). One PoP pair
alone: 19 % of bytes have a greener path, bill -18 % bytes-weighted (p90 -29 %; per source PoP
p10/50/90 -3.6/-15/-34 %), movers' hops x0.97 (tunnel paths are not shortest). Everyone: links empty
0 -> 45 of 134, spans dark 0 -> 22.6 of 67 (a spanning tree leaves 25), dark spans' emissions 15.8 %,
hops 2.25 -> 2.39, bridges 2 -> 34.5, no link > 100 %, > 80 % in 5.7 % of hours (cap: 0.3 %), 77 % of
bytes pay more; continuous: 24.6 dark, 15 dark all four weeks, lit set changes 21 times in 671 h. Same
with ECMP as today: 22.1 dark, 14.5 %; with the matrix estimated from counters (EM, ECMP; OD-level
error 105 % of bytes): 22.2 dark, 15.9 %, hops 2.38, bridges 32, unilateral -13.2 % vs -13.8 % measured
(same baseline). Dark share follows today's load (Spearman -0.66), not the grid (+0.13). Both networks
end near a spanning tree: the dark share is the topology's redundancy (2021 34 %, 2025-26 51 %).

Paper evaluation (2026-11-06, `artifacts/paper`, `recipes/paper`; test blocks Switch 2026-03-22 -> 07-10,
I2 2026-07-19 -> 10-31; traffic and per-link forecasts Timestral 24 w zero-shot, grid Timestral median + 7-day
correction). Per GB over a year: Switch 0.013 g, I2 0.107 g; backbone 0.47 / 0.33 Wh/GB. Per byte, p10-p90 over
hours 15x (Switch) / 3.2x (I2), over city pairs 4x / 7x (bytes-weighted); log-rate variance from the grid 82 % (Switch),
from traffic 78 % (I2); hour-of-day median rate max/min 4.6 / 2.3. Entities: Switch 218 (median 10 kg/yr, top 10
= 50 %, access equipment 26 % of bills, 5 of the top 10 are missing routers), I2 124 (median 1.4 t/yr, MAX 40.5 t);
path vs flat moves 32 % / 20 % between entities. Bill error by period (hour / day / 7-day / 30-day): Switch 41.3 / 25.5
/ 9.4 / 3.9 % (perfect traffic 23.1 % a day, perfect grid 6.6 %), bias +5.3 %; I2 5.2 / 3.6 / 1.4 / 0.5 %; daily abs
error 4.8 kg (Switch) vs 61 kg (I2). Idle-link cap on I2: 5.6x the flat rate (0.60 g/GB, 9 % of resource-hours
capped, 8.9 % of emissions spread flat; classic mu + 2 sigma would be 108x); day-ahead resource rate error factor p90
at < 0.01 % load 4,200x uncapped, 1.44x capped; path rate a host sees 21 % (bytes-weighted WAPE; flat 6.3 %); router
path bills do not cancel (30-day 9.0 % vs flat 0.6 %). On Switch the same rule caps at 9,000x (107 of 388 resources
< 1 Mbit/s). CIRo-style (I2, PoP pairs, chosen on day-ahead rates, scored settled): 64 % of pairs gain, 33 % >= 20 %,
12 % >= 50 %, median 7 % (oracle 16 %), bytes 14 %, 20 % of pairs worse. Everyone on day-ahead rates (closed loop,
80 % cap): dark spans 35 -> 53 of 107 within two weeks (block mean 51.7, 37 dark >= 90 % of hours = 17.6 % of
emissions), no link > 80 %, hops 2.54 -> 3.15, 70 % of bytes pay more; centralized static sleeping (Hypnos-like,
greedy, 80 %): 48 spans, 21.7 %. Dark share vs today's load Spearman -0.40.

Margin (2026-11-08, `artifacts/paper/margin/`; own share = 1 - sum bills / sum shared emissions; DEV SWITCHlan 2025-03-22 ->
2026-03-21, I2 2026-03-01 -> 07-18; SWITCHlan traffic Jul 11 -> Oct 28 2026 rebuilt from the S3 cache, matches the mirror to 3e-8):
m = 6.0 % / 0.5 %; test own share -5.32 -> +1.00 % / +0.14 -> +0.64 %; 30-day error (every window) 3.91 -> 3.42 % / 0.53 -> 0.81 %,
full calendar months 5.59 -> 1.70 % / 0.77 -> 0.92 %; SWITCHlan Jul-Oct own share -4.17 -> +2.08 %. Path choices unchanged under a
uniform margin (checked: one pair 0/200 hours differ, closed loop identical dark spans in all 2,492 hours). I2 path rates over-bill:
posted path rate x actual traffic = 108.7 % of emissions over the test block (own share -8.7 %, every 30-day window negative): the
cap's flat add-on is set from forecast loads while capped/idle resources are billed at actual loads. Energy per byte (site, 12 months
to block end): SWITCHlan 0.56 Wh/GB shared pool / 0.78 all equipment, I2 0.33 (0.58x / 0.42x, not "similar"). SWITCHlan 2025 mean
factor 21.2 g/kWh energy-weighted (21.6 = CH zone alone). Controller: 48 spans asleep (21.0 %), 49 dark (21.7 %; + core1.dall5-core2.dall3,
never used). Hosts: dark spans block mean 51.7, from day 15 on 53.2, >= 90 % of hours 37 (17.6 %).

Paper round 2, common periods (2026-11-09, `artifacts/paper_common/`, every number with its source in `report2_numbers.json`;
test 2026-04-01 -> 06-30 for both networks). Margin on DEV 2025-04 -> 2026-03: SWITCHlan 6.0 %, I2 1.0 % (was 0.5 % on a 4.5-month DEV).
Test own share no margin -> margin -5.59 -> +0.75 % / +0.35 -> +1.35 %; posted bill error day / 7-day block / calendar month 26.5 / 10.7
/ 1.7 % and 4.9 / 1.6 / 1.35 % (no margin 28.0 / 11.4 / 5.6 and 4.8 / 1.3 / 0.6 %); perfect traffic 25.8 / 9.8 / 3.0 %, perfect grid
7.2 / 2.9 / 2.2 % (SWITCHlan); later period Jul-Sep 2026 own share +3.69 / +1.14 %. Bill bias +5.6 / -0.35 %. SWITCHlan with dedicated
equipment posted from yesterday's zone grid (A_k runs +3.7 % high on the test block): own share over all emissions +1.24 %, months
-1.32 / +4.04 / +2.01 %, monthly error 2.30 %; m chosen on all emissions would be 4.5 % (test -0.34 %). Grid, SWITCHlan: corrected
Timestral median DEV bias +1.5 %, daily 17.8 % vs rule 22.1 %, calendar-month 2.0 vs 2.1 %; test +3.0 %, later +1.4 %. I2 path rates:
posted own share -11.45 %, 7-day correction -0.61 %, + 1 % margin +0.40 % (months +0.17 / -0.18 / +1.14 %), choices unchanged (0 / 200
hours); c_d p10/50/90 0.81 / 0.92 / 0.95. Cap k = 2: 5.2x (0.57 g/GB), 8 % capped; host-seen error 26.4 % (flat 8.8 %). Steering: alone
61.5 % of pairs gain >= 1 %, 34.1 % >= 20 %, bytes 12.6 % vs oracle 17.4 %, same choice 88.4 % of bytes (k = 1/2/3/none 88.3-88.7 %:
"> 90 %" no longer holds); everyone, from day 15: 55.7 spans dark, 42 dark >= 90 % of hours = 18.7 % of emissions; controller 54 spans
without traffic (52 asleep) = 24.7 % (ratio 0.76, was ~0.8). Totals (hourly WAPE): Timestral 10.4 / 5.9 %, best trained 11.0 (GBDT) /
7.8 % (TCN). Links (median link, a day): Timestral 29.7 / 17.8 %, PatchTST 38.2 / 21.8 %. Annual window 2025-07 -> 2026-06: SWITCHlan
84.7 kW, 17.4 t, 0.0131 g/GB, 0.565 / 0.779 Wh/GB; I2 204.6 kW, 592.6 t, 0.1168 g/GB, 0.353 Wh/GB (8.9x the rate). Context use, last 13
training weeks / test (one seed): I2 24 w 17.3 / 22.1, 4 w 16.7 / 21.2 (no over-fitting signature now), Timestral 15.6 / 17.8; SWITCHlan
24 w 33.8 / 36.6, 4 w 34.7 / 35.0, Timestral 29.2 / 29.7; blanking Timestral's weeks 5-24 +0.44 / +1.37. Fine-tuned Timestral 17.8 % =
zero-shot. Learning curve (I2): 4 w PatchTST 20.2 -> 19.8 %, 24 w 22.3 -> 20.2 %, best trained 19.4 % (profile MLP 24 w + doors).

Paper appendix (2026-11-06, `recipes/paper/appendix.py`, `artifacts/paper/appendix/`). Day-ahead daily bill with the paper's
grid forecast, every model: Switch 25.5-28.3 % (persistence 28.2, GBDT 26.4, Timestral 25.5: the grid dominates), I2 12.5
(persistence) / 6.0 (GBDT) / 3.9 (Chronos-2) / 3.6 % (Timestral). Per-link training (GPU minutes, jobs sharing one GPU):
Switch DLinear 22, PatchTST 64, GRU 65, TCN 82; I2 17 / 46 / 23 / 61; Switch total 4-10. Horizon (median link hourly WAPE,
Timestral): I2 1 h 9.4, 8 h 14.5, day 16.2, week 20.6; Switch 8 h 24.7, day 28.1, week 32.5. Decision quality (I2, one pair
alone): 93 % of bytes make the same choice on day-ahead as on settled rates, 83 % of the oracle reduction captured; worse
pairs 20 % (12 % of bytes), median loss 5.8 %. Cap k = 1 / 2 / 3 / none: cap 1.3x / 5.6x / 24x / -, path-rate error 14 / 21 / 36
% / 6e7 %, pairs >= 20 % 26 / 33 / 33 / 24 %, dark spans (closed loop, block mean) 46.6 / 51.7 / 53.3 / 54.0.
Four networks on the hourly totals (`recipes/paper/total_neural.py`, 4 w hourly input, mean scaling, L1, 3 seeds): I2
DLinear 6.2 / GRU 5.7 / TCN 5.3 / PatchTST 5.4 % hourly WAPE (GBDT 7.1, Chronos-2 4.8, Timestral 4.3), daily bill 4.4 / 4.2 /
4.1 / 3.9 %; Switch hourly check 12.7 / 12.4 / 11.7 / 12.9 % (5-min ladder 12.9 / 11.7 / 11.3 / 12.5). Parameters: per-link
DLinear 8.1 M, PatchTST 13.6 M, GRU 1.8 / 0.4 M, TCN 1.7 / 0.4 M (Switch with calendar / I2); hourly totals 0.03-2.3 M;
Chronos-2 120 M, Timestral 209 M. Switch grid bias by month over DEV + TEST + Jul-Oct 2026 (`appendix.py`, 19 months):
Timestral median low in 17 of 19 (to -16 %), corrected median within +-4 % in 18 of 19, period bias +1.1 / +4.3 / +2.5 %;
rule -0.5 / +3.9 / -0.5 %; 50/50 blend +0.3 / +4.1 / +1.0 %, 30-day 1.4 / 3.0 / 1.5 %.
Accuracy vs cost (`recipes/paper/tradeoff.py`, RTX PRO 4500, one day of forecasts): totals, zero-shot by context 1 / 2 / 4 /
12 / 24 / 48 w: Timestral Switch 14.4 / 12.5 / 11.2 / 10.6 / 10.6 / 10.6 %, I2 7.5 / 5.1 / 4.5 / 4.5 / 4.3 / 4.3 %; Chronos-2 12.9 /
11.4 / 11.1 / 10.7 / 10.7 / 10.7 and 7.6 / 5.7 / 5.0 / 4.9 / 4.8 / 4.8 %; TimesFM-2.5 13.1 ... 10.8 and 8.4 ... 4.9-5.0 %: with 1 week
every pretrained model is worse than the trained ones. Time per day: totals 10-30 ms (Chronos-2, Timestral), 65-410 ms
(TimesFM); all links Timestral 24 w 3.3 s (Switch, 534) / 0.70 s (I2, 113), 2 w 0.53 / 0.12 s; trained nets < 1-170 ms
(+ training). Pareto: per link Timestral >= 2 w is the frontier below the best trained (32.5 / 18.4 %); totals: GBDT ties
on Switch (10.7 vs 10.6), pretrained lead on I2 (4.3 vs 5.3 %). Learning curve (I2, data_size_runs): 6 months -> 5 years
of history moves 4-w PatchTST 18.7 -> 18.4 %, 24-w 22.0 -> 19.0 %; + doors 18.4 / 18.7; best trained ~18 %; Timestral
16.2, fine-tuned on 5 years 16.1. On the last 13 training weeks: trained 24 w 17.8, 4 w 18.4, Timestral (unseen) 16.6 %.

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
35. **Compare models on equal information.** The ladder (1 week of input) vs Timestral (24 weeks)
    read as "trained models are weak"; Timestral with 1 week is no better than the ladder, and a
    trained model with 1 week beats it. Before ranking models, match context length and resolution
    (`recipes/ladder_diagnosis/timestral_ctx.py` runs Timestral at any context).
36. **5-minute log-space targets on small, bursty links.** On Switch LAN 12.5 % of the log 5-minute
    variance is within-hour (I2 6.6 %); a day-ahead model trained on 5-minute values in log1p is
    biased low (the hourly mean is above the within-hour geometric mean) and 2.6 points worse than
    the same model on hourly means (~2.0 points from the 5-minute input, ~0.7 from the 5-minute
    targets). The bias is a symptom: an oracle per-link rescale to zero bias
    raises WAPE (absolute error wants the median). For day/week horizons, forecast hourly means.
37. **Long context needs history before the panel.** With only the panel window (~30 weeks before
    the training end), most 24-week training windows are padding and long-context models over-fit
    (I2 linear 24 w 26.1 vs 20.5 at 12 w). The extra history exists: I2 `data/links_hourly.csv.gz`
    from 2024-11 (u->v + v->u = the panel link, +-0.3 %), Switch hourly 2024-01 -> 2025-07 from the
    frozen S3 cache (`recipes/ladder_diagnosis/switch_history.py`).
38. **Fine-tuning Timestral with a context-mean-scaled pinball loss** explodes on windows whose
    context is ~0 but whose target is not (a link waking up): loss 0.08 -> 1,742 in three epochs.
    Cap the per-window loss and floor the scale (1 Mbit/s); `ft_timestral.py` does.
39. **2021 tunnel paths change within the month.** 1,061 of the 1,977 traceable tunnels cross more
    than one circuit set over March 2021 (re-routes), so a walk over the union of a tunnel's circuits
    is no path, and `tunnels.py`'s walk iterates a Python `set` (hash-seeded): its own-path numbers
    (`own_paths_*`) change from run to run (the bills there use ECMP and are not affected). Trace each
    tunnel-hour from that hour's circuits, sorted (`green_paths/setup_2021.py`): 72.6 % of bytes,
    link-load WAPE 20 %.
40. **EIA-930 files before mid-2024 use coarser fuel stems** ("Solar", "Wind", "Hydropower and
    Pumped Storage"); `us_grid.load_balance` crashed on them. It now reads both (exact match, newer
    files bit-identical).
41. **Recent-mean scaling explodes with long context.** Dividing a window by the mean of its last
    week leaves old weeks unbounded: on a link that is idle now and was busy months ago they reach
    1e5-1e7 (p99 of a window's largest input on I2: 46 at 1 week, 739,439 at 24 weeks). A model
    without normalisation layers diverges (profile MLP, loss in the thousands, I2 27.5 % at 24 w).
    Bound the scaled input (`long5m.py --squash 10`: identity to 10, then 10 + log1p): 18.6 %.
42. **Validation can favour long context that the test does not.** A 24-week model fits the training
    months better than a 4-week one (I2 -0.64) and ties or beats it on validation, then loses on the
    test block (+0.30). Score the last training weeks next to validation and test
    (`uses_context.py`) before trusting a long-context gain; compare with Timestral's scores on the
    same periods (period difficulty).
43. **Context ablations on pretrained models: blank, do not tile.** Replacing old weeks by copies of
    recent ones makes Timestral repeat the exact pattern (error 37-55 % vs 16-28 %); blanking them gives
    its shorter-context score. TSDS node names are FQDNs (`core1.chic.net.internet2.edu`); a short
    name returns nothing, without an error.
44. **Electricity Maps' forecast cannot be backtested.** `/v3/carbon-intensity/forecast` returns the
    forecast made now (25 hourly points); a past `datetime` is ignored and the current one comes back.
    Log it going forward (`recipes/carbon/em_forecast.py log`, hourly; the last fetch before 00:00 UTC
    is the day-ahead forecast of the next day) and score it later (`... score`). Use
    `emissionFactorType=direct` like the history. First log 2026-11-06.
45. **A mu + k sigma cap on usage rates is set by the outliers it should cut.** Per-resource log rates are not
    normal (I2 skew 3.2): nearly idle resources sit at 1e5-1e6x the flat rate and inflate sigma (cap 108x at 2 sigma,
    1,500x at 3). Use median + k x 1.4826 MAD (`path_rate.idle_cap`). On a network of many thin resources (Switch:
    107 of 388 under 1 Mbit/s) even the robust cap is loose (9,000x): the spread is real, not outliers.
46. **The Overleaf checkout is the owner's paper repo.** `6aea077f0929c9a3a1ac5660/` is a git clone of Overleaf,
    gitignored here; edit only what the owner asks for, never commit or push there.

47. **Overlapping windows are not billing periods.** The paper's "per week / per month" errors are WAPE over every 7- / 30-day
    window (one-day step); edge days count in fewer windows, so the result can fall below the period bias: SWITCHlan test 3.9 %
    "monthly" vs +5.3 % bias (the over-billed, high-emission late March is under-weighted); non-overlapping windows 5.8 %, full
    calendar months 5.6 % (`factcheck.py` `window_definitions`). State which definition a number uses.
48. **Matplotlib's style forces a tight bbox.** `paper.mplstyle` sets `savefig.bbox: tight`; `bbox_inches=None` still means tight. For
    an exact size use `plt.rc_context({"savefig.bbox": "standard"})` (as `figures.fig_power_traffic`).
49. **`pivot_table(aggfunc="sum")` turns an all-missing group into 0.** `recipes/internet2/carbon.py` summed each plant's circuits that
    way, so hours with every reading missing became 0 W and the median fill never applied (97 hours; +0.05 % on the scored emissions,
    and I2's power persistence read 1.6 % instead of 0.48 %). Use `groupby(...).sum(min_count=1)`.
50. **Do not edit a bash script while it runs.** Bash reads a script as it goes; an edit shifts the offsets and the running copy executes
    fragments (here a harmless syntax error after the last step). Copy it, or wait for it to finish.
51. **Wait on markers the job really prints.** A watcher that waited for a log line that never appeared stalled a chain of runs for hours;
    two jobs sharing a temporary file name (`fm5m` on both networks) collided on the rename. Let each step print an explicit DONE line,
    chain on it, and give parallel jobs their own temporary names.
52. **One periods file, read by every script.** Round 2 put every window in `recipes/paper/periods.py`; scripts that hard-code dates
    drift apart (round 1 had a different test block per network, a different development period for the margin and for the grid choice).
    Look a timing or score up by every key that identifies it (`tradeoff.points` once took the hourly 8,064-step timing for the
    5-minute one).

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
- **Foundation models** (TimesFM, Chronos) zero-shot on the same origins. Timestral done
  (2026-11-03/04); the ladder diagnosis says: per-link day/week forecasts should use it, or a
  trained model rebuilt on hourly means with long context and all history (Section 6).
- **Ladder rebuild** (if a trained model is kept): hourly targets for day/week, mean scaling +
  L1/pinball, extended history, profile features; port from `recipes/ladder_diagnosis/hourly_train.py`
  into `prediction/` behind flags (default = old behaviour). Not done: LoRA fine-tuning of
  Timestral, Switch fine-tuning, calendar covariates per link. At 5 minutes use 4 weeks of context
  (multi-resolution transformer in `long5m.py`, bounded inputs); 24 weeks gave nothing more, even
  with 5 years of I2 history and door series (why: `docs/reports/why-long-context.html`). Fine-tuning
  Timestral on I2 (5 years) and Switch: done, no gain.
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
