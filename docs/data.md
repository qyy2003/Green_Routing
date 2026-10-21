# The data

SWITCHlan's Cricket monitoring exports 5-minute SNMP measurements for every router,
interface, power supply and optical shelf of the Swiss research backbone. A nightly job at
SWITCH ([`tools/upstream/parse_rrd.py`](../tools/upstream/parse_rrd.py)) converts the
RRD files to cumulative CSVs and uploads them to object storage. `green_routing.dataset`
reads them straight from there.

## Where it is

| | |
|---|---|
| Bucket | `s3://network-energy-efficiency-research` on SWITCHengines (`https://zhw-b.s3.cloud.switch.ch`) |
| Upstream | the uploader writes `switchlan-load-timeseries`; the project's credentials read the copy above |
| Updated | daily — every CSV is rewritten with the new day appended |
| Size | `router-interfaces/`: 1,986 CSVs on 215 devices, ~20 GB (2026-08) |
| Span | late 2023 → yesterday for most series; coverage is very uneven (some devices stop in 2025) |

Credentials and setup: [setup.md](setup.md#3-access-to-the-data-switchengines-s3).

### Layout

The keys mirror a directory tree; a local mirror (`GR_DATA_SOURCE=local`) has the same
shape under `GR_LOCAL_ROOT`.

| Prefix | Content | Used by |
|---|---|---|
| `router-interfaces/<device>/<iface>.csv` | per-interface octets, errors, packets | forecasting (traffic per link) |
| `routers/<device>.csv` | CPU, memory, temperature | topology viewer |
| `router-power/<device>_psu<k>.csv` | PSU voltage / current | topology viewer, power |
| `eci/apollo/<shelf>.csv` | optical shelf power | power |
| `transceiver-monitoring/`, `cpu-usage/` | optics, CPU | not used yet |

Device names drop the `swi` prefix in the path (`ag1` = `swiag1`); a link is
`<device>/<iface>` with the interface name lower-cased and `/` → `_`
(`ag1/tengige0_0_1_0` = TenGigE0/0/1/0 on swiag1).

### The CSV format and its two "eras"

```text
timestamp,ds0,ds1,ds2,ds3,ds4,ds5,ifHCInOctets,ifHCOutOctets,ifInErrors,ifOutErrors,...
1703858100,61463996.88,115685196.52,0.0,0.0,71107.65,110340.72,,,,,,,,
...                                          (later rows: ds* empty, named columns filled)
```

- `timestamp`: Unix seconds, UTC, on a 5-minute grid.
- Values are **5-minute averages per second**: `ifHCInOctets` is bytes/s →
  `Mbit/s = bytes/s × 8 / 10⁶`.
- **Two schema eras in every file**: the first ~1–3 days of each series are in raw
  `ds0..dsN` columns, everything after in the named columns. `dsK` is the K-th named
  column. `loaders.read_series` coalesces them; never read these CSVs with a plain
  `pd.read_csv` and use the named columns alone — you would lose the first days.
- A stray `*.csv.<hex>` partial sometimes sits next to a file; the iterators ignore it.

## The package: `green_routing.dataset`

| Module | Role |
|---|---|
| `config` | data source (S3 / local), every environment setting, sampling constants (`STEP_SECONDS=300`, `DAILY_STEPS=288`, `WEEKLY_STEPS=2016`), horizons, column names |
| `storage` | one read API over `s3://` URIs and local paths; listing, ETag-validated object cache, parallel prefetch, `check()` |
| `loaders` | `read_series` (era coalescing), `interface_throughput` (→ `in/out/total_mbps`), `psu_power`, `eci_shelf_power`, subtree iterators |
| `cohort` | coverage scan over a window, cohort selection, the common 5-min grid |
| `topology` | parse `switch-network-topology.txt` → links, nodes, adjacency with speed/cost weights |
| `tensor`, `splits`, `build` | node tensors `X[T,N,F]` + adjacency + leak-free splits (`gr-build-tensor`) |

```python
from green_routing.dataset import loaders

s = loaders.interface_throughput(
    "s3://network-energy-efficiency-research/router-interfaces/ag1/tengige0_0_1_0.csv"
)["total_mbps"]                      # UTC-indexed, both eras coalesced, Mbit/s

for dev, iface, uri in loaders.iter_interface_files():   # the configured source
    ...
```

Local paths and `s3://` URIs are interchangeable everywhere.

### Configuration

Every setting is an environment variable ([`.env.example`](../.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `GR_DATA_SOURCE` | `s3` | `s3`, `local`, or `auto` (local mirror if it exists) |
| `GR_S3_BUCKET` | `network-energy-efficiency-research` | bucket |
| `GR_S3_PREFIX` | — | key prefix inside the bucket |
| `GR_S3_ENDPOINT` | `https://zhw-b.s3.cloud.switch.ch` | endpoint |
| `GR_S3_PROFILE` | default chain | AWS profile (else `AWS_PROFILE`, env keys, instance role) |
| `GR_S3_REGION` | `eu-central-1` | region |
| `GR_S3_ADDRESSING` | `path` | `path` or `virtual` |
| `GR_S3_CACHE` | `1` | cache objects + listings on disk (`0`: stream every read) |
| `GR_S3_VALIDATE` | `1` | re-download a cached object when its ETag changed |
| `GR_CACHE_DIR` | `~/.cache/green-routing/s3` | cache location |
| `GR_S3_LIST_TTL` | `43200` | seconds a cached bucket listing stays valid |
| `GR_S3_WORKERS` | `8` | parallel downloads |
| `GR_LOCAL_ROOT` | `~/switch` | local mirror root |
| `GR_RESEARCH_ROOT` | `../network-energy-efficiency-research` | research checkout (topology, usage logs), used only if present |
| `GR_ARTIFACTS_DIR` | `<repo>/artifacts` | where runs and panel caches go |

**Cache.** A coverage scan reads every interface CSV, so each object is downloaded once
and reused. Freshness is checked against the ETag that listing a subtree returns anyway,
so validation costs no extra requests during a scan. `gr-data warm` pre-downloads the
interface subtree in parallel; `gr-data cache-info` / `cache-clear` manage it.

**Topology.** The router topology is not in the bucket. `config.TOPOLOGY_TXT` resolves
to the research checkout's file when present, else to the copy committed in the package
(`switch-network-topology.txt`, a device-level reconstruction — enough for the Graph
WaveNet adjacency, but without interface names).

## From CSVs to a forecasting panel

`green_routing.prediction.data.build_panel(start, end, min_coverage=0.9)` scans every
interface series, keeps those covering at least 90 % of the 5-minute grid of the window,
and stacks their `total_mbps` into `values[T, L]` (NaN = missing). The panel is cached as
`$GR_ARTIFACTS_DIR/panel_<hash>.npz`, the hash covering the window and the selection
arguments; `gr-benchmark --min-mean-mbps 1` then drops idle ports (mean < 1 Mbit/s).

| Window | Panel | After `--min-mean-mbps 1` |
|---|---|---|
| 2025-07-01 → 2026-07-11 (the year benchmark) | 835 links × 108,001 steps, `panel_19ea24bef019.npz`, 206 MB | **534 links** |
| 2025-12-01 → 2026-07-11 | 886 links on 141 devices (830 at ≥ 0.99 coverage) | 547 links |

A run's `config.json` records the window and its link list;
`load_run_panel(run_dir)` returns exactly the panel a finished run used. The year panel's
fingerprint (link-list hash, finite-value count and sum) is in
[`recipes/year/expected.json`](../recipes/year/expected.json) — a new machine checks it
with `recipes/year/run.sh check` before training anything.

## Node tensors for other models (`gr-build-tensor`)

```bash
gr-build-tensor --start 2024-01-01 --end 2025-01-01 \
    --train-end 2024-08-31 --val-start 2024-09-08 --val-end 2024-10-20 \
    --test-start 2024-10-27 --min-coverage 0.9 --horizon 288 --lookback 2016 \
    --out artifacts/traffic2024
```

Writes `cohort.csv`, `coverage_all.csv`, `tensor.npz` (`X`, `timestamps`, `node_index`,
`feature_names`: interfaces summed per device), `adjacency.npz` (`A`, `W_speed`,
`W_cost`), `splits.json` (train/val/test + purge gaps + rolling origins + federated
partitions) and `meta.json` (including the data source used).

```python
import json, numpy as np
d = np.load("artifacts/traffic2024/tensor.npz", allow_pickle=True)
X, ts = d["X"], d["timestamps"]
A = np.load("artifacts/traffic2024/adjacency.npz")["A"]
sp = json.load(open("artifacts/traffic2024/splits.json"))
X_train = X[sp["split"]["train"][0]:sp["split"]["train"][1]]   # normalise with train stats only
```

## Notes

- **Power**: `config.PSU_POWER_SCALE` assumes `U`[mV] × `I`[mA] → W (~173 W per PSU on
  swiag1 — plausible, unverified).
- **Gaps** stay NaN in the panel; models fill short gaps (≤ 1 h) for their inputs only.
- **Interface-level graphs** would need the topology's interface names matched to the CSV
  names; the committed topology copy has none.
