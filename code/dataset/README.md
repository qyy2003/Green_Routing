# `dataset/` — SWITCH Cricket dataset processing

Turns the parsed Cricket CSVs into clean series and benchmark-ready artifacts for
network-prediction experiments. Implements the design in
[../../benchmark_design.md](../../benchmark_design.md); data described in
[../../dataset.md](../../dataset.md).

## Data source

The CSVs are produced daily by the research repo's `parse_rrd.py`, which uploads every
cumulative file to **SWITCHengines object storage** (bucket `switchlan-load-timeseries`).
A local `~/switch` tree is only ever a mirror of that bucket, so **this package reads S3
directly by default** and takes the filesystem as an opt-in override. Object keys mirror
the directory layout exactly, so nothing but the root changes:

    s3://switchlan-load-timeseries/router-interfaces/ag1/tengige0_0_1_0.csv
    ~/switch/router-interfaces/ag1/tengige0_0_1_0.csv

Check the source before anything else on a new box — it separates a credentials/endpoint
problem from a pipeline problem:

```bash
python -m dataset check        # resolve source, list subtree, read a few CSVs
python -m dataset warm         # pull router-interfaces into the local cache (parallel)
python -m dataset cache-info   # where the cache is and how big
python -m dataset cache-clear
```

### Configuration

All by environment variable, so a cloud deployment configures itself without editing
`config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `GR_DATA_SOURCE` | `s3` | `s3`, `local`, or `auto` (local mirror if it exists) |
| `GR_LOCAL_ROOT` | `~/switch` | filesystem mirror root |
| `GR_S3_BUCKET` | `switchlan-load-timeseries` | bucket holding the CSVs |
| `GR_S3_PREFIX` | *(none)* | key prefix inside the bucket |
| `GR_S3_ENDPOINT` | `https://zhw-b.s3.cloud.switch.ch` | S3 endpoint |
| `GR_S3_PROFILE` | *(default chain)* | AWS profile; else `AWS_PROFILE`, else env keys / instance role |
| `GR_S3_REGION` | `eu-central-1` | region |
| `GR_S3_ADDRESSING` | `path` | `path` or `virtual` |
| `GR_S3_CACHE` | `1` | cache objects + listings on disk; `0` = no disk writes |
| `GR_S3_VALIDATE` | `1` | check cached objects against the bucket ETag |
| `GR_CACHE_DIR` | `~/.cache/green-routing/s3` | cache location |
| `GR_S3_LIST_TTL` | `43200` | seconds a cached bucket listing stays valid |
| `GR_S3_WORKERS` | `8` | parallel download threads |
| `GR_RESEARCH_ROOT` | *(workstation path)* | research checkout, for the topology file |

**Caching.** The coverage scan reads every interface CSV (~2 300 objects, ~19 GB), so
each object is downloaded once into `GR_CACHE_DIR` and reused. Freshness is checked
against the ETag that listing a subtree already returns, so validation costs no extra
requests during a scan; a one-off read of an unlisted object pays a single HEAD. Budget
~20 GB of disk, or set `GR_S3_CACHE=0` to stream every read instead.

**Topology.** `switch-network-topology.txt` is **not** in the bucket (the pipeline
uploads time-series only). It resolves to the research checkout when present, else to the
copy committed next to this package — refresh that copy when the topology changes.

## Environment

Needs `pandas` + `numpy`, and `boto3` for the S3 source (`pip install boto3`). The
`RRDparsing` conda env is **only** for rrdtool parsing and has a broken pandas/numpy
binary combo — use the base conda python instead:

```bash
/home/yuyqin/anaconda3/bin/python    # pandas 2.2.3, numpy 2.1.3  ✓
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | **Data source** (S3 / local mirror), `STEP_SECONDS=300`, `DAILY_STEPS=288`, `WEEKLY_STEPS=2016`, target columns, `swi` name helpers |
| `storage.py` | Uniform reads over `s3://` and local paths; subtree listing, ETag-validated object cache, parallel prefetch, `check()` |
| `loaders.py` | **Coalesce the two schema eras** (`dsK` ↔ K-th named col); `read_series`, `interface_throughput`, `psu_power`, `eci_shelf_power`; subtree iterators |
| `topology.py` | Parse `switch-network-topology.txt` → links, nodes, node adjacency (+ speed/cost edge weights) |
| `cohort.py` | Coverage scan over a window; select dense cohort; build the common 5-min grid |
| `tensor.py` | Aggregate interfaces → device nodes; build `X[T, N, F]` + aligned adjacency |
| `splits.py` | Chronological split w/ purge, `chronological_split_by_date`, `rolling_origins`, `federated_partition`, `inductive_node_split` |
| `build.py` | End-to-end CLI writing all artifacts |

## Quickstart

Per-series (for ARIMA / FARIMA / SVR / Kalman — univariate, own span):
```python
import sys; sys.path.insert(0, "/home/yuyqin/ETH_Master_Study/Green_Routing/Green_Routing/code")
from dataset import loaders
s = loaders.interface_throughput(
    "s3://switchlan-load-timeseries/router-interfaces/ag1/tengige0_0_1_0.csv"
)["total_mbps"]          # clean, UTC-indexed, both eras coalesced
```

Paths and `s3://` URIs are interchangeable everywhere, so a local path recorded in an
older `cohort.csv` still reads fine. To let the config pick the source instead of
hard-coding a URI:

```python
from dataset import config, loaders
for dev, iface, uri in loaders.iter_interface_files():
    s = loaders.interface_throughput(uri)["total_mbps"]
```

End-to-end tensor + splits (for LSTM/ConvLSTM/GNN/Transformer/diffusion/federated):
```bash
cd /home/yuyqin/ETH_Master_Study/Green_Routing/Green_Routing/code
/home/yuyqin/anaconda3/bin/python -m dataset.build \
    --start 2024-01-01 --end 2025-01-01 \
    --train-end 2024-08-31 --val-start 2024-09-08 \
    --val-end 2024-10-20 --test-start 2024-10-27 \
    --min-coverage 0.9 --horizon 288 --lookback 2016 \
    --out ../artifacts/traffic2024
```

Writes to `--out`: `cohort.csv`, `coverage_all.csv`, `tensor.npz`
(`X`, `timestamps`, `node_index`, `feature_names`), `adjacency.npz`
(`A`, `W_speed`, `W_cost`, `node_index`), `splits.json` (train/val/test + purge +
rolling origins + PoP clients), `meta.json`.

> The coverage scan reads every interface CSV (~19 GB) once — expect it to take a while
> on the first run (it warms the object cache in parallel first; later runs read from
> disk). Narrow with `--min-coverage` / the window to control cohort size.
> `meta.json` records which source the artifact was built from.

## Loading artifacts downstream

```python
import numpy as np, json
d = np.load("artifacts/traffic2024/tensor.npz", allow_pickle=True)
X, ts, nodes, feats = d["X"], d["timestamps"], d["node_index"], d["feature_names"]
A = np.load("artifacts/traffic2024/adjacency.npz")["A"]
sp = json.load(open("artifacts/traffic2024/splits.json"))
tr, va, te = sp["split"]["train"], sp["split"]["val"], sp["split"]["test"]
X_train = X[tr[0]:tr[1]]     # normalize with THESE stats only
```

## Notes / assumptions

- **PSU wattage** uses `config.PSU_POWER_SCALE` (assumes `U`[mV]·`I`[mA] → W). Gives
  ~173 W/PSU on `swiag1` — plausible, but verify SNMP scaling for absolute values.
- Node tensor aggregates (sums) interface throughput **per device**; adjacency is
  node-level physical topology. Interface-level graphs need CSV↔topology iface-name
  normalization (not done here).
- Stray `*.csv.<hex>` files are ignored (iterators keep `*.csv` only).
- `cohort.csv` records the source URI per series, so an artifact says where it came from.
- Genuine gaps stay `NaN`; short gaps (≤ ~1 h) are ffilled/bfilled in the tensor —
  tune `fill_limit` in `tensor.build_node_tensor`.
