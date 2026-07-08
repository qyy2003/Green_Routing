# `dataset/` — SWITCH Cricket dataset processing

Turns the raw parsed Cricket CSVs at `/media/yuyqin/share/switch` into clean series and
benchmark-ready artifacts for network-prediction experiments. Implements the design in
[../../benchmark_design.md](../../benchmark_design.md); data described in
[../../dataset.md](../../dataset.md).

## Environment

Needs `pandas` + `numpy`. The `RRDparsing` conda env is **only** for rrdtool parsing and
has a broken pandas/numpy binary combo — use the base conda python instead:

```bash
/home/yuyqin/anaconda3/bin/python    # pandas 2.2.3, numpy 2.1.3  ✓
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Paths, `STEP_SECONDS=300`, `DAILY_STEPS=288`, `WEEKLY_STEPS=2016`, target columns, `swi` name helpers |
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
    "/media/yuyqin/share/switch/router-interfaces/ag1/tengige0_0_1_0.csv"
)["total_mbps"]          # clean, UTC-indexed, both eras coalesced
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
> on the first run. Narrow with `--min-coverage` / the window to control cohort size.

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
- Stray `*.csv.<hex>` files are ignored (iterators glob `*.csv` only).
- Genuine gaps stay `NaN`; short gaps (≤ ~1 h) are ffilled/bfilled in the tensor —
  tune `fill_limit` in `tensor.build_node_tensor`.
