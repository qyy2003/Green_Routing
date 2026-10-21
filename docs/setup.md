# Setting up a machine

From a fresh Linux box to a verified install in about half an hour (plus a one-time
~20 GB data download). Everything below was done on the reference machine: a
SWITCHengines VM with an NVIDIA L40S (46 GB), 40 cores, 339 GB RAM, Ubuntu, driver 580.

## 1. What you need

| | Minimum | Used for the published results |
|---|---|---|
| OS | Linux (x86-64) | Ubuntu, kernel 6.8 |
| Python | 3.10+ (via conda) | 3.11.15, Anaconda |
| GPU | any CUDA GPU for tiers 4–6; 16 GB runs the year benchmark with `PARALLEL=2` | L40S 46 GB, driver 580.173, CUDA 13.0 |
| RAM | 32 GB (one training job at a time) | 339 GB; ~6–10 GB per parallel training job |
| Disk | ~25 GB data cache + ~10 GB per year run (+ ~15 GB for the experiments' logs) | |
| Network | HTTPS to `zhw-b.s3.cloud.switch.ch` | |

Tiers 1–3 (baselines, GBDT) need no GPU. Tiers 4–6 fall back to the CPU, very slowly.

## 2. Python environment

The project uses a dedicated conda environment; don't install into the system Python.

```bash
git clone git@github.com:qyy2003/Green_Routing.git
cd Green_Routing
conda env create -f environment.yml          # env "green": pinned pandas/numpy/sklearn
conda activate green
```

Install PyTorch with pip, choosing the build that matches your driver
([pytorch.org/get-started](https://pytorch.org/get-started/locally/)). The reference run
used CUDA 13.0:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # driver >= 580
# pip install torch --index-url https://download.pytorch.org/whl/cu126 # older drivers
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # no GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On a fresh cloud VM, `nvidia-smi` missing does not mean there is no GPU — check
`lspci | grep -i nvidia`; the VM may simply have no driver yet (the reference box needed
`nvidia-driver-580-server-open` from apt).

Then install this package (editable, with the test tools):

```bash
pip install -e ".[dev]"
gr-benchmark --help            # the command-line tools are now on PATH
```

| Tool | What it does |
|---|---|
| `gr-data` | check the data source, warm or clear the download cache |
| `gr-benchmark` | train the model tiers and backtest them on identical origins |
| `gr-dense` | log every model's forecast at every 5-min origin, verify, score |
| `gr-viewer` | web viewer for a run, link by link |
| `gr-zoo` | keep good models (weights + provenance) in a model zoo |
| `gr-build-tensor` | build node tensors + adjacency + splits for other models |

Each is also `python -m green_routing.<package>.<module>`.

## 3. Access to the data (SWITCHengines S3)

The telemetry is in SWITCHengines object storage, an S3-compatible service:

| Setting | Value |
|---|---|
| Endpoint | `https://zhw-b.s3.cloud.switch.ch` |
| Region | `eu-central-1` (any value works; boto3 needs one) |
| Bucket | `network-energy-efficiency-research` |
| Addressing | path-style |

You need an **EC2-style key pair** (access key + secret) for a SWITCHengines project that
can read the bucket. Create one in the SWITCHengines dashboard (*API Access → EC2
Credentials*) or with the OpenStack CLI (`openstack ec2 credentials create`), or ask the
project owner for read access.

Put the keys in the standard AWS files, which the package (boto3) reads:

```ini
# ~/.aws/credentials   (chmod 600)
[default]
aws_access_key_id = <access key>
aws_secret_access_key = <secret key>
```

```ini
# ~/.aws/config
[default]
region = eu-central-1
# optional, so the plain `aws s3 ...` CLI also talks to SWITCH:
services = switch-s3

[services switch-s3]
s3 =
  endpoint_url = https://zhw-b.s3.cloud.switch.ch
```

The package passes the endpoint and bucket itself, so nothing else is required. Use a
named profile with `GR_S3_PROFILE=<name>`, or the `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` variables instead of the files. All settings are listed in
[`.env.example`](../.env.example) and [data.md](data.md#configuration).

Check access — this separates a credentials problem from everything else:

```bash
gr-data check          # resolves the source, lists the bucket, reads a few CSVs
```

Then download the interface subtree once (~20 GB, ~10–15 min; later runs read the cache):

```bash
gr-data warm           # into ~/.cache/green-routing/s3 (GR_CACHE_DIR to move it)
gr-data cache-info
```

The bucket is rewritten daily (each CSV is cumulative), so the cache validates every
object against the bucket's ETag; a changed object is downloaded again.

## 4. Check the install

```bash
pytest -m "not e2e and not live"     # unit tests, seconds
pytest -m e2e                         # the whole pipeline on synthetic data, ~5 min (CPU)
GR_TEST_LIVE=1 pytest -m live         # reads the real bucket
```

The e2e tests need no credentials: they generate a small Cricket-format mirror and run
`gr-benchmark → gr-dense → gr-viewer → gr-zoo` on it.

## 5. Where outputs go

Runs, panel caches and the model zoo go to `artifacts/` in the checkout (gitignored).
Point `GR_ARTIFACTS_DIR` at a bigger disk if needed:

```bash
export GR_ARTIFACTS_DIR=/data/green-routing/artifacts
```

## 6. Optional extras

- **The research checkout** (`network-energy-efficiency-research`, a GitLab repo at
  SWITCH): the real router topology (needed to rebuild the topology viewer's link map)
  and the backbone usage logs (needed for the spectral analysis). Clone it next to this
  repository or set `GR_RESEARCH_ROOT`. Without it, a committed copy of the topology is
  used, which is enough for everything else.
- **A local mirror** instead of S3: `GR_DATA_SOURCE=local GR_LOCAL_ROOT=/path/to/switch`.
- **Topology map tools** (`tools/topology/`): `pip install -e ".[topology]"`.

## 7. Opening the viewers from your laptop

The viewers listen on all interfaces and the prediction viewer asks for a password
(Basic auth; generated into `~/.config/green-routing/viewer_password`). Two ways in:

- **SSH tunnel** (works everywhere): `ssh -N -L 8138:localhost:8138 <user>@<host>`, then
  open http://localhost:8138.
- **Public IP**: allow TCP 8138 (prediction viewer) / 8137 (topology viewer) in the
  SWITCHengines security group. Plain HTTP — prefer the tunnel on untrusted networks.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `NoSuchBucket` | wrong bucket: the upstream uploader's bucket is `switchlan-load-timeseries`; the project's copy is `network-energy-efficiency-research` (the default). Set `GR_S3_BUCKET`. |
| `403` / `InvalidAccessKeyId` | credentials missing or for another project; check `~/.aws/credentials` and `GR_S3_PROFILE`. |
| first run takes >10 min before training | the coverage scan downloads every interface CSV once; run `gr-data warm` first. |
| strange pandas errors | pandas 3 in the environment — use `environment.yml` (pandas 2.2.3). |
| `torch.cuda.is_available()` is False | CPU-only wheel or driver mismatch; reinstall torch from the right index URL. |
| `CUDA out of memory` with parallel jobs | lower `PARALLEL` in the recipes; the year benchmark fits five jobs in ~12 GB, the month-window experiments need up to ~20 GB per job. |
| load average far above the core count | too many CPU threads per job; the recipes set `THREADS_PER_JOB` (default 4). |
| `calendar table covers ...` | `--calendar` on dates outside the Swiss holiday table (2025-06 → 2026-08); extend `HOLIDAYS` in `prediction/calendar_features.py`. |
