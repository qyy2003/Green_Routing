"""Shared fixtures: a synthetic Cricket mirror and a small benchmark run on it.

The mirror reproduces the real CSV layout — ``timestamp, ds0..ds5, ifHCInOctets,
ifHCOutOctets, ...`` with the first two days in the ``ds*`` columns and the rest in the
named ones (the "two schema eras" every SWITCH CSV has), values in bytes/s — so the
whole pipeline runs offline: GR_DATA_SOURCE=local points the package at it.

Links (6 series on 4 devices):
  ag1/te0_0_0_1, ag1/te0_0_0_2, be1/hu0_0_0_1, zh2/te0_0_1_0   busy, diurnal+weekly
  zh2/te0_0_1_1                                                idle (mean < 1 Mbps)
  ge1/te0_0_0_9                                                gappy (coverage ~0.8)
so a run with --min-coverage 0.9 --min-mean-mbps 1 keeps exactly the first four.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
START, END = "2026-01-01", "2026-03-15"
NAMED = ["ifHCInOctets", "ifHCOutOctets", "ifInErrors", "ifOutErrors",
         "ifInUcastPackets", "ifOutUcastPackets", "ifHCInUcastPkts", "ifHCOutUcastPkts"]
LINKS = {  # link -> (mean Mbps, coverage)
    "ag1/te0_0_0_1": (80.0, 1.0),
    "ag1/te0_0_0_2": (450.0, 1.0),
    "be1/hu0_0_0_1": (2600.0, 1.0),
    "zh2/te0_0_1_0": (35.0, 1.0),
    "zh2/te0_0_1_1": (0.3, 1.0),
    "ge1/te0_0_0_9": (120.0, 0.8),
}
KEPT = ["ag1/te0_0_0_1", "ag1/te0_0_0_2", "be1/hu0_0_0_1", "zh2/te0_0_1_0"]

# The benchmark the e2e tests share (a scaled-down pred_year: same flags, tiny budget)
BENCH_FLAGS = [
    "--start", START, "--end", END,
    "--train-end", "2026-02-10", "--val-start", "2026-02-13",
    "--val-end", "2026-02-24", "--test-start", "2026-02-27",
    "--min-mean-mbps", "1", "--tiers", "1", "3", "4", "5",
    "--horizons", "15min", "1h", "8h", "day", "--max-origins", "12",
    "--lookback", "288", "--epochs", "2", "--max-train-samples", "3000",
    "--calendar", "--under-weight", "2",
]


def _series(mean_mbps: float, coverage: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(START, END, freq="5min", tz="UTC", inclusive="left")
    tod = (t.hour * 60 + t.minute).to_numpy() / 1440.0
    weekend = (t.dayofweek >= 5).astype(float)
    shape = (1 + 0.6 * np.sin(2 * np.pi * (tod - 0.35))) * (1 - 0.35 * weekend)
    noise = rng.lognormal(0, 0.25, len(t))
    spikes = np.where(rng.random(len(t)) < 0.002, rng.uniform(2, 5, len(t)), 1.0)
    mbps = np.clip(mean_mbps * shape * noise * spikes, 0, None)
    bps = mbps * 1e6 / 8                              # Mbit/s -> bytes/s
    frac_in = 0.4 + 0.2 * rng.random()
    df = pd.DataFrame({"timestamp": (t.asi8 // 10**9)})
    if coverage < 1:                                  # drop whole hours at random
        hours = (df.timestamp // 3600).to_numpy()
        keep_h = rng.random(hours.max() - hours.min() + 1) < coverage
        df = df[keep_h[hours - hours.min()]]
        bps = bps[df.index]
    vin, vout = bps * frac_in, bps * (1 - frac_in)
    era1 = df.timestamp.to_numpy() < df.timestamp.iloc[0] + 2 * 86400
    out = pd.DataFrame({"timestamp": df.timestamp.to_numpy()})
    for k in range(6):
        out[f"ds{k}"] = np.nan
    for c in NAMED:
        out[c] = np.nan
    for col, v in (("ds0", vin), ("ds1", vout)):
        out.loc[era1, col] = v[era1]
    for col, v in (("ifHCInOctets", vin), ("ifHCOutOctets", vout)):
        out.loc[~era1, col] = v[~era1]
    return out


def write_mirror(root: Path) -> Path:
    for i, (link, (mean, cov)) in enumerate(LINKS.items()):
        dev, iface = link.split("/")
        p = root / "router-interfaces" / dev / f"{iface}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        _series(mean, cov, seed=100 + i).to_csv(p, index=False)
    return root


@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("gr")


@pytest.fixture(scope="session")
def mirror(workdir) -> Path:
    return write_mirror(workdir / "mirror")


@pytest.fixture(scope="session")
def env(workdir, mirror) -> dict:
    """Environment for the CLI subprocesses: offline, CPU, isolated artifacts."""
    e = dict(os.environ)
    for k in list(e):
        if k.startswith("GR_"):
            del e[k]
    e.update({
        "GR_DATA_SOURCE": "local", "GR_LOCAL_ROOT": str(mirror),
        "GR_ARTIFACTS_DIR": str(workdir / "artifacts"), "GR_ZOO_DIR": str(workdir / "zoo"),
        "PRED_DEVICE": "cpu", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "GR_VIEWER_PASSWORD": "smoke-test",
    })
    return e


def run_cli(args: list[str], env: dict, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run one of the installed gr-* tools (falls back to python -m when not on PATH)."""
    exe = shutil.which(args[0])
    modules = {"gr-benchmark": "green_routing.prediction.run",
               "gr-dense": "green_routing.prediction.dense",
               "gr-data": "green_routing.dataset",
               "gr-zoo": "green_routing.prediction.zoo",
               "gr-viewer": "green_routing.viewer.server"}
    cmd = [exe, *args[1:]] if exe else [sys.executable, "-m", modules[args[0]], *args[1:]]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout,
                       cwd=REPO)
    if p.returncode != 0:
        raise AssertionError(f"{' '.join(args)} failed ({p.returncode}):\n"
                             f"{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p


@pytest.fixture(scope="session")
def bench(workdir, env) -> Path:
    """One small benchmark run (all tiers but STGNN) shared by the e2e tests."""
    out = workdir / "artifacts" / "smoke_run"
    run_cli(["gr-benchmark", *BENCH_FLAGS, "--out", str(out)], env)
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
