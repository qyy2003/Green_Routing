"""End-to-end smoke tests: the installed tools on the synthetic mirror, in order.

    pytest -m e2e          # ~3-6 min on CPU

benchmark -> dense logs (both modes) -> verify against the sampled records -> metrics
on common targets -> viewer API -> model zoo, plus the experiment flags (month-window
knobs, smoothed targets, tagged/strided/post-smoothed dense logs).
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from conftest import BENCH_FLAGS, KEPT, free_port, run_cli

pytestmark = pytest.mark.e2e
MODELS = ["persistence", "seasonal_naive_day", "seasonal_naive_week",
          "historical_average", "gbdt", "gru", "tcn", "dlinear", "patchtst"]


def test_data_check_on_the_mirror(env):
    p = run_cli(["gr-data", "check", "--sample", "2"], env)
    assert "OK" in p.stdout or "ok" in p.stdout.lower()


def test_benchmark_outputs(bench):
    cfg = json.loads((bench / "config.json").read_text())
    assert cfg["links"] == KEPT                        # coverage + idle filters applied
    for f in ("aggregate.csv", "per_link.csv", "ratio_vs_persistence_mae.csv",
              "wape.csv", "peaks.csv", "run.log"):
        assert (bench / f).exists(), f
    agg = pd.read_csv(bench / "aggregate.csv")
    assert set(agg.model) == set(MODELS)
    assert np.isfinite(agg.mae).all() and (agg.mae > 0).all()
    ratio = pd.read_csv(bench / "ratio_vs_persistence_mae.csv")
    assert np.allclose(ratio[ratio.model == "persistence"].median_ratio, 1.0)
    for m in ("gbdt", "gru", "tcn", "dlinear", "patchtst"):  # checkpoints for every learned model
        assert list((bench / "checkpoints").glob(f"{m}.*")), m
    assert "steps skipped" not in (bench / "run.log").read_text()


def test_resume_rebuilds_tables_without_training(bench, env):
    p = run_cli(["gr-benchmark", *BENCH_FLAGS, "--resume", "--out", str(bench)], env)
    assert "epoch 1/" not in p.stdout                  # nothing retrained
    assert "resume: cached records" in p.stdout


def test_dense_logs_match_the_sampled_records(bench, env):
    run_cli(["gr-dense", "predict", "--run", str(bench)], env)
    meta = json.loads((bench / "dense" / "meta.json").read_text())
    assert set(meta["modes"]["rolling"]) == set(MODELS)
    assert set(meta["modes"]["block"]) == set(MODELS) - {"gbdt"}   # GBDT has no path
    p = run_cli(["gr-dense", "verify", "--run", str(bench)], env)
    diffs = {m: float(d) for m, d in re.findall(r"^\s+(\S+)\s+n=.*max rel diff=(\S+)",
                                                  p.stdout, re.M)}
    assert set(diffs) == set(MODELS)
    assert max(diffs.values()) < 1e-3, diffs           # same forecasts as the harness
    run_cli(["gr-dense", "metrics", "--run", str(bench), "--common"], env)
    for mode in ("rolling", "block"):
        agg = pd.read_csv(bench / "dense" / mode / "aggregate.csv")
        assert {"wape_1h", "wape_1d", "rough"} <= set(agg.columns)
        assert np.isfinite(agg[agg.model == "tcn"].wape_1h).all()


def test_experiment_flags_and_variant_logs(bench, workdir, env):
    out = workdir / "artifacts" / "smoke_month"
    flags = [f for f in BENCH_FLAGS]
    flags[flags.index("--lookback") + 1] = "576"
    run_cli(["gr-benchmark", *flags, "--batch-size", "32", "--tcn-max-levels", "8",
             "--patchtst-patch", "32", "--patchtst-stride", "16", "--gbdt-context", "month",
             "--target-smooth", "12", "--only", "persistence,gbdt,tcn,patchtst",
             "--epochs", "1", "--out", str(out)], env)
    assert "[overrides]" in (out / "run.log").read_text()
    # a variant run logged into the main run's dense dir, every 3rd origin, + smoothing
    run_cli(["gr-dense", "predict", "--run", str(out), "--out", str(bench / "dense"),
             "--models", "tcn", "--tag", "@month", "--origin-stride", "3"], env)
    run_cli(["gr-dense", "verify", "--run", str(out), "--out", str(bench / "dense"),
             "--tag", "@month"], env)
    run_cli(["gr-dense", "predict", "--run", str(bench), "--models", "dlinear",
             "--smooth", "3,12", "--skip-raw", "--origin-stride", "3"], env)
    meta = json.loads((bench / "dense" / "meta.json").read_text())
    assert {"tcn@month", "dlinear+ma15m", "dlinear+ma1h"} <= set(meta["modes"]["block"])
    run_cli(["gr-dense", "metrics", "--run", str(bench), "--common"], env)
    agg = pd.read_csv(bench / "dense" / "rolling" / "aggregate.csv")
    assert (agg.rough_lag == 3).all()                  # strided logs -> 15-min roughness
    run_cli(["gr-dense", "metrics", "--run", str(bench), "--models", "untagged", "--common",
             "--tables-dir", str(bench / "dense_untagged")], env)
    assert set(pd.read_csv(bench / "dense_untagged" / "block" / "aggregate.csv").model) \
        == set(MODELS) - {"gbdt"}


def _get(url, auth=True):
    req = urllib.request.Request(url)
    if auth:
        tok = base64.b64encode(b"x:smoke-test").decode()
        req.add_header("Authorization", f"Basic {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read()


def test_viewer_serves_the_run(bench, env):
    port = free_port()
    proc = subprocess.Popen(["gr-viewer", str(bench), "--port", str(port), "--host",
                             "127.0.0.1"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                _get(base + "/api/meta")
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(1)
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/api/meta", auth=False)
        assert e.value.code == 401
        code, body = _get(base + "/api/meta")
        meta = json.loads(body)
        assert meta["modes"][:2] == ["block", "rolling"] and len(meta["links"]) == len(KEPT)
        code, body = _get(f"{base}/api/link?id={KEPT[0]}&h=96&mode=block")
        d = json.loads(body)
        assert code == 200 and "tcn" in d["models"] and d["window_steps"] == 96
        assert _get(base + "/api/leaderboard?mode=rolling")[0] == 200
        assert _get(base + "/")[0] == 200                  # the page itself
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def test_zoo_promotes_a_model(bench, env, workdir):
    run_cli(["gr-zoo", "add", str(bench), "--models", "gbdt", "--tag", "smoke"], env)
    p = run_cli(["gr-zoo", "list"], env)
    assert "smoke" in p.stdout
    man = json.loads((workdir / "zoo" / "gbdt" / "smoke" / "manifest.json").read_text())
    assert man["weights"]["sha256"]
