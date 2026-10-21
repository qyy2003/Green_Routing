"""Fast unit tests: no data source, no subprocesses."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import NAMED, _series


def test_read_series_coalesces_the_two_schema_eras(tmp_path):
    from green_routing.dataset import loaders
    df = _series(100.0, 1.0, seed=1)
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    s = loaders.read_series(p)
    assert list(s.columns[:2]) == NAMED[:2]           # ds* folded into the named columns
    assert not any(c.startswith("ds") for c in s.columns)
    assert s.index.tz is not None and s.index.is_monotonic_increasing
    assert s["ifHCInOctets"].notna().all()            # both eras present, no gap at the seam
    thr = loaders.interface_throughput(p)
    raw = df["ds0"].fillna(df["ifHCInOctets"]) + df["ds1"].fillna(df["ifHCOutOctets"])
    np.testing.assert_allclose(thr["total_mbps"].to_numpy(), raw.to_numpy() * 8 / 1e6)


def test_point_metrics_and_skill():
    from green_routing.prediction import metrics as M
    y, p = np.array([10.0, 20.0, 30.0, np.nan]), np.array([12.0, 18.0, 33.0, 5.0])
    assert M.mae(y, p) == pytest.approx(7 / 3)
    assert M.wape(y, p) == pytest.approx(100 * 7 / 60)    # sum|e| / sum|y|
    assert M.skill(5.0, 10.0) == pytest.approx(0.5)
    assert np.isnan(M.skill(1.0, 0.0))


def test_calendar_features_and_span_guard():
    from green_routing.prediction.calendar_features import K_CAL, build_calendar
    ts = pd.date_range("2026-04-03", periods=288 * 4, freq="5min", tz="UTC").asi8 // 10**9
    cal = build_calendar(ts)
    assert cal.shape == (len(ts), K_CAL)
    assert np.allclose(cal[:, 2:9].sum(1), 1)          # one day-of-week flag per step
    assert cal[:, 9].max() == 1                        # Good Friday 2026 is a holiday
    with pytest.raises(ValueError):                    # outside the holiday table
        build_calendar(pd.date_range("2030-01-01", periods=10, freq="5min",
                                     tz="UTC").asi8 // 10**9)


def test_gbdt_batch_prediction_equals_per_link():
    from green_routing.dataset import splits
    from green_routing.prediction.harness import Context
    from green_routing.prediction.models_gbdt import GlobalGBDT
    rng = np.random.default_rng(0)
    T, L = 288 * 20, 5
    tod = np.arange(T) % 288 / 288
    vals = (100 * (1 + 0.5 * np.sin(2 * np.pi * tod))[:, None]
            * rng.lognormal(0, 0.2, (T, L)) * np.arange(1, L + 1))
    vals[1000:1010, 2] = np.nan
    ts = pd.date_range("2026-01-01", periods=T, freq="5min", tz="UTC").asi8 // 10**9
    split = splits.chronological_split(T, purge_steps=288)
    m = GlobalGBDT(horizons=(3, 12, 96), max_train_rows=4000)
    m.fit(vals, split, ts)
    for o in (split.test[0], (split.test[0] + split.test[1]) // 2):
        batch = m.predict_batch(o, 96, L)
        for j in range(L):
            ctx = Context(link=j, origin=o, horizon=96, horizons=(3, 12, 96),
                          timestamps=ts, values_filled=vals, train_end=split.train[1])
            np.testing.assert_array_equal(m.predict(ctx), batch[j])


def test_path_moving_average_matches_numpy():
    torch = pytest.importorskip("torch")
    from green_routing.prediction.dense import _path_ma
    path = torch.tensor(np.random.default_rng(1).random((3, 40)), dtype=torch.float32)
    w = 12
    got = _path_ma(torch, path, w).numpy()
    x = path.numpy()
    for i in range(x.shape[1]):                       # centred, truncated at both ends
        lo, hi = max(0, i - (w - 1) // 2), min(x.shape[1] - 1, i + w // 2)
        np.testing.assert_allclose(got[:, i], x[:, lo:hi + 1].mean(1), rtol=1e-5)


def test_artifacts_dir_follows_the_environment(tmp_path):
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c", "from green_routing.paths import ARTIFACTS_DIR; print(ARTIFACTS_DIR)"],
        env={"GR_ARTIFACTS_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(tmp_path.resolve())
