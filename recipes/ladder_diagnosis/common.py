"""Shared loaders and the block-layout hourly evaluator for the ladder diagnosis.

Datasets (the two per-link runs the ladder and Timestral were scored on):
  i2      artifacts/internet2/links/merged      113 directed backbone links
  switch  artifacts/pred_year_sym/merged        534 links (Switch LAN year run)

Everything here works on HOURLY means (12 five-minute values, >= 6 finite, the same rule
as dense._predict_foundation), and scores a forecast in the block layout exactly like
`gr-dense metrics` does for wape_1h: windows of H hours from the test start, actual hour =
mean of 12 raw values when all 12 are finite, per-link WAPE = sum|F-A| / sum A.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.dataset import splits as splits_mod
from green_routing.paths import ARTIFACTS_DIR

RUNS = {"i2": ARTIFACTS_DIR / "internet2/links/merged",
        "switch": ARTIFACTS_DIR / "pred_year_sym/merged"}
PANELS = {"i2": ARTIFACTS_DIR / "internet2/cache_links/panel_0faa1bd57c28.npz",
          "switch": ARTIFACTS_DIR / "panel_19ea24bef019.npz"}
# other runs (the paper's common periods, recipes/paper/appendix_runs.sh): GR_LD_RUN_<DS>, GR_LD_PANEL_<DS>,
# and GR_LD_FM_<DS> for the folder of the pretrained models' block logs (default <run>/dense_fm)
for _k in list(RUNS):
    RUNS[_k] = Path(os.environ.get(f"GR_LD_RUN_{_k.upper()}", RUNS[_k]))
    PANELS[_k] = Path(os.environ.get(f"GR_LD_PANEL_{_k.upper()}", PANELS[_k]))
CACHE = ARTIFACTS_DIR / "ladder_diagnosis"


class DS:
    """One dataset: raw 5-min panel, hourly means, split, the run's dense logs."""

    def __init__(self, name: str):
        self.name = name
        self.run = RUNS[name]
        self.cfg = json.loads((self.run / "config.json").read_text())
        d = np.load(PANELS[name], allow_pickle=True)
        links = list(d["links"])
        pos = {l: i for i, l in enumerate(links)}
        idx = np.array([pos[l] for l in self.cfg["links"]])
        self.links = self.cfg["links"]
        self.V = d["values"][:, idx].astype(np.float32)          # [T, L] Mbit/s, NaN gaps
        self.ts = d["timestamps"].astype(np.int64)
        self.T, self.L = self.V.shape
        c = self.cfg
        self.split = splits_mod.chronological_split_by_date(
            self.ts, c["train_end"], c["val_start"], c["val_end"], c["test_start"])
        self.lo = self.split.test[0]
        # hourly means (>= 6 of 12 finite) and complete-hour actuals (12 of 12)
        off = int(np.flatnonzero(self.ts % 3600 == 0)[0])
        nh = (self.T - off) // 12
        blk = self.V[off:off + nh * 12].reshape(nh, 12, self.L)
        cnt = np.isfinite(blk).sum(axis=1)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(blk, axis=1)
        self.off = off
        self.Hm = np.where(cnt >= 6, m, np.nan).astype(np.float32)    # [nh, L] input
        self.Ha = np.where(cnt == 12, m, np.nan).astype(np.float32)   # [nh, L] actual
        self.hts = self.ts[off::12][:nh]
        assert (self.lo - off) % 12 == 0, "test start not on the hour"
        self.hlo = (self.lo - off) // 12                       # first test hour
        self.htrain = ((self.split.train[0] - off) // 12, (self.split.train[1] - off) // 12)
        self.hval = ((self.split.val[0] - off) // 12, (self.split.val[1] - off) // 12)
        self.nh = nh

    # -- test-block bookkeeping (hour index relative to hlo) ----------------- #
    def n_test_hours(self) -> int:
        return self.nh - self.hlo

    def actual_test(self) -> np.ndarray:
        """[L, n_test_hours] complete-hour actuals of the test block."""
        return self.Ha[self.hlo:].T

    def block_starts(self, H: int) -> np.ndarray:
        """Absolute hour indices of block-window starts for an H-hour horizon."""
        return np.arange(self.hlo, self.nh, H)

    def fm_dir(self, sub: str) -> Path:
        e = os.environ.get(f"GR_LD_FM_{self.name.upper()}")
        return Path(e) if (e and sub == "dense_fm") else self.run / sub

    def load_dense_block(self, model: str, horizon: str, sub: str = "dense_fm") -> np.ndarray:
        """A model's block log at ``horizon`` (5-min) -> hourly means [L, n_test_hours]."""
        meta = json.loads((self.fm_dir(sub) / "meta.json").read_text())
        hn = [h["name"] for h in meta["horizons"]]
        P = np.load(self.fm_dir(sub) / f"pred_block_{model}.npy", mmap_mode="r")
        p = np.asarray(P[:, hn.index(horizon), :], np.float64)          # [L, n_t]
        n = self.n_test_hours()
        p = p[:, :n * 12].reshape(self.L, n, 12)
        ok = np.isfinite(p).all(axis=2)
        return np.where(ok, p.mean(axis=2), np.nan)


    # -- 5-minute views of the test block (long5m.py) ----------------------- #
    def n_test_steps(self) -> int:
        return self.T - self.lo

    def actual5_test(self) -> np.ndarray:
        """[L, n_test_steps] raw 5-minute actuals of the test block (NaN gaps)."""
        return self.V[self.lo:].T

    def load_dense_block5(self, model: str, horizon: str, sub: str = "dense_fm") -> np.ndarray:
        """A model's block log at ``horizon``, 5-minute values [L, n_test_steps]."""
        meta = json.loads((self.fm_dir(sub) / "meta.json").read_text())
        hn = [h["name"] for h in meta["horizons"]]
        P = np.load(self.fm_dir(sub) / f"pred_block_{model}.npy", mmap_mode="r")
        return np.asarray(P[:, hn.index(horizon), :self.n_test_steps()], np.float32)


def score(preds: dict[str, np.ndarray], A: np.ndarray, common: bool = True,
          ref: str | None = None) -> pd.DataFrame:
    """Per-model summary of hourly WAPE over the test block.

    preds: name -> [L, n_test_hours] hourly forecasts. Returns median/p75 link WAPE, pooled
    WAPE, pooled bias, and (with ``ref``) the share of links where the model beats ``ref``."""
    ok = np.isfinite(A)
    if common:
        for p in preds.values():
            ok &= np.isfinite(p)
    rows, per = [], {}
    for name, p in preds.items():
        m = ok if common else ok & np.isfinite(p)
        e = np.where(m, p - A, 0.0)
        a = np.where(m, A, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = 100 * np.abs(e).sum(1) / a.sum(1)
        w = np.where(a.sum(1) > 1e-8, w, np.nan)
        per[name] = w
        rows.append({"model": name, "median_wape": np.nanmedian(w),
                     "p75_wape": np.nanpercentile(w, 75),
                     "pooled_wape": 100 * np.abs(e).sum() / a.sum(),
                     "pooled_bias": 100 * e.sum() / a.sum(),
                     "median_link_bias": np.nanmedian(100 * e.sum(1) / np.maximum(a.sum(1), 1e-8)),
                     "hours": int(m.sum())})
    df = pd.DataFrame(rows)
    if ref is not None and ref in per:
        df["beats_ref"] = [np.nanmean(per[n] < per[ref]) for n in df.model]
    return df, per
