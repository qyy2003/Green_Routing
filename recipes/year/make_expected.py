#!/usr/bin/env python
"""Snapshot a reference pred_year run into expected.json (what verify.py checks).

    python recipes/year/make_expected.py --run artifacts/pred_year/merged \
        [--dense-tables artifacts/pred_year/merged/dense] [--note "..."]

Records the data fingerprint, the sampled (160-origin) tables and, when given, the
dense tables (the year models only, scored with ``gr-dense metrics --models untagged
--common``). Only re-run this on a run you trust: it is the reference.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from verify import HORIZONS, data_fingerprint   # same fingerprint code as the check

HERE = Path(__file__).resolve().parent


def _ratio_table(path: Path) -> dict:
    r = pd.read_csv(path)
    out: dict = {}
    for _, row in r.iterrows():
        out.setdefault(row.model, {})[row.horizon_name] = {
            "median_ratio": round(float(row.median_ratio), 5),
            **({"share_beats": round(float(row.share_beats_base), 4)}
               if "share_beats_base" in r and pd.notna(row.share_beats_base) else {}),
        }
    return out


def _agg(path: Path) -> dict:
    a = pd.read_csv(path)
    out: dict = {}
    for _, row in a.iterrows():
        out.setdefault(row.model, {})[row.horizon_name] = {
            "mae": round(float(row.mae), 4), "wape": round(float(row.wape), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dense-tables", default=None,
                    help="directory holding rolling/ and block/ tables of the year models")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=HERE, text=True).strip()
    except Exception:
        commit = None
    exp = {
        "about": ("Reference numbers for recipes/year/verify.py, from the pred_year run "
                  "this repository's results come from. " + args.note).strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_machine": {"python": platform.python_version(), "commit": commit},
        "config": {k: cfg[k] for k in ("start", "end", "value_col", "min_coverage",
                                       "min_mean_mbps", "train_end", "val_start",
                                       "val_end", "test_start", "lookback", "max_origins",
                                       "horizons", "calendar", "under_weight", "epochs")},
        "data": data_fingerprint(cfg),
        "sampled": {"aggregate": _agg(run / "aggregate.csv"),
                    "ratio": _ratio_table(run / "ratio_vs_persistence_mae.csv")},
    }
    if args.dense_tables:
        d = Path(args.dense_tables)
        exp["dense"] = {mode: {"aggregate": _agg(d / mode / "aggregate.csv"),
                               "ratio": _ratio_table(d / mode / "ratio_vs_persistence_mae.csv")}
                        for mode in ("rolling", "block") if (d / mode).exists()}
    out = HERE / "expected.json"
    out.write_text(json.dumps(exp, indent=1) + "\n")
    print(f"wrote {out}: {exp['data']['n_links']} links, horizons {HORIZONS}")


if __name__ == "__main__":
    main()
