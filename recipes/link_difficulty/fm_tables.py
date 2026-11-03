"""Compare zero-shot pretrained models' block logs (dense_fm) with the trained ladder.

usage: python fm_tables.py TABLES_DIR [TABLES_DIR ...]
Each TABLES_DIR is a `gr-dense metrics --tables-dir` output (block/per_link.csv,
block/aggregate.csv, block/ratio_vs_persistence_mae.csv). Prints, per horizon: median link hourly
WAPE, pooled hourly WAPE, median 5-min MAE ratio vs persistence, and the share of links where
each model beats PatchTST on hourly WAPE.
"""
import sys
from pathlib import Path

import pandas as pd

ORDER = ["persistence", "seasonal_naive_day", "seasonal_naive_week", "historical_average",
         "dlinear", "gru", "tcn", "patchtst"]

for d in map(Path, sys.argv[1:]):
    p = pd.read_csv(d / "block" / "per_link.csv")
    a = pd.read_csv(d / "block" / "aggregate.csv")
    r = pd.read_csv(d / "block" / "ratio_vs_persistence_mae.csv")
    for hn, g in p.groupby("horizon_name", sort=False):
        models = [m for m in ORDER if m in set(g.model)] + sorted(set(g.model) - set(ORDER))
        w = g.pivot_table(index="link", columns="model", values="wape_1h")
        rows = []
        for m in models:
            rows.append({"model": m,
                         "median_link_hourly_WAPE": w[m].median(),
                         "pooled_hourly_WAPE": a[(a.model == m) & (a.horizon_name == hn)].wape_1h.iloc[0],
                         "median_ratio_5min_MAE": r[(r.model == m) & (r.horizon_name == hn)].median_ratio.iloc[0]
                         if ((r.model == m) & (r.horizon_name == hn)).any() else float("nan"),
                         "beats_patchtst_share": (w[m] < w["patchtst"]).mean() if "patchtst" in w else float("nan"),
                         "links": int(w[m].notna().sum())})
        print(f"== {d}  horizon {hn}")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
