#!/usr/bin/env python
"""Cross-check a pred_year reproduction against the report and the reference run.

    python recipes/year/verify.py --data-only            # data fingerprint only
    python recipes/year/verify.py --data-only --dump-links links.txt   # + the cohort
    python recipes/year/verify.py --run artifacts/pred_year/merged

Three layers, from strict to tolerant:

  data       the panel this machine builds from the bucket must be the reference's:
             same 534 links (hash of the list), same finite-value count and sum.
             Any difference = the bucket's history changed; stop and investigate.
  baselines  persistence / seasonal-naive / historical average are deterministic
             numpy: their MAE and per-link ratios must match to rounding.
  models     the trained networks differ across GPUs (cuDNN algorithms, TF32) and
             runs: per-link median error ratio vs persistence within 0.02 = PASS,
             within 0.04 = WARN, beyond = FAIL. The reference run's own noise floor
             was about ±1% of the ratio.

It also prints the report's headline numbers next to this run's. Exit status 1 when
anything FAILs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HORIZONS = ["15min", "1h", "8h", "day", "week"]
BASELINES = {"persistence", "seasonal_naive_day", "seasonal_naive_week",
             "historical_average"}
# The report (docs/reports/carbon-accounting-report.html), §05: best per-link median
# ratio vs persistence per horizon and pooled MAE (best model vs persistence), Mbps.
REPORT = {
    "1h":   {"best": "gru",  "ratio": 0.801, "pooled": 585, "persistence": 712},
    "8h":   {"best": "gbdt / tcn / patchtst", "ratio": 0.545, "pooled": 836,
             "persistence": 1541},
    "day":  {"best": "tcn",  "ratio": 0.743, "pooled": 850, "persistence": 1061},
    "week": {"best": "gbdt", "ratio": 0.842, "pooled": 973, "persistence": 1145},
}
TOL_PASS, TOL_WARN = 0.02, 0.04


def data_fingerprint(cfg: dict, dump_links: str | None = None) -> dict:
    """Identity of the run's panel: the cached panel (built from the bucket on first
    use), restricted to the run's links. Same code for the reference and the check.
    dump_links: also write the link names there, one per line (to diff a mismatch)."""
    from green_routing.prediction import data as data_mod
    panel = data_mod.build_panel(cfg["start"], cfg["end"], value_col=cfg["value_col"],
                                 min_coverage=cfg["min_coverage"], verbose=True)
    if cfg.get("min_mean_mbps"):
        panel = panel.subset(panel.mean_mbps() >= cfg["min_mean_mbps"])
    if dump_links:
        Path(dump_links).write_text("\n".join(panel.links) + "\n")
    v = np.asarray(panel.values, dtype=np.float64)
    fin = np.isfinite(v)
    return {
        "n_links": int(panel.L),
        "n_steps": int(panel.T),
        "links_sha1": hashlib.sha1("\n".join(panel.links).encode()).hexdigest(),
        "finite_count": int(fin.sum()),
        "finite_sum_mbps": float(v[fin].sum()),
        "t_first": int(panel.timestamps[0]),
        "t_last": int(panel.timestamps[-1]),
    }


class Report:
    def __init__(self):
        self.rows = []

    def add(self, section, item, status, detail=""):
        self.rows.append((section, item, status, detail))

    def close(self, section, item, got, want, tol_pass, tol_warn, fmt="{:.4f}"):
        if want is None or got is None or not np.isfinite(got):
            self.add(section, item, "MISSING", f"got {got}, expected {want}")
            return
        d = abs(got - want)
        status = "PASS" if d <= tol_pass else "WARN" if d <= tol_warn else "FAIL"
        self.add(section, item, status,
                 f"{fmt.format(got)} vs {fmt.format(want)} (Δ {d:.4f})")

    def show(self) -> int:
        width = max(len(r[1]) for r in self.rows) if self.rows else 10
        for sec in dict.fromkeys(r[0] for r in self.rows):
            print(f"\n== {sec}")
            for _, item, st, det in (r for r in self.rows if r[0] == sec):
                print(f"  {st:7s} {item:{width}s}  {det}")
        n = {s: sum(r[2] == s for r in self.rows) for s in ("PASS", "WARN", "FAIL", "MISSING")}
        print(f"\n{n['PASS']} pass, {n['WARN']} warn, {n['FAIL']} fail, {n['MISSING']} missing")
        return 1 if n["FAIL"] else 0


def check_data(rep: Report, exp: dict, dump_links: str | None = None):
    fp = data_fingerprint(exp["config"], dump_links)
    want = exp["data"]
    for k in ("n_links", "n_steps", "links_sha1", "finite_count", "t_first", "t_last"):
        rep.add("data", k, "PASS" if fp[k] == want[k] else "FAIL",
                f"{fp[k]} vs {want[k]}" if fp[k] != want[k] else str(fp[k]))
    rel = abs(fp["finite_sum_mbps"] - want["finite_sum_mbps"]) / max(want["finite_sum_mbps"], 1)
    rep.add("data", "finite_sum_mbps", "PASS" if rel < 1e-9 else "FAIL",
            f"{fp['finite_sum_mbps']:.6e} (relative Δ {rel:.1e})")


def _ratios(path: Path) -> dict:
    r = pd.read_csv(path)
    return {(row.model, row.horizon_name): float(row.median_ratio) for _, row in r.iterrows()}


def check_tables(rep: Report, section: str, tables: Path, ref: dict):
    """Compare one table set (sampled run dir, or a dense mode dir) with the reference."""
    if not (tables / "ratio_vs_persistence_mae.csv").exists():
        rep.add(section, "tables", "MISSING", f"no tables in {tables}")
        return
    agg = pd.read_csv(tables / "aggregate.csv")
    got_mae = {(r.model, r.horizon_name): float(r.mae) for _, r in agg.iterrows()}
    got_ratio = _ratios(tables / "ratio_vs_persistence_mae.csv")
    for h in HORIZONS:
        want = ref["aggregate"].get("persistence", {}).get(h, {}).get("mae")
        rep.close(section, f"persistence MAE {h} (Mbps)", got_mae.get(("persistence", h)),
                  want, 0.05, 0.5, "{:.2f}")
    for model, per_h in sorted(ref["ratio"].items()):
        if model == "persistence":
            continue
        tp, tw = (1e-4, 1e-3) if model in BASELINES else (TOL_PASS, TOL_WARN)
        for h in HORIZONS:
            if h in per_h:
                rep.close(section, f"{model} {h}", got_ratio.get((model, h)),
                          per_h[h]["median_ratio"], tp, tw)


def report_side_by_side(run: Path):
    ratio = pd.read_csv(run / "ratio_vs_persistence_mae.csv")
    agg = pd.read_csv(run / "aggregate.csv")
    learned = ratio[~ratio.model.isin(BASELINES)]
    print("\n== against the report (sampled origins, best learned model per horizon)")
    print(f"  {'horizon':7s} {'report':>26s} {'this run':>26s}   pooled MAE report / this run")
    for h, r in REPORT.items():
        g = learned[learned.horizon_name == h]
        if g.empty:
            continue
        b = g.loc[g.median_ratio.idxmin()]
        a = agg[agg.horizon_name == h].set_index("model")
        lm = a.drop(index=[m for m in BASELINES if m in a.index])
        print(f"  {h:7s} {r['best']:>18s} {r['ratio']:.3f} {b.model:>18s} {b.median_ratio:.3f}"
              f"   {r['pooled']} / {lm.mae.min():.0f} (persistence {r['persistence']} / "
              f"{a.loc['persistence', 'mae']:.0f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="merged run directory (artifacts/pred_year/merged)")
    ap.add_argument("--data-only", action="store_true", help="only the data fingerprint")
    ap.add_argument("--dump-links", metavar="FILE",
                    help="also write the cohort's link names, one per line, to diff "
                         "against the reference run's config.json")
    ap.add_argument("--expected", default=str(HERE / "expected.json"))
    ap.add_argument("--dense-tables", default=None,
                    help="directory with rolling/ and block/ tables (default: "
                         "<run>/dense_year_tables, written by run.sh dense)")
    args = ap.parse_args()
    exp = json.loads(Path(args.expected).read_text())
    rep = Report()
    check_data(rep, exp, args.dump_links)
    if not args.data_only:
        if not args.run:
            ap.error("--run is required unless --data-only")
        run = Path(args.run)
        check_tables(rep, "sampled origins (harness, 160 origins)", run, exp["sampled"])
        dense = (Path(args.dense_tables) if args.dense_tables
                 else run / "dense_year_tables" if (run / "dense_year_tables").exists()
                 else run / "dense")
        for mode, ref in exp.get("dense", {}).items():
            check_tables(rep, f"dense logs, {mode} mode", dense / mode, ref)
    code = rep.show()
    if not args.data_only:
        report_side_by_side(Path(args.run))
    sys.exit(code)


if __name__ == "__main__":
    main()
