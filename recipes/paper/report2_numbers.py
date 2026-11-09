"""Every number of REPORT2 (round 2, common periods) in one place, each with its source: reads the outputs of
run_common.sh and writes $PAPER_COMMON/report2_numbers.json as {key: {"value": ..., "source": "file :: field"}}.
Missing inputs are listed under "_missing" instead of stopping.

usage: GR_ARTIFACTS_DIR=$PAPER_COMMON/root python report2_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR, REPO_ROOT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import periods as PER                                                   # noqa: E402

PC = PER.ROOT
PP = ARTIFACTS_DIR / "paper"                                            # overlay paper outputs
N: dict = {}
MISSING: list = []


def put(key, value, source):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    N[key] = {"value": value, "source": source}


def rel(p: Path) -> str:
    """The path below <repo>/artifacts (paper_common/... ; paper_common/root/paper/... for the overlay)."""
    return str(p).replace(str(REPO_ROOT / "artifacts") + "/", "")


def load_json(p: Path):
    if not p.exists():
        MISSING.append(rel(p))
        return None
    return json.load(open(p))


def flat():
    f = PC / "flat" / "flat.json"
    r = load_json(f)
    if r is None:
        return
    src = rel(f)
    for net in ("switch", "i2"):
        x = r[net]
        put(f"{net}.m", x["m"], f"{src} :: {net}.m")
        put(f"{net}.m_rule_robustness", x["rule_robustness"], f"{src} :: {net}.rule_robustness")
        put(f"{net}.m_dev_zero", x["m_dev_zero"], f"{src} :: {net}.m_dev_zero")
        for tag in ("no_margin", "margin"):
            for v in ("forecast", "perfect_traffic", "perfect_grid", "timestral_uncorrected", "rule_grid"):
                for per in ("DEV", "TEST", "LATER"):
                    s = x[f"{tag}/{v}/{per}"]
                    for k in ("own_share", "own_share_all", "own_month", "own_month_all", "err_day", "err_week",
                              "err_month", "err_day_abs_kg", "share_30d_negative", "days", "months", "weeks"):
                        put(f"{net}.{tag}.{v}.{per}.{k}", s[k], f"{src} :: {net}['{tag}/{v}/{per}'].{k}")
        for per in ("DEV", "TEST", "LATER"):
            for k, v in x[f"bias/{per}"].items():
                put(f"{net}.bias.{per}.{k}", v, f"{src} :: {net}['bias/{per}'].{k}")
        for k, v in x["bias_table"].items():
            put(f"{net}.bias_table.{k}", [v["bias %"], v["95 % low"], v["95 % high"]], f"{src} :: {net}.bias_table.{k}")
            put(f"{net}.bias_table.{k}.top10", v["bias on top-10 % days"], f"{src} :: {net}.bias_table.{k}")
            put(f"{net}.bias_table.{k}.by_month", v["by month"], f"{src} :: {net}.bias_table.{k}")
    f = PC / "flat" / "bill_periods.csv"
    if f.exists():
        B = pd.read_csv(f)
        for _, b in B.iterrows():
            put(f"{b.network}.periods.{b.variant}.{b.period}", float(b.wape), f"{rel(f)} :: wape")
    f = PC / "flat" / "grid_scores.csv"
    if f.exists():
        G = pd.read_csv(f)
        for _, g in G.iterrows():
            for c in ("bias %", "worst month %", "day %", "week %", "month %", "months within 4 %", "months low"):
                put(f"grid.{g.forecast}.{g.period}.{c}", float(g[c]), f"{rel(f)} :: {c}")
    f = PC / "flat" / "grid_three_periods.csv"
    if f.exists():
        G = pd.read_csv(f)
        for _, g in G.iterrows():
            for c in ("bias %", "worst month %", "day %", "week %", "month %", "months within 4 %", "months low", "months"):
                put(f"grid3.{g.forecast}.{g.period}.{c}", float(g[c]), f"{rel(f)} :: {c}")
            put(f"grid3.{g.forecast}.{g.period}.by_month", json.loads(g.by_month), f"{rel(f)} :: by_month")


def facts():
    f = PP / "facts.json"
    r = load_json(f)
    if r:
        for net in ("switch", "i2"):
            for k, v in r[net].items():
                put(f"facts.{net}.{k}", v, f"{rel(f)} :: {net}.{k}")
    f = PP / "text_numbers.json"
    r = load_json(f)
    if r:
        for k, v in r.items():
            put(f"text.{k}", v, f"{rel(f)} :: {k}")
    f = PP / "factcheck.json"
    r = load_json(f)
    if r:
        for k, v in r.items():
            put(f"fc.{k}", v, f"{rel(f)} :: {k}")


def rates():
    for net in ("switch", "i2"):
        f = PP / net / "summary.json"
        r = load_json(f)
        if r:
            for k, v in r.items():
                put(f"rates.{net}.{k}", v, f"{rel(f)} :: {k}")
    f = PP / "i2" / "rate_error_factor_by_load.csv"
    if f.exists():
        F = pd.read_csv(f)
        for _, x in F.iterrows():
            put(f"rates.i2.error_factor.{x.load}.{x.kind}", {"median": x["median"], "p90": x.p90, "p99": x.p99},
                f"{rel(f)} :: median, p90, p99")
    f = PC / "paths" / "correction.json"
    r = load_json(f)
    if r:
        for k, v in r.items():
            put(f"pathcorr.{k}", v, f"{rel(f)} :: {k}")
    f = PC / "switch_routing_check.csv"
    if f.exists():
        R = pd.read_csv(f).groupby("routing").holdout_WAPE.mean()
        put("switch.routing_holdout", R.to_dict(), f"{rel(f)} :: holdout_WAPE (mean of 3 seeds)")
    f = PC / "i2_carbon" / "summary.json"
    r = load_json(f)
    if r:
        for k in ("holdout_link_WAPE", "W_per_core_router_median", "power_systems_used", "pops_imputed", "dependence_median_r2"):
            put(f"i2carbon.{k}", r[k], f"{rel(f)} :: {k}")


def steering():
    for k, tag in (("2", ""), ("1", "_k1.0"), ("3", "_k3.0"), ("none", "_kNone")):
        for what in ("unilateral", "everyone"):
            f = PP / "i2" / f"{what}{tag}.json"
            r = load_json(f)
            if r:
                for a, v in r.items():
                    put(f"steer.k{k}.{what}.{a}", v, f"{rel(f)} :: {a}")
    f = PP / "i2" / "central.json"
    r = load_json(f)
    if r:
        for a, v in r.items():
            if a != "asleep_spans":
                put(f"steer.central.{a}", v, f"{rel(f)} :: {a}")


def appendix():
    A = PP / "appendix"
    for name in ("total_models", "link_models", "horizons", "context", "cap_sensitivity", "train_cost",
                 "grid_bias_periods", "timestral_context", "model_params", "sweep_totals", "tradeoff_points", "data_size_runs"):
        f = A / f"{name}.csv"
        if f.exists():
            put(f"app.{name}", pd.read_csv(f).to_dict("records"), rel(f))
        else:
            MISSING.append(rel(f))
    for name in ("numbers", "bias"):
        f = A / f"{name}.json"
        r = load_json(f)
        if r:
            put(f"app.{name}", r, rel(f))


def extra():
    """Files the report cites that the stages above do not read."""
    for name in ("dedicated", ):
        f = PC / "flat" / f"{name}.json"
        r = load_json(f)
        if r:
            put(f"flat.{name}", r, rel(f))
    for name, f in (("flat.sensitivity", PC / "flat" / "sensitivity.csv"),
                    ("app.grid_bias_months", PP / "appendix" / "grid_bias_months.csv"),
                    ("app.timing", PP / "appendix" / "timing.csv")):
        if f.exists():
            put(name, pd.read_csv(f).to_dict("records"), rel(f))
        else:
            MISSING.append(rel(f))
    for net in ("i2", "switch"):
        f = PC / "link_difficulty" / f"{net}_patchtst.json"
        r = load_json(f)
        if r:
            put(f"ld.{net}", r, rel(f))
        for f in sorted((ARTIFACTS_DIR / "ladder_diagnosis" / net).glob("uses_context_*.csv")):
            put(f"uc.{net}.{f.stem.replace('uses_context_', '')}", pd.read_csv(f).to_dict("records"), rel(f))
        for f in sorted((ARTIFACTS_DIR / "ladder_diagnosis" / net / "runs").glob("F2_ft_*/summary.json")):
            r = json.load(open(f))
            put(f"ft.{f.parent.name}", {"lr": r["args"]["lr"], "best_val": r["best_val"],
                                        "test_day_median_wape": r["test"]["day"]["median_wape"]}, rel(f))


def main():
    for f in (flat, facts, rates, steering, appendix, extra):
        f()
    N["_missing"] = MISSING
    out = PC / "report2_numbers.json"
    json.dump(N, open(out, "w"), indent=1, default=float)
    print(f"{len(N) - 1} numbers -> {out}; missing: {MISSING}")


if __name__ == "__main__":
    main()
