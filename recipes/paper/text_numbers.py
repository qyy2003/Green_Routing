"""Numbers of the evaluation text that combine the outputs of the other scripts: percentiles of
the rate per byte over hours and over city pairs, the spans that stay dark when hosts follow the
day-ahead rates and their emissions, what predicts a span going dark, named entity bills.

usage: python text_numbers.py      (after i2_rates.py, switch_rates.py, facts.py, steering.py)
writes $GR_ARTIFACTS_DIR/paper/text_numbers.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "green_paths"))
import i2_rates as IR                     # noqa: E402
import simulate as SIM                    # noqa: E402

P = IR.ARTIFACTS_DIR / "paper"


def wq(v: pd.Series, w: pd.Series, ps) -> list:
    o = np.argsort(v.to_numpy())
    c = np.cumsum(w.to_numpy()[o]) / w.sum()
    return [round(float(np.interp(p, c, v.to_numpy()[o])), 5) for p in ps]


def main():
    out = {}
    for net in ("switch", "i2"):
        c = pd.read_csv(P / net / "citypairs.csv")
        out[f"{net}_citypair_bytes_p10_p50_p90"] = wq(c.g_per_GB, c.GB, (.1, .5, .9))
        out[f"{net}_citypair_unweighted_p10_p50_p90"] = [round(x, 5) for x in c.g_per_GB.quantile([.1, .5, .9])]
        h = pd.read_csv(P / f"hourly_year_{net}.csv", index_col=0)
        out[f"{net}_hours_bytes_p10_p50_p90"] = wq(h.r, h.T_gbps, (.1, .5, .9))
        d = pd.read_csv(P / f"hourly_{net}.csv", index_col=0, parse_dates=True).dropna()
        bill = d.Ef * d["T"] / d.Tf
        D = pd.DataFrame({"b": bill.groupby(d.index.floor("D")).sum(), "e": d.E.groupby(d.index.floor("D")).sum()})
        out[f"{net}_daily_emissions_kg"] = float(D.e.mean() / 1e3)
        out[f"{net}_daily_abs_bill_error_kg"] = float((D.b - D.e).abs().mean() / 1e3)
    R = IR.load_rates()
    net = SIM.Net()
    s = pd.read_csv(P / "i2" / "everyone_spans.csv", index_col=0).share_hours_dark
    idx = [net.res.index(r) for r in s.index]
    e_share = R["E"][:, idx].sum(0) / R["E"].sum()
    for thr in (0.9, 0.99):
        m = (s >= thr).to_numpy()
        out[f"spans_dark_ge{int(thr * 100)}"] = int(m.sum())
        out[f"E_share_spans_dark_ge{int(thr * 100)}"] = float(e_share[m].sum())
    out["spearman_dark_vs_load_today"] = float(spearmanr(s, R["U"][:, idx].mean(0)).correlation)
    E = pd.read_csv(P / "i2" / "entity_bills.csv", index_col=0)
    out["i2_named_examples_kg_per_year"] = E.loc[["MAX", "CENIC", "GPN"], ["flat_kg_per_year", "path_kg_per_year"]].round(0).to_dict("index")
    sw = pd.read_csv(P / "switch" / "entity_bills.csv", index_col=0)
    out["switch_median_kg_per_year_by_type"] = sw.groupby("type").flat_kg_per_year.median().round(2).to_dict()
    out["switch_entities_by_type"] = sw.groupby("type").size().to_dict()
    out["switch_top10_hidden_routers"] = int(sw.index[:10].str.startswith("hidden:").sum())
    json.dump(out, open(P / "text_numbers.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
