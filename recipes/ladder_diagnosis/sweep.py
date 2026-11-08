"""Run a list of hourly_train.py configurations one after another (skip finished tags)
and print a summary table of every finished run of a dataset.

usage: python sweep.py run GRIDFILE [SCRIPT]   (one argument string per line, '#' comments;
                                               SCRIPT defaults to hourly_train.py)
       python sweep.py table DS [PATTERN]
       python sweep.py table5 DS             (the long5m.py runs, tags L5*)
"""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import CACHE

HERE = Path(__file__).resolve().parent


def run(gridfile, script="hourly_train.py"):
    for line in Path(gridfile).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        a = shlex.split(line)
        ds = a[a.index("--ds") + 1]
        tag = a[a.index("--tag") + 1]
        if (CACHE / ds / "runs" / tag / "summary.json").exists():
            print(f"skip {tag}", flush=True)
            continue
        subprocess.run([sys.executable, "-W", "ignore", str(HERE / script), *a],
                       check=False)


def table(ds, pat=""):
    rows = []
    for p in sorted((CACHE / ds / "runs").glob("*/summary.json")):
        s = json.loads(p.read_text())
        if pat and pat not in p.parent.name:
            continue
        a = s["args"]
        rows.append({"tag": p.parent.name, "model": a["model"], "ctx": a["ctx"],
                     "norm": a["norm"], "loss": a["loss"], "hist": a["hist"],
                     "ep": s["epochs_run"], "val": round(s["best_val"], 2),
                     "day": round(s["test"]["day"]["median_wape"], 2),
                     "day_bias": round(s["test"]["day"]["median_link_bias"], 1),
                     "day_beats_pt": round(s["test"]["day"]["beats_ref"], 2),
                     "week": round(s["test"]["week"]["median_wape"], 2),
                     "week_bias": round(s["test"]["week"]["median_link_bias"], 1),
                     "sec": round(s["seconds"])})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def table5(ds):
    rows = []
    for p in sorted((CACHE / ds / "runs").glob("L5*/summary.json")):
        s = json.loads(p.read_text())
        a, t = s["args"], s["test"]
        rows.append({"tag": p.parent.name, "weeks": a["ctx"] // 2016, "hist": a["hist"],
                     "ep": s["epochs_run"], "val": round(s["best_val"], 2),
                     "day": round(t["day"]["median_wape"], 2), "day5": round(t["day"]["wape5"], 2),
                     "week": round(t["week"]["median_wape"], 2),
                     "week5": round(t["week"]["wape5"], 2),
                     "day_bias": round(t["day"]["median_link_bias"], 1),
                     "beats_pt": round(t["day"]["beats_ref"], 2),
                     "Mparams": round(s["params"] / 1e6, 1), "min": round(s["seconds"] / 60, 1)})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["weeks", "tag"])
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:4])
    elif sys.argv[1] == "table5":
        table5(sys.argv[2])
    else:
        table(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
