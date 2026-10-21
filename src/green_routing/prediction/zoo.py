"""Model zoo — a curated, self-describing home for the runs worth keeping.

A backtest writes everything it produced into ``--out``: metric CSVs, plots, raw
records, and ``checkpoints/`` with one weights file per fitted model. That directory is
a *run*: it mixes twelve models together, is named after an experiment rather than a
result, and gets overwritten or pruned. The zoo is the other half — you finish a run,
decide which models were actually good, and **promote** those into

    artifacts/zoo/<model>/<tag>/

where each entry holds that one model's weights, its learning curve, its slice of the
training log, its metrics, and a manifest recording exactly which data window, code
commit and environment produced it. Entries are immutable and stable-pathed, so an entry
can be uploaded once and cited later.

Promotion is always explicit — nothing lands in the zoo unless you ask:

    gr-zoo add artifacts/pred_online --models gbdt+online,tcn \\
        --note "best day-horizon so far"
    gr-zoo list
    gr-zoo pack gbdt+online/pred_online

``pack`` builds a single ``.tar.gz`` and prints the upload command for you to run —
the zoo never uploads anything itself.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import config as ds_config

# --- Layout ------------------------------------------------------------------

from ..paths import ARTIFACTS_DIR as ARTIFACTS
ZOO_DIR = Path(os.environ.get("GR_ZOO_DIR") or (ARTIFACTS / "zoo"))

MANIFEST = "manifest.json"
# Files copied verbatim from the run directory into an entry, when present.
RUN_FILES = ("config.json",)
# Metric tables that carry a "model" column and are therefore sliceable per entry.
MODEL_TABLES = {
    "aggregate.csv": "aggregate.csv",
    "per_link.csv": "per_link.csv",
    "peaks.csv": "peaks.csv",
    "wape.csv": "wape.csv",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    """Filesystem-safe directory name that still reads like the model name."""
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", text).strip("_") or "unnamed"


def entry_dir(model: str, tag: str) -> Path:
    return ZOO_DIR / _slug(model) / _slug(tag)


# --- Provenance --------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_info() -> dict:
    """Commit the entry was produced at — the only reliable link back to the code."""
    def _run(*args):
        try:
            return subprocess.run(args, cwd=Path(__file__).resolve().parent,
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = _run("git", "rev-parse", "HEAD")
    if not commit:
        return {}
    return {
        "commit": commit,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("git", "status", "--porcelain")),
    }


def _env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    try:
        import sklearn
        info["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    return info


# --- Reading a run directory -------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def checkpoint_for(run: Path, model: str) -> Path | None:
    """The final-weights file for ``model``, ignoring transient training state."""
    ckpt = run / "checkpoints"
    if not ckpt.is_dir():
        return None
    hits = [p for p in sorted(ckpt.glob(f"{model}.*"))
            if not p.name.endswith(".train.pt")]
    return hits[0] if hits else None


def models_in_run(run: Path) -> list[str]:
    """Every model the run scored, newest-results-first order not implied."""
    agg = _read_csv(run / "aggregate.csv")
    if agg is None or "model" not in agg:
        return []
    return sorted(agg["model"].astype(str).unique())


def _headline_metrics(run: Path, model: str, metric: str = "mae") -> dict:
    """Per-horizon metrics for one model, plus skill vs the naive baselines.

    Skill is recomputed here from ``aggregate.csv`` rather than parsed out of the
    run's wide ``skill_*.csv`` — same definition (1 - model/baseline), but robust to
    that file's pivoted layout.
    """
    agg = _read_csv(run / "aggregate.csv")
    out: dict = {"by_horizon": {}, "skill_vs": {}}
    if agg is None or "model" not in agg:
        return out
    mine = agg[agg.model.astype(str) == model]
    for _, r in mine.iterrows():
        hn = str(r.get("horizon_name", r.get("horizon")))
        out["by_horizon"][hn] = {
            k: (float(r[k]) if k in r and pd.notna(r[k]) else None)
            for k in ("mae", "rmse", "smape", "wape")
        }
        out["by_horizon"][hn]["n"] = int(r["n"]) if "n" in r and pd.notna(r["n"]) else None
    for base in ("persistence", "seasonal_naive_day", "seasonal_naive_week"):
        b = agg[agg.model.astype(str) == base]
        if b.empty or mine.empty:
            continue
        bser = b.set_index("horizon_name")[metric]
        mser = mine.set_index("horizon_name")[metric]
        skills = {}
        for hn in mser.index:
            if hn in bser.index and pd.notna(bser[hn]) and bser[hn] != 0:
                skills[str(hn)] = float(1.0 - mser[hn] / bser[hn])
        if skills:
            out["skill_vs"][base] = skills
    out["metric"] = metric

    ratio = None
    for p in sorted(run.glob("ratio_vs_persistence_*.csv")):
        ratio = _read_csv(p)
        break
    if ratio is not None and "model" in ratio:
        mr = ratio[ratio.model.astype(str) == model]
        if not mr.empty:
            out["ratio_vs_persistence"] = {
                str(r["horizon_name"]): {
                    k: (float(r[k]) if pd.notna(r[k]) else None)
                    for k in ("median_ratio", "geomean_ratio", "logv_ratio",
                              "median_wape", "share_beats_base")
                    if k in r
                }
                for _, r in mr.iterrows()
            }
    peaks = _read_csv(run / "peaks.csv")
    if peaks is not None and "model" in peaks:
        mp = peaks[peaks.model.astype(str) == model]
        if not mp.empty:
            out["peaks"] = {
                str(r["horizon_name"]): {
                    k: (float(r[k]) if pd.notna(r[k]) else None)
                    for k in ("peak_bias_pct", "peak_underpred_rate", "peak_mae",
                              "peak_wape", "off_bias_pct")
                    if k in r
                }
                for _, r in mp.iterrows()
            }
    return out


_LOG_TAG = re.compile(r"\[([^\]]+)\]")


def _log_slice(run: Path, model: str) -> str | None:
    """This model's lines from the run log.

    Training progress is tagged ``[<model>]`` / ``[<model>:online]`` and the harness
    tags its own lines ``[harness]``; we keep the model's own lines plus the harness
    lines that name it, which is the readable story of one model's training.
    """
    log = run / "run.log"
    if not log.is_file():
        return None
    keep: list[str] = []
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        m = _LOG_TAG.match(line.strip())
        if not m:
            continue
        tag = m.group(1)
        if tag == model or tag.startswith(f"{model}:"):
            keep.append(line)
        elif tag in ("harness", "cleanup") and re.search(
                rf"(?<![\w+]){re.escape(model)}(?![\w+])", line):
            keep.append(line)
    return "\n".join(keep) + "\n" if keep else None


def _history(run: Path, model: str) -> pd.DataFrame | None:
    """Per-epoch learning curve, written by run.py from the trainer's train_history."""
    p = run / "history" / f"{model}.csv"
    return _read_csv(p) if p.is_file() else None


# --- Promotion ---------------------------------------------------------------

def add(
    run_dir: str | Path,
    models: list[str] | None = None,
    tag: str | None = None,
    note: str = "",
    with_records: bool = False,
    force: bool = False,
    metric: str = "mae",
    verbose: bool = True,
) -> list[Path]:
    """Promote models from a finished run into the zoo. Returns the entry dirs.

    ``models``: names to promote (default: every model with a weights file — the
    baselines and classical tiers have nothing to save, so they are skipped unless
    named explicitly). ``tag``: entry name under ``zoo/<model>/`` (default: the run
    directory's name), so re-promoting the same run to the same tag is a deliberate
    ``--force`` rather than a silent overwrite.
    """
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise SystemExit(f"not a run directory: {run}")
    scored = models_in_run(run)
    if not scored:
        raise SystemExit(f"{run}/aggregate.csv missing or empty — is the run finished?")

    if models:
        unknown = [m for m in models if m not in scored]
        if unknown:
            raise SystemExit(f"not in this run: {unknown}\navailable: {scored}")
        wanted = list(models)
    else:
        wanted = [m for m in scored if checkpoint_for(run, m) is not None]
        if not wanted:
            raise SystemExit(
                f"no model in {run.name} has saved weights "
                f"(checkpoints/ is empty or pruned).\n"
                f"Name models explicitly to promote weightless entries: "
                f"--models {','.join(scored[:3])}")

    tag = tag or run.name
    run_cfg = {}
    cfg_path = run / "config.json"
    if cfg_path.is_file():
        try:
            run_cfg = json.loads(cfg_path.read_text())
        except ValueError:
            pass

    made: list[Path] = []
    for model in wanted:
        dest = entry_dir(model, tag)
        if dest.exists():
            if not force:
                raise SystemExit(
                    f"entry already exists: {dest}\n"
                    f"pass --force to replace it, or --tag <name> to keep both")
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        agg = _read_csv(run / "aggregate.csv")
        tier = None
        if agg is not None and "tier" in agg:
            t = agg[agg.model.astype(str) == model]["tier"]
            if not t.empty and pd.notna(t.iloc[0]):
                tier = int(t.iloc[0])

        # weights
        weights = {"file": None, "bytes": None, "sha256": None, "source": None}
        ck = checkpoint_for(run, model)
        if ck is not None:
            target = dest / f"weights{ck.suffix}"
            shutil.copy2(ck, target)
            weights = {
                "file": target.name,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "source": str(ck.relative_to(run)),
            }
        elif verbose:
            print(f"  [{model}] no weights file (stateless model or pruned checkpoint)")

        # metric tables, sliced to this model
        for src, dst in MODEL_TABLES.items():
            df = _read_csv(run / src)
            if df is None or "model" not in df:
                continue
            sub = df[df.model.astype(str) == model]
            if not sub.empty:
                sub.to_csv(dest / dst, index=False)
        for p in sorted(run.glob("ratio_vs_persistence_*.csv")):
            df = _read_csv(p)
            if df is not None and "model" in df:
                sub = df[df.model.astype(str) == model]
                if not sub.empty:
                    sub.to_csv(dest / "ratio_vs_persistence.csv", index=False)

        # learning curve + this model's log lines
        hist = _history(run, model)
        if hist is not None and not hist.empty:
            hist.to_csv(dest / "history.csv", index=False)
        slice_ = _log_slice(run, model)
        if slice_:
            (dest / "train.log").write_text(slice_)

        # raw per-point predictions: big, so opt-in, and gzipped when kept
        records_info = None
        rec = run / "records" / f"{model}.csv"
        if with_records and rec.is_file():
            target = dest / "records.csv.gz"
            with open(rec, "rb") as fin, gzip.open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            records_info = {"file": target.name, "bytes": target.stat().st_size,
                            "rows": sum(1 for _ in open(rec)) - 1}

        for name in RUN_FILES:
            if (run / name).is_file():
                shutil.copy2(run / name, dest / f"run_{name}")

        train_summary = None
        sp = run / "history" / f"{model}.summary.json"
        if sp.is_file():
            try:
                train_summary = json.loads(sp.read_text())
            except ValueError:
                pass

        manifest = {
            "schema": 1,
            "model": model,
            "tag": tag,
            "tier": tier,
            "created_utc": _utcnow(),
            "note": note,
            "run": {
                "dir": str(run),
                "name": run.name,
                "config": run_cfg,
            },
            "data": {
                # The source the RUN read, recorded by run.py into config.json.
                # Falling back to the live config would describe this machine at
                # promotion time, which is not where the weights came from.
                "source": run_cfg.get("data_source") or ds_config.describe_source(),
                "source_recorded_by_run": "data_source" in run_cfg,
                "start": run_cfg.get("start"),
                "end": run_cfg.get("end"),
                "value_col": run_cfg.get("value_col"),
                "min_coverage": run_cfg.get("min_coverage"),
                "n_links": run_cfg.get("n_links"),
                "n_steps": run_cfg.get("n_steps"),
                "split": {k: run_cfg.get(k) for k in
                          ("train_end", "val_start", "val_end", "test_start")},
            },
            "training": train_summary,
            "weights": weights,
            "records": records_info,
            "metrics": _headline_metrics(run, model, metric=metric),
            "env": _env_info(),
            "git": _git_info(),
        }
        (dest / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str))
        made.append(dest)
        if verbose:
            mb = (weights["bytes"] or 0) / 1e6
            print(f"  + {model}/{tag}  ({mb:.1f} MB weights)"
                  if weights["file"] else f"  + {model}/{tag}  (no weights)")

    write_index()
    return made


# --- Reading the zoo ---------------------------------------------------------

def entries() -> list[dict]:
    """Every manifest in the zoo, with its directory attached as ``_dir``."""
    out = []
    if not ZOO_DIR.is_dir():
        return out
    for mf in sorted(ZOO_DIR.glob(f"*/*/{MANIFEST}")):
        try:
            m = json.loads(mf.read_text())
        except ValueError:
            continue
        m["_dir"] = str(mf.parent)
        out.append(m)
    return out


def index_frame(metric: str = "mae") -> pd.DataFrame:
    """One row per entry: identity, data window, and headline numbers."""
    rows = []
    for m in entries():
        by_h = (m.get("metrics") or {}).get("by_horizon", {}) or {}
        skill = ((m.get("metrics") or {}).get("skill_vs", {}) or {}).get("persistence", {})
        row = {
            "model": m.get("model"),
            "tag": m.get("tag"),
            "tier": m.get("tier"),
            "created": (m.get("created_utc") or "")[:16].replace("T", " "),
            "window": f"{(m.get('data') or {}).get('start')}..{(m.get('data') or {}).get('end')}",
            "links": (m.get("data") or {}).get("n_links"),
            "weights_mb": round(((m.get("weights") or {}).get("bytes") or 0) / 1e6, 1),
            "note": m.get("note") or "",
        }
        for hn, vals in by_h.items():
            row[f"{metric}@{hn}"] = vals.get(metric)
            if metric != "wape":
                row[f"wape@{hn}"] = vals.get("wape")      # sum|e|/sum|y|, %
        for hn, v in skill.items():
            row[f"skill@{hn}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["model", "created"]).reset_index(drop=True) if not df.empty else df


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown``, which needs ``tabulate`` —
    not installed in the prediction env, and not worth a dependency for one table.
    """
    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
            return ""
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v).replace("|", "\\|")

    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(cell(r[c]) for c in df.columns) + " |"
            for _, r in df.iterrows()]
    return "\n".join([head, rule] + body)


def write_index(metric: str = "mae") -> None:
    """Refresh ``index.json`` + ``README.md`` so the zoo is browsable on disk."""
    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    all_entries = entries()
    (ZOO_DIR / "index.json").write_text(json.dumps(
        [{k: v for k, v in m.items() if k != "_dir"} | {"dir": m["_dir"]}
         for m in all_entries],
        indent=2, default=str))

    df = index_frame(metric=metric)
    lines = [
        "# Model zoo",
        "",
        f"{len(all_entries)} entr{'y' if len(all_entries) == 1 else 'ies'} · "
        f"regenerated {_utcnow()}",
        "",
        "Each entry is one model from one run: weights, learning curve, that model's",
        "slice of the training log, its metrics, and a manifest recording the data",
        "window, git commit and environment behind it.",
        "",
        "```bash",
        "gr-zoo add <run_dir> --models <name> --note '...'",
        "gr-zoo pack <model>/<tag>      # -> .tar.gz to upload",
        "```",
        "",
    ]
    if df.empty:
        lines.append("_Empty — promote a run with `gr-zoo add`._")
    else:
        lines += [_markdown_table(df), ""]
        lines += ["Lower `mae@<horizon>` is better; `wape@<horizon>` is the pooled "
                  "WAPE, sum|e|/sum|y| in %; `skill@<horizon>` is "
                  "`1 - mae/persistence` (positive beats the naive baseline).", ""]
    (ZOO_DIR / "README.md").write_text("\n".join(lines))


def resolve(selector: str) -> Path:
    """``<model>/<tag>`` -> entry directory (errors list what is available)."""
    if "/" not in selector:
        raise SystemExit(f"selector must be <model>/<tag>, got {selector!r}")
    model, tag = selector.split("/", 1)
    d = entry_dir(model, tag)
    if not (d / MANIFEST).is_file():
        have = [f"{m['model']}/{m['tag']}" for m in entries()]
        raise SystemExit(f"no such entry: {selector}\navailable: {have or '(zoo empty)'}")
    return d


def remove(selector: str) -> None:
    d = resolve(selector)
    shutil.rmtree(d)
    parent = d.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    write_index()


# --- Packing for a manual upload ---------------------------------------------

def pack(selectors: list[str], dest: str | Path | None = None) -> Path:
    """Bundle entries into one ``.tar.gz`` for manual upload. Uploads nothing."""
    dirs = [resolve(s) for s in selectors]
    if dest is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = (_slug(selectors[0].replace("/", "__")) if len(dirs) == 1
                else f"zoo-{len(dirs)}-entries")
        dest = ZOO_DIR / "_packs" / f"{name}_{stamp}.tar.gz"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        for d in dirs:
            tf.add(d, arcname=str(Path(d).relative_to(ZOO_DIR)))
    return dest


def upload_hint(archive: Path) -> str:
    """The command to run by hand. Printing beats doing: an upload is outward-facing
    and the bucket/prefix is the operator's call, not this tool's."""
    bucket = ds_config.S3_BUCKET
    endpoint = ds_config.S3_ENDPOINT
    profile = ds_config.S3_PROFILE
    prof = f" --profile {profile}" if profile else ""
    return (
        f"aws s3 cp {archive} s3://<your-model-bucket>/zoo/{archive.name} \\\n"
        f"    --endpoint-url {endpoint}{prof}\n"
        f"  (data bucket is {bucket!r}; publish models to a bucket you own)\n"
        f"  or: rsync -avP {archive} <host>:<path>/"
    )


# --- CLI ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="gr-zoo", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="promote models from a finished run")
    a.add_argument("run_dir")
    a.add_argument("--models", help="comma-separated names (default: those with weights)")
    a.add_argument("--tag", help="entry name (default: the run directory's name)")
    a.add_argument("--note", default="", help="why this run is worth keeping")
    a.add_argument("--with-records", action="store_true",
                   help="also keep the raw per-point predictions (gzipped; large)")
    a.add_argument("--force", action="store_true", help="replace an existing entry")
    a.add_argument("--metric", default="mae")

    l = sub.add_parser("list", help="table of every entry")
    l.add_argument("--model", help="filter by model name")
    l.add_argument("--metric", default="mae")

    s = sub.add_parser("show", help="print one entry's manifest")
    s.add_argument("selector", help="<model>/<tag>")

    pk = sub.add_parser("pack", help="bundle entries into a .tar.gz to upload by hand")
    pk.add_argument("selectors", nargs="+", help="<model>/<tag> ...")
    pk.add_argument("--out", help="archive path")

    rm = sub.add_parser("remove", help="delete an entry")
    rm.add_argument("selector", help="<model>/<tag>")
    rm.add_argument("--yes", action="store_true", help="skip the confirmation")

    sub.add_parser("index", help="regenerate index.json and README.md")

    args = ap.parse_args(argv)

    if args.cmd == "add":
        names = [m.strip() for m in args.models.split(",")] if args.models else None
        print(f"[zoo] promoting from {args.run_dir} -> {ZOO_DIR}")
        made = add(args.run_dir, models=names, tag=args.tag, note=args.note,
                   with_records=args.with_records, force=args.force,
                   metric=args.metric)
        print(f"[zoo] {len(made)} entr{'y' if len(made) == 1 else 'ies'} written; "
              f"index at {ZOO_DIR / 'README.md'}")
        return 0

    if args.cmd == "list":
        df = index_frame(metric=args.metric)
        if df.empty:
            print(f"zoo is empty ({ZOO_DIR})")
            return 0
        if args.model:
            df = df[df.model == args.model]
        pd.set_option("display.width", 200, "display.max_columns", 40)
        print(df.to_string(index=False))
        return 0

    if args.cmd == "show":
        print((resolve(args.selector) / MANIFEST).read_text())
        return 0

    if args.cmd == "pack":
        archive = pack(args.selectors, args.out)
        print(f"[zoo] packed {len(args.selectors)} entr"
              f"{'y' if len(args.selectors) == 1 else 'ies'} -> {archive} "
              f"({archive.stat().st_size / 1e6:.1f} MB)")
        print("\nUpload it yourself with:\n  " + upload_hint(archive))
        return 0

    if args.cmd == "remove":
        d = resolve(args.selector)
        if not args.yes:
            reply = input(f"delete {d}? [y/N] ").strip().lower()
            if reply not in ("y", "yes"):
                print("aborted")
                return 1
        remove(args.selector)
        print(f"[zoo] removed {args.selector}")
        return 0

    write_index()
    print(f"[zoo] index regenerated -> {ZOO_DIR / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
