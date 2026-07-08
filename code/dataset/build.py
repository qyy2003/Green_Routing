"""End-to-end builder: scan coverage -> select cohort -> build tensor + adjacency ->
compute splits -> write artifacts.

Usage (run as a module from the code/ dir so relative imports resolve):
    cd /home/yuyqin/ETH_Master_Study/Green_Routing/Green_Routing/code
    python -m dataset.build --start 2024-01-01 --end 2025-01-01 \
        --train-end 2024-08-31 --val-start 2024-09-08 --val-end 2024-10-20 \
        --test-start 2024-10-27 --min-coverage 0.9 --out ../artifacts/traffic2024

Artifacts written to --out:
    cohort.csv       selected series + coverage stats
    tensor.npz       X[T,N,F], timestamps, node_index, feature_names
    adjacency.npz    A, W_speed, W_cost, node_index
    splits.json      train/val/test boundaries + rolling origins
    meta.json        config used
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config, cohort as cohort_mod, tensor as tensor_mod, splits as splits_mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="cohort window start (UTC date)")
    ap.add_argument("--end", required=True, help="cohort window end (UTC date)")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--features", nargs="+", default=["in_mbps", "out_mbps"])
    # explicit calendar split (optional; falls back to fractional split)
    ap.add_argument("--train-end")
    ap.add_argument("--val-start")
    ap.add_argument("--val-end")
    ap.add_argument("--test-start")
    ap.add_argument("--horizon", type=int, default=config.DAILY_STEPS)
    ap.add_argument("--lookback", type=int, default=config.WEEKLY_STEPS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] scanning interface coverage over {args.start}..{args.end} ...")
    cov = cohort_mod.scan_interfaces(args.start, args.end)
    cohort = cohort_mod.select_cohort(cov, min_coverage=args.min_coverage)
    cohort_mod.coverage_frame(cov).to_csv(out / "coverage_all.csv", index=False)
    cohort_mod.coverage_frame(cohort).to_csv(out / "cohort.csv", index=False)
    print(f"      {len(cohort)}/{len(cov)} series pass coverage >= {args.min_coverage}")
    if not cohort:
        raise SystemExit("empty cohort; lower --min-coverage or widen the window")

    print("[2/5] building node tensor ...")
    t = tensor_mod.build_node_tensor(
        cohort, args.start, args.end, features=tuple(args.features)
    )
    np.savez_compressed(
        out / "tensor.npz",
        X=t["X"],
        timestamps=t["timestamps"],
        node_index=np.array(t["node_index"]),
        feature_names=np.array(t["feature_names"]),
    )
    print(
        f"      X shape {t['X'].shape} (T,N,F); "
        f"pre-fill NaN fraction {t['nan_fraction']:.3f}"
    )

    print("[3/5] building adjacency ...")
    adj = tensor_mod.build_adjacency(t["node_index"])
    np.savez_compressed(out / "adjacency.npz", **{
        k: v for k, v in adj.items() if k != "n_isolated"
    })
    print(
        f"      {len(t['node_index'])} nodes, "
        f"{int(adj['A'].sum() // 2)} undirected links, "
        f"{adj['n_isolated']} nodes not in topology"
    )

    print("[4/5] computing splits ...")
    n = len(t["timestamps"])
    if all([args.train_end, args.val_start, args.val_end, args.test_start]):
        sp = splits_mod.chronological_split_by_date(
            t["timestamps"], args.train_end, args.val_start,
            args.val_end, args.test_start,
        )
    else:
        sp = splits_mod.chronological_split(
            n, max_lookback=args.lookback, max_horizon=args.horizon
        )
    origins = [
        o.__dict__ for o in splits_mod.rolling_origins(
            sp.test, horizon=args.horizon, lookback=args.lookback
        )
    ]
    federated = splits_mod.federated_partition(t["node_index"], by="pop")
    with open(out / "splits.json", "w") as fh:
        json.dump(
            {
                "split": sp.as_dict(),
                "horizon": args.horizon,
                "lookback": args.lookback,
                "n_rolling_origins": len(origins),
                "rolling_origins": origins,
                "federated_pops": {k: v for k, v in federated.items()},
            },
            fh,
            indent=2,
        )
    print(
        f"      train={sp.train} val={sp.val} test={sp.test} "
        f"purge={sp.purge_steps}; {len(origins)} rolling origins; "
        f"{len(federated)} PoP clients"
    )

    print("[5/5] writing meta ...")
    with open(out / "meta.json", "w") as fh:
        json.dump(vars(args) | {
            "step_seconds": config.STEP_SECONDS,
            "n_nodes": len(t["node_index"]),
            "n_steps": n,
        }, fh, indent=2)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
