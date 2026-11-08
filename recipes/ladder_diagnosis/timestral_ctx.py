"""Timestral zero-shot on the hourly means at several context lengths (block layout, same
windows as the ladder): how much of its edge needs long context?

usage: python timestral_ctx.py DS [CTX ...]      (CTX in hours; default 168 336 672 2016 4032)
Writes <CACHE>/<ds>/runs/timestral_c<ctx>/{pred_day.npy, pred_week.npy, summary.json}.
"""
import json
import sys
import time

import numpy as np
import torch

from common import DS, CACHE, score


def forecast(fm, Hm, starts, ctx, H, L, batch=256):
    out = np.full((L, len(starts), H), np.nan, np.float32)
    jobs = [(j, b) for j in range(L) for b in range(len(starts))]
    for b0 in range(0, len(jobs), batch):
        jb = jobs[b0:b0 + batch]
        X = np.full((len(jb), 1, ctx), np.nan, np.float32)
        for r, (j, b) in enumerate(jb):
            s = starts[b]
            a = max(0, s - ctx)
            X[r, 0, ctx - (s - a):] = Hm[a:s, j]
        with torch.inference_mode():
            f = fm.forecast(X, prediction_length=H)
        med = f.median[:, 0, :].float().cpu().numpy()
        for r, (j, b) in enumerate(jb):
            if np.isfinite(X[r, 0, -24:]).any():
                out[j, b] = med[r]
    return out.reshape(L, -1)


def main():
    from timestral import TimestralForecaster
    name = sys.argv[1]
    ctxs = [int(c) for c in sys.argv[2:]] or [168, 336, 672, 2016, 4032]
    ds = DS(name)
    fm = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda")
    A = ds.actual_test()
    n = ds.n_test_hours()
    for ctx in ctxs:
        t0 = time.time()
        tag = f"timestral_c{ctx}"
        od = CACHE / name / "runs" / tag
        od.mkdir(parents=True, exist_ok=True)
        res = {}
        for hz, H in (("day", 24), ("week", 168)):
            P = forecast(fm, ds.Hm, ds.block_starts(H), ctx, H, ds.L)[:, :n]
            np.save(od / f"pred_{hz}.npy", P)
            refs = {m: ds.load_dense_block(m, hz) for m in ("patchtst", "timestral@1h_24w")}
            df, _ = score({**refs, tag: P}, A, ref="patchtst")
            print(df.round(3).to_string(index=False), flush=True)
            r = df.set_index("model").loc[tag]
            res[hz] = {k: float(v) for k, v in r.items()}
        (od / "summary.json").write_text(json.dumps(
            {"args": {"model": "timestral-209M", "ctx": ctx, "norm": "-", "loss": "-",
                      "hist": "panel"}, "epochs_run": 0, "best_val": float("nan"), "test": res,
             "seconds": time.time() - t0}, indent=1))
        print(f"[{tag}] DONE day {res['day']['median_wape']:.2f} week "
              f"{res['week']['median_wape']:.2f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
