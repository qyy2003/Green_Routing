"""Does a model use the old weeks of its context, and does it over-fit them?

For trained 24-week models (long5m.py runs saved with --save 1) and for Timestral (hourly input,
24 weeks, zero-shot), day-ahead forecasts at 00:00 UTC are scored three ways:
  full    the real context
  tile    weeks 5-24 replaced by copies of the last 4 weeks (same length, only 4 weeks of
          information, every value still a real traffic value)
  blank   weeks 5-24 missing
and in three periods: the last 13 weeks of the training period, the validation block, the test
block. Score: median link hourly WAPE (complete hours). Sensitivity: median over links of
mean |forecast(full) - forecast(tile)| / mean actual, in %.

usage: python uses_context.py DS TAG [TAG ...]     (TAG of a saved long5m run, or "timestral")
Writes <CACHE>/<ds>/uses_context_<tag>.csv.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch

from common import DS, CACHE
from long5m import Data5, Windows5, MODELS, W, D, HW

WK = 4                                        # weeks kept by tile / blank
TRAIN_WEEKS = 13


def wape_med(P, A):
    """P, A: [L, n_hours] hourly forecast / actual (NaN = missing). Median link WAPE."""
    ok = np.isfinite(P) & np.isfinite(A)
    e = np.where(ok, np.abs(P - A), 0).sum(1)
    a = np.where(ok, A, 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(a > 1e-8, 100 * e / a, np.nan)
    return float(np.nanmedian(w)), w


def sensitivity(P1, P2, A):
    ok = np.isfinite(P1) & np.isfinite(P2) & np.isfinite(A)
    d = np.where(ok, np.abs(P1 - P2), 0).sum(1)
    a = np.where(ok, A, 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmedian(np.where(a > 1e-8, 100 * d / a, np.nan)))


def periods(data):
    """Midnight origins (absolute 5-minute indices) of the three periods."""
    te = np.arange(data.test0, data.N - D + 1, D)
    va = np.arange(data.val[0], data.val[1] - D + 1, D)
    tr = np.arange(data.train_end - TRAIN_WEEKS * W, data.train_end - D + 1, D)
    return {"train": tr, "val": va, "test": te}


def hourly_actual(data, starts):
    """[L, len(starts)*24] complete-hour means of the 5-minute targets."""
    L = data.L
    out = np.full((L, len(starts), 24), np.nan, np.float32)
    for k, s in enumerate(starts):
        blk = data.Y[s:s + D].reshape(24, 12, L)
        ok = np.isfinite(blk).all(1)
        out[:, k] = np.where(ok, blk.mean(1), np.nan).T
    return out.reshape(L, -1)


def trained(ds_name, tag):
    s = json.loads((CACHE / ds_name / "runs" / tag / "summary.json").read_text())
    a = s["args"]
    ds = DS(ds_name)
    data = Data5(ds, a["hist"])
    C = a["ctx"]
    Wn = Windows5(data, C, W, "cuda", a.get("squash", 0.0), a.get("norm", "lastmean"))
    net = MODELS[a["model"]](C, W, seg=a.get("seg"), qprof=a.get("qprof", 0), dropout=a["dropout"],
                             d=a.get("d", 128), layers=a.get("layers", 3)).cuda()
    net.load_state_dict(torch.load(CACHE / ds_name / "runs" / tag / "model.pt"))
    net.eval()

    def predict(starts, ablate):
        P = np.full((data.L, len(starts) * 24), np.nan, np.float32)
        s_t = torch.tensor(starts, device="cuda")
        with torch.no_grad():
            for j in range(data.L):
                j_t = torch.full_like(s_t, j)
                ps = []
                for b in range(0, len(starts), 128):
                    xn, mk, yn, ym, inv, y = Wn.get(s_t[b:b + 128], j_t[b:b + 128], ablate, WK * W)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = net(xn, mk, j_t[b:b + 128])
                    p = inv(p.float())[:, :D]
                    p[(mk[:, -D:] > 0).sum(1) == 0] = float("nan")
                    ps.append(p.view(-1, 24, 12).mean(-1))
                P[j] = torch.cat(ps).cpu().numpy().reshape(-1)
        return P
    return data, predict


def timestral(ds_name):
    """Timestral 209M on hourly means, 4,032 hours of context; inputs from the hourly harness's
    extended history (NaN gaps, no filling), so training-period days have full context."""
    from timestral import TimestralForecaster
    from hourly_train import Data as HData
    ds = DS(ds_name)
    hd = HData(ds, "ext")
    data = Data5(ds, "ext")
    fm = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda")
    C, keep = 4032, WK * HW
    # 5-minute origin -> hourly row of the extended hourly array

    def to_hour(s):
        # data (5-min) index s is ds.ts[s - data.pre] for panel rows; hourly row = hd.pre + (s - pre - off)/12
        return hd.pre + (s - data.pre - ds.off) // 12

    def predict(starts, ablate):
        P = np.full((data.L, len(starts) * 24), np.nan, np.float32)
        jobs = [(j, k) for j in range(data.L) for k in range(len(starts))]
        for b0 in range(0, len(jobs), 256):
            jb = jobs[b0:b0 + 256]
            X = np.full((len(jb), 1, C), np.nan, np.float32)
            for r, (j, k) in enumerate(jb):
                h = to_hour(starts[k])
                a0 = max(0, h - C)
                X[r, 0, C - (h - a0):] = hd.Y[a0:h, j]
            if ablate == "tile":
                old = C - keep
                src = (np.arange(C) - old) % keep + old
                X = X[:, :, src]
            elif ablate == "blank":
                X[:, :, :C - keep] = np.nan
            with torch.inference_mode():
                f = fm.forecast(X, prediction_length=24)
            med = f.median[:, 0, :].float().cpu().numpy()
            for r, (j, k) in enumerate(jb):
                if np.isfinite(X[r, 0, -24:]).any():
                    P[j, k * 24:(k + 1) * 24] = med[r]
        return P
    return data, predict


def main():
    name, tags = sys.argv[1], sys.argv[2:]
    for tag in tags:
        data, predict = timestral(name) if tag == "timestral" else trained(name, tag)
        rows = []
        for per, starts in periods(data).items():
            A = hourly_actual(data, starts)
            P = {ab: predict(starts, ab) for ab in ("full", "tile", "blank")}
            r = {"period": per, "days": len(starts)}
            for ab, p in P.items():
                r[ab] = wape_med(p, A)[0]
            r["sens_tile"] = sensitivity(P["full"], P["tile"], A)
            r["sens_blank"] = sensitivity(P["full"], P["blank"], A)
            rows.append(r)
            print(name, tag, r, flush=True)
        pd.DataFrame(rows).to_csv(CACHE / name / f"uses_context_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
