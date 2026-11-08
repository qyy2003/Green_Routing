"""Fine-tune Timestral on one network's training period (hourly means, pinball loss over its
21 quantiles, one 32-hour patch ahead), select on the validation block, score on the test
block exactly like the zero-shot runs (fm.forecast, symmetric averaging, median).

usage: python ft_timestral.py --ds i2 --hist ext --ctx 4032 [--size 209M] [--lr 1e-5]
Writes <CACHE>/<ds>/runs/<tag>/{pred_day.npy, pred_week.npy, summary.json, history.csv}.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from common import DS, CACHE, score
from hourly_train import Data, valid_origins, WEEK
from timestral_ctx import forecast


def pinball(q, y, levels, ym):
    """q [B, H, Q] quantiles, y [B, H] targets (NaN-free where ym) -> mean pinball loss."""
    lv = torch.tensor(levels, device=q.device, dtype=q.dtype)
    d = y[..., None] - q
    loss = torch.maximum(lv * d, (lv - 1) * d)
    per_win = (loss.mean(-1) * ym).sum(1) / ym.sum(1).clamp_min(1)
    # robust: a window whose loss exceeds `cap` context means (a link waking up from ~0)
    # must not dominate the batch -- it made the first fine-tune's loss explode
    return per_win.clamp(max=2.0).mean()


def run(args):
    from timestral import TimestralForecaster
    from timestral.forecaster import ffill_with_mask
    torch.manual_seed(args.seed)
    dev = "cuda"
    ds = DS(args.ds)
    data = Data(ds, args.hist)
    fm = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{args.size}", device=dev)
    model = fm.model
    ps = fm.patch_size
    assert args.ctx % ps == 0
    Hp = ps                                                 # one patch = 32 h ahead
    levels = fm.quantile_levels
    Y = torch.tensor(np.vstack([np.full((args.ctx, data.L), np.nan, np.float32), data.Y]),
                     device=dev)
    pad = args.ctx
    tr_s, tr_j = valid_origins(data, 0, data.train_end, args.ctx, Hp)
    print(f"[{args.tag}] train windows {len(tr_s):,}", flush=True)
    tr_s_t, tr_j_t = torch.tensor(tr_s, device=dev), torch.tensor(tr_j, device=dev)
    ar_c, ar_h = torch.arange(args.ctx, device=dev), torch.arange(Hp, device=dev)

    def batch(idx):
        s = tr_s_t[idx] + pad
        j = tr_j_t[idx]
        x = Y[(s[:, None] - args.ctx) + ar_c[None, :], j[:, None]]
        y = Y[s[:, None] + ar_h[None, :], j[:, None]]
        m = torch.isfinite(x)
        x = ffill_with_mask(torch.nan_to_num(x)[:, None, :], m[:, None, :])[:, 0, :]
        sc = (torch.nan_to_num(x).abs() * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
        return x, m, y, sc.clamp_min(1.0)                  # >= 1 Mbit/s (cohort floor)

    # validation: midnight origins of the val block, day-ahead, median link WAPE
    v_starts = np.arange(data.val[0], data.val[1] - 24 + 1, 24)

    def val_wape():
        model.eval()
        P = forecast(fm, data.Y, v_starts, args.ctx, 24, data.L, batch=64)  # [L, nb*24]
        A = np.stack([data.A[s:s + 24] for s in v_starts]).transpose(2, 0, 1).reshape(data.L, -1)
        ok = np.isfinite(P) & np.isfinite(A)
        num = np.where(ok, np.abs(P - A), 0).sum(1)
        den = np.where(ok, A, 0).sum(1)
        w = 100 * num / np.where(den > 1e-6, den, np.nan)
        model.train()
        return float(np.nanmedian(w))

    params = [p for p in model.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=max(0.05, 3 / total))
    hist = []
    best = val_wape()
    print(f"[{args.tag}] zero-shot val day {best:.2f}", flush=True)
    hist.append({"epoch": 0, "train_loss": float("nan"), "val_wape_day": best})
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad, t0 = 0, time.time()
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    model.train()
    for ep in range(args.epochs):
        tl = 0.0
        for it in range(args.steps):
            idx = torch.randint(0, tr_s_t.numel(), (args.bs,), device=dev, generator=g)
            x, m, y, sc = batch(idx)
            gid = torch.arange(x.shape[0], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = model.predict_block(x, m, gid, 1)               # [B, 32, Q] raw space
            ym = torch.isfinite(y)
            loss = pinball(q.float() / sc[..., None], torch.nan_to_num(y) / sc, levels, ym)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tl += float(loss)
        v = val_wape()
        hist.append({"epoch": ep + 1, "train_loss": tl / args.steps, "val_wape_day": v,
                     "sec": time.time() - t0})
        print(f"[{args.tag}] ep {ep + 1} train {tl / args.steps:.4f} val day {v:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if v < best - 1e-3:
            best, bad = v, 0
            best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    n = ds.n_test_hours()
    A = ds.actual_test()
    od = CACHE / args.ds / "runs" / args.tag
    od.mkdir(parents=True, exist_ok=True)
    res = {}
    for hz, H in (("day", 24), ("week", 168)):
        starts = np.arange(data.test0, data.N, H)
        P = forecast(fm, data.Y, starts, args.ctx, H, data.L, batch=64)[:, :n]
        np.save(od / f"pred_{hz}.npy", P)
        refs = {m: ds.load_dense_block(m, hz) for m in ("patchtst", "timestral@1h_24w")}
        df, _ = score({**refs, args.tag: P}, A, ref="patchtst")
        print(df.round(3).to_string(index=False), flush=True)
        r = df.set_index("model").loc[args.tag]
        res[hz] = {k: float(v) for k, v in r.items()}
    pd.DataFrame(hist).to_csv(od / "history.csv", index=False)
    a = vars(args) | {"model": f"timestral-{args.size}-ft", "norm": "-", "loss": "pinball"}
    (od / "summary.json").write_text(json.dumps(
        {"args": a, "epochs_run": len(hist) - 1, "best_val": best, "test": res,
         "seconds": time.time() - t0}, indent=1))
    print(f"[{args.tag}] DONE day {res['day']['median_wape']:.2f} week "
          f"{res['week']['median_wape']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--hist", default="ext")
    ap.add_argument("--ctx", type=int, default=4032)
    ap.add_argument("--size", default="209M")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    args.tag = args.tag or f"F_ft_{args.size}_c{args.ctx}_{args.hist}_lr{args.lr:g}"
    run(args)


if __name__ == "__main__":
    main()
