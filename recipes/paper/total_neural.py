"""DLinear, GRU, TCN and PatchTST trained on a network's hourly total traffic, day ahead (paper appendix).

Internet2 has no 5-minute door data, so its total can only be forecast from hourly values; the
four networks are the hourly versions of the ladder's architectures (recipes/ladder_diagnosis/
hourly_train.py, classes reused unchanged). Each reads 4 weeks of hourly history, scaled by the
mean of that context (the ladder diagnosis's best normalisation), and forecasts the next 24 hours;
L1 loss, AdamW (lr 1e-3, PatchTST 1e-4; weight decay 1e-4, none on weight-norm directions), batch
256, early stopping on the validation days (patience 5, at most 40 epochs). Training origins are
every hour of the training range, validation and test origins 00:00 UTC of each day; three seeds,
averaged. Splits as the per-link models: Internet2 train to 2026-06-14, validation 06-21 -> 07-12,
test 07-19 -> 10-31; SWITCHlan (also run, as a check against its 5-minute ladder) train
2025-02 -> 2026-02-15, validation 02-22 -> 03-15, test 03-22 -> 07-10.

usage: python total_neural.py [i2|switch|all]    writes $GR_ARTIFACTS_DIR/paper/appendix/total_neural_<net>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from green_routing.paths import ARTIFACTS_DIR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ladder_diagnosis"))
OUT = ARTIFACTS_DIR / "paper" / "appendix"
CTX, H = 672, 24
SPLITS = {"i2": ("2026-06-14", "2026-06-21", "2026-07-12", "2026-07-19", "2026-10-31"),
          "switch": ("2026-02-15", "2026-02-22", "2026-03-15", "2026-03-22", "2026-07-10")}


def series(net: str) -> pd.Series:
    if net == "i2":
        sys.path.insert(0, str(HERE.parent / "internet2"))
        import timestral_total as TT
        return TT.total_series()
    sys.path.insert(0, str(HERE.parent / "flat_forecast"))
    import run as R
    s = R.S
    return s[s.index >= pd.Timestamp("2025-02-01", tz="UTC")]                # pitfall 28


def windows(v: np.ndarray, origins: np.ndarray):
    X = np.stack([v[o - CTX:o] for o in origins]).astype(np.float32)
    Y = np.stack([v[o:o + H] for o in origins]).astype(np.float32)
    return X, Y


def scale(X):
    sc = np.clip(np.nanmean(X, axis=1, keepdims=True), 1e-3, None)
    return X / sc, sc


def train_one(model: str, Xtr, Ytr, Xva, Yva, seed: int, dev="cuda"):
    import hourly_train as HT
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = HT.MODELS[model](CTX, H, dropout=0.1, patch=24, n_links=0).to(dev)
    nowd = [p for n, p in net.named_parameters() if "original1" in n]
    rest = [p for n, p in net.named_parameters() if "original1" not in n]
    lr = 1e-4 if model == "patchtst" else 1e-3
    opt = torch.optim.AdamW([{"params": rest, "weight_decay": 1e-4}, {"params": nowd, "weight_decay": 0.0}], lr=lr)
    xs, ss = scale(Xtr)
    ys = Ytr / ss
    xv, sv = scale(Xva)
    yv = Yva / sv
    t = lambda a: torch.tensor(a, device=dev)                                      # noqa: E731
    Xt, Yt, Xv, Yv = t(xs), t(ys), t(xv), t(yv)
    Mt, Mv = torch.ones_like(Xt), torch.ones_like(Xv)
    best, best_state, bad = np.inf, None, 0
    for ep in range(40):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = (net(Xt[b], Mt[b]) - Yt[b]).abs().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float((net(Xv, Mv) - Yv).abs().mean())
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state)
    return net, ep + 1, sum(p.numel() for p in net.parameters())


def run(net_name: str):
    s = series(net_name)
    tr_end, va0, va1, te0, te1 = SPLITS[net_name]
    idx = s.index
    v = s.to_numpy(np.float64)
    pos = lambda ts: idx.get_indexer([pd.Timestamp(ts, tz="UTC")])[0]             # noqa: E731
    ok = np.isfinite(v)
    def origins(lo, hi, daily):
        o = np.arange(max(lo, CTX), min(hi, len(v) - H) + 1)
        if daily:
            o = o[idx[o].hour == 0]
        good = np.array([ok[i - CTX:i + H].all() for i in o])
        return o[good]
    o_tr = origins(0, pos(tr_end) - H, False)
    o_va = origins(pos(va0), pos(va1), True)
    o_te = origins(pos(te0), pos(te1), True)
    Xtr, Ytr = windows(v, o_tr)
    Xva, Yva = windows(v, o_va)
    Xte, _ = windows(np.nan_to_num(v), o_te)
    print(f"[{net_name}] train {len(o_tr):,} windows from {idx[o_tr[0]].date()}, val {len(o_va)}, test {len(o_te)} days", flush=True)
    th = pd.DatetimeIndex(np.concatenate([idx[o:o + H] for o in o_te]))
    out = pd.DataFrame(index=th)
    xs, ss = scale(Xte)
    for model in ("dlinear", "gru", "tcn", "patchtst"):
        preds = []
        for seed in (0, 1, 2):
            net, eps, npar = train_one(model, Xtr, Ytr, Xva, Yva, seed)
            with torch.no_grad():
                p = net(torch.tensor(xs, device="cuda"), torch.ones(xs.shape, device="cuda")).cpu().numpy() * ss
            preds.append(np.clip(p, 0, None).ravel())
            print(f"  {model} seed {seed}: {eps} epochs, {npar:,} parameters", flush=True)
        out[model] = np.mean(preds, axis=0)
    out["actual"] = s.reindex(th).to_numpy()
    out.to_csv(OUT / f"total_neural_{net_name}.csv")
    for m in ("dlinear", "gru", "tcn", "patchtst"):
        print(f"  {m}: hourly WAPE {100 * (out[m] - out.actual).abs().sum() / out.actual.sum():.2f} %", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    for n in ("i2", "switch"):
        if what in (n, "all"):
            run(n)
