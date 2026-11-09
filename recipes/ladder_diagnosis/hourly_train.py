"""Train one global forecaster on HOURLY means and score it in the block layout.

A lean, fast harness for controlled ablations of what separates the ladder (5-min, 1-week
input, log1p + RevIN, Huber) from Timestral (hourly, 24-week input, zero-shot): every factor
is a flag, everything else is held fixed.

  --ds        i2 | switch
  --ctx       input length in hours (168 = 1 week ... 4032 = 24 weeks)
  --norm      log_revin (the ladder: log1p, per-window mean/std, floor 0.05)
              mean      (x / mean of the window, linear space; Chronos-style)
              revin_lin (per-window mean/std in linear space)
  --loss      huber | l1 | mse  (in normalised space)
  --model     linear | mlp | patchtst
  --hist      panel (hours of the run's panel only) | ext (I2: + hourly file from 2024-11) |
              ext21 (I2: + i2_history.py's 2021 -> 2024 hours)
  --H         output path in hours (168: day blocks use the first 24 hours)

Model selection: the epoch with the lowest validation WAPE (val block, targets inside it).
Outputs <CACHE>/<ds>/runs/<tag>/{pred_day.npy, pred_week.npy, summary.json, history.csv}.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import DS, CACHE, score

WEEK = 168


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def i2_ext_history(ds: DS, since_2021: bool = False) -> tuple[np.ndarray, int]:
    """I2 hourly loads (u->v + v->u) from the 2024-11 file, before the panel's first hour
    (``since_2021``: preceded by i2_history.py's 2021-01 -> 2024-11 file).
    Returns (rows [n_pre, L], n_pre)."""
    from green_routing.paths import ARTIFACTS_DIR
    h = pd.read_csv(ARTIFACTS_DIR / "internet2/data/links_hourly.csv.gz", index_col=0,
                    parse_dates=True)
    if since_2021:
        old = pd.read_csv(CACHE / "i2" / "links_hourly_2021.csv.gz", index_col=0, parse_dates=True)
        old = old[old.index < h.index.min()]
        h = pd.concat([old.reindex(columns=h.columns), h])
    cols = {}
    for l in ds.links:
        u, v = l.split("/to-")
        cols[l] = h[f"{u}->{v}"].add(h[f"{v}->{u}"])     # NaN if either side is missing
    S = pd.DataFrame(cols)
    t0 = pd.Timestamp(int(ds.hts[0]), unit="s", tz="UTC")
    pre = S[S.index < t0]
    full = pd.date_range(pre.index.min(), t0, freq="1h", inclusive="left")
    pre = pre.reindex(full)
    return pre.to_numpy(np.float32), len(pre)


def i2_door_series(ds: DS, t0: pd.Timestamp, n: int, train_end: int) -> np.ndarray:
    """Every I2 door's hourly ingress and egress (connector ports and virtual doors, 2021 ->
    2026: i2_history.py's file, then the recipe's), on the hourly grid starting at ``t0``
    (``n`` rows). Kept: series with >= 20 % of the training hours (about a year) and a training mean >= 1
    Mbit/s. Training-only extra series for a global model."""
    from green_routing.paths import ARTIFACTS_DIR
    cols = []
    for d in ("in", "out"):
        new = pd.read_csv(ARTIFACTS_DIR / f"internet2/data/doors_{d}_hourly.csv.gz", index_col=0,
                          parse_dates=True)
        old = pd.read_csv(CACHE / "i2" / f"doors_{d}_hourly_2021.csv.gz", index_col=0,
                          parse_dates=True)
        old = old[old.index < new.index.min()]
        S = pd.concat([old, new.reindex(columns=old.columns.union(new.columns))])
        S = S.reindex(pd.date_range(t0, periods=n, freq="1h"))
        S.columns = [f"{d}:{c}" for c in S.columns]
        cols.append(S)
    X = pd.concat(cols, axis=1).to_numpy(np.float32)
    tr = X[:train_end]
    with np.errstate(invalid="ignore"):
        keep = (np.isfinite(tr).mean(0) >= 0.2) & (np.nanmean(tr, 0) >= 1.0)
    return X[:, keep]


def causal_fill(Y: np.ndarray, week: int = WEEK, ff: int = 6) -> np.ndarray:
    """Inputs only: forward-fill up to ``ff`` steps, then same step of earlier weeks (<= 4)."""
    F = pd.DataFrame(Y).ffill(limit=ff).to_numpy(np.float32)
    for _ in range(4):
        miss = ~np.isfinite(F)
        if not miss.any():
            break
        sh = np.full_like(F, np.nan)
        sh[week:] = F[:-week]
        F = np.where(miss, sh, F)
    return F


class Data:
    def __init__(self, ds: DS, hist: str, res: str = "1h", extra: str = ""):
        self.sph = 1 if res == "1h" else 12                 # steps per hour
        if res == "5m":
            if hist != "panel":
                raise SystemExit("--res 5m uses the panel only")
            self.pre, self.Y, self.A = 0, ds.V.copy(), ds.V.copy()
            self.F = causal_fill(self.Y, WEEK * 12, 12)
            self.N, self.L = self.Y.shape
            self.train_end = ds.split.train[1]
            self.val = ds.split.val
            self.test0 = ds.lo
            self.ds = ds
            return
        Y = ds.Hm.copy()                                    # hourly input means
        A = ds.Ha.copy()                                    # complete-hour actuals
        pre = 0
        if hist.startswith("ext"):
            if ds.name == "i2":
                P, pre = i2_ext_history(ds, since_2021=hist == "ext21")
            else:
                h = np.load(CACHE / "switch" / "history_hourly.npz", allow_pickle=True)
                assert list(h["links"]) == list(ds.links)
                assert int(h["hts"][-1]) + 3600 == int(ds.hts[0])
                P = h["values"].astype(np.float32)
                if hist == "ext2502":                       # from 2025-02 (pitfall 28)
                    P = P[h["hts"] >= pd.Timestamp("2025-02-01", tz="UTC").value // 10**9]
                pre = len(P)
            Y = np.vstack([P, Y])
            A = np.vstack([P, A])                           # history hours count as actuals
        self.pre = pre
        if extra == "i2doors":                              # training-only extra series
            t0 = pd.Timestamp(int(ds.hts[0]) - 3600 * pre, unit="s", tz="UTC")
            X = i2_door_series(ds, t0, len(Y), pre + ds.htrain[1])
            self.tcols = list(range(Y.shape[1]))            # scored columns
            print(f"extra series: {X.shape[1]} I2 door directions", flush=True)
            Y, A = np.hstack([Y, X]), np.hstack([A, X])
        self.Y = Y                                          # [N, L] raw (NaN)
        self.A = A
        self.F = causal_fill(Y)                             # inputs
        self.N, self.L = Y.shape
        self.train_end = pre + ds.htrain[1]
        self.val = (pre + ds.hval[0], pre + ds.hval[1])
        self.test0 = pre + ds.hlo
        self.ds = ds


class JointData:
    """Several networks on one absolute hourly grid (columns side by side). Training uses
    every network's training windows, but no target after the EVALUATED network's training
    end (so no network contributes data from the evaluated test period); validation and
    test are the evaluated network's only."""
    def __init__(self, target: str, names: list[str], hist: str):
        parts = {n: Data(DS(n), hist) for n in names}
        t0 = {n: int(p.ds.hts[0]) - 3600 * p.pre for n, p in parts.items()}
        T0 = min(t0.values())
        T1 = max(t0[n] + 3600 * p.N for n, p in parts.items())
        self.N = (T1 - T0) // 3600
        self.L = sum(p.L for p in parts.values())
        self.Y = np.full((self.N, self.L), np.nan, np.float32)
        self.F = self.Y.copy()
        self.A = self.Y.copy()
        self.off, self.col, c = {}, {}, 0
        for n, p in parts.items():
            o = (t0[n] - T0) // 3600
            self.Y[o:o + p.N, c:c + p.L] = p.Y
            self.F[o:o + p.N, c:c + p.L] = p.F
            self.A[o:o + p.N, c:c + p.L] = p.A
            self.off[n], self.col[n] = o, c
            c += p.L
        self.parts, self.target = parts, target
        pt, o, c = parts[target], self.off[target], self.col[target]
        self.tcols = list(range(c, c + pt.L))
        self.cutoff = o + pt.train_end                      # absolute row
        self.train_end = self.cutoff
        self.val = (o + pt.val[0], o + pt.val[1])
        self.test0 = o + pt.test0
        self.pre, self.sph, self.ds = 0, 1, pt.ds

    def origins(self, ctx, H, min_recent):
        S, J = [], []
        for n, p in self.parts.items():
            o, c = self.off[n], self.col[n]
            s, j = valid_origins(p, 0, min(p.train_end, self.cutoff - o), ctx, H, min_recent)
            S.append(s + o)
            J.append(j + c)
        vs, vj = valid_origins(self.parts[self.target], self.parts[self.target].val[0],
                               self.parts[self.target].val[1], ctx, H, min_recent, step=6)
        return (np.concatenate(S), np.concatenate(J), vs + self.off[self.target],
                vj + self.col[self.target])


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class Linear(nn.Module):
    def __init__(self, ctx, H, **_):
        super().__init__()
        self.lin = nn.Linear(ctx, H)

    def forward(self, x, m, lid=None):
        return self.lin(x)


class MLP(nn.Module):
    def __init__(self, ctx, H, hidden=1024, dropout=0.1, n_links=0, **_):
        super().__init__()
        self.emb = nn.Embedding(n_links, 16) if n_links else None
        e = 16 if n_links else 0
        self.net = nn.Sequential(nn.Linear(2 * ctx + e, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, H))
        self.skip = nn.Linear(ctx, H)

    def forward(self, x, m, lid=None):
        z = [x, m] if self.emb is None else [x, m, self.emb(lid)]
        return self.net(torch.cat(z, -1)) + self.skip(x)


class PatchTST(nn.Module):
    """Patch transformer on hourly input: daily patches (24 h, stride 24) of values and
    mask, d=128, 3 layers, low-rank flatten head."""
    def __init__(self, ctx, H, patch=24, d=128, heads=8, layers=3, d_ff=256, dropout=0.1,
                 bottleneck=512, n_links=0, **_):
        super().__init__()
        assert ctx % patch == 0
        self.emb = nn.Embedding(n_links, 16) if n_links else None
        self.patch, self.n = patch, ctx // patch
        self.embed = nn.Linear(2 * patch, d)
        self.pos = nn.Parameter(torch.zeros(1, self.n, d))
        enc = nn.TransformerEncoderLayer(d, heads, d_ff, dropout, batch_first=True,
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.neck = nn.Linear(self.n * d, bottleneck)
        self.head = nn.Linear(bottleneck + (16 if n_links else 0), H)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(ctx, H)

    def forward(self, x, m, lid=None):
        B = x.shape[0]
        tok = torch.cat([x.view(B, self.n, self.patch), m.view(B, self.n, self.patch)], -1)
        z = self.tr(self.embed(tok) + self.pos)
        z = torch.nn.functional.gelu(self.neck(self.drop(z.flatten(1))))
        if self.emb is not None:
            z = torch.cat([z, self.emb(lid)], -1)
        return self.head(self.drop(z)) + self.skip(x)


class WStack(nn.Module):
    """Week-stacked long-context model: ONE small MLP shared by every target hour k. Its
    input for hour k: the values at the same hour of the week in each of the K past weeks
    (+-nb hours, with masks), the last 24 hours, the K weekly means, the last 7 daily
    means, and an embedding of k. Parameter sharing across the 168 target positions and a
    weekly inductive bias make 24 weeks of context learnable from little data."""
    def __init__(self, ctx, H, nb=2, hidden=256, dropout=0.1, emb=16, **_):
        super().__init__()
        assert ctx % WEEK == 0
        K = ctx // WEEK
        self.K, self.ctx, self.H = K, ctx, H
        k = torch.arange(H)[:, None, None]
        i = torch.arange(1, K + 1)[None, :, None]
        d = torch.arange(-nb, nb + 1)[None, None, :]
        idx = ctx - WEEK * i + k + d                      # [H, K, 2nb+1]
        valid = (idx >= 0) & (idx < ctx)
        self.register_buffer("idx", idx.clamp(0, ctx - 1).reshape(-1))
        self.register_buffer("valid", valid.reshape(-1).float())
        self.nfeat = K * (2 * nb + 1)
        in_dim = 2 * self.nfeat + 48 + K + 7 + emb
        self.pos = nn.Embedding(H, emb)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x, m, lid=None):
        B, H, K = x.shape[0], self.H, self.K
        sv = x[:, self.idx] * self.valid                  # [B, H*K*(2nb+1)]
        sm = m[:, self.idx] * self.valid
        sv = sv.view(B, H, self.nfeat)
        sm = sm.view(B, H, self.nfeat)
        rec = torch.cat([x[:, -24:], m[:, -24:]], -1)[:, None, :].expand(B, H, 48)
        wk_m = m.view(B, K, WEEK)
        wk = (x.view(B, K, WEEK) * wk_m).sum(-1) / wk_m.sum(-1).clamp_min(1)
        dm = m[:, -WEEK:].view(B, 7, 24)
        dy = (x[:, -WEEK:].view(B, 7, 24) * dm).sum(-1) / dm.sum(-1).clamp_min(1)
        glob = torch.cat([wk, dy], -1)[:, None, :].expand(B, H, K + 7)
        pe = self.pos.weight[None].expand(B, H, -1)
        z = torch.cat([sv, sm, rec, glob, pe], -1)
        return self.mlp(z).squeeze(-1)


def _nanmedian_weeks(x, m, K):
    """[B, K*168] values + mask -> per-hour-of-week median over the K weeks [B, 168]
    (0 where no week is observed)."""
    B = x.shape[0]
    v = torch.where(m.view(B, K, WEEK) > 0, x.view(B, K, WEEK),
                    torch.full_like(x.view(B, K, WEEK), float("nan")))
    return torch.nan_to_num(v.nanmedian(dim=1).values)


class ProfMLP(nn.Module):
    """Long context through robust summaries: the last 2 weeks raw, the per-hour-of-week
    median over all K weeks and over the last 4, and the K weekly means. Targets k = 0..H-1
    line up with profile column k (the context ends at the origin)."""
    def __init__(self, ctx, H, hidden=1024, dropout=0.1, n_links=0, recent=336, **_):
        super().__init__()
        assert ctx % WEEK == 0 and H <= WEEK
        self.K, self.H, self.recent = ctx // WEEK, H, min(recent, ctx)
        in_dim = 2 * self.recent + 2 * WEEK + self.K
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, H))
        self.skip = nn.Linear(3 * WEEK, H)

    def forward(self, x, m, lid=None):
        B, K = x.shape[0], self.K
        pK = _nanmedian_weeks(x, m, K)
        k4 = min(4, K)
        p4 = _nanmedian_weeks(x[:, -k4 * WEEK:], m[:, -k4 * WEEK:], k4)
        wm = m.view(B, K, WEEK)
        wk = (x.view(B, K, WEEK) * wm).sum(-1) / wm.sum(-1).clamp_min(1)
        r = self.recent
        z = torch.cat([x[:, -r:], m[:, -r:], pK, p4, wk], -1)
        lastw = x[:, -WEEK:]
        return self.net(z) + self.skip(torch.cat([pK, p4, lastw], -1))[:, :self.H]


class DLinear(nn.Module):
    """The ladder's DLinear (Zeng et al. 2023): moving-average trend / remainder split, one
    linear map each (kernel 25 steps)."""
    def __init__(self, ctx, H, kernel=25, **_):
        super().__init__()
        self.k = kernel
        self.lin_t, self.lin_s = nn.Linear(ctx, H), nn.Linear(ctx, H)

    def forward(self, x, m, lid=None):
        pad = self.k // 2
        xp = torch.nn.functional.pad(x[:, None, :], (pad, pad), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(xp, self.k, 1)[:, 0, :x.shape[1]]
        return self.lin_t(trend) + self.lin_s(x - trend)


class GRUNet(nn.Module):
    """The ladder's GRU: 2 layers, 128 hidden, last state -> linear head."""
    def __init__(self, ctx, H, hidden=128, layers=2, dropout=0.1, **_):
        super().__init__()
        self.gru = nn.GRU(2, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, H)

    def forward(self, x, m, lid=None):
        out, _ = self.gru(torch.stack([x, m], -1))
        return self.head(out[:, -1])


class TCNNet(nn.Module):
    """The ladder's TCN (Bai et al. 2018): residual blocks of weight-norm dilated causal convs,
    64 channels, kernel 3, dropout 0.2, depth sized so the receptive field covers the input
    (no cap: 10 levels at 4,032 steps). Weight decay is kept off the weight-norm directions
    (pitfall 1) by run()."""
    def __init__(self, ctx, H, ch=64, k=3, dropout=0.2, **_):
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm
        levels = 1
        while 1 + 2 * (k - 1) * (2 ** levels - 1) < ctx:
            levels += 1
        self.blocks = nn.ModuleList()
        c_in = 2
        for i in range(levels):
            d, pad = 2 ** i, (k - 1) * 2 ** i
            self.blocks.append(nn.ModuleDict({
                "c1": weight_norm(nn.Conv1d(c_in, ch, k, padding=pad, dilation=d)),
                "c2": weight_norm(nn.Conv1d(ch, ch, k, padding=pad, dilation=d)),
                "down": nn.Conv1d(c_in, ch, 1) if c_in != ch else nn.Identity()}))
            c_in = ch
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(ch, H)
        self.levels = levels

    def forward(self, x, m, lid=None):
        z = torch.stack([x, m], 1)                       # [B, 2, L]
        L = z.shape[-1]
        for b in self.blocks:
            h = self.drop(torch.relu(b["c1"](z)[..., :L]))
            h = self.drop(torch.relu(b["c2"](h)[..., :L]))
            z = torch.relu(h + b["down"](z))
        return self.head(z[:, :, -1])


MODELS = {"linear": Linear, "mlp": MLP, "patchtst": PatchTST, "wstack": WStack,
          "profmlp": ProfMLP, "dlinear": DLinear, "gru": GRUNet, "tcn": TCNNet}


# --------------------------------------------------------------------------- #
# windows (on the GPU)
# --------------------------------------------------------------------------- #
class Windows:
    def __init__(self, data: Data, ctx: int, H: int, norm: str, dev, agg: int = 1):
        self.ctx, self.H, self.norm, self.dev, self.agg = ctx, H, norm, dev, agg
        pad = ctx                                           # left pad: partial contexts
        F = np.vstack([np.full((pad, data.L), np.nan, np.float32), data.F])
        Yt = np.vstack([np.full((pad, data.L), np.nan, np.float32), data.Y])
        self.pad = pad
        self.F = torch.tensor(F, device=dev)
        self.Yt = torch.tensor(Yt, device=dev)
        self.ar_c = torch.arange(ctx, device=dev)
        self.ar_h = torch.arange(H, device=dev)

    def get(self, s, j):
        """s: [B] absolute hour index of the origin (first target hour), j: [B] link.
        Returns normalised input, mask, normalised target, target mask, and the
        de-normalisation closure."""
        s = s + self.pad
        xi = (s[:, None] - self.ctx) + self.ar_c[None, :]
        x = self.F[xi, j[:, None]]                          # [B, ctx]
        ti = s[:, None] + self.ar_h[None, :]
        ti = ti.clamp(max=self.Yt.shape[0] - 1)
        y = self.Yt[ti, j[:, None]]                         # [B, H]
        if self.agg > 1:                                    # targets: means of agg steps
            yv = y.view(y.shape[0], -1, self.agg)
            c = torch.isfinite(yv).sum(-1)
            ym_ = torch.nan_to_num(yv).sum(-1) / c.clamp_min(1)
            y = torch.where(c >= self.agg // 2, ym_, torch.full_like(ym_, float("nan")))
        m = torch.isfinite(x)
        mf = m.float()
        cnt = mf.sum(1, keepdim=True).clamp_min(1)
        if self.norm == "log_revin":
            z = torch.log1p(torch.nan_to_num(x).clamp_min(0))
            mu = (z * mf).sum(1, keepdim=True) / cnt
            sd = (((z - mu) ** 2 * mf).sum(1, keepdim=True) / cnt).sqrt().clamp_min(0.05)
            xn = torch.where(m, (z - mu) / sd, torch.zeros_like(z))
            yn = (torch.log1p(torch.nan_to_num(y).clamp_min(0)) - mu) / sd
            yn = yn.clamp(-20, 20)
            inv = lambda p: torch.expm1(p * sd + mu).clamp_min(0)
        elif self.norm == "mean":
            xv = torch.nan_to_num(x).clamp_min(0)
            sc = ((xv * mf).sum(1, keepdim=True) / cnt).clamp_min(1e-3)
            xn = torch.where(m, xv / sc, torch.zeros_like(xv))
            yn = torch.nan_to_num(y).clamp_min(0) / sc
            yn = yn.clamp(max=50)
            inv = lambda p: (p * sc).clamp_min(0)
        elif self.norm == "revin_lin":
            xv = torch.nan_to_num(x).clamp_min(0)
            mu = (xv * mf).sum(1, keepdim=True) / cnt
            sd = (((xv - mu) ** 2 * mf).sum(1, keepdim=True) / cnt).sqrt()
            sd = torch.maximum(sd, 0.05 * mu).clamp_min(1e-3)
            xn = torch.where(m, (xv - mu) / sd, torch.zeros_like(xv))
            yn = ((torch.nan_to_num(y).clamp_min(0) - mu) / sd).clamp(-50, 50)
            inv = lambda p: (p * sd + mu).clamp_min(0)
        else:
            raise ValueError(self.norm)
        ym = torch.isfinite(y)
        return xn, mf, yn, ym, inv, y


def valid_origins(data: Data, lo: int, hi: int, ctx: int, H: int, min_recent: int = 168,
                  step: int = 1):
    """(s, j) pairs: origin s in [lo, hi - H] (target inside), >= half of the last
    min(ctx, min_recent) hours finite, >= 90 % of the target finite."""
    fin = np.isfinite(data.Y)
    cs = np.vstack([np.zeros((1, data.L)), np.cumsum(fin, 0)])
    S = np.arange(max(lo, 1), hi - H + 1, step)
    r = min(ctx, min_recent)
    a = np.clip(S - r, 0, None)
    recent = (cs[S] - cs[a]) / r
    tgt = (cs[np.minimum(S + H, data.N)] - cs[S]) / H
    ok = (recent >= 0.5) & (tgt >= 0.9) & (S[:, None] - r >= 0)
    ss, jj = np.nonzero(ok)
    return S[ss], jj


# --------------------------------------------------------------------------- #
# train / predict
# --------------------------------------------------------------------------- #
def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    ds = DS(args.ds)
    data = (JointData(args.ds, args.joint.split(","), args.hist) if args.joint
            else Data(ds, args.hist, args.res, args.extra))
    sph, agg = data.sph, args.agg
    out_sph = sph // agg                                    # output steps per hour
    Hout = args.H // agg
    W = Windows(data, args.ctx, args.H, args.norm, dev, agg)
    if args.joint:
        tr_s, tr_j, va_s, va_j = data.origins(args.ctx, args.H, WEEK)
    else:
        lo = 0 if not args.train_months else max(0, data.train_end - int(args.train_months * 730.5 * sph))
        if args.train_from:                 # first training origin (context may reach further back)
            t = pd.Timestamp(args.train_from, tz="UTC").value // 10**9
            lo = max(lo, int((t - (int(data.ds.hts[0]) - 3600 * data.pre)) // (3600 // sph)))
        tr_s, tr_j = valid_origins(data, lo, data.train_end, args.ctx, args.H, WEEK * sph)
        va_s, va_j = valid_origins(data, data.val[0], data.val[1], args.ctx, args.H,
                                   WEEK * sph, step=6 * sph)
        if hasattr(data, "tcols"):                          # validate on the scored links only
            keep = va_j < len(data.tcols)
            va_s, va_j = va_s[keep], va_j[keep]
    print(f"[{args.tag}] train windows {len(tr_s):,}  val windows {len(va_s):,}  "
          f"first train origin hour {tr_s.min()} (pre={data.pre})", flush=True)
    net = MODELS[args.model](args.ctx, Hout, dropout=args.dropout, patch=args.patch,
                             n_links=data.L if args.lemb else 0).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    nowd = [p for n, p in net.named_parameters() if "original1" in n]   # weight-norm directions
    rest = [p for n, p in net.named_parameters() if "original1" not in n]
    opt = torch.optim.AdamW([{"params": rest, "weight_decay": args.wd},
                             {"params": nowd, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = args.samples // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    lossf = {"huber": nn.SmoothL1Loss(reduction="none"), "l1": nn.L1Loss(reduction="none"),
             "mse": nn.MSELoss(reduction="none")}[args.loss]
    hw = torch.ones(Hout, device=dev)
    if args.hweight == "sqrt":
        hw = 1 / torch.sqrt(torch.arange(1, Hout + 1, device=dev, dtype=torch.float32))
        hw = hw * Hout / hw.sum()

    def hourly(p, y, ym):
        """5-min outputs -> hourly means (an hour counts if all its steps are finite)."""
        if out_sph == 1:
            return p, y, ym
        B = p.shape[0]
        ok = ym.view(B, -1, out_sph).all(-1)
        return (p.view(B, -1, out_sph).mean(-1), torch.nan_to_num(y).view(B, -1, out_sph).mean(-1),
                ok)
    tr_s_t = torch.tensor(tr_s, device=dev)
    tr_j_t = torch.tensor(tr_j, device=dev)
    va_s_t = torch.tensor(va_s, device=dev)
    va_j_t = torch.tensor(va_j, device=dev)

    def evaluate(s_t, j_t, Hn=None):
        """Median over links of per-link WAPE (the test metric's definition)."""
        net.eval()
        num = torch.zeros(data.L, device=dev, dtype=torch.float64)
        den = torch.zeros(data.L, device=dev, dtype=torch.float64)
        with torch.no_grad():
            eb = args.eval_bs
            for b in range(0, s_t.numel(), eb):
                jb = j_t[b:b + eb]
                xn, mf, yn, ym, inv, y = W.get(s_t[b:b + eb], jb)
                p, y, ym = hourly(inv(net(xn, mf, jb)), y, ym)
                e = torch.where(ym, (p - torch.nan_to_num(y)).abs(), torch.zeros_like(p))
                a = torch.where(ym, torch.nan_to_num(y), torch.zeros_like(p))
                if Hn:
                    e, a = e[:, :Hn], a[:, :Hn]
                num.index_add_(0, jb, e.sum(1).double())
                den.index_add_(0, jb, a.sum(1).double())
        net.train()
        w = (num / den)[den > 1e-6]
        return 100 * float(w.median())

    hist, best, best_state, bad = [], float("inf"), None, 0
    t0 = time.time()
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randint(0, tr_s_t.numel(), (steps_per_epoch * args.bs,), device=dev,
                             generator=g)
        tl = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.bs:(i + 1) * args.bs]
            xn, mf, yn, ym, inv, y = W.get(tr_s_t[idx], tr_j_t[idx])
            if args.drop_ctx > 0:
                # random context truncation: teaches the model to work with partial history
                keep = torch.rand(xn.shape[0], 1, device=dev) > args.drop_ctx
                cut = torch.randint(WEEK * sph, args.ctx + 1, (xn.shape[0], 1), device=dev)
                pos = torch.arange(args.ctx, device=dev)[None, :]
                vis = keep | (pos >= args.ctx - cut)
                xn, mf = xn * vis, mf * vis
            p = net(xn, mf, tr_j_t[idx])
            per = lossf(p, torch.nan_to_num(yn)) * ym * hw
            loss = per.sum() / ym.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss)
        v_day = evaluate(va_s_t, va_j_t, 24)
        v_wk = evaluate(va_s_t, va_j_t)
        crit = v_day if args.select == "day" else v_wk
        hist.append({"epoch": ep + 1, "train_loss": tl / steps_per_epoch, "val_wape_day": v_day,
                     "val_wape_week": v_wk, "sec": time.time() - t0})
        print(f"[{args.tag}] ep {ep + 1:2d} train {tl / steps_per_epoch:.4f}  val day {v_day:.2f} "
              f"week {v_wk:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        if crit < best - 1e-4:
            best, bad = crit, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    # ---- test: block layout, day (first 24 h of the path) and week --------- #
    n = ds.n_test_hours()
    out = {}
    for hz, Hb in (("day", 24), ("week", 168)):
        starts = np.arange(data.test0, data.N, Hb * sph)
        cols = getattr(data, "tcols", list(range(data.L)))
        P = np.full((len(cols), len(starts) * Hb), np.nan, np.float32)
        with torch.no_grad():
            for jj, j in enumerate(cols):
                s_t = torch.tensor(starts, device=dev)
                j_t = torch.full_like(s_t, j)
                xn, mf, yn, ym, inv, y = W.get(s_t, j_t)
                p = inv(net(xn, mf, j_t))[:, :Hb * out_sph]
                p = p.view(p.shape[0], Hb, out_sph).mean(-1)
                # same rule as the ladder: no forecast without the last day of input
                recent = mf[:, -24 * sph:].sum(1) > 0
                p[~recent] = float("nan")
                P[jj] = p.cpu().numpy().reshape(-1)
        out[hz] = P[:, :n]
    od = CACHE / args.ds / "runs" / args.tag
    od.mkdir(parents=True, exist_ok=True)
    np.save(od / "pred_day.npy", out["day"])
    np.save(od / "pred_week.npy", out["week"])
    pd.DataFrame(hist).to_csv(od / "history.csv", index=False)
    A = ds.actual_test()
    res = {}
    for hz in ("day", "week"):
        refs = {m: ds.load_dense_block(m, hz) for m in ("patchtst", "timestral@1h_24w")}
        df, _ = score({**refs, args.tag: out[hz]}, A, ref="patchtst")
        print(df.round(3).to_string(index=False))
        r = df.set_index("model").loc[args.tag]
        res[hz] = {k: float(v) for k, v in r.items()}
        res[hz]["timestral_median"] = float(df.set_index("model").loc["timestral@1h_24w",
                                                                      "median_wape"])
        res[hz]["patchtst_median"] = float(df.set_index("model").loc["patchtst", "median_wape"])
    summ = {"args": vars(args), "params": nparam, "train_windows": int(len(tr_s)),
            "epochs_run": len(hist), "best_val": best, "test": res,
            "seconds": time.time() - t0}
    (od / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[{args.tag}] DONE day {res['day']['median_wape']:.2f} week "
          f"{res['week']['median_wape']:.2f}  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--ctx", type=int, default=168)
    ap.add_argument("--H", type=int, default=168)
    ap.add_argument("--norm", default="log_revin")
    ap.add_argument("--loss", default="huber")
    ap.add_argument("--model", default="linear")
    ap.add_argument("--hist", default="panel")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--hweight", default="flat")
    ap.add_argument("--drop-ctx", type=float, default=0.0)
    ap.add_argument("--select", default="day")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--res", default="1h", help="1h (hourly means) | 5m (raw 5-min panel)")
    ap.add_argument("--agg", type=int, default=1, help="targets: means of this many steps")
    ap.add_argument("--lemb", type=int, default=0, help="1: learned per-link embedding")
    ap.add_argument("--patch", type=int, default=24)
    ap.add_argument("--joint", default="", help="train on several networks, e.g. i2,switch")
    ap.add_argument("--eval-bs", type=int, default=4096, help="validation batch (GRU/TCN at long context: 256)")
    ap.add_argument("--extra", default="", help="i2doors: train also on every I2 door series")
    ap.add_argument("--train-from", default="", help="first training origin (date)")
    ap.add_argument("--train-months", type=float, default=0,
                    help="training targets only from the last N months before the cutoff")
    args = ap.parse_args()
    if args.tag is None:
        args.tag = (f"{args.model}_c{args.ctx}_{args.norm}_{args.loss}_{args.hist}"
                    f"_s{args.seed}")
    run(args)


if __name__ == "__main__":
    main()
