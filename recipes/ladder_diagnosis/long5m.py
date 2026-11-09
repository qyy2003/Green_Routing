"""Long context at 5-minute resolution: one global forecaster on raw 5-minute input (up to
24 weeks = 48,384 steps) with a 5-minute output path of one week (2,016 steps), scored in the
block layout on hourly WAPE (as everything in this recipe) and on 5-minute WAPE.

Why the ladder stops at one week: its heads are dense in the input length (DLinear 2016 x
2016 weights; 24 weeks would be 97M per horizon), the GRU steps through every input value,
the TCN needs one dilation level per doubling, and PatchTST flattens every token into its
head. These methods keep the input at 5 minutes and grow the context without that cost:

  --model seg       patch transformer over a set of segments of the context, then 168 query
                    tokens (one per target hour) read by a per-token head (12 five-minute
                    values each). Tokens carry an hour-of-week, an age (weeks back) and a
                    resolution embedding, so the same hour of earlier weeks lines up.
       --seg full      1-hour patches over the whole context (C/12 tokens: 4,032 at 24 weeks;
                       flash attention makes that cheap)
       --seg coarse    168 tokens whatever the context (patch C/168: 1 day at 24 weeks)
       --seg multires  last week in 1-hour patches, older weeks in 1-day patches
       --seg mrh       as multires, but an older day enters as its 24 hourly means (the
                       old weeks need no 5-minute detail, and hourly-filled history - I2
                       before the panel - then looks exactly like real 5-minute data)
       --seg recent    last week only (long context enters only through --qprof)
       --qprof 1       query tokens carry the per-5-minute-of-week median over all K weeks and
                       over the last 4 (robust long-context summaries, aligned with the target)
  --model fold      period folding (SparseTSF / WStack style): ONE MLP shared by all 2,016
                    target steps; its input for step k is the hour-smoothed value at the same
                    5-minute-of-week in each of the K past weeks (and +-1 h), the last day's
                    hourly means, weekly and daily means, and an embedding of k.
  --model profmlp   MLP on the last week (5-min), the per-5-minute-of-week medians (K weeks,
                    last 4) and the weekly means; linear skip from the profiles and last week.

Inputs: --hist ext adds history before the panel (Switch: 5-minute from 2024-01, I2: only the
hourly file exists before the panel, repeated over its 12 steps and flagged to the model with
mask 0.5). Scaling: x / mean of the last week of input (linear space); --squash T bounds the
scaled input (identity up to T, T + log1p(x - T) above; old weeks of a link that is nearly idle
now otherwise reach 1e5-1e7 at 24 weeks and destabilise training); loss L1 on 5-minute
targets (+ --aux1h w x L1 on hourly means). Selection: validation median link hourly WAPE at a
day, origins at 00:00 UTC. Outputs <CACHE>/<ds>/runs/<tag>/{pred_day.npy, pred_week.npy}
(hourly, for compare.py), {pred5_day.npy, pred5_week.npy} (5-minute), summary.json, history.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from common import DS, CACHE, score
from hourly_train import causal_fill, i2_ext_history, valid_origins
from timestral.scaler import apply_affine_asinh_masked, causal_patch_loc_scale

W = 2016          # 5-minute steps per week
D = 288           # per day
HW = 168          # hours per week


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class Data5:
    def __init__(self, ds: DS, hist: str):
        V = ds.V
        pre, coarse = 0, np.zeros(0, bool)
        if hist == "ext":
            if ds.name == "switch":
                m = np.load(CACHE / "switch" / "history_5m.npz", allow_pickle=True)
                assert list(m["links"]) == list(ds.links)
                assert int(m["ts"][-1]) + 300 == int(ds.ts[0])
                P = np.load(CACHE / "switch" / "history_5m_values.npy")
                coarse = np.zeros(len(P), bool)
            else:
                Ph, _ = i2_ext_history(ds)                    # hours before ds.hts[0]
                P = np.repeat(Ph, 12, axis=0)
                if ds.off:                                    # panel starts off steps before hts[0]
                    P = P[:-ds.off]
                coarse = np.ones(len(P), bool)
            pre = len(P)
            V = np.vstack([P, V])
        self.pre, self.Y, self.ds = pre, V, ds
        self.coarse = np.concatenate([coarse, np.zeros(len(ds.V), bool)])
        self.F = causal_fill(V, W, 12)
        self.N, self.L = V.shape
        self.train_end = pre + ds.split.train[1]
        self.val = (pre + ds.split.val[0], pre + ds.split.val[1])
        self.test0 = pre + ds.lo
        self.h0 = (pre + ds.off) % 12                          # first index on the hour


class Windows5:
    """Window gather on the GPU. --norm lastmean: x / mean of the last week of input (with
    --squash); log_revin: the ladder's log1p + per-window mean/std (std floor 0.05); ts:
    Timestral's scaler (causal per-patch mean/std over 1-hour patches, asinh; the target in the
    last patch's frame)."""
    def __init__(self, data: Data5, ctx: int, H: int, dev, squash: float = 0.0,
                 norm: str = "lastmean"):
        self.ctx, self.H, self.squash, self.norm = ctx, H, squash, norm
        pad = ctx
        nanpad = np.full((pad, data.L), np.nan, np.float32)
        self.F = torch.tensor(np.vstack([nanpad, data.F]), device=dev)
        self.Yt = torch.tensor(np.vstack([nanpad, data.Y]), device=dev)
        self.C = torch.tensor(np.concatenate([np.zeros(pad, bool), data.coarse]), device=dev)
        self.pad = pad
        self.ar_c = torch.arange(ctx, device=dev)
        self.ar_h = torch.arange(H, device=dev)

    def get(self, s, j, ablate: str | None = None, keep: int = 4 * W):
        """Returns normalised input, model mask, normalised target, target mask, the inverse
        transform of a normalised forecast, and the raw target. ``ablate`` (analysis only):
        "tile" replaces everything older than the last ``keep`` steps by copies of them,
        "blank" marks it missing."""
        s = s + self.pad
        xi = (s[:, None] - self.ctx) + self.ar_c[None, :]
        x = self.F[xi, j[:, None]]
        if ablate and self.ctx > keep:
            old = self.ctx - keep
            if ablate == "tile":
                src = (torch.arange(self.ctx, device=x.device) - old) % keep + old
                x = x[:, src]
            elif ablate == "blank":
                x = x.clone()
                x[:, :old] = float("nan")
        ti = (s[:, None] + self.ar_h[None, :]).clamp(max=self.Yt.shape[0] - 1)
        y = self.Yt[ti, j[:, None]]
        m = torch.isfinite(x)
        mf = m.float()
        xv = torch.nan_to_num(x).clamp_min(0)
        yv = torch.nan_to_num(y).clamp_min(0)
        mk = mf * torch.where(self.C[xi], 0.5, 1.0)            # 0.5: hourly-filled input
        if self.norm == "log_revin":                           # the ladder: log1p + RevIN
            z = torch.log1p(xv)
            cnt = mf.sum(1, keepdim=True).clamp_min(1)
            mu = (z * mf).sum(1, keepdim=True) / cnt
            sd = (((z - mu) ** 2 * mf).sum(1, keepdim=True) / cnt).sqrt().clamp_min(0.05)
            xn = torch.where(m, (z - mu) / sd, torch.zeros_like(z))
            yn = ((torch.log1p(yv) - mu) / sd).clamp(-20, 20)
            inv = lambda p: torch.expm1(p * sd + mu).clamp_min(0)
            return xn, mk, yn, torch.isfinite(y), inv, y
        if self.norm == "ts":                                  # Timestral's own scaler
            loc, scale = causal_patch_loc_scale(xv, m, 12, 1e-6, False)
            xn = apply_affine_asinh_masked(xv, loc, scale, m, use_asinh=True)
            l1, s1 = loc[:, -1:], scale[:, -1:]                # the last patch's frame
            yn = torch.asinh((yv - l1) / s1.clamp_min(1e-6)).clamp(-30, 30)
            inv = lambda p: (torch.sinh(p.clamp(-30, 30)) * s1 + l1).clamp_min(0)
            return xn, mk, yn, torch.isfinite(y), inv, y
        r = min(W, self.ctx)
        cr = mf[:, -r:].sum(1, keepdim=True)
        sc_r = (xv[:, -r:] * mf[:, -r:]).sum(1, keepdim=True) / cr.clamp_min(1)
        sc_a = (xv * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp_min(1)
        sc = torch.where(cr > 0, sc_r, sc_a).clamp_min(1e-3)
        xn = torch.where(m, xv / sc, torch.zeros_like(xv))
        if self.squash > 0:
            t = self.squash
            xn = torch.where(xn > t, t + torch.log1p((xn - t).clamp_min(0)), xn)
        yn = (yv / sc).clamp(max=50)
        inv = lambda p: (p * sc).clamp_min(0)
        return xn, mk, yn, torch.isfinite(y), inv, y


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def week_median(x, m, K):
    """[B, K*W] values + mask -> per-5-minute-of-week median over K weeks [B, W], and the
    share of weeks observed [B, W]."""
    B = x.shape[0]
    xv, mv = x[:, -K * W:].reshape(B, K, W), m[:, -K * W:].reshape(B, K, W) > 0
    v = torch.where(mv, xv, torch.full_like(xv, float("nan")))
    return torch.nan_to_num(v.nanmedian(dim=1).values), mv.float().mean(1)


class SegTST(nn.Module):
    def __init__(self, C, Hout=W, seg="full", qprof=0, d=128, heads=8, layers=3, d_ff=256,
                 dropout=0.1, **_):
        super().__init__()
        assert C % W == 0 and Hout == W
        self.C, self.K, self.qprof, self.seg = C, C // W, qprof, seg
        if seg == "full":
            groups = [(12, range(0, C, 12))]
        elif seg == "coarse":
            groups = [(C // HW, range(0, C, C // HW))]
        elif seg in ("multires", "mrh"):
            groups = ([(D, range(0, C - W, D))] if C > W else []) + [(12, range(C - W, C, 12))]
        elif seg == "recent":
            groups = [(12, range(C - W, C, 12))]
        else:
            raise ValueError(seg)
        self.embeds = nn.ModuleList()
        hows, ages, ress = [], [], []
        for g, (P, starts) in enumerate(groups):
            st = torch.tensor(list(starts))
            self.register_buffer(f"idx{g}", (st[:, None] + torch.arange(P)[None, :]).reshape(-1))
            hourly = seg == "mrh" and P == D
            self.embeds.append(nn.Linear(2 * (24 if hourly else P), d))
            hows.append(((st - C) // 12) % HW)
            ages.append((C - 1 - st) // W)
            ress.append(torch.full_like(st, g))
        self.P = [P for P, _ in groups]
        self.ntok = [len(s) for _, s in groups]
        self.register_buffer("how", torch.cat(hows + [torch.arange(HW)]))
        self.register_buffer("age", torch.cat(ages + [torch.full((HW,), self.K)]))
        self.register_buffer("res", torch.cat(ress + [torch.full((HW,), len(groups))]))
        self.e_how = nn.Embedding(HW, d)
        self.e_age = nn.Embedding(self.K + 1, d)
        self.e_res = nn.Embedding(len(groups) + 1, d)
        self.q0 = nn.Parameter(torch.zeros(1, 1, d))
        if qprof:
            self.qemb = nn.Linear(4 * 12, d)
        enc = nn.TransformerEncoderLayer(d, heads, d_ff, dropout, batch_first=True,
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 12)
        self.skip = nn.Linear(W, W)

    def forward(self, x, m, lid=None):
        B = x.shape[0]
        toks = []
        for g, (P, n) in enumerate(zip(self.P, self.ntok)):
            ix = getattr(self, f"idx{g}")
            xv, mv = x[:, ix].view(B, n, P), m[:, ix].view(B, n, P)
            if self.seg == "mrh" and P == D:                   # day -> 24 hourly means
                mb = (mv > 0).float().view(B, n, 24, 12)
                c = mb.sum(-1)
                xv = (xv.view(B, n, 24, 12) * mb).sum(-1) / c.clamp_min(1)
                mv = (c > 0).float()
            toks.append(self.embeds[g](torch.cat([xv, mv], -1)))
        q = self.q0.expand(B, HW, -1)
        if self.qprof:
            pK, oK = week_median(x, m, self.K)
            k4 = min(4, self.K)
            p4, o4 = week_median(x, m, k4)
            f = torch.stack([pK, oK, p4, o4], -1).view(B, HW, 12 * 4)
            q = q + self.qemb(f)
        z = torch.cat(toks + [q], 1)
        z = z + self.e_how(self.how) + self.e_age(self.age) + self.e_res(self.res)
        z = self.tr(z)
        out = self.head(self.norm(z[:, -HW:])).reshape(B, W)
        return out + self.skip(x[:, -W:])


class Fold(nn.Module):
    def __init__(self, C, Hout=W, hidden=256, dropout=0.1, emb=16, nb=(-12, 0, 12), sm=13,
                 **_):
        super().__init__()
        assert C % W == 0
        K = C // W
        self.K, self.C, self.H, self.sm = K, C, Hout, sm
        k = torch.arange(Hout)[:, None, None]
        i = torch.arange(1, K + 1)[None, :, None]
        dl = torch.tensor(nb)[None, None, :]
        idx = C - W * i + k + dl                                # [H, K, nb]
        valid = (idx >= 0) & (idx < C)
        self.register_buffer("idx", idx.clamp(0, C - 1).reshape(-1))
        self.register_buffer("valid", valid.reshape(-1).float())
        self.nf = K * len(nb)
        in_dim = 2 * self.nf + 24 + 12 + K + 7 + emb
        self.pos = nn.Embedding(Hout, emb)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x, m, lid=None):
        B, H, K = x.shape[0], self.H, self.K
        ker = torch.ones(1, 1, self.sm, device=x.device)
        mm = (m > 0).float()
        num = Fn.conv1d((x * mm)[:, None], ker, padding=self.sm // 2)[:, 0]
        den = Fn.conv1d(mm[:, None], ker, padding=self.sm // 2)[:, 0]
        xs = num / den.clamp_min(1)
        ms = (den > 0).float()
        sv = (xs[:, self.idx] * self.valid).view(B, H, self.nf)
        sm_ = (ms[:, self.idx] * self.valid).view(B, H, self.nf)
        hm = (x[:, -D:] * mm[:, -D:]).view(B, 24, 12).sum(-1) / mm[:, -D:].view(B, 24, 12).sum(-1).clamp_min(1)
        wk = (x.view(B, K, W) * mm.view(B, K, W)).sum(-1) / mm.view(B, K, W).sum(-1).clamp_min(1)
        dy = (x[:, -W:].view(B, 7, D) * mm[:, -W:].view(B, 7, D)).sum(-1) / mm[:, -W:].view(B, 7, D).sum(-1).clamp_min(1)
        glob = torch.cat([hm, x[:, -12:], wk, dy], -1)[:, None, :].expand(B, H, -1)
        pe = self.pos.weight[None].expand(B, H, -1)
        return self.mlp(torch.cat([sv, sm_, glob, pe], -1)).squeeze(-1)


class ProfMLP5(nn.Module):
    def __init__(self, C, Hout=W, hidden=1024, dropout=0.1, **_):
        super().__init__()
        assert C % W == 0
        self.K = C // W
        in_dim = 4 * W + self.K
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, Hout))
        self.skip = nn.Linear(3 * W, Hout)

    def forward(self, x, m, lid=None):
        B, K = x.shape[0], self.K
        pK, _ = week_median(x, m, K)
        p4, _ = week_median(x, m, min(4, K))
        mm = (m > 0).float()
        wk = (x.view(B, K, W) * mm.view(B, K, W)).sum(-1) / mm.view(B, K, W).sum(-1).clamp_min(1)
        z = torch.cat([x[:, -W:], m[:, -W:], pK, p4, wk], -1)
        return self.net(z) + self.skip(torch.cat([pK, p4, x[:, -W:]], -1))


MODELS = {"seg": SegTST, "fold": Fold, "profmlp": ProfMLP5}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def wape5(preds: dict[str, np.ndarray], A: np.ndarray) -> dict[str, float]:
    """Median link 5-minute WAPE on the 5-minute targets every forecast covers."""
    ok = np.isfinite(A)
    for p in preds.values():
        ok &= np.isfinite(p)
    out = {}
    for k, p in preds.items():
        e = np.where(ok, np.abs(p - A), 0).sum(1)
        a = np.where(ok, A, 0).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(a > 1e-8, 100 * e / a, np.nan)
        out[k] = float(np.nanmedian(w))
    return out


def to_hourly(P5: np.ndarray, n_hours: int) -> np.ndarray:
    L = P5.shape[0]
    p = P5[:, :n_hours * 12].reshape(L, n_hours, 12)
    return np.where(np.isfinite(p).all(2), p.mean(2), np.nan)


# --------------------------------------------------------------------------- #
# train / predict
# --------------------------------------------------------------------------- #
def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    ds = DS(args.ds)
    data = Data5(ds, args.hist)
    C, H = args.ctx, W
    Wn = Windows5(data, C, H, dev, args.squash, args.norm)
    lo = 12 + data.h0
    if args.train_from:                     # first training origin (context may reach further back)
        t = pd.Timestamp(args.train_from, tz="UTC").value // 10**9
        lo = max(lo, int((t - (int(ds.ts[0]) - 300 * data.pre)) // 300))
        lo += (data.h0 - lo) % 12
    tr_s, tr_j = valid_origins(data, lo, data.train_end, C, H, W, step=12)
    va_s, va_j = valid_origins(data, data.val[0], data.val[1], C, H, W, step=D)
    print(f"[{args.tag}] train windows {len(tr_s):,}  val windows {len(va_s):,}  "
          f"pre={data.pre}  coarse share of train contexts ~"
          f"{data.coarse[:data.train_end].mean():.2f}", flush=True)
    net = MODELS[args.model](C, H, seg=args.seg, qprof=args.qprof, dropout=args.dropout,
                             d=args.d, layers=args.layers).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    spe = args.samples // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * spe,
                                                pct_start=max(0.05, 3 / (args.epochs * spe)))
    tr_s_t, tr_j_t = torch.tensor(tr_s, device=dev), torch.tensor(tr_j, device=dev)
    va_s_t, va_j_t = torch.tensor(va_s, device=dev), torch.tensor(va_j, device=dev)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.bf16))

    def predict(s_t, j_t, bs):
        outs = []
        for b in range(0, s_t.numel(), bs):
            xn, mk, yn, ym, inv, y = Wn.get(s_t[b:b + bs], j_t[b:b + bs])
            with torch.autocast(**amp):
                p = net(xn, mk, j_t[b:b + bs])
            p = inv(p.float())
            p[(mk[:, -D:] > 0).sum(1) == 0] = float("nan")     # the ladder's rule
            outs.append((p, y, mk))
        return outs

    def evaluate(s_t, j_t):
        """Validation median link hourly WAPE at a day and a week, and 5-min WAPE at a day."""
        net.eval()
        acc = {k: [torch.zeros(data.L, device=dev, dtype=torch.float64) for _ in range(2)]
               for k in ("d", "w", "d5")}
        with torch.no_grad():
            for b in range(0, s_t.numel(), args.eval_bs):
                jb = j_t[b:b + args.eval_bs]
                ((p, y, _),) = predict(s_t[b:b + args.eval_bs], jb, args.eval_bs)
                B = p.shape[0]
                ok5 = torch.isfinite(y) & torch.isfinite(p)
                okh = ok5.view(B, HW, 12).all(-1)
                ph = torch.nan_to_num(p).view(B, HW, 12).mean(-1)
                yh = torch.nan_to_num(y).view(B, HW, 12).mean(-1)
                for k, (e, a) in {"w": ((ph - yh).abs() * okh, yh * okh),
                                  "d": (((ph - yh).abs() * okh)[:, :24], (yh * okh)[:, :24]),
                                  "d5": (((torch.nan_to_num(p) - torch.nan_to_num(y)).abs() * ok5)[:, :D],
                                         (torch.nan_to_num(y) * ok5)[:, :D])}.items():
                    acc[k][0].index_add_(0, jb, e.sum(1).double())
                    acc[k][1].index_add_(0, jb, a.sum(1).double())
        net.train()
        out = {}
        for k, (n, d) in acc.items():
            out[k] = 100 * float((n / d)[d > 1e-6].median())
        return out

    hist, best, best_state, bad = [], float("inf"), None, 0
    t0 = time.time()
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randint(0, tr_s_t.numel(), (spe * args.bs,), device=dev, generator=g)
        tl = 0.0
        for i in range(spe):
            idx = perm[i * args.bs:(i + 1) * args.bs]
            xn, mk, yn, ym, inv, y = Wn.get(tr_s_t[idx], tr_j_t[idx])
            if args.drop_ctx > 0 and C > W:
                # random truncation to 1 week .. C: the model must not rely on old weeks
                Bb = xn.shape[0]
                cut = torch.randint(W, C + 1, (Bb, 1), device=dev)
                vis = (torch.rand(Bb, 1, device=dev) > args.drop_ctx) | \
                      (torch.arange(C, device=dev)[None, :] >= C - cut)
                xn, mk = xn * vis, mk * vis
            with torch.autocast(**amp):
                p = net(xn, mk, tr_j_t[idx])
            p = p.float()
            ymf = ym.float()
            err = (p - yn).abs() if args.loss == "l1" else Fn.smooth_l1_loss(p, yn, reduction="none")
            loss = (err * ymf).sum() / ymf.sum().clamp_min(1)
            if args.aux1h > 0:
                B = p.shape[0]
                okh = ym.view(B, HW, 12).all(-1).float()
                ph, yh = p.view(B, HW, 12).mean(-1), yn.view(B, HW, 12).mean(-1)
                loss = loss + args.aux1h * ((ph - yh).abs() * okh).sum() / okh.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss)
        v = evaluate(va_s_t, va_j_t)
        hist.append({"epoch": ep + 1, "train_loss": tl / spe, "val_day": v["d"],
                     "val_week": v["w"], "val_day5": v["d5"], "sec": time.time() - t0})
        print(f"[{args.tag}] ep {ep + 1:2d} train {tl / spe:.4f}  val day {v['d']:.2f} "
              f"week {v['w']:.2f} day5 {v['d5']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        if v["d"] < best - 1e-4:
            best, bad = v["d"], 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    od = CACHE / args.ds / "runs" / args.tag
    od.mkdir(parents=True, exist_ok=True)
    if args.save:
        torch.save(best_state, od / "model.pt")

    # ---- test: block layout, day (first 288 steps of the path) and week ----- #
    n5, nh = ds.n_test_steps(), ds.n_test_hours()
    out5 = {}
    with torch.no_grad():
        for hz, Hb in (("day", D), ("week", W)):
            starts = np.arange(data.test0, data.N, Hb)
            P = np.full((data.L, len(starts) * Hb), np.nan, np.float32)
            s_t = torch.tensor(starts, device=dev)
            for j in range(data.L):
                j_t = torch.full_like(s_t, j)
                p = torch.cat([o[0] for o in predict(s_t, j_t, args.eval_bs)])[:, :Hb]
                P[j] = p.cpu().numpy().reshape(-1)
            out5[hz] = P[:, :n5]
    A5, A = ds.actual5_test(), ds.actual_test()
    res = {}
    for hz in ("day", "week"):
        ph = to_hourly(out5[hz], nh)
        np.save(od / f"pred_{hz}.npy", ph.astype(np.float32))
        np.save(od / f"pred5_{hz}.npy", out5[hz])
        refs = {m: ds.load_dense_block(m, hz) for m in ("patchtst", "timestral@1h_24w")}
        df, _ = score({**refs, args.tag: ph}, A, ref="patchtst")
        print(df.round(3).to_string(index=False))
        r = df.set_index("model").loc[args.tag]
        res[hz] = {k: float(v) for k, v in r.items()}
        w5 = wape5({"patchtst": ds.load_dense_block5("patchtst", hz), args.tag: out5[hz]}, A5)
        res[hz]["wape5"] = w5[args.tag]
        res[hz]["patchtst_wape5"] = w5["patchtst"]
        print(f"  5-min WAPE {hz}: {args.tag} {w5[args.tag]:.2f}  ladder patchtst "
              f"{w5['patchtst']:.2f}")
    pd.DataFrame(hist).to_csv(od / "history.csv", index=False)
    summ = {"args": vars(args), "params": nparam, "train_windows": int(len(tr_s)),
            "epochs_run": len(hist), "best_val": best, "test": res,
            "seconds": time.time() - t0}
    (od / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[{args.tag}] DONE day {res['day']['median_wape']:.2f} (5-min {res['day']['wape5']:.2f}) "
          f"week {res['week']['median_wape']:.2f} (5-min {res['week']['wape5']:.2f})  "
          f"({time.time() - t0:.0f}s, {nparam / 1e6:.1f}M params)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--ctx", type=int, default=W, help="input length in 5-minute steps")
    ap.add_argument("--model", default="seg")
    ap.add_argument("--seg", default="full")
    ap.add_argument("--qprof", type=int, default=0)
    ap.add_argument("--hist", default="ext")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--eval-bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--aux1h", type=float, default=0.0)
    ap.add_argument("--squash", type=float, default=0.0, help="bound scaled inputs above T")
    ap.add_argument("--norm", default="lastmean", help="lastmean | log_revin | ts")
    ap.add_argument("--loss", default="l1", help="l1 | huber (in normalised space)")
    ap.add_argument("--drop-ctx", type=float, default=0.0,
                    help="share of training windows truncated to a random length >= 1 week")
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=int, default=0, help="1: keep the selected weights (model.pt)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-from", default="", help="first training origin (date); default the first possible")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
