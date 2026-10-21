"""Tier 4 (deep temporal) and Tier 5 (transformer + linear control) forecasters.

All are **global** sequence models: one shared network over every link, RevIN per window,
direct multi-horizon head (see torch_common). They are the roadmap's Tiers 4–5:

  Tier 4  GRU, TCN         — learn nonlinear temporal dynamics end-to-end.
  Tier 5  DLinear, PatchTST-lite — a linear control (DLinear beat many transformers in
          "Are Transformers Effective for TSF?") and a patch-token transformer.

The roadmap is explicit that these pay off most on *multi-step / bursty* series and are
often overkill at 1-step — the harness will show exactly where (if anywhere) they earn
their cost over persistence and the Tier-3 booster.
"""
from __future__ import annotations

import numpy as np

from ..dataset import config

from .harness import Context, Forecaster
from . import torch_common as tc
from .torch_common import link_scale_weights
from .calendar_features import build_calendar


# --------------------------------------------------------------------------- #
# nn.Modules (built lazily so importing this file never requires torch)
# --------------------------------------------------------------------------- #
def _modules():
    torch = tc._torch()
    import torch.nn as nn
    from torch.nn.utils.parametrizations import weight_norm

    class MovingAvg(nn.Module):
        def __init__(self, kernel):
            super().__init__()
            self.kernel = kernel

        def forward(self, x):                 # x [B, L]
            pad = self.kernel // 2
            xp = torch.nn.functional.pad(x[:, None, :], (pad, pad), mode="replicate")
            return torch.nn.functional.avg_pool1d(xp, self.kernel, 1)[:, 0, :x.shape[1]]

    class DLinear(nn.Module):
        """Trend/seasonal decomposition + a linear map per component (Zeng et al.
        2023). Already its designed size — DLinear is linear; its only capacity
        knob is the lookback."""
        def __init__(self, lin, h, kernel=25):
            super().__init__()
            self.decomp = MovingAvg(kernel)
            self.lin_t = nn.Linear(lin, h)
            self.lin_s = nn.Linear(lin, h)

        def forward(self, x):
            trend = self.decomp(x)
            seasonal = x - trend
            return self.lin_t(trend) + self.lin_s(seasonal)

    class LinkEmb(nn.Module):
        """Learned per-link identity vector concatenated before the head — the
        deep-net analogue of GBDT's link-level features. One global net over
        hundreds of heterogeneous links otherwise has no way to specialise
        (RevIN deliberately erases the only identity signal, the scale)."""
        def __init__(self, n_links, dim):
            super().__init__()
            self.emb = nn.Embedding(n_links, dim) if n_links else None
            self.dim = dim if n_links else 0

        def cat(self, z, lid):
            if self.emb is None:
                return z
            if lid is None:
                raise ValueError("net has a link embedding but no link ids given")
            return torch.cat([z, self.emb(lid)], dim=-1)

    class CalendarCond(nn.Module):
        """Calendar conditioning (designs A + C from the pred_547f report).

        A — anchor vector: the K_CAL covariates (sin/cos tod, dow one-hot,
            holiday, workday) at the forecast origin.
        C — future-known path code: the calendar of the TARGET path is known in
            advance, so the model is told (not left to infer) that the path
            crosses a Saturday or Whit Monday: covariates at every ``stride``-th
            future step (hourly — dow/holiday only change daily) are flattened
            through a small linear into a ``code``-dim vector.

        Both are concatenated onto the pre-head features. The full-grid
        covariate matrix rides along as a buffer (~3 MB); samples only carry
        their anchor index, so window RAM does not grow.
        """
        def __init__(self, cal, h, stride=12, code=32):
            super().__init__()
            self.register_buffer("cal", torch.tensor(np.asarray(cal, np.float32)))
            offs = torch.arange(0, h, stride, dtype=torch.int64)
            self.register_buffer("offs", offs)
            k = self.cal.shape[1]
            self.lin = nn.Linear(offs.numel() * k, code)
            self.act = nn.GELU()
            self.dim = k + code

        def forward(self, aid):                        # aid [B] int64
            idx = (aid[:, None] + self.offs[None, :]).clamp(max=self.cal.shape[0] - 1)
            path = self.cal[idx]                       # [B, n_path, k]
            code = self.act(self.lin(path.flatten(1)))  # [B, code]
            return torch.cat([self.cal[aid.clamp(max=self.cal.shape[0] - 1)],
                              code], dim=-1)           # [B, k + code]

    class GRUNet(nn.Module):
        """2-layer, 128-hidden GRU with inter-layer dropout — the standard strong
        RNN-forecaster configuration (there is no single canonical paper)."""
        def __init__(self, lin, h, n_links=0, emb_dim=16, hidden=128, layers=2,
                     dropout=0.1, cal=None):
            super().__init__()
            self.gru = nn.GRU(1, hidden, layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.lemb = LinkEmb(n_links, emb_dim)
            self.calc = None if cal is None else CalendarCond(cal, h)
            cdim = 0 if self.calc is None else self.calc.dim
            self.head = nn.Linear(hidden + self.lemb.dim + cdim, h)

        def forward(self, x, lid=None, aid=None):
            out, _ = self.gru(x[:, :, None])     # [B, L, hidden]
            z = self.lemb.cat(out[:, -1], lid)
            if self.calc is not None:
                z = torch.cat([z, self.calc(aid)], dim=-1)
            return self.head(z)

    class Chomp(nn.Module):
        """Trim the future-side padding so the dilated conv stays causal."""
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            return x[:, :, :-self.n] if self.n else x

    class TCNBlock(nn.Module):
        """Bai et al. 2018 residual block: 2 x (weight-norm dilated causal conv
        -> chomp -> ReLU -> dropout), plus a 1x1 downsample on the residual."""
        def __init__(self, c_in, ch, k, d, dropout):
            super().__init__()
            pad = (k - 1) * d
            self.net = nn.Sequential(
                weight_norm(nn.Conv1d(c_in, ch, k, padding=pad, dilation=d)),
                Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                weight_norm(nn.Conv1d(ch, ch, k, padding=pad, dilation=d)),
                Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            )
            self.down = nn.Conv1d(c_in, ch, 1) if c_in != ch else None
            self.relu = nn.ReLU()

        def forward(self, x):
            res = x if self.down is None else self.down(x)
            return self.relu(self.net(x) + res)

    class TCNNet(nn.Module):
        """Bai et al. 2018 TCN at its designed shape: residual blocks, 64 channels,
        dropout 0.2, and depth auto-sized so the receptive field covers the
        lookback (the paper's sizing rule), capped at ``max_levels`` (default 9 =
        RF 2045, one week; a 1-month window needs 12 = RF 16381 to be seen)."""
        def __init__(self, lin, h, n_links=0, emb_dim=16, ch=64, k=3, dropout=0.2,
                     cal=None, max_levels=9):
            super().__init__()
            levels = 1
            while 1 + 2 * (k - 1) * (2 ** levels - 1) < lin and levels < max_levels:
                levels += 1
            blocks, c_in = [], 1
            for i in range(levels):
                blocks.append(TCNBlock(c_in, ch, k, 2 ** i, dropout))
                c_in = ch
            self.net = nn.Sequential(*blocks)
            self.lemb = LinkEmb(n_links, emb_dim)
            self.calc = None if cal is None else CalendarCond(cal, h)
            cdim = 0 if self.calc is None else self.calc.dim
            self.head = nn.Linear(ch + self.lemb.dim + cdim, h)

        def forward(self, x, lid=None, aid=None):
            z0 = self.net(x[:, None, :])
            z = self.lemb.cat(z0[:, :, -1], lid)
            if self.calc is not None:
                z = torch.cat([z, self.calc(aid)], dim=-1)
            return self.head(z)

    class PatchTST(nn.Module):
        """Nie et al. 2023, patches 16/8, d_model 128, 16 heads, 3 encoder layers,
        d_ff 256, dropout 0.2 (LayerNorm in place of the paper's BatchNorm; RevIN
        applied per window by the forecaster). Head history on the 547 cohort:
        the paper's flatten head (n_patch*d -> h, ~65 M params at week horizons)
        memorised the train regime by epoch 2; the attention-pooled replacement
        (d -> h) fixed that but squeezed 251 patch tokens through ONE 128-d
        vector before emitting a 2016-step path — an information bottleneck that
        made it the worst learned tier (lost to GRU on 94% of links, flattest
        peaks). Now: a LOW-RANK flatten head — flatten -> bottleneck -> h — which
        keeps a direct per-patch path to every output step at ~13 M params,
        guarded by fc_dropout and the class's weight decay."""
        def __init__(self, lin, h, n_links=0, emb_dim=16, patch=16, stride=8,
                     d=128, heads=16, layers=3, d_ff=256, dropout=0.2,
                     fc_dropout=0.2, bottleneck=384, cal=None):
            super().__init__()
            self.patch, self.stride = patch, stride
            self.pad = (-(lin - patch)) % stride      # front-pad so patches tile lin
            self.n_patch = (lin + self.pad - patch) // stride + 1
            self.embed = nn.Linear(patch, d)
            self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d))
            enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=d_ff,
                                             batch_first=True, dropout=dropout)
            self.tr = nn.TransformerEncoder(enc, layers)
            self.drop = nn.Dropout(fc_dropout)
            self.neck = nn.Linear(self.n_patch * d, bottleneck)
            self.act = nn.GELU()
            self.lemb = LinkEmb(n_links, emb_dim)
            self.calc = None if cal is None else CalendarCond(cal, h)
            cdim = 0 if self.calc is None else self.calc.dim
            self.head = nn.Linear(bottleneck + self.lemb.dim + cdim, h)

        def forward(self, x, lid=None, aid=None):
            if self.pad:
                x = torch.nn.functional.pad(x, (self.pad, 0), mode="replicate")
            tokens = x.unfold(-1, self.patch, self.stride)   # [B, n_patch, patch]
            z = self.tr(self.embed(tokens) + self.pos)       # [B, n_patch, d]
            z = self.act(self.neck(self.drop(z.flatten(1)))) # [B, bottleneck]
            z = self.lemb.cat(z, lid)
            if self.calc is not None:
                z = torch.cat([z, self.calc(aid)], dim=-1)
            return self.head(self.drop(z))

    return dict(DLinear=DLinear, GRUNet=GRUNet, TCNNet=TCNNet, PatchTST=PatchTST)


# --------------------------------------------------------------------------- #
# Shared global sequence forecaster
# --------------------------------------------------------------------------- #
class SeqForecaster(Forecaster, tc.TorchTrainer):
    is_global = True
    supports_online = True      # warm-start fine-tune on recent windows at test time

    # per-link training-loss weight from mean traffic: 0 = uniform (pure RevIN
    # equalisation), an exponent (e.g. 0.5/1) = weight ~ Mbps**exp (chases pooled
    # MAE), or "log" = weight ~ log10(Mbps) — the training-side mirror of the
    # log-volume-weighted ratio metric (normalise-then-weight).
    scale_weight: float | str = 0.0

    def __init__(self, lookback=config.DAILY_STEPS, horizons=tuple(config.HORIZONS.values()),
                 epochs=12, max_train_samples=100_000, scale_weight=None):
        self.lookback = lookback
        self.horizons = tuple(sorted(set(horizons)))
        self.max_h = max(self.horizons)
        self.epochs = epochs
        self.max_train_samples = max_train_samples
        if scale_weight is not None:
            self.scale_weight = scale_weight
        # Activation memory scales with the input length (worst for the GRU's
        # cuDNN workspace); shrink batches as the lookback grows so a week-long
        # window still fits a 6 GB GPU.
        self.batch_size = max(32, min(256, (256 * 672) // max(1, lookback)))
        self.net = None
        self._train_start = 0
        self._link_w = None        # per-link loss weight (raw Mbps ** scale_weight)
        self._n_links = 0          # set in fit()/load(); sizes the link embedding
        self._cal = None           # calendar covariates [T, K_CAL] (use_calendar)

    # learned link-identity embedding conditioning the head (DLinear opts out to
    # stay the pure-linear control)
    use_link_embedding: bool = True
    # calendar conditioning (day-of-week + Swiss holidays, designs A+C): set via
    # run.py --calendar; DLinear opts out (control)
    use_calendar: bool = False

    def _build_net(self, lin, h):
        raise NotImplementedError

    # Smoothed-target training (--target-smooth W): the model is fit to a centred
    # W-step moving average of the traffic (taken in Mbps, then log1p) instead of
    # the raw 5-min series, i.e. it is never asked to reproduce the spikes. Inputs
    # stay raw. 0 = off (raw targets, the default).
    target_smooth: int = 0

    def _target_vals(self, values_filled, log_vals):
        """log1p of the training *target* series: raw, or its centred moving
        average when ``target_smooth`` > 1 (NaN kept where the raw value is NaN,
        so the set of valid windows is unchanged)."""
        w = int(self.target_smooth or 0)
        if w <= 1:
            return log_vals
        import pandas as pd
        ma = (pd.DataFrame(np.clip(values_filled, 0, None))
              .rolling(w, center=True, min_periods=1).mean().to_numpy(np.float32))
        ma[~np.isfinite(values_filled)] = np.nan
        return tc.log1p(ma)

    def _windows(self, log_vals, anchors, tgt_vals=None):
        """RevIN-normalised (input, target) windows plus the source link index per
        sample (so callers can attach per-link loss weights). ``tgt_vals`` (same
        shape as ``log_vals``) supplies the targets when they differ from the
        inputs (smoothed-target training)."""
        L = log_vals.shape[1]
        tv = log_vals if tgt_vals is None else tgt_vals
        X, Y, src, anc = [], [], [], []
        for j in range(L):
            for t in anchors:
                win = log_vals[t - self.lookback:t, j]
                tgt = tv[t:t + self.max_h, j]
                if not (np.isfinite(win).all() and np.isfinite(tgt).all()):
                    continue
                # Variance floor: intermittent links (off for the whole window,
                # live in the target) otherwise divide by ~0 and produce 1e5+
                # normalised targets that monopolise the loss. 0.05 log1p-units
                # leaves normal windows (std 0.2-2) untouched.
                m, s = win.mean(), max(win.std(), 0.05)
                X.append((win - m) / s)
                Y.append(np.clip((tgt - m) / s, -20.0, 20.0))
                src.append(j)
                anc.append(t)
        return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
                np.asarray(src, np.int64), np.asarray(anc, np.int64))

    def _weights_for(self, src):
        """Per-sample loss weights from the source-link scale, or None if uniform."""
        if self.scale_weight == 0.0 or self._link_w is None or src.size == 0:
            return None
        return self._link_w[src]

    def _ids_for(self, src):
        return src if self.use_link_embedding else None

    def _aids_for(self, anc):
        return anc if (self.use_calendar and self._cal is not None) else None

    def fit(self, values_filled, split, timestamps):
        # order-independent init/dropout; --seed offsets for multi-seed runs
        tc.seed_model(self.name, getattr(self, 'seed_offset', 0))
        log_vals = tc.log1p(values_filled)
        tgt_vals = self._target_vals(values_filled, log_vals)
        L = log_vals.shape[1]
        self._n_links = L
        self._train_start = split.train[0]
        if self.use_calendar:
            self._cal = build_calendar(timestamps)
        if self.scale_weight:
            # link scale = mean raw throughput over the train slice (no test leak)
            tr_mean = np.nanmean(values_filled[split.train[0]:split.train[1]], axis=0)
            tr_mean = np.nan_to_num(np.clip(tr_mean, 0, None), nan=0.0)
            self._link_w = link_scale_weights(tr_mean, self.scale_weight)
        cap = max(1, self.max_train_samples // L)

        # Chronological validation: windows anchored in the harness's val block
        # (targets strictly later than every training target) drive early stopping,
        # LR reduction and checkpoint selection — forward generalisation, not
        # in-distribution fit.
        val = None
        v_anchors = tc.make_val_anchors(split.val, self.lookback, self.max_h,
                                        cap=max(1, (self.max_train_samples // 5) // L))
        if v_anchors.size:
            Xv, Yv, srcv, ancv = self._windows(log_vals, v_anchors, tgt_vals)
            if Xv.shape[0] >= 50:
                val = (Xv, Yv, self._weights_for(srcv), self._ids_for(srcv),
                       self._aids_for(ancv))

        # Per-epoch anchor resampling: a fixed anchor set means every epoch
        # revisits the same windows (epochs add optimisation, not information);
        # drawing fresh anchors each epoch multiplies the distinct windows seen.
        lo = max(split.train[0] + self.lookback, self.lookback)
        hi = split.train[1] - self.max_h
        pool = np.arange(lo, hi)

        def sampler(ep):
            # seeded per epoch (not sequentially): epoch k draws identical anchors
            # whether the run got there directly or via a mid-training resume
            rng = np.random.default_rng(1_000_003 + ep)
            anchors = np.sort(rng.choice(pool, size=min(cap, pool.size),
                                         replace=False))
            X, Y, src, anc = self._windows(log_vals, anchors, tgt_vals)
            return (X, Y, self._weights_for(src), self._ids_for(src),
                    self._aids_for(anc))

        if val is not None:
            self.net = self._fit_torch(
                self._build_net(self.lookback, self.max_h),
                verbose_name=self.name, val=val, sampler=sampler,
                val_steps=[h - 1 for h in self.horizons])
        else:
            print(f"[{self.name}] warning: val block unusable for chronological "
                  f"validation; fixed anchors + random train-window split")
            anchors = tc.make_anchors(split.train, self.lookback, self.max_h,
                                      min_ctx=self.lookback, cap=cap)
            X, Y, src, anc = self._windows(log_vals, anchors, tgt_vals)
            if X.shape[0] < 100:
                raise ValueError(f"{self.name}: too few training windows ({X.shape[0]})")
            self.net = self._fit_torch(self._build_net(self.lookback, self.max_h),
                                       X, Y, verbose_name=self.name,
                                       W=self._weights_for(src), ids=self._ids_for(src),
                                       aids=self._aids_for(anc))

    def online_update(self, values_filled, timestamps, upto: int) -> None:
        if self.net is None:
            return
        log_vals = tc.log1p(values_filled)
        L = log_vals.shape[1]
        start = self._train_start if self.online_window is None \
            else max(self._train_start, upto - self.online_window)
        cap = max(1, self.max_train_samples // L)
        # anchors drawn from [start, upto): target path kept strictly < upto (no leakage)
        anchors = tc.make_anchors((start, upto), self.lookback, self.max_h,
                                  min_ctx=self.lookback, cap=cap)
        X, Y, src, anc = self._windows(log_vals, anchors)
        if X.shape[0] < 20:
            return                              # too little recent data: keep current net
        # warm-start: continue training the existing net briefly at a low LR
        self.net = self._fit_torch(self.net, X, Y, verbose_name=f"{self.name}:online",
                                   epochs=self.online_epochs, lr=self.online_lr,
                                   W=self._weights_for(src), ids=self._ids_for(src),
                                   aids=self._aids_for(anc))

    def load(self, path, values_filled, split, timestamps) -> bool:
        if self.use_calendar:
            self._cal = build_calendar(timestamps)   # net rebuild needs the buffer
        return super().load(path, values_filled, split, timestamps)

    def predict(self, ctx: Context) -> np.ndarray:
        torch = tc._torch()
        out = np.full(ctx.horizon, np.nan, dtype=float)
        o = ctx.origin
        if o < self.lookback or self.net is None:
            return out
        win = tc.log1p(ctx.values_filled[o - self.lookback:o, ctx.link])
        if not np.isfinite(win).all():
            return out
        m, s = win.mean(), max(win.std(), 0.05)   # same variance floor as training
        xn = ((win - m) / s).astype(np.float32)
        device = next(self.net.parameters()).device
        with torch.no_grad():
            xb = torch.tensor(xn[None], device=device)
            if self.use_link_embedding:
                lid = torch.tensor([ctx.link], dtype=torch.int64, device=device)
                if self.use_calendar and self._cal is not None:
                    aid = torch.tensor([o], dtype=torch.int64, device=device)
                    yn = self.net(xb, lid, aid).cpu().numpy().ravel()
                else:
                    yn = self.net(xb, lid).cpu().numpy().ravel()
            else:
                yn = self.net(xb).cpu().numpy().ravel()
        path = np.expm1(yn * s + m)
        n = min(ctx.horizon, path.size)
        out[:n] = np.clip(path[:n], 0, None)
        return out


# -- concrete tiers --------------------------------------------------------- #
class GRUForecaster(SeqForecaster):
    name = "gru"
    tier = 4

    def _build_net(self, lin, h):
        return _modules()["GRUNet"](lin, h, n_links=self._n_links, cal=self._cal)


class TCNForecaster(SeqForecaster):
    name = "tcn"
    tier = 4

    max_levels: int = 9         # run.py --tcn-max-levels (12 covers a 1-month window)

    def _build_net(self, lin, h):
        return _modules()["TCNNet"](lin, h, n_links=self._n_links, cal=self._cal,
                                    max_levels=self.max_levels)


class DLinearForecaster(SeqForecaster):
    name = "dlinear"
    tier = 5
    use_link_embedding = False   # the control stays purely linear, no identity

    def _build_net(self, lin, h):
        return _modules()["DLinear"](lin, h)


class PatchTSTForecaster(SeqForecaster):
    name = "patchtst"
    tier = 5
    # Seed-clean ablation (pred_pt_a/b, 2026-07-16): calendar conditioning is a
    # NET NEGATIVE for PatchTST -- weekend targets improve (-4..-6%) but weekdays
    # degrade (+1%), per-link median worse at 15min/1h/week (p<=1e-6). Its
    # attention over a full week of patches already infers the calendar; the
    # extra head inputs are redundant. --calendar therefore skips this model.
    calendar_opt_out = True
    lr = 1e-4                   # the paper's supervised LR (1e-3 diverges at d=128)
    weight_decay = 1e-3         # the pooled head still rides a 3-layer encoder;
                                # decay guards the epoch-2 memorisation seen at 547

    def __init__(self, lookback=config.DAILY_STEPS,
                 horizons=tuple(config.HORIZONS.values()), patch=16, stride=8, **kw):
        super().__init__(lookback=lookback, horizons=horizons, **kw)
        self.patch = patch
        self.stride = stride

    def _build_net(self, lin, h):
        return _modules()["PatchTST"](lin, h, n_links=self._n_links, cal=self._cal,
                                      patch=self.patch, stride=self.stride)


def default_deep(horizons, lookback=config.DAILY_STEPS):
    return [
        GRUForecaster(lookback=lookback, horizons=horizons),
        TCNForecaster(lookback=lookback, horizons=horizons),
    ]


def default_transformer(horizons, lookback=config.DAILY_STEPS):
    return [
        DLinearForecaster(lookback=lookback, horizons=horizons),
        PatchTSTForecaster(lookback=lookback, horizons=horizons),
    ]
