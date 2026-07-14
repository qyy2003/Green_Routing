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

from dataset import config

from .harness import Context, Forecaster
from . import torch_common as tc
from .torch_common import link_scale_weights


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

    class GRUNet(nn.Module):
        """2-layer, 128-hidden GRU with inter-layer dropout — the standard strong
        RNN-forecaster configuration (there is no single canonical paper)."""
        def __init__(self, lin, h, hidden=128, layers=2, dropout=0.1):
            super().__init__()
            self.gru = nn.GRU(1, hidden, layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden, h)

        def forward(self, x):
            out, _ = self.gru(x[:, :, None])     # [B, L, hidden]
            return self.head(out[:, -1])

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
        lookback (the paper's sizing rule), capped at 9 levels (RF 2045)."""
        def __init__(self, lin, h, ch=64, k=3, dropout=0.2):
            super().__init__()
            levels = 1
            while 1 + 2 * (k - 1) * (2 ** levels - 1) < lin and levels < 9:
                levels += 1
            blocks, c_in = [], 1
            for i in range(levels):
                blocks.append(TCNBlock(c_in, ch, k, 2 ** i, dropout))
                c_in = ch
            self.net = nn.Sequential(*blocks)
            self.head = nn.Linear(ch, h)

        def forward(self, x):
            z = self.net(x[:, None, :])
            return self.head(z[:, :, -1])

    class PatchTST(nn.Module):
        """Nie et al. 2023 at the paper's supervised config: overlapping patches
        (16/8), d_model 128, 16 heads, 3 encoder layers, d_ff 256, dropout 0.2
        (LayerNorm in place of the paper's BatchNorm; RevIN is applied per window
        by the forecaster)."""
        def __init__(self, lin, h, patch=16, stride=8, d=128, heads=16,
                     layers=3, d_ff=256, dropout=0.2, fc_dropout=0.2):
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
            self.head = nn.Linear(self.n_patch * d, h)

        def forward(self, x):
            if self.pad:
                x = torch.nn.functional.pad(x, (self.pad, 0), mode="replicate")
            tokens = x.unfold(-1, self.patch, self.stride)   # [B, n_patch, patch]
            z = self.embed(tokens) + self.pos
            z = self.tr(z).reshape(x.shape[0], -1)
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

    def _build_net(self, lin, h):
        raise NotImplementedError

    def _windows(self, log_vals, anchors):
        """RevIN-normalised (input, target) windows plus the source link index per
        sample (so callers can attach per-link loss weights)."""
        L = log_vals.shape[1]
        X, Y, src = [], [], []
        for j in range(L):
            for t in anchors:
                win = log_vals[t - self.lookback:t, j]
                tgt = log_vals[t:t + self.max_h, j]
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
        return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
                np.asarray(src, np.int64))

    def _weights_for(self, src):
        """Per-sample loss weights from the source-link scale, or None if uniform."""
        if self.scale_weight == 0.0 or self._link_w is None or src.size == 0:
            return None
        return self._link_w[src]

    def fit(self, values_filled, split, timestamps):
        log_vals = tc.log1p(values_filled)
        L = log_vals.shape[1]
        self._train_start = split.train[0]
        if self.scale_weight:
            # link scale = mean raw throughput over the train slice (no test leak)
            tr_mean = np.nanmean(values_filled[split.train[0]:split.train[1]], axis=0)
            tr_mean = np.nan_to_num(np.clip(tr_mean, 0, None), nan=0.0)
            self._link_w = link_scale_weights(tr_mean, self.scale_weight)
        cap = max(1, self.max_train_samples // L)
        anchors = tc.make_anchors(split.train, self.lookback, self.max_h,
                                  min_ctx=self.lookback, cap=cap)
        X, Y, src = self._windows(log_vals, anchors)
        if X.shape[0] < 100:
            raise ValueError(f"{self.name}: too few training windows ({X.shape[0]})")
        self.net = self._fit_torch(self._build_net(self.lookback, self.max_h), X, Y,
                                   verbose_name=self.name, W=self._weights_for(src))

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
        X, Y, src = self._windows(log_vals, anchors)
        if X.shape[0] < 20:
            return                              # too little recent data: keep current net
        # warm-start: continue training the existing net briefly at a low LR
        self.net = self._fit_torch(self.net, X, Y, verbose_name=f"{self.name}:online",
                                   epochs=self.online_epochs, lr=self.online_lr,
                                   W=self._weights_for(src))

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
            yn = self.net(torch.tensor(xn[None], device=device)).cpu().numpy().ravel()
        path = np.expm1(yn * s + m)
        n = min(ctx.horizon, path.size)
        out[:n] = np.clip(path[:n], 0, None)
        return out


# -- concrete tiers --------------------------------------------------------- #
class GRUForecaster(SeqForecaster):
    name = "gru"
    tier = 4

    def _build_net(self, lin, h):
        return _modules()["GRUNet"](lin, h)


class TCNForecaster(SeqForecaster):
    name = "tcn"
    tier = 4

    def _build_net(self, lin, h):
        return _modules()["TCNNet"](lin, h)


class DLinearForecaster(SeqForecaster):
    name = "dlinear"
    tier = 5

    def _build_net(self, lin, h):
        return _modules()["DLinear"](lin, h)


class PatchTSTForecaster(SeqForecaster):
    name = "patchtst"
    tier = 5
    lr = 1e-4                   # the paper's supervised LR (1e-3 diverges at d=128)

    def __init__(self, lookback=config.DAILY_STEPS,
                 horizons=tuple(config.HORIZONS.values()), patch=16, stride=8, **kw):
        super().__init__(lookback=lookback, horizons=horizons, **kw)
        self.patch = patch
        self.stride = stride

    def _build_net(self, lin, h):
        return _modules()["PatchTST"](lin, h, patch=self.patch, stride=self.stride)


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
