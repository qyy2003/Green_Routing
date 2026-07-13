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


# --------------------------------------------------------------------------- #
# nn.Modules (built lazily so importing this file never requires torch)
# --------------------------------------------------------------------------- #
def _modules():
    torch = tc._torch()
    import torch.nn as nn

    class MovingAvg(nn.Module):
        def __init__(self, kernel):
            super().__init__()
            self.kernel = kernel

        def forward(self, x):                 # x [B, L]
            pad = self.kernel // 2
            xp = torch.nn.functional.pad(x[:, None, :], (pad, pad), mode="replicate")
            return torch.nn.functional.avg_pool1d(xp, self.kernel, 1)[:, 0, :x.shape[1]]

    class DLinear(nn.Module):
        """Trend/seasonal decomposition + a linear map per component (Zeng et al. 2023)."""
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
        def __init__(self, lin, h, hidden=64, layers=1):
            super().__init__()
            self.gru = nn.GRU(1, hidden, layers, batch_first=True)
            self.head = nn.Linear(hidden, h)

        def forward(self, x):
            out, _ = self.gru(x[:, :, None])     # [B, L, hidden]
            return self.head(out[:, -1])

    class TCNNet(nn.Module):
        """Stacked dilated causal 1-D convolutions (exponential receptive field)."""
        def __init__(self, lin, h, ch=32, levels=4, k=3):
            super().__init__()
            layers = []
            c_in = 1
            for i in range(levels):
                d = 2 ** i
                layers += [nn.Conv1d(c_in, ch, k, padding=(k - 1) * d, dilation=d),
                           nn.ReLU()]
                c_in = ch
            self.net = nn.Sequential(*layers)
            self.lin = lin
            self.head = nn.Linear(ch, h)

        def forward(self, x):
            z = self.net(x[:, None, :])          # causal padding -> trim to length
            z = z[:, :, :self.lin]
            return self.head(z[:, :, -1])

    class PatchTST(nn.Module):
        """Patch the series into tokens, encode with a small transformer (Nie et al. 2023)."""
        def __init__(self, lin, h, patch=24, d=48, heads=4, layers=2):
            super().__init__()
            self.patch = patch
            self.n_patch = (lin + patch - 1) // patch
            self.pad = self.n_patch * patch - lin
            self.embed = nn.Linear(patch, d)
            self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d))
            enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                             batch_first=True, dropout=0.0)
            self.tr = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(self.n_patch * d, h)

        def forward(self, x):
            if self.pad:
                x = torch.nn.functional.pad(x, (self.pad, 0), mode="replicate")
            b = x.shape[0]
            tokens = x.reshape(b, self.n_patch, self.patch)
            z = self.embed(tokens) + self.pos
            z = self.tr(z).reshape(b, -1)
            return self.head(z)

    return dict(DLinear=DLinear, GRUNet=GRUNet, TCNNet=TCNNet, PatchTST=PatchTST)


# --------------------------------------------------------------------------- #
# Shared global sequence forecaster
# --------------------------------------------------------------------------- #
class SeqForecaster(Forecaster, tc.TorchTrainer):
    is_global = True
    supports_online = True      # warm-start fine-tune on recent windows at test time

    def __init__(self, lookback=config.DAILY_STEPS, horizons=tuple(config.HORIZONS.values()),
                 epochs=12, max_train_samples=40_000):
        self.lookback = lookback
        self.horizons = tuple(sorted(set(horizons)))
        self.max_h = max(self.horizons)
        self.epochs = epochs
        self.max_train_samples = max_train_samples
        self.net = None
        self._train_start = 0

    def _build_net(self, lin, h):
        raise NotImplementedError

    def _windows(self, log_vals, anchors):
        """RevIN-normalised (input, target) windows for the given anchor times."""
        L = log_vals.shape[1]
        X, Y = [], []
        for j in range(L):
            for t in anchors:
                win = log_vals[t - self.lookback:t, j]
                tgt = log_vals[t:t + self.max_h, j]
                if not (np.isfinite(win).all() and np.isfinite(tgt).all()):
                    continue
                m, s = win.mean(), win.std() + 1e-5
                X.append((win - m) / s)
                Y.append((tgt - m) / s)
        return np.asarray(X, np.float32), np.asarray(Y, np.float32)

    def fit(self, values_filled, split, timestamps):
        log_vals = tc.log1p(values_filled)
        L = log_vals.shape[1]
        self._train_start = split.train[0]
        cap = max(1, self.max_train_samples // L)
        anchors = tc.make_anchors(split.train, self.lookback, self.max_h,
                                  min_ctx=self.lookback, cap=cap)
        X, Y = self._windows(log_vals, anchors)
        if X.shape[0] < 100:
            raise ValueError(f"{self.name}: too few training windows ({X.shape[0]})")
        self.net = self._fit_torch(self._build_net(self.lookback, self.max_h), X, Y,
                                   verbose_name=self.name)

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
        X, Y = self._windows(log_vals, anchors)
        if X.shape[0] < 20:
            return                              # too little recent data: keep current net
        # warm-start: continue training the existing net briefly at a low LR
        self.net = self._fit_torch(self.net, X, Y, verbose_name=f"{self.name}:online",
                                   epochs=self.online_epochs, lr=self.online_lr)

    def predict(self, ctx: Context) -> np.ndarray:
        torch = tc._torch()
        out = np.full(ctx.horizon, np.nan, dtype=float)
        o = ctx.origin
        if o < self.lookback or self.net is None:
            return out
        win = tc.log1p(ctx.values_filled[o - self.lookback:o, ctx.link])
        if not np.isfinite(win).all():
            return out
        m, s = win.mean(), win.std() + 1e-5
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

    def __init__(self, lookback=config.DAILY_STEPS,
                 horizons=tuple(config.HORIZONS.values()), patch=24, **kw):
        super().__init__(lookback=lookback, horizons=horizons, **kw)
        self.patch = patch

    def _build_net(self, lin, h):
        return _modules()["PatchTST"](lin, h, patch=self.patch)


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
