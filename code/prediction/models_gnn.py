"""Tier 6 — spatiotemporal GNN (the link-level frontier of the roadmap).

A link's load is physically coupled to adjacent links: they share flows and common
routing, and a reroute/failure event shifts load between them together. A per-link model
can't see that; a graph model can. This is the Andreoletti-et-al.-2019 DCRNN-on-backbone
idea, here as a compact **Graph-WaveNet-lite**:

  * a shared **dilated-TCN** temporal encoder per link (node),
  * graph mixing over a **fixed** link adjacency (links sharing a router, or on
    physically-adjacent routers per ``dataset.topology``) **plus a learned adaptive
    adjacency** (Graph WaveNet's contribution — usually beats fixed-topology-only),
  * a direct multi-horizon head per node.

The whole panel is forecast jointly at each origin (cached), then the harness reads the
requested link's path — so it is scored on the exact same rolling origins as every other
tier.
"""
from __future__ import annotations

import numpy as np

from dataset import config, topology

from .harness import Context, Forecaster
from . import torch_common as tc


def link_adjacency(devices: list[str]) -> np.ndarray:
    """Symmetric-normalised link-level adjacency (self-loops + shared/adjacent routers)."""
    links = topology.parse_links()
    dev_adj = set()
    for lk in links:
        dev_adj.add((lk.a, lk.b))
        dev_adj.add((lk.b, lk.a))
    n = len(devices)
    A = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if devices[i] == devices[j] or (devices[i], devices[j]) in dev_adj:
                A[i, j] = A[j, i] = 1.0
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    return (A * dinv[:, None] * dinv[None, :]).astype(np.float32)


def _build_module(n_nodes, lin, h, A_norm, ch=32, levels=3, d_embed=10, gc_layers=2):
    torch = tc._torch()
    import torch.nn as nn

    class GraphWaveNetLite(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("A", torch.tensor(A_norm))
            self.E1 = nn.Parameter(torch.randn(n_nodes, d_embed) * 0.05)
            self.E2 = nn.Parameter(torch.randn(n_nodes, d_embed) * 0.05)
            # temporal encoder (shared across nodes): dilated causal convs
            tlayers, c_in = [], 1
            for i in range(levels):
                dl = 2 ** i
                tlayers += [nn.Conv1d(c_in, ch, 3, padding=2 * dl, dilation=dl), nn.ReLU()]
                c_in = ch
            self.tcn = nn.Sequential(*tlayers)
            self.lin = lin
            self.gc = nn.ModuleList([nn.Linear(ch, ch) for _ in range(gc_layers)])
            self.head = nn.Linear(ch, h)

        def forward(self, x):                     # x [B, N, lin]
            b, n, lw = x.shape
            z = self.tcn(x.reshape(b * n, 1, lw))[:, :, :self.lin]
            z = z[:, :, -1].reshape(b, n, -1)     # [B, N, ch]
            adp = torch.softmax(torch.relu(self.E1 @ self.E2.t()), dim=1)   # [N, N]
            A = self.A + adp
            for lin_layer in self.gc:
                agg = torch.einsum("nm,bmc->bnc", A, z)      # graph mixing
                z = torch.relu(z + lin_layer(agg))           # residual
            return self.head(z)                   # [B, N, h]

    return GraphWaveNetLite()


class STGNNForecaster(Forecaster, tc.TorchTrainer):
    """Graph-WaveNet-lite over the link graph. Needs the panel's device list."""

    name = "stgnn"
    tier = 6
    is_global = True

    def __init__(self, devices: list[str], lookback=config.DAILY_STEPS,
                 horizons=tuple(config.HORIZONS.values()), epochs=15,
                 max_train_samples=1500):
        self.devices = list(devices)
        self.lookback = lookback
        self.horizons = tuple(sorted(set(horizons)))
        self.max_h = max(self.horizons)
        self.epochs = epochs
        self.batch_size = 64
        self.max_train_samples = max_train_samples
        self.net = None
        self._A = link_adjacency(self.devices)
        self._cache_o = None
        self._cache_out = None

    @staticmethod
    def _revin_matrix(win):                       # win [N, lin] -> normalised + stats
        m = np.nanmean(win, axis=1, keepdims=True)
        s = np.nanstd(win, axis=1, keepdims=True) + 1e-5
        m = np.where(np.isfinite(m), m, 0.0)
        xn = (win - m) / s
        xn[~np.isfinite(xn)] = 0.0
        return xn.astype(np.float32), m, s

    def fit(self, values_filled, split, timestamps):
        log_vals = tc.log1p(values_filled)
        T, N = log_vals.shape
        anchors = tc.make_anchors(split.train, self.lookback, self.max_h,
                                  min_ctx=self.lookback, cap=self.max_train_samples)
        X, Y = [], []
        for t in anchors:
            win = log_vals[t - self.lookback:t, :].T            # [N, lin]
            tgt = log_vals[t:t + self.max_h, :].T               # [N, max_h]
            if np.isnan(win).all() or np.isnan(tgt).all():
                continue
            xn, m, s = self._revin_matrix(win)
            yn = (tgt - m) / s
            yn[~np.isfinite(yn)] = 0.0
            X.append(xn)
            Y.append(yn.astype(np.float32))
        X = np.asarray(X, np.float32)
        Y = np.asarray(Y, np.float32)
        if X.shape[0] < 50:
            raise ValueError(f"stgnn: too few training samples ({X.shape[0]})")
        net = _build_module(N, self.lookback, self.max_h, self._A)
        self.net = self._fit_torch(net, X, Y, verbose_name=self.name)

    def _forecast_all(self, ctx: Context) -> np.ndarray:
        torch = tc._torch()
        o = ctx.origin
        log_vals = tc.log1p(ctx.values_filled)
        win = log_vals[o - self.lookback:o, :].T
        xn, m, s = self._revin_matrix(win)
        with torch.no_grad():
            yn = self.net(torch.tensor(xn[None])).numpy()[0]    # [N, max_h]
        return np.clip(np.expm1(yn * s + m), 0, None)           # [N, max_h]

    def predict(self, ctx: Context) -> np.ndarray:
        out = np.full(ctx.horizon, np.nan, dtype=float)
        if ctx.origin < self.lookback or self.net is None:
            return out
        if ctx.origin != self._cache_o:
            self._cache_out = self._forecast_all(ctx)
            self._cache_o = ctx.origin
        path = self._cache_out[ctx.link]
        n = min(ctx.horizon, path.size)
        out[:n] = path[:n]
        return out


def default_gnn(devices, horizons, lookback=config.DAILY_STEPS):
    return [STGNNForecaster(devices, lookback=lookback, horizons=horizons)]
