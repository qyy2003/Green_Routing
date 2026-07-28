"""Tier 6 — spatiotemporal GNN (the link-level frontier of the roadmap).

A link's load is physically coupled to adjacent links: they share flows and common
routing, and a reroute/failure event shifts load between them together. A per-link model
can't see that; a graph model can. This is the Andreoletti-et-al.-2019 DCRNN-on-backbone
idea, here as **Graph WaveNet at its paper design** (Wu et al. 2019):

  * **gated** (tanh*sigmoid) dilated-conv temporal encoder, residual + skip streams,
  * **order-2 diffusion** over a **fixed** link adjacency (links sharing a router, or
    on physically-adjacent routers per ``dataset.topology``) **plus a learned adaptive
    adjacency** (Graph WaveNet's contribution — usually beats fixed-topology-only),
  * a direct multi-horizon 256->512->h head per node.

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


def _build_module(n_nodes, lin, h, A_norm, ch=32, skip_ch=256, end_ch=512,
                  k=2, d_embed=10, order=2, dropout=0.3):
    """Graph WaveNet at the paper's design (Wu et al. 2019, official repo):
    gated tanh*sigmoid dilated convs (residual/dilation 32ch), per-layer skip
    connections into a 256-ch skip stream, order-2 diffusion over the fixed +
    learned-adaptive adjacency, dropout 0.3, and a 256->512->h output MLP.

    GWN is a short-window model (the paper runs seq_len=12): only the last
    output position feeds the head, so input beyond the receptive field is dead
    compute — we size depth to the lookback (<= the paper's 8 layers) and
    truncate the input to the receptive field.
    """
    torch = tc._torch()
    import torch.nn as nn

    n_layers = min(8, max(1, int(np.log2(max(2, lin)))))   # valid convs must fit lin
    rf = 1 + (k - 1) * sum(2 ** i for i in range(n_layers))

    class GraphWaveNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("A", torch.tensor(A_norm))
            self.E1 = nn.Parameter(torch.randn(n_nodes, d_embed))
            self.E2 = nn.Parameter(torch.randn(d_embed, n_nodes))
            self.start = nn.Conv2d(1, ch, (1, 1))
            self.filt = nn.ModuleList()
            self.gate = nn.ModuleList()
            self.skip = nn.ModuleList()
            self.gconv = nn.ModuleList()
            for i in range(n_layers):
                d = 2 ** i
                self.filt.append(nn.Conv2d(ch, ch, (1, k), dilation=(1, d)))
                self.gate.append(nn.Conv2d(ch, ch, (1, k), dilation=(1, d)))
                self.skip.append(nn.Conv2d(ch, skip_ch, (1, 1)))
                # diffusion GCN: 1x1 conv mixes [z, Az, A^2 z, ~Az, ~A^2 z]
                self.gconv.append(nn.Conv2d(ch * (2 * order + 1), ch, (1, 1)))
            self.end1 = nn.Conv2d(skip_ch, end_ch, (1, 1))
            self.end2 = nn.Conv2d(end_ch, h, (1, 1))
            self.drop = nn.Dropout(dropout)

        def _layer(self, i, z, adp):
            """One residual block -> (next z, skip contribution). Isolated so it can
            be gradient-checkpointed: each layer's [B, 160, N, rf] gconv-concat
            activation is recomputed in the backward pass instead of all n_layers
            being held at once (the term that OOMs a 6 GB card at large N)."""
            res = z
            z = torch.tanh(self.filt[i](z)) * torch.sigmoid(self.gate[i](z))
            s = self.skip[i](z[:, :, :, -1:])                  # last position only
            outs = [z]
            for A in (self.A, adp):                            # fixed + adaptive
                zk = z
                for _ in range(order):
                    zk = torch.einsum("nm,bcmt->bcnt", A, zk)
                    outs.append(zk)
            z = self.drop(self.gconv[i](torch.cat(outs, dim=1)))
            z = z + res[:, :, :, -z.shape[-1]:]                # residual (align T)
            return z, s

        def forward(self, x):                     # x [B, N, lin]
            from torch.utils.checkpoint import checkpoint
            z = self.start(x[:, None, :, -rf:])   # [B, ch, N, T<=rf]
            adp = torch.softmax(torch.relu(self.E1 @ self.E2), dim=1)
            skip = None
            for i in range(len(self.filt)):
                if self.training and z.requires_grad:
                    z, s = checkpoint(self._layer, i, z, adp, use_reentrant=False)
                else:
                    z, s = self._layer(i, z, adp)
                skip = s if skip is None else skip + s
            out = self.end2(torch.relu(self.end1(torch.relu(skip))))   # [B, h, N, 1]
            return out[:, :, :, 0].transpose(1, 2)                     # [B, N, h]

    return GraphWaveNet()


class STGNNForecaster(Forecaster, tc.TorchTrainer):
    """Graph-WaveNet-lite over the link graph. Needs the panel's device list."""

    name = "stgnn"
    tier = 6
    is_global = True
    supports_online = True      # warm-start fine-tune on recent windows at test time

    def __init__(self, devices: list[str], lookback=config.DAILY_STEPS,
                 horizons=tuple(config.HORIZONS.values()), epochs=15,
                 max_train_samples=3000, scale_weight=0.0):
        self.devices = list(devices)
        self.scale_weight = scale_weight   # per-NODE loss weight (see torch_common)
        self._node_w = None
        self.lookback = lookback
        self.horizons = tuple(sorted(set(horizons)))
        self.max_h = max(self.horizons)
        self.epochs = epochs
        self.weight_decay = 1e-4
        self.max_train_samples = max_train_samples
        # Batch adaptively. With gradient checkpointing the retained activations
        # are the n_layers inter-layer states [B, ch=32, N, rf] plus one recomputed
        # gconv-concat [B, 160, N, rf]; the former dominates. Keep it ~1.5 GB so a
        # large node set N still fits a 6 GB card (rf<=256 is the receptive field).
        n = len(self.devices)
        n_layers = min(8, max(1, int(np.log2(max(2, lookback)))))
        rf = min(lookback, 1 + sum(2 ** i for i in range(n_layers)))
        budget_elems = 1_500_000_000 // 4          # ~1.5 GB of float32 activations
        per_sample = n_layers * 32 * n * rf        # ch=32 inter-layer states
        self.batch_size = int(max(4, min(64, budget_elems // max(1, per_sample))))
        self.net = None
        self._A = link_adjacency(self.devices)
        self._cache_o = None
        self._cache_out = None
        self._train_start = 0

    def _build_net(self, lin, h):
        return _build_module(len(self.devices), lin, h, self._A)

    @staticmethod
    def _revin_matrix(win):                       # win [N, lin] -> normalised + stats
        m = np.nanmean(win, axis=1, keepdims=True)
        # same variance floor as the sequence tiers (see models_deep._windows)
        s = np.maximum(np.nanstd(win, axis=1, keepdims=True), 0.05)
        m = np.where(np.isfinite(m), m, 0.0)
        xn = (win - m) / s
        xn[~np.isfinite(xn)] = 0.0
        return xn.astype(np.float32), m, s

    def _windows(self, log_vals, anchors):
        """Per-anchor whole-panel (input, target) tensors, RevIN-normalised per node.

        Preallocated and filled in place: at 547 nodes a week-lookback window set
        is ~13 GB per tensor, so the list-of-arrays + np.asarray-copy pattern
        transiently doubles that and OOM-killed the 62 GB box."""
        N = log_vals.shape[1]
        X = np.empty((len(anchors), N, self.lookback), np.float32)
        Y = np.empty((len(anchors), N, self.max_h), np.float32)
        k = 0
        for t in anchors:
            win = log_vals[t - self.lookback:t, :].T            # [N, lin]
            tgt = log_vals[t:t + self.max_h, :].T               # [N, max_h]
            if np.isnan(win).all() or np.isnan(tgt).all():
                continue
            xn, m, s = self._revin_matrix(win)
            yn = np.clip((tgt - m) / s, -20.0, 20.0)   # bounded targets (see models_deep)
            yn[~np.isfinite(yn)] = 0.0
            X[k] = xn
            Y[k] = yn
            k += 1
        return X[:k], Y[:k]

    def fit(self, values_filled, split, timestamps):
        # order-independent init/dropout; --seed offsets for multi-seed runs
        tc.seed_model(self.name, getattr(self, 'seed_offset', 0))
        log_vals = tc.log1p(values_filled)
        self._train_start = split.train[0]
        if self.scale_weight:
            # per-node loss weights (train-window means; picked up by the shared
            # trainer's _wloss for [b, N, h] predictions)
            tr_mean = np.nanmean(values_filled[split.train[0]:split.train[1]], axis=0)
            self._node_w = tc.link_scale_weights(tr_mean, self.scale_weight)
        anchors = tc.make_anchors(split.train, self.lookback, self.max_h,
                                  min_ctx=self.lookback, cap=self.max_train_samples)
        X, Y = self._windows(log_vals, anchors)
        if X.shape[0] < 50:
            raise ValueError(f"stgnn: too few training samples ({X.shape[0]})")
        # chronological validation on the harness's val block (see models_deep.fit);
        # capped at 1/10 of the train budget — whole-panel val windows cost ~9 MB
        # each at 547 nodes, so a generous cap OOMs the box
        val = None
        v_anchors = tc.make_val_anchors(split.val, self.lookback, self.max_h,
                                        cap=max(50, self.max_train_samples // 10))
        if v_anchors.size:
            Xv, Yv = self._windows(log_vals, v_anchors)
            if Xv.shape[0] >= 10:
                val = (Xv, Yv, None)   # node weights apply inside the loss
        if val is None:
            print(f"[{self.name}] warning: val block unusable for chronological "
                  f"validation; falling back to a random train-window split")
        self.net = self._fit_torch(self._build_net(self.lookback, self.max_h), X, Y,
                                   verbose_name=self.name, val=val,
                                   val_steps=[h - 1 for h in self.horizons])

    def online_update(self, values_filled, timestamps, upto: int) -> None:
        if self.net is None:
            return
        log_vals = tc.log1p(values_filled)
        start = self._train_start if self.online_window is None \
            else max(self._train_start, upto - self.online_window)
        # anchors from [start, upto): target path kept strictly < upto (no leakage)
        anchors = tc.make_anchors((start, upto), self.lookback, self.max_h,
                                  min_ctx=self.lookback, cap=self.max_train_samples)
        X, Y = self._windows(log_vals, anchors)
        if X.shape[0] < 20:
            return
        self._cache_o = None                    # invalidate the per-origin forecast cache
        self.net = self._fit_torch(self.net, X, Y, verbose_name=f"{self.name}:online",
                                   epochs=self.online_epochs, lr=self.online_lr)

    def _forecast_all(self, ctx: Context) -> np.ndarray:
        torch = tc._torch()
        o = ctx.origin
        log_vals = tc.log1p(ctx.values_filled)
        win = log_vals[o - self.lookback:o, :].T
        xn, m, s = self._revin_matrix(win)
        device = next(self.net.parameters()).device
        with torch.no_grad():
            yn = self.net(torch.tensor(xn[None], device=device)).cpu().numpy()[0]  # [N, max_h]
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
