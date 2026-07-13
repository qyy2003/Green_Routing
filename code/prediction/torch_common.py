"""Shared machinery for the torch-based tiers (4/5/6).

All deep models here follow the roadmap's best practice for per-link ISP traffic:

  * **one global model** shared across all links (more data, cross-link transfer,
    generalises to unseen links) — never one model per link;
  * **log1p** the target (traffic is log-normal), model in log space, invert for metrics;
  * **RevIN / instance normalisation**: normalise each input *window* by its own
    mean/std and de-normalise the output, so the model is robust to slow level drift and
    to per-link magnitude differences spanning orders of magnitude (core vs. access);
  * **direct multi-horizon**: a single head emits the whole ``max_horizon`` path at once.

Training samples are drawn only from the train slice, with the target path kept inside
the train block (no leakage across the purge gap). Runs on GPU when one is available
(override with ``PRED_DEVICE=cpu``); the nets are still small, so a CPU box works too.
"""
from __future__ import annotations

import os

import numpy as np

_SEEDED = False
_DEVICE = None


def _torch():
    import torch
    global _SEEDED
    if not _SEEDED:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        _SEEDED = True
    return torch


def get_device():
    """Resolve (and cache) the torch device for the deep/GNN tiers.

    Defaults to CUDA when available, else CPU. Set ``PRED_DEVICE=cpu`` (or ``cuda``)
    to force a choice — an unavailable CUDA request silently falls back to CPU.
    """
    global _DEVICE
    if _DEVICE is None:
        torch = _torch()
        want = os.environ.get("PRED_DEVICE", "").strip().lower()
        if want == "cpu":
            _DEVICE = "cpu"
        elif want == "cuda":
            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def log1p(x):
    return np.log1p(np.clip(x, 0, None))


def make_anchors(train_slice, lookback, max_h, min_ctx, cap, n_links_stride=1):
    """Valid anchor indices t: window [t-lookback, t) in-range, target [t, t+max_h) in train."""
    tr_start, tr_end = train_slice
    lo = max(tr_start + lookback, min_ctx)
    hi = tr_end - max_h
    if hi <= lo:
        raise ValueError("train slice too short for lookback+horizon")
    anchors = np.arange(lo, hi, n_links_stride)
    if cap and anchors.size > cap:
        sel = np.linspace(0, anchors.size - 1, cap).round().astype(int)
        anchors = anchors[np.unique(sel)]
    return anchors


def revin_norm(win_log, eps=1e-5):
    """Per-window instance norm. win_log: [..., L_win]. Returns (xn, mean, std)."""
    m = win_log.mean(axis=-1, keepdims=True)
    s = win_log.std(axis=-1, keepdims=True) + eps
    return (win_log - m) / s, m, s


class TorchTrainer:
    """Mixin giving a :class:`~prediction.harness.Forecaster` a standard train loop.

    Subclasses implement ``_build_net(lookback, horizon)`` returning an nn.Module that
    maps normalised input -> normalised path, and ``_prep_batch`` producing tensors.
    """

    lookback: int = 288
    max_train_samples: int = 60_000
    epochs: int = 12
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    # online fine-tune: fewer epochs + lower LR (warm-start, avoid forgetting)
    online_epochs: int = 3
    online_lr: float = 3e-4

    def _fit_torch(self, net, X, Y, verbose_name="", epochs=None, lr=None):
        """Train (or, warm-started from an already-trained ``net``, fine-tune) the
        network. ``epochs``/``lr`` override the defaults — online fine-tuning passes
        a smaller LR and fewer epochs so the model nudges toward the recent regime
        without forgetting the train-set fit."""
        torch = _torch()
        import torch.nn as nn

        epochs = self.epochs if epochs is None else epochs
        lr = self.lr if lr is None else lr
        device = get_device()
        net = net.to(device)
        Xt = torch.tensor(np.asarray(X, np.float32), device=device)
        Yt = torch.tensor(np.asarray(Y, np.float32), device=device)
        n = Xt.shape[0]
        n_val = max(1, int(0.1 * n))
        perm = torch.randperm(n, device=device)
        tr, va = perm[n_val:], perm[:n_val]
        opt = torch.optim.Adam(net.parameters(), lr=lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.SmoothL1Loss()          # Huber: robust to traffic bursts
        best = float("inf")
        best_state = None
        for ep in range(epochs):
            net.train()
            idx = tr[torch.randperm(tr.shape[0], device=device)]
            for i in range(0, idx.shape[0], self.batch_size):
                b = idx[i:i + self.batch_size]
                opt.zero_grad()
                out = net(Xt[b])
                loss = loss_fn(out, Yt[b])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                # Batch the val forward too: a single full-set pass can OOM the GPU
                # at long horizons (the head emits a max_h-wide path per sample).
                vsum, vcount = 0.0, 0
                for i in range(0, va.shape[0], self.batch_size):
                    b = va[i:i + self.batch_size]
                    vsum += loss_fn(net(Xt[b]), Yt[b]).item() * b.shape[0]
                    vcount += b.shape[0]
                vloss = vsum / max(1, vcount)
            if vloss < best:
                best = vloss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net
