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
from pathlib import Path

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
            torch.backends.cuda.matmul.allow_tf32 = True   # free speedup on Ampere+
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True          # fixed input shapes
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


def link_scale_weights(tr_mean_mbps, scale_weight):
    """Per-link training-loss weights from train-window mean traffic.

    ``scale_weight``: an exponent (``w ~ Mbps**exp`` — chases the raw pooled-MAE
    vote) or the string ``"log"`` (``w ~ log10(max(Mbps, 1))`` — the training-side
    mirror of the log-volume-weighted ratio metric: gentle size preference,
    Gini ~0.33 on the 547-link cohort vs 0.65 at exp=0.5 and 0.85 at exp=1).
    """
    v = np.nan_to_num(np.clip(np.asarray(tr_mean_mbps, np.float64), 0, None))
    if isinstance(scale_weight, str):
        if scale_weight != "log":
            raise ValueError(f"unknown scale_weight {scale_weight!r} (use 'log' or a float)")
        w = np.log10(np.maximum(v, 1.0)) + 0.01   # floor links keep a small say
    else:
        w = v ** float(scale_weight)
    return w.astype(np.float32)


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
    max_train_samples: int = 100_000
    epochs: int = 12
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10          # early stop after this many non-improving epochs
    clip_grad: float = 5.0
    cosine_lr: bool = True      # cosine-anneal LR over the full run (helps stability)
    max_sample_weight: float = 10.0   # cap scale weights: no link dominates a batch
    # online fine-tune: fewer epochs + lower LR (warm-start, avoid forgetting)
    online_epochs: int = 3
    online_lr: float = 3e-4

    # -- checkpointing (see harness.Forecaster's save/load protocol) --------- #
    def save(self, dir_path) -> Path | None:
        """Persist the fitted net's weights (CPU tensors) as ``<name>.pt``."""
        net = getattr(self, "net", None)
        if net is None:
            return None
        torch = _torch()
        p = Path(dir_path) / f"{self.name}.pt"
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, p)
        return p

    def load(self, path, values_filled, split, timestamps) -> bool:
        """Rebuild the net (host class's ``_build_net``) and restore :meth:`save`d
        weights — a shape mismatch (e.g. a changed --lookback) raises, and the
        harness falls back to fitting."""
        torch = _torch()
        state = torch.load(path, map_location="cpu")
        net = self._build_net(self.lookback, self.max_h)
        net.load_state_dict(state)
        self.net = net.to(get_device())
        self.net.eval()
        self._train_start = split.train[0]
        return True

    def _fit_torch(self, net, X, Y, verbose_name="", epochs=None, lr=None, W=None):
        """Train (or, warm-started from an already-trained ``net``, fine-tune) the
        network. ``epochs``/``lr`` override the defaults — online fine-tuning passes
        a smaller LR and fewer epochs so the model nudges toward the recent regime
        without forgetting the train-set fit.

        ``W`` (optional, shape ``[n]``): per-sample loss weights (normalised to mean
        1 internally). Used for scale-weighted training so high-traffic links pull
        the fit more — the pooled-MAE objective — instead of RevIN's equal say.
        Both the train step and the val criterion use the same weights, so early
        stopping tracks the weighted objective too."""
        torch = _torch()
        import torch.nn as nn

        import time

        epochs = self.epochs if epochs is None else epochs
        lr = self.lr if lr is None else lr
        device = get_device()
        net = net.to(device)
        # Keep the dataset on the CPU and stream mini-batches to the GPU: samples ×
        # a week-long target path is gigabytes, so residency on a laptop GPU would
        # OOM — while the per-batch copy (a few MB) is negligible next to compute.
        Xt = torch.tensor(np.asarray(X, np.float32))
        Yt = torch.tensor(np.asarray(Y, np.float32))
        if W is None:
            Wt = torch.ones(Xt.shape[0])
        else:
            Wt = torch.tensor(np.asarray(W, np.float32)).clamp_min(0)
            Wt = Wt * (Wt.numel() / Wt.sum().clamp_min(1e-8))   # normalise mean -> 1
            # cap so a single high-traffic link can't dominate a batch and blow up
            # the gradient (scale-weighting otherwise diverges the deeper nets), then
            # re-normalise mean -> 1 to keep the effective LR unchanged.
            Wt = Wt.clamp(max=self.max_sample_weight)
            Wt = Wt * (Wt.numel() / Wt.sum().clamp_min(1e-8))
        n = Xt.shape[0]
        n_val = max(1, int(0.1 * n))
        perm = torch.randperm(n)
        tr, va = perm[n_val:], perm[:n_val]
        opt = torch.optim.Adam(net.parameters(), lr=lr,
                               weight_decay=self.weight_decay)
        sched = None
        if self.cosine_lr and epochs > 3:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.SmoothL1Loss(reduction="none")   # Huber: robust to traffic bursts
        # Optional per-node weights (STGNN whole-panel samples: pred is [b, N, h]).
        # Sequence tiers weight per *sample* via W instead; both mirror the
        # normalise-then-weight evaluation metric (log-volume weighted ratio).
        nw = getattr(self, "_node_w", None)
        nw_t = None if nw is None else \
            torch.tensor(np.asarray(nw, np.float32), device=device)

        def _wloss(pred, tgt, w):
            per = loss_fn(pred, tgt)                   # [b, ...]
            if per.dim() == 3 and nw_t is not None:    # [b, N, h]: weighted node mean
                per = (per.mean(dim=2) * nw_t).sum(dim=1) / nw_t.sum().clamp_min(1e-8)
            else:
                per = per.flatten(1).mean(dim=1) if per.dim() > 1 else per
            return (per * w).sum() / w.sum().clamp_min(1e-8)

        best = float("inf")
        best_state = None
        bad = 0
        chatty = bool(verbose_name) and epochs > 3   # quiet during online refits
        for ep in range(epochs):
            t0 = time.time()
            net.train()
            idx = tr[torch.randperm(tr.shape[0])]
            tsum, twsum = 0.0, 0.0
            for i in range(0, idx.shape[0], self.batch_size):
                b = idx[i:i + self.batch_size]
                xb = Xt[b].to(device, non_blocking=True)
                yb = Yt[b].to(device, non_blocking=True)
                wb = Wt[b].to(device, non_blocking=True)
                opt.zero_grad()
                loss = _wloss(net(xb), yb, wb)
                if not torch.isfinite(loss):
                    continue            # skip a bad batch rather than corrupt weights
                tsum += loss.item() * float(wb.sum())
                twsum += float(wb.sum())
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(net.parameters(), self.clip_grad)
                if torch.isfinite(gnorm):
                    opt.step()
            tloss = tsum / max(1e-8, twsum)
            if sched is not None:
                sched.step()
            net.eval()
            with torch.no_grad():
                # Batch the val forward too: a single full-set pass can OOM the GPU
                # at long horizons (the head emits a max_h-wide path per sample).
                vsum, wsum = 0.0, 0.0
                for i in range(0, va.shape[0], self.batch_size):
                    b = va[i:i + self.batch_size]
                    xb = Xt[b].to(device, non_blocking=True)
                    yb = Yt[b].to(device, non_blocking=True)
                    wb = Wt[b].to(device, non_blocking=True)
                    vsum += _wloss(net(xb), yb, wb).item() * float(wb.sum())
                    wsum += float(wb.sum())
                vloss = vsum / max(1e-8, wsum)
            if chatty:
                print(f"[{verbose_name}] epoch {ep + 1}/{epochs} train={tloss:.4f} "
                      f"val={vloss:.4f} ({time.time() - t0:.1f}s)", flush=True)
            if vloss < best - 1e-5:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if self.patience and bad >= self.patience:
                    if chatty:
                        print(f"[{verbose_name}] early stop at epoch {ep + 1} "
                              f"(best val={best:.4f})")
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        if device == "cuda" and chatty:
            print(f"[{verbose_name}] peak GPU mem "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()
        return net
