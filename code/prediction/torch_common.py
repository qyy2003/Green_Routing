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


def seed_model(name: str, offset: int = 0) -> None:
    """Deterministic per-model RNG seed, called at the top of each fit().

    Models train sequentially in one process, so with only the global
    process-level seed, model k+1's weight init and dropout draws depend on how
    many RNG draws model k consumed — e.g. a GRU early-stopping at epoch 37
    instead of 71 shifted the *unchanged* DLinear control by +0.6-1.1% per-link
    (p~1e-16) between pred_547f and pred_547g. Seeding by model name makes each
    model's init/dropout stream independent of training order and of other
    models' epoch counts.
    """
    import hashlib
    torch = _torch()
    seed = (int(hashlib.md5(name.encode()).hexdigest()[:8], 16) + offset) % (2 ** 31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def make_val_anchors(val_slice, lookback, max_h, cap):
    """Anchor indices for the CHRONOLOGICAL validation block.

    The input window [t-lookback, t) may reach back before the block (that is
    real, permissible history — exactly what the harness's predict does); the
    target path [t, t+max_h) stays strictly inside the block, so validation
    never sees a test-block value.
    """
    lo = max(val_slice[0], lookback)
    hi = val_slice[1] - max_h
    if hi <= lo:
        return np.empty(0, np.int64)
    anchors = np.arange(lo, hi)
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
    # LR schedule: ReduceLROnPlateau on the (chronological) val loss. Early stop is
    # gated on >= 2 LR reductions so a high-LR plateau triggers refinement instead
    # of killing the run (the pred_547c failure mode: cosine T_max=epochs never
    # decayed before patience fired).
    plateau_factor: float = 0.3
    plateau_patience: int = 3
    max_sample_weight: float = 10.0   # cap scale weights: no link dominates a batch
    # Asymmetric Huber: multiply the loss of UNDER-predicted steps (pred < target)
    # by this factor. 1.0 = symmetric. >1 moves the point forecast above the
    # conditional median (pinball-style) — the counterweight to peak shaving:
    # on pred_547d every learned tier under-shot top-decile traffic by -36..-46%
    # at day/week vs -28% for seasonal-naive (the irreducible part). Applies to
    # train AND val, so early stopping selects for the same objective.
    under_weight: float = 1.0
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
        # the file is a plain tensor state_dict, so the safe loader suffices
        # (and silences torch's weights_only FutureWarning)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._n_links = values_filled.shape[1]   # sizes the link embedding, if any
        net = self._build_net(self.lookback, self.max_h)
        net.load_state_dict(state)
        self.net = net.to(get_device())
        self.net.eval()
        self._train_start = split.train[0]
        return True

    def _fit_torch(self, net, X=None, Y=None, verbose_name="", epochs=None, lr=None,
                   W=None, val=None, ids=None, aids=None, sampler=None,
                   val_steps=None):
        """Train (or, warm-started from an already-trained ``net``, fine-tune) the
        network. ``epochs``/``lr`` override the defaults — online fine-tuning passes
        a smaller LR and fewer epochs so the model nudges toward the recent regime
        without forgetting the train-set fit.

        ``W`` (optional, shape ``[n]``): per-sample loss weights (normalised to mean
        1 internally). Used for scale-weighted training so high-traffic links pull
        the fit more, mirroring the log-volume-weighted evaluation.

        ``val`` (optional): ``(Xv, Yv, Wv[, idsv])`` — an explicit CHRONOLOGICAL
        validation set (windows anchored in the harness's val block, i.e. strictly
        later than every training target). This is the criterion for early stopping,
        LR reduction, and best-checkpoint selection, so model selection measures
        forward generalisation, not in-distribution fit. When given, ALL training
        samples take gradient steps. Without it (online fine-tunes), a random 10%
        of the training windows is held out as a stability guard only.

        ``ids`` (optional, shape ``[n]`` int64): per-sample link indices, forwarded
        to nets with a link embedding as ``net(x, lid)``.

        ``sampler`` (optional): callable ``sampler(epoch) -> (X, Y, W, ids)`` that
        draws a FRESH training set every epoch (anchor resampling). With a fixed
        anchor set, every epoch revisits the same windows and epochs add
        optimisation but no information; resampling multiplies the distinct windows
        seen by the number of epochs. Requires ``val``.

        Loss: Huber on RevIN-normalised paths, horizon-weighted by 1/sqrt(h)
        (mean-1 normalised) so the near steps — the scored horizons — are not
        drowned by the (mostly irreducible) far tail of a week-long path.

        ``val_steps`` (optional): 0-based path positions (the scored horizons,
        h-1) at which the VAL criterion is evaluated, equally weighted. Without
        it, val uses the same 1/sqrt(h) weighting as training — which gives the
        week tail ~no say in selection: two TCNs whose day-horizon test MAE
        differed by 13% had val losses 0.3952 vs 0.3990. Training keeps the
        dense 1/sqrt(h) supervision either way.

        Schedule: ReduceLROnPlateau on the val loss (factor 0.3, patience 3).
        Early stopping fires only after the LR has been reduced at least twice —
        a val plateau at high LR triggers refinement, not death."""
        torch = _torch()
        import torch.nn as nn

        import time

        epochs = self.epochs if epochs is None else epochs
        lr = self.lr if lr is None else lr
        device = get_device()
        net = net.to(device)
        if sampler is not None and val is None:
            raise ValueError("per-epoch sampler requires an explicit val set")

        def _prep_w(Warr, n_samples):
            """mean-1 normalised, capped sample weights (train and val alike)."""
            if Warr is None:
                return torch.ones(n_samples)
            t = torch.tensor(np.asarray(Warr, np.float32)).clamp_min(0)
            t = t * (t.numel() / t.sum().clamp_min(1e-8))   # normalise mean -> 1
            # cap so a single high-traffic link can't dominate a batch and blow up
            # the gradient, then re-normalise so the effective LR is unchanged.
            t = t.clamp(max=self.max_sample_weight)
            return t * (t.numel() / t.sum().clamp_min(1e-8))

        def _tensors(Xa, Ya, Wa, Ia, Aa=None):
            # CPU-resident; mini-batches are streamed to the GPU (a full week-path
            # dataset is gigabytes and would OOM a laptop GPU). from_numpy shares
            # the numpy buffer — torch.tensor() would COPY, and at STGNN scale
            # (13 GB per tensor) that transient copy OOM-killed the box.
            Xt = torch.from_numpy(np.ascontiguousarray(np.asarray(Xa, np.float32)))
            Yt = torch.from_numpy(np.ascontiguousarray(np.asarray(Ya, np.float32)))
            Wt = _prep_w(Wa, Xt.shape[0])
            It = None if Ia is None else \
                torch.from_numpy(np.ascontiguousarray(np.asarray(Ia, np.int64)))
            At = None if Aa is None else \
                torch.from_numpy(np.ascontiguousarray(np.asarray(Aa, np.int64)))
            return Xt, Yt, Wt, It, At

        tr = va = None
        if sampler is None:
            Xt, Yt, Wt, It, At = _tensors(X, Y, W, ids, aids)
            n = Xt.shape[0]
            if val is None:
                # legacy fallback (online fine-tunes): random 10% stability guard
                n_val = max(1, int(0.1 * n))
                perm = torch.randperm(n)
                tr, va = perm[n_val:], perm[:n_val]
                Xva, Yva, Wva, Iva, Ava = Xt, Yt, Wt, It, At
        if val is not None:
            Xv, Yv, Wv, *rest = val
            Xva, Yva, Wva, Iva, Ava = _tensors(
                Xv, Yv, Wv, rest[0] if rest else None,
                rest[1] if len(rest) > 1 else None)
            va = torch.arange(Xva.shape[0])

        opt = torch.optim.Adam(net.parameters(), lr=lr,
                               weight_decay=self.weight_decay)
        sched = None
        if epochs > 3:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=self.plateau_factor,
                patience=self.plateau_patience)
        loss_fn = nn.SmoothL1Loss(reduction="none")   # Huber: robust to traffic bursts
        # Optional per-node weights (STGNN whole-panel samples: pred is [b, N, h]).
        # Sequence tiers weight per *sample* via W instead; both mirror the
        # normalise-then-weight evaluation metric (log-volume weighted ratio).
        nw = getattr(self, "_node_w", None)
        nw_t = None if nw is None else \
            torch.tensor(np.asarray(nw, np.float32), device=device)
        hw_cache = {}

        def _hw(h):                       # horizon weights 1/sqrt(h), mean 1
            if h not in hw_cache:
                w = 1.0 / torch.sqrt(torch.arange(1, h + 1, dtype=torch.float32,
                                                  device=device))
                hw_cache[h] = w * (h / w.sum())
            return hw_cache[h]

        uw = float(self.under_weight)

        vsteps_t = None
        if val_steps:
            vsteps_t = torch.tensor(sorted({int(v) for v in val_steps}),
                                    dtype=torch.int64, device=device)

        def _wloss(pred, tgt, w, steps=None):
            per = loss_fn(pred, tgt)                   # [b, h] or [b, N, h]
            if uw != 1.0:
                # under-predicted steps cost uw x (see under_weight above)
                per = per * ((pred < tgt).to(per.dtype) * (uw - 1.0) + 1.0)
            if steps is not None and per.dim() >= 2:
                # selection criterion: scored horizons only, equal weight each
                per = per.index_select(-1, steps.clamp(max=per.shape[-1] - 1))
            elif per.dim() >= 2:
                per = per * _hw(per.shape[-1])         # near horizons matter more
            if per.dim() == 3 and nw_t is not None:    # [b, N, h]: weighted node mean
                per = (per.mean(dim=2) * nw_t).sum(dim=1) / nw_t.sum().clamp_min(1e-8)
            else:
                per = per.flatten(1).mean(dim=1) if per.dim() > 1 else per
            return (per * w).sum() / w.sum().clamp_min(1e-8)

        def _fwd(xb, ib, ab=None):
            if ib is None:
                return net(xb)
            if ab is not None:
                return net(xb, ib, ab)     # calendar-conditioned nets
            return net(xb, ib)

        best = float("inf")
        best_state = None
        bad = 0
        lr_now = lr
        min_lr_for_stop = lr * (self.plateau_factor ** 2) * 1.001
        chatty = bool(verbose_name) and epochs > 3   # quiet during online refits

        # ---- training history ------------------------------------------------
        # The per-epoch curve is otherwise only printed; recording it lets a run be
        # promoted to the model zoo with its learning curve attached (prediction.zoo).
        # ``verbose_name`` carries ":online" for warm-start refits, so the phases stay
        # separable in the saved history.
        phase = "online" if ":" in (verbose_name or "") else "fit"
        if not hasattr(self, "train_history"):
            self.train_history: list[dict] = []
        stop_reason = "epochs_exhausted"

        # ---- mid-training checkpointing -------------------------------------
        # The full training state (weights, Adam moments, plateau-scheduler state,
        # early-stop counters, RNG) is flushed to disk after every epoch, so an
        # interruption — Ctrl-C, OOM kill, suspend-induced CUDA loss — costs at
        # most one epoch: rerunning the same command resumes exactly where it
        # stopped. The file is transient: deleted when training completes (the
        # harness then saves the final weights-only checkpoint). Online
        # fine-tunes (":online", few epochs) are excluded.
        state_path = None
        ckpt_dir = getattr(self, "train_ckpt_dir", None)
        if ckpt_dir and verbose_name and ":" not in verbose_name and epochs > 3:
            state_path = Path(ckpt_dir) / f"{verbose_name}.train.pt"
        start_ep = 0
        if state_path is not None and state_path.exists():
            try:
                # our own trusted bundle (optimiser state, RNG blobs) — needs the
                # full unpickler, so pin weights_only against torch's default flip
                st = torch.load(state_path, map_location="cpu", weights_only=False)
                net.load_state_dict(st["net"])
                opt.load_state_dict(st["opt"])
                if sched is not None and st.get("sched") is not None:
                    sched.load_state_dict(st["sched"])
                best, bad, lr_now = st["best"], st["bad"], st["lr_now"]
                best_state = st["best_state"]
                start_ep = st["epoch"] + 1
                torch.set_rng_state(st["torch_rng"])
                if torch.cuda.is_available() and st.get("cuda_rng") is not None:
                    try:
                        torch.cuda.set_rng_state_all(st["cuda_rng"])
                    except Exception:
                        pass          # device count changed: keep fresh CUDA rng
                print(f"[{verbose_name}] resuming training at epoch "
                      f"{start_ep + 1}/{epochs} (best val={best:.4f}, "
                      f"lr={lr_now:.2e})", flush=True)
            except Exception as e:    # incompatible shapes/config: start fresh
                print(f"[{verbose_name}] training-state resume failed ({e}); "
                      f"starting fresh")
                start_ep, best, bad, lr_now, best_state = 0, float("inf"), 0, lr, None

        def _save_train_state(ep):
            if state_path is None:
                return
            bundle = {
                "epoch": ep,
                "net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched is not None else None,
                "best": best, "bad": bad, "lr_now": lr_now,
                "best_state": None if best_state is None else
                              {k: v.cpu() for k, v in best_state.items()},
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None,
            }
            tmp = state_path.with_suffix(".tmp")
            torch.save(bundle, tmp)
            os.replace(tmp, state_path)    # atomic: never a half-written file

        for ep in range(start_ep, epochs):
            t0 = time.time()
            if sampler is not None:
                smp = sampler(ep)                     # (X, Y, W, ids[, aids])
                Xt, Yt, Wt, It, At = _tensors(*smp) if len(smp) == 5 else \
                    _tensors(*smp, None)
            if tr is None or sampler is not None:
                tr_idx = torch.randperm(Xt.shape[0])
            else:
                tr_idx = tr[torch.randperm(tr.shape[0])]
            net.train()
            tsum, twsum = 0.0, 0.0
            for i in range(0, tr_idx.shape[0], self.batch_size):
                b = tr_idx[i:i + self.batch_size]
                xb = Xt[b].to(device, non_blocking=True)
                yb = Yt[b].to(device, non_blocking=True)
                wb = Wt[b].to(device, non_blocking=True)
                ib = None if It is None else It[b].to(device, non_blocking=True)
                ab = None if At is None else At[b].to(device, non_blocking=True)
                opt.zero_grad()
                loss = _wloss(_fwd(xb, ib, ab), yb, wb)
                if not torch.isfinite(loss):
                    continue            # skip a bad batch rather than corrupt weights
                tsum += loss.item() * float(wb.sum())
                twsum += float(wb.sum())
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(net.parameters(), self.clip_grad)
                if torch.isfinite(gnorm):
                    opt.step()
            tloss = tsum / max(1e-8, twsum)
            net.eval()
            with torch.no_grad():
                # Batch the val forward too: a single full-set pass can OOM the GPU
                # at long horizons (the head emits a max_h-wide path per sample).
                vsum, wsum = 0.0, 0.0
                for i in range(0, va.shape[0], self.batch_size):
                    b = va[i:i + self.batch_size]
                    xb = Xva[b].to(device, non_blocking=True)
                    yb = Yva[b].to(device, non_blocking=True)
                    wb = Wva[b].to(device, non_blocking=True)
                    ib = None if Iva is None else Iva[b].to(device, non_blocking=True)
                    ab = None if Ava is None else Ava[b].to(device, non_blocking=True)
                    vsum += _wloss(_fwd(xb, ib, ab), yb, wb,
                                   steps=vsteps_t).item() * float(wb.sum())
                    wsum += float(wb.sum())
                vloss = vsum / max(1e-8, wsum)
            if sched is not None:
                sched.step(vloss)
                new_lr = opt.param_groups[0]["lr"]
                if chatty and new_lr < lr_now * 0.999:
                    print(f"[{verbose_name}] lr {lr_now:.2e} -> {new_lr:.2e}")
                lr_now = new_lr
            secs = time.time() - t0
            if chatty:
                print(f"[{verbose_name}] epoch {ep + 1}/{epochs} train={tloss:.4f} "
                      f"val={vloss:.4f} ({secs:.1f}s)", flush=True)
            self.train_history.append({
                "phase": phase, "epoch": ep + 1, "epochs": epochs,
                "train_loss": float(tloss), "val_loss": float(vloss),
                "lr": float(lr_now), "seconds": round(secs, 2),
            })
            if vloss < best - 1e-5:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                _save_train_state(ep)
            else:
                bad += 1
                _save_train_state(ep)
                # stop only once the plateau schedule has had its chance: LR must
                # have been reduced >= 2 times before patience can kill the run
                if (self.patience and bad >= self.patience
                        and (sched is None or lr_now <= min_lr_for_stop)):
                    stop_reason = "early_stop"
                    if chatty:
                        print(f"[{verbose_name}] early stop at epoch {ep + 1} "
                              f"(best val={best:.4f})")
                    break
        if state_path is not None:
            state_path.unlink(missing_ok=True)   # completed: transient state gone
        rows = [r for r in self.train_history if r["phase"] == phase]
        self.train_summary = {
            **getattr(self, "train_summary", {}),
            phase: {
                "best_val_loss": None if best == float("inf") else float(best),
                "epochs_run": len(rows),
                "epochs_requested": epochs,
                "final_lr": float(lr_now),
                "stop_reason": stop_reason,
                "seconds": round(sum(r["seconds"] for r in rows), 1),
            },
        }
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        if device == "cuda" and chatty:
            print(f"[{verbose_name}] peak GPU mem "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()
        return net
