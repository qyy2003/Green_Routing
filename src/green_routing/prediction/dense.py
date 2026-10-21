"""Dense prediction log: every model, every link, every 5-min step of the test block.

The harness (``harness.backtest``) scores a *sample* of forecast origins
(``--max-origins``, ~18 h apart on pred_year) and keeps one point per scored
horizon. This module re-runs the fitted models of a finished run at **every** 5-min
origin of the test block and logs the predictions in two layouts, both indexed by
the *target* time t (so they line up with the real series):

  rolling  fixed lead: pred[t] is the forecast for t made exactly h steps before
           (origin o = t - h + 1). Same metric definition as the harness's
           horizon-h scores — just at every origin instead of a sample.
  block    window by window: for horizon H the test block is cut into consecutive
           H-step windows; at each window start the model forecasts the whole
           window (leads 1..H) at 5-min resolution, then re-forecasts at the next
           window. pred[t] is therefore a lead-(1..H) forecast and the metrics
           average over the window's leads. GBDT is direct-per-horizon (no path),
           so it has no block log.

Inference mirrors each model's ``predict`` exactly (same windows, RevIN floor,
calendar ids, clipping) but batched on the GPU; ``verify`` checks the rolling log
against the harness's records at the sampled origins.

Output (``--out``, default ``<run>/dense``)::

    meta.json                      links, horizons, target grid, models done
    pred_<mode>_<model>.npy        float32 [L, n_horizons, n_targets], NaN = none
    <mode>/aggregate.csv per_link.csv ratio_vs_persistence_mae.csv wape.csv peaks.csv

Usage::

    gr-dense predict --run artifacts/pred_year/merged \
        [--models tcn,gru] [--batch 2048]
    gr-dense verify  --run artifacts/pred_year/merged
    gr-dense metrics --run artifacts/pred_year/merged
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import config, splits as splits_mod

from . import data as data_mod, harness, metrics as M

MODES = ("rolling", "block")
BASELINE_NAMES = ("persistence", "seasonal_naive_day", "seasonal_naive_week",
                  "historical_average")


# --------------------------------------------------------------------------- #
# Run context: the exact panel / split / models of a finished run
# --------------------------------------------------------------------------- #
class RunCtx:
    def __init__(self, run: Path, out: Path | None = None):
        self.run = Path(run).resolve()
        self.out = Path(out).resolve() if out else self.run / "dense"
        self.cfg = json.loads((self.run / "config.json").read_text())
        c = self.cfg
        panel = data_mod.build_panel(c["start"], c["end"], value_col=c["value_col"],
                                     min_coverage=c["min_coverage"],
                                     max_links=c["max_links"], devices=c["devices"],
                                     verbose=False)
        if c.get("min_mean_mbps"):
            panel = panel.subset(panel.mean_mbps() >= c["min_mean_mbps"])
        if c.get("connected_links"):
            panel = data_mod.select_connected(panel, c["connected_links"], verbose=False)
        if panel.links != c["links"]:
            raise SystemExit("rebuilt panel does not match the run's link list")
        self.panel = panel
        self.split = splits_mod.chronological_split_by_date(
            panel.timestamps, c["train_end"], c["val_start"], c["val_end"],
            c["test_start"])
        self.hnames = list(c["horizons"])
        self.hsteps = [config.HORIZONS[h] for h in self.hnames]
        order = np.argsort(self.hsteps)
        self.hnames = [self.hnames[i] for i in order]
        self.hsteps = [self.hsteps[i] for i in order]
        self.values_raw = panel.values
        self.values_filled = harness._fill(panel.values, config.DAILY_STEPS // 24)
        self.T, self.L = panel.values.shape
        self.lo = self.split.test[0]                 # first origin / first target
        self.n_t = self.T - self.lo                  # targets t in [lo, T)

    # -- target-time bookkeeping -------------------------------------------- #
    def rolling_origin(self, h: int) -> np.ndarray:
        """origin for each target t (index t - lo); -1 where it precedes the test."""
        t = np.arange(self.lo, self.T)
        o = t - h + 1
        return np.where(o >= self.lo, o, -1)

    def block_origin(self, H: int) -> np.ndarray:
        t = np.arange(self.lo, self.T)
        return self.lo + ((t - self.lo) // H) * H

    def models(self, names):
        """Instantiate the run's models by name and reload their checkpoints."""
        from .run import build_models
        c = self.cfg
        ms = build_models(c["tiers"], tuple(sorted(self.hsteps)), False, self.panel,
                          c["lookback"], c["epochs"],
                          max_train_samples=c.get("max_train_samples"),
                          scale_weight=c.get("scale_weight", 0.0),
                          under_weight=c.get("under_weight", 1.0),
                          calendar=c.get("calendar", False))
        from .run import apply_overrides
        apply_overrides(ms, c)
        out = []
        ck = self.run / "checkpoints"
        for m in ms:
            if m.name not in names:
                continue
            hits = [p for p in sorted(ck.glob(f"{m.name}.*"))
                    if not p.name.endswith(".train.pt")]
            if getattr(m, "is_global", False):
                if not hits:
                    raise SystemExit(f"{m.name}: no checkpoint in {ck}")
                m.load(hits[0], self.values_filled, self.split, self.panel.timestamps)
            out.append(m)
        return out


# --------------------------------------------------------------------------- #
# Baselines, vectorised over (origin, lead) — same rules as baselines.py
# --------------------------------------------------------------------------- #
class _BaselineEngine:
    def __init__(self, rc: RunCtx):
        vf = rc.values_filled.astype(np.float64)
        T, L = vf.shape
        fin = np.isfinite(vf)
        # last finite value strictly before origin o:  lastfin[o] (0.0 if none)
        ff = pd.DataFrame(vf).ffill().to_numpy()
        self.lastfin = np.vstack([np.zeros((1, L)), np.nan_to_num(ff, nan=0.0)])
        self.vf, self.fin = vf, fin
        # historical average: weekly (dow, tod) bins on the UTC grid
        idx = pd.to_datetime(rc.panel.timestamps, unit="s", utc=True)
        sod = ((idx.hour * 3600 + idx.minute * 60 + idx.second)
               // config.STEP_SECONDS).to_numpy()
        self.bin = idx.dayofweek.to_numpy() * config.DAILY_STEPS + sod
        W = config.WEEKLY_STEPS
        x = np.where(fin, vf, 0.0)
        n_w = -(-T // W)
        pad = n_w * W - T
        xs = np.concatenate([x, np.zeros((pad, L))]).reshape(n_w, W, L)
        cs = np.concatenate([fin.astype(np.float64), np.zeros((pad, L))]).reshape(n_w, W, L)
        # cumulative over weeks: S[t] = sum of finite values at t, t-W, t-2W, ...
        self.S = np.cumsum(xs, axis=0).reshape(-1, L)[:T]
        self.C = np.cumsum(cs, axis=0).reshape(-1, L)[:T]
        # running global mean of finite history: G[o] over [0, o)
        gs = np.vstack([np.zeros((1, L)), np.cumsum(x, axis=0)])
        gc = np.vstack([np.zeros((1, L)), np.cumsum(fin, axis=0)])
        with np.errstate(invalid="ignore", divide="ignore"):
            self.G = np.where(gc > 0, gs / gc, 0.0)
        self.W = W

    def predict(self, name: str, o: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Forecasts [n, L] for origins ``o`` at 0-based leads ``k`` (target o+k)."""
        if name == "persistence":
            return self.lastfin[o]
        if name.startswith("seasonal_naive"):
            s = config.DAILY_STEPS if name.endswith("day") else config.WEEKLY_STEPS
            src = o - s + k
            fb = self.lastfin[o]
            ok = (k < s) & (src >= 0)
            v = np.where(ok[:, None], self.vf[np.clip(src, 0, None)], np.nan)
            return np.where(np.isfinite(v), v, fb)
        if name == "historical_average":
            tgt = o + k
            b = self.bin[tgt]
            last = o - 1
            # last history index t' < o sharing the target's weekly bin
            tp = last - ((self.bin[last] - b) % self.W)
            okp = tp >= 0
            tpc = np.clip(tp, 0, None)
            S, C = self.S[tpc], self.C[tpc]
            with np.errstate(invalid="ignore", divide="ignore"):
                prof = np.where(okp[:, None] & (C > 0), S / C, np.nan)
            return np.where(np.isfinite(prof), prof, self.G[o])
        raise KeyError(name)


# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
def _alloc(rc: RunCtx, mode: str, name: str):
    rc.out.mkdir(parents=True, exist_ok=True)
    p = rc.out / f"pred_{mode}_{name}.tmp.npy"
    arr = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32,
                                    shape=(rc.L, len(rc.hsteps), rc.n_t))
    arr[:] = np.nan
    return arr, p


def _finish(arr, tmp: Path):
    arr.flush()
    del arr
    tmp.replace(tmp.with_name(tmp.name.replace(".tmp.npy", ".npy")))


def _predict_baseline(rc: RunCtx, name: str, eng: _BaselineEngine):
    for mode in MODES:
        arr, tmp = _alloc(rc, mode, name)
        for hi, h in enumerate(rc.hsteps):
            o = rc.rolling_origin(h) if mode == "rolling" else rc.block_origin(h)
            t = np.arange(rc.lo, rc.T)
            ok = o >= 0
            k = t - o
            pred = np.full((rc.n_t, rc.L), np.nan)
            pred[ok] = eng.predict(name, o[ok], k[ok])
            arr[:, hi, :] = np.clip(pred, 0, None).T.astype(np.float32)
        _finish(arr, tmp)


def _predict_gbdt(rc: RunCtx, m, tag: str = "", stride: int = 1):
    arr, tmp = _alloc(rc, "rolling", m.name + tag)
    origins = np.arange(rc.lo, rc.T, stride)             # every (s-th) origin in the test
    for j in range(rc.L):
        X = m._rows(origins - 1, j)                      # anchor = last observed step
        ok = np.isfinite(X).all(axis=1)
        for hi, h in enumerate(rc.hsteps):
            if h not in m.models:
                continue
            p = np.full(origins.size, np.nan)
            if ok.any():
                p[ok] = np.clip(np.expm1(m.models[h].predict(X[ok])), 0, None)
            tgt = origins + h - 1                        # target index
            keep = tgt < rc.T
            arr[j, hi, tgt[keep] - rc.lo] = p[keep]
        if (j + 1) % 50 == 0:
            print(f"  [{m.name}] {j + 1}/{rc.L} links", flush=True)
    _finish(arr, tmp)


SMOOTH_LABEL = {3: "15m", 6: "30m", 12: "1h", 24: "2h", 36: "3h", 72: "6h"}


def _smooth_name(name: str, w: int) -> str:
    return f"{name}+ma{SMOOTH_LABEL.get(w, f'{w}s')}"


def _path_ma(torch, path, w: int):
    """Centred w-step moving average along each forecast path [B, max_h], the
    window truncated at the path's ends (lead 0 averages leads 0..w//2). Every
    value used comes from the same origin's path — no extra information."""
    B, H = path.shape
    c = torch.nn.functional.pad(torch.nan_to_num(path).cumsum(1), (1, 0))
    i = torch.arange(H, device=path.device)
    lo = (i - (w - 1) // 2).clamp(min=0)
    hi = (i + w // 2).clamp(max=H - 1)
    out = (c[:, hi + 1] - c[:, lo]) / (hi - lo + 1).to(path.dtype)
    out[~torch.isfinite(path)] = float("nan")
    return out


def _predict_torch(rc: RunCtx, m, batch: int, tag: str = "", smooth=(),
                   skip_raw: bool = False, tf32: bool = False, stride: int = 1):
    from . import torch_common as tc
    torch = tc._torch()
    # Full fp32 by default: _torch() enables TF32, which large batched matmuls/convs
    # then use (the harness's batch-of-1 predict effectively doesn't). With TF32 the
    # dense log drifted ~3e-4 relative from the scored records; without it, ~1e-6.
    # --tf32 trades that fidelity for speed (experiment variants with no records).
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    dev = next(m.net.parameters()).device
    lb, max_h = m.lookback, m.max_h
    LV = torch.tensor(tc.log1p(rc.values_filled), device=dev)   # [T, L]
    use_lid = m.use_link_embedding
    use_cal = m.use_calendar and m._cal is not None
    hs = rc.hsteps
    # one output per variant: the raw path and each post-smoothed path
    variants = ([] if skip_raw else [(m.name + tag, 0)]) + \
               [(_smooth_name(m.name + tag, w), int(w)) for w in smooth]
    outs = {v: (_alloc(rc, "rolling", v), _alloc(rc, "block", v)) for v, _ in variants}
    # stride > 1: forecast only from origins lo, lo+s, ... Block logs stay exact and
    # complete when every horizon is a multiple of s (window starts are multiples of
    # H); rolling logs then hold every s-th target per horizon.
    if any(H % stride for H in hs):
        raise SystemExit(f"--origin-stride {stride} must divide every horizon {hs}")
    origins = np.arange(rc.lo, rc.T, stride)
    origins = origins[origins >= lb]
    ar = torch.arange(lb, device=dev)
    t0 = time.time()
    m.net.eval()
    for j in range(rc.L):
        bufs = {v: (np.full((len(hs), rc.n_t), np.nan, np.float32),
                    np.full((len(hs), rc.n_t), np.nan, np.float32)) for v, _ in variants}
        col = LV[:, j]
        for b0 in range(0, origins.size, batch):
            ob = origins[b0:b0 + batch]
            o_t = torch.as_tensor(ob, device=dev)
            win = col[(o_t[:, None] - lb) + ar[None, :]]           # [B, lb]
            fin = torch.isfinite(win).all(dim=1)
            mu = win.mean(dim=1, keepdim=True)
            sd = win.std(dim=1, unbiased=False, keepdim=True).clamp_min(0.05)
            xn = torch.nan_to_num((win - mu) / sd)
            with torch.no_grad():
                if use_lid:
                    lid = torch.full((ob.size,), j, dtype=torch.int64, device=dev)
                    yn = m.net(xn, lid, o_t) if use_cal else m.net(xn, lid)
                else:
                    yn = m.net(xn)
                path0 = torch.expm1(yn * sd + mu).clamp_min(0)      # [B, max_h]
                path0[~fin] = float("nan")
            sel = torch.as_tensor([h - 1 for h in hs], device=dev)
            for v, w in variants:
                path = path0 if w <= 1 else _path_ma(torch, path0, w)
                roll, blk = bufs[v]
                # rolling: lead h-1 of every origin -> target o + h - 1
                pr = path.index_select(1, sel).cpu().numpy().T      # [H, B]
                for hi, h in enumerate(hs):
                    tgt = ob + h - 1
                    keep = tgt < rc.T
                    roll[hi, tgt[keep] - rc.lo] = pr[hi, keep]
                # block: window starts (o - lo) % H == 0 -> whole path[:H]
                for hi, H in enumerate(hs):
                    starts = np.nonzero((ob - rc.lo) % H == 0)[0]
                    if starts.size == 0:
                        continue
                    seg = path[torch.as_tensor(starts, device=dev), :H].cpu().numpy()
                    tgt = (ob[starts] - rc.lo)[:, None] + np.arange(H)[None, :]
                    keep = tgt < rc.n_t                   # last window may run off the end
                    blk[hi, tgt[keep]] = seg[keep]
        for v, _ in variants:
            outs[v][0][0][j] = bufs[v][0]
            outs[v][1][0][j] = bufs[v][1]
        if (j + 1) % 25 == 0 or j + 1 == rc.L:
            el = time.time() - t0
            print(f"  [{m.name}] {j + 1}/{rc.L} links  {el:.0f}s "
                  f"(~{el / (j + 1) * (rc.L - j - 1):.0f}s left)", flush=True)
    for v, _ in variants:
        for arr, tmp in outs[v]:
            _finish(arr, tmp)


def _write_meta(rc: RunCtx):
    done = sorted({p.stem.split("_", 2)[2] for p in rc.out.glob("pred_*_*.npy")
                   if not p.name.endswith(".tmp.npy")})
    meta = {
        "run": str(rc.run), "links": rc.panel.links, "devices": rc.panel.devices,
        "horizons": [{"name": n, "steps": s} for n, s in zip(rc.hnames, rc.hsteps)],
        "lo": int(rc.lo), "T": int(rc.T), "n_targets": int(rc.n_t),
        "t0": int(rc.panel.timestamps[rc.lo]), "step": config.STEP_SECONDS,
        "test_start": rc.cfg["test_start"], "models": done,
        "modes": {m: sorted(p.stem.split("_", 2)[2] for p in rc.out.glob(f"pred_{m}_*.npy")
                            if not p.name.endswith(".tmp.npy")) for m in MODES},
    }
    (rc.out / "meta.json").write_text(json.dumps(meta, indent=1))


def cmd_predict(args):
    rc = RunCtx(args.run, args.out)
    names = ([n.strip() for n in args.models.split(",")] if args.models else None)
    avail = {p.stem.split(".")[0] for p in (rc.run / "records").glob("*.csv")}
    names = sorted(avail) if names is None else names
    print(f"[dense] {rc.L} links x {rc.n_t} targets ({rc.n_t * config.STEP_SECONDS / 86400:.0f} d) "
          f"x horizons {rc.hsteps} -> {rc.out}")
    eng = None
    for name in names:
        t0 = time.time()
        if name in BASELINE_NAMES:
            eng = eng or _BaselineEngine(rc)
            _predict_baseline(rc, name, eng)
        else:
            (m,) = rc.models({name})
            smooth = [int(w) for w in args.smooth.split(",")] if args.smooth else []
            if hasattr(m, "net"):
                _predict_torch(rc, m, args.batch, tag=args.tag, smooth=smooth,
                               skip_raw=args.skip_raw, tf32=args.tf32,
                               stride=args.origin_stride)
            else:
                _predict_gbdt(rc, m, tag=args.tag, stride=args.origin_stride)
        print(f"[dense] {name} done in {time.time() - t0:.0f}s", flush=True)
        _write_meta(rc)


# --------------------------------------------------------------------------- #
# Verify against the harness's sampled records
# --------------------------------------------------------------------------- #
def cmd_verify(args):
    rc = RunCtx(args.run, args.out)
    meta = json.loads((rc.out / "meta.json").read_text())
    link_idx = {l: i for i, l in enumerate(meta["links"])}
    hidx = {h["steps"]: i for i, h in enumerate(meta["horizons"])}
    tag = getattr(args, "tag", "") or ""
    for name in meta["modes"]["rolling"]:
        if tag and not name.endswith(tag):
            continue                      # --tag: check only that run's variant logs
        if not tag and ("@" in name or "+" in name):
            continue
        rec = rc.run / "records" / f"{name[:len(name) - len(tag)] if tag else name}.csv"
        if not rec.exists():
            continue
        r = pd.read_csv(rec, usecols=["link", "horizon", "origin", "y_pred"])
        arr = np.load(rc.out / f"pred_rolling_{name}.npy", mmap_mode="r")
        j = r.link.map(link_idx).to_numpy()
        hi = r.horizon.map(hidx).to_numpy()
        t = r.origin.to_numpy() + r.horizon.to_numpy() - 1 - rc.lo
        d = np.asarray(arr[j, hi, t], dtype=np.float64)
        y = r.y_pred.to_numpy()
        ok = np.isfinite(d) & np.isfinite(y)
        rel = np.abs(d[ok] - y[ok]) / np.maximum(np.abs(y[ok]), 1.0)
        print(f"  {name:20s} n={ok.sum():7d} missing={int((~np.isfinite(d)).sum()):5d} "
              f"max rel diff={rel.max():.2e}  p99={np.quantile(rel, .99):.2e}")


# --------------------------------------------------------------------------- #
# Metrics over the dense logs (same definitions as harness.BacktestResult)
# --------------------------------------------------------------------------- #
def cmd_metrics(args):
    rc = RunCtx(args.run, args.out)
    meta = json.loads((rc.out / "meta.json").read_text())
    tiers = {}
    for p in (rc.run / "records").glob("*.csv"):
        tiers[p.stem] = int(pd.read_csv(p, usecols=["tier"], nrows=1).tier.iloc[0])
    Y = rc.values_raw[rc.lo:].T.astype(np.float64)               # [L, n_t]
    tr = rc.values_raw[:rc.split.train[1]]
    scale = []
    for j in range(rc.L):
        yt = tr[:, j][np.isfinite(tr[:, j])]
        scale.append(np.mean(np.abs(np.diff(yt))) if yt.size > 1 else np.nan)
    scale = np.asarray(scale)
    vol = np.nanmean(np.abs(Y), axis=1)
    links = meta["links"]
    base_name = lambda n: n.split("@")[0].split("+")[0]
    # clock-hour / UTC-day bins of the target grid, for volume-aggregated WAPE
    tsec = rc.panel.timestamps[rc.lo:]
    bins = {"1h": (tsec // 3600) - tsec[0] // 3600, "1d": (tsec // 86400) - tsec[0] // 86400}
    bin_len = {"1h": 3600 // config.STEP_SECONDS, "1d": 86400 // config.STEP_SECONDS}

    def agg_wape(p, ok, key):
        """WAPE of bin totals: sum_b |sum_t F - sum_t A| / sum_b sum_t A, over bins
        whose every step is scored (a partial bin would compare unlike totals)."""
        b = bins[key]
        starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])   # bins are contiguous
        cnt = np.add.reduceat(ok.astype(np.float64), starts, axis=1)
        F = np.add.reduceat(np.where(ok, p, 0.0), starts, axis=1)
        A = np.add.reduceat(np.where(ok, Y, 0.0), starts, axis=1)
        # a strided (every s-th origin) rolling log scores bin_len/s steps per bin;
        # F and A always sum the same scored points, so "full" = every scored slot
        s_ = 1 if (ok[:, 1:] & ok[:, :-1]).any() else 3
        full = cnt >= bin_len[key] // s_
        num = np.where(full, np.abs(F - A), 0.0)
        den = np.where(full, np.abs(A), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            per = np.where(den.sum(1) > M.EPS, 100 * num.sum(1) / den.sum(1), np.nan)
        return per, num.sum(), den.sum()

    def roughness(p, ok):
        """mean |dF| / mean |dA| over consecutive scored steps: 1 = as jagged as
        the traffic, < 1 = smoother than the traffic. Lag 1 (5 min) when adjacent
        targets are scored, else lag 3 (a stride-3 rolling log); the lag used is
        returned so tables can say which."""
        for lag in (1, 3):
            both = ok[:, lag:] & ok[:, :-lag]
            if both.any():
                break
        dF = np.where(both, np.abs(p[:, lag:] - p[:, :-lag]), 0.0).sum(1)
        dA = np.where(both, np.abs(Y[:, lag:] - Y[:, :-lag]), 0.0).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(dA > M.EPS, dF / dA, np.nan), dF.sum(), dA.sum(), lag
    for mode in MODES:
        names = _select(meta["modes"][mode], getattr(args, "models", None))
        if not names:
            continue
        od = (Path(args.tables_dir) if getattr(args, "tables_dir", None) else rc.out) / mode
        od.mkdir(parents=True, exist_ok=True)
        per_rows, agg_rows, pk_rows = [], [], []
        common = {}
        if getattr(args, "common", False):
            # targets every model of this mode forecast, per horizon (per link)
            for hi in range(len(rc.hsteps)):
                cm = np.isfinite(Y).copy()
                for name in names:
                    P = np.load(rc.out / f"pred_{mode}_{name}.npy", mmap_mode="r")
                    cm &= np.isfinite(np.asarray(P[:, hi, :]))
                common[hi] = cm
                print(f"  [{mode}] {rc.hnames[hi]}: common targets/link "
                      f"{cm.sum(1).mean():.0f}", flush=True)
        for name in names:
            P = np.load(rc.out / f"pred_{mode}_{name}.npy", mmap_mode="r")
            for hi, (hn, h) in enumerate(zip(rc.hnames, rc.hsteps)):
                p = np.asarray(P[:, hi, :], dtype=np.float64)
                ok = np.isfinite(p) & np.isfinite(Y)
                if common:
                    ok &= common[hi]
                e = np.where(ok, p - Y, 0.0)
                ae, ay = np.abs(e), np.where(ok, np.abs(Y), 0.0)
                n = ok.sum(axis=1)
                den = np.abs(Y) + np.abs(p)
                with np.errstate(invalid="ignore", divide="ignore"):
                    sm = np.where(ok, np.abs(e) / np.where(den < M.EPS, np.nan, den), 0.0)
                    mae = ae.sum(1) / n
                    rmse = np.sqrt((e ** 2).sum(1) / n)
                    wape = np.where(ay.sum(1) > M.EPS, 100 * ae.sum(1) / ay.sum(1), np.nan)
                    smape = 200 * sm.sum(1) / n
                    mase = np.where(scale > M.EPS, mae / scale, np.nan)
                w1h, n1h, d1h = agg_wape(p, ok, "1h")
                w1d, n1d, d1d = agg_wape(p, ok, "1d")
                rgh, rF, rA, rlag = roughness(p, ok)
                for j in range(rc.L):
                    if n[j] == 0:
                        continue
                    per_rows.append({"model": name, "tier": tiers.get(base_name(name), 0),
                                     "horizon": h, "horizon_name": hn, "link": links[j],
                                     "mae": mae[j], "rmse": rmse[j], "smape": smape[j],
                                     "wape": wape[j], "wape_1h": w1h[j], "wape_1d": w1d[j],
                                     "rough": rgh[j], "mase": mase[j], "n": int(n[j])})
                N = ok.sum()
                with np.errstate(invalid="ignore", divide="ignore"):
                    agg_rows.append({
                        "model": name, "tier": tiers.get(base_name(name), 0), "horizon": h,
                        "horizon_name": hn, "mae": ae.sum() / N,
                        "rmse": np.sqrt((e ** 2).sum() / N),
                        "smape": 200 * sm.sum() / N, "wape": 100 * ae.sum() / ay.sum(),
                        "wape_1h": 100 * n1h / d1h if d1h > 0 else np.nan,
                        "wape_1d": 100 * n1d / d1d if d1d > 0 else np.nan,
                        "rough": rF / rA if rA > 0 else np.nan, "rough_lag": rlag,
                        "n": int(N)})
                # peaks: per-link q90 of the scored actuals
                yq = np.where(ok, Y, np.nan)
                with np.errstate(all="ignore"):
                    thr = np.nanquantile(yq, 0.9, axis=1)
                pk = ok & (Y >= thr[:, None])
                off = ok & ~pk
                pe = np.where(pk, e, 0.0)
                pk_rows.append({
                    "model": name, "tier": tiers.get(base_name(name), 0), "horizon": h,
                    "horizon_name": hn,
                    "peak_bias_pct": 100 * pe.sum() / np.where(pk, Y, 0).sum(),
                    "peak_underpred_rate": float((pe < 0).sum() / max(pk.sum(), 1)),
                    "peak_mae": float(np.abs(pe).sum() / max(pk.sum(), 1)),
                    "peak_wape": 100 * np.abs(pe).sum() / max(np.where(pk, Y, 0).sum(), 1e-9),
                    "off_bias_pct": 100 * np.where(off, e, 0).sum()
                                    / max(np.where(off, Y, 0).sum(), 1e-9),
                    "n_peak": int(pk.sum())})
            print(f"  [{mode}] {name}", flush=True)
        pl = pd.DataFrame(per_rows)
        agg = pd.DataFrame(agg_rows).sort_values(["horizon", "mae"]).reset_index(drop=True)
        pl.to_csv(od / "per_link.csv", index=False)
        agg.to_csv(od / "aggregate.csv", index=False)
        pd.DataFrame(pk_rows).to_csv(od / "peaks.csv", index=False)
        # ratio vs persistence + WAPE tables (harness definitions)
        base = pl[pl.model == "persistence"].set_index(["horizon", "link"])["mae"]
        w_all = pd.Series(np.log10(np.maximum(np.nan_to_num(vol, nan=1.0), 1.0)) + 0.01,
                          index=links)
        rrows, wrows = [], []
        for (model, tier, h, hn), g in pl.groupby(["model", "tier", "horizon", "horizon_name"]):
            gw = g.wape.to_numpy(dtype=float)
            gw = gw[np.isfinite(gw)]
            a = agg[(agg.model == model) & (agg.horizon == h)]
            wrows.append({"model": model, "tier": tier, "horizon": h, "horizon_name": hn,
                          "pooled_wape": float(a.wape.iloc[0]) if len(a) else np.nan,
                          "median_wape": float(np.median(gw)) if gw.size else np.nan,
                          "p90_wape": float(np.quantile(gw, .9)) if gw.size else np.nan,
                          "n_links": int(gw.size)})
            if h not in base.index.get_level_values(0):
                continue
            b = base.loc[h]
            g = g.set_index("link")
            common = [l for l in g.index if l in b.index and np.isfinite(b[l])
                      and b[l] > 1e-3 and np.isfinite(g.mae[l])]
            if not common:
                continue
            r = (g.mae.loc[common] / b.loc[common]).astype(float)
            w = w_all.reindex(common).fillna(0.01)
            wl = g.wape.reindex(common)
            wl = wl[np.isfinite(wl)]
            rrows.append({"model": model, "tier": tier, "horizon": h, "horizon_name": hn,
                          "median_ratio": float(r.median()),
                          "geomean_ratio": float(np.exp(np.log(r.clip(lower=1e-9)).mean())),
                          "logv_ratio": float((r.clip(upper=20.0) * w).sum() / w.sum()),
                          "median_wape": float(wl.median()) if len(wl) else np.nan,
                          "share_beats_base": float((r < 1.0).mean()),
                          "n_links": len(common)})
        (pd.DataFrame(rrows).sort_values(["horizon", "logv_ratio"])
         .to_csv(od / "ratio_vs_persistence_mae.csv", index=False))
        (pd.DataFrame(wrows).sort_values(["horizon", "median_wape"])
         .to_csv(od / "wape.csv", index=False))
        _print_summary(mode, od, rc.hnames)


def _select(names, spec):
    """--models filter: None = all, "untagged" = the run's own models (no @/+ variant
    tags), else a comma-separated list of exact names."""
    if not spec:
        return list(names)
    if spec == "untagged":
        return [n for n in names if "@" not in n and "+" not in n]
    want = {x.strip() for x in spec.split(",") if x.strip()}
    return [n for n in names if n in want]


def _print_summary(mode, od: Path, hnames):
    r = pd.read_csv(od / "ratio_vs_persistence_mae.csv")
    w = pd.read_csv(od / "wape.csv")
    print(f"\n===== dense [{mode}] per-link median MAE ratio vs persistence =====")
    print(r.pivot_table(index="model", columns="horizon_name", values="median_ratio")
          [hnames].round(3).to_string())
    print(f"\n===== dense [{mode}] WAPE % pooled / median per-link =====")
    pw = w.pivot_table(index="model", columns="horizon_name",
                       values=["pooled_wape", "median_wape"])
    pw = pw.reindex(columns=pd.MultiIndex.from_product(
        [["pooled_wape", "median_wape"], hnames]))
    print(pw.round(1).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("predict", "verify", "metrics"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True, help="finished gr-benchmark run directory")
        s.add_argument("--out", default=None, help="default: <run>/dense")
        if name == "verify":
            s.add_argument("--tag", default="", help="check the logs carrying this tag")
        if name == "predict":
            s.add_argument("--models", default=None,
                           help="comma-separated (default: every model with records)")
            s.add_argument("--batch", type=int, default=2048,
                           help="origins per GPU forward pass")
            s.add_argument("--tag", default="",
                           help="suffix for the logged model names, e.g. '@month' "
                                "(log a variant run into another run's --out)")
            s.add_argument("--smooth", default="",
                           help="also log post-smoothed paths: comma-separated "
                                "centred moving-average widths in steps, e.g. 3,12,36")
            s.add_argument("--skip-raw", action="store_true",
                           help="with --smooth: don't (re)write the raw path log")
            s.add_argument("--tf32", action="store_true",
                           help="allow TF32 (faster; ~3e-4 relative drift)")
            s.add_argument("--origin-stride", type=int, default=1,
                           help="forecast from every s-th origin (block logs exact "
                                "when s divides every horizon; rolling every s-th target)")
        if name == "metrics":
            s.add_argument("--common", action="store_true",
                           help="score every model of a mode on the points ALL of them "
                                "forecast (identical targets across full and strided logs)")
            s.add_argument("--models", default=None,
                           help="score only these: 'untagged' (the run's own models, no "
                                "@/+ variants) or a comma-separated list (default: all)")
            s.add_argument("--tables-dir", default=None,
                           help="write the per-mode tables under this directory instead "
                                "of the dense dir (which the viewer reads)")
    args = ap.parse_args()
    {"predict": cmd_predict, "verify": cmd_verify, "metrics": cmd_metrics}[args.cmd](args)


if __name__ == "__main__":
    main()
