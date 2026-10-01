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

    python -m prediction.dense predict --run ../artifacts/pred_year/merged \
        [--models tcn,gru] [--batch 2048]
    python -m prediction.dense verify  --run ../artifacts/pred_year/merged
    python -m prediction.dense metrics --run ../artifacts/pred_year/merged
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import config, splits as splits_mod

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


def _predict_gbdt(rc: RunCtx, m):
    arr, tmp = _alloc(rc, "rolling", m.name)
    origins = np.arange(rc.lo, rc.T)                     # every origin in the test
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


def _predict_torch(rc: RunCtx, m, batch: int):
    from . import torch_common as tc
    torch = tc._torch()
    # Full fp32: _torch() enables TF32, which large batched matmuls/convs then use
    # (the harness's batch-of-1 predict effectively doesn't). With TF32 the dense
    # log drifted ~3e-4 relative from the scored records; without it, ~1e-6.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = next(m.net.parameters()).device
    lb, max_h = m.lookback, m.max_h
    LV = torch.tensor(tc.log1p(rc.values_filled), device=dev)   # [T, L]
    use_lid = m.use_link_embedding
    use_cal = m.use_calendar and m._cal is not None
    hs = rc.hsteps
    arr_r, tmp_r = _alloc(rc, "rolling", m.name)
    arr_b, tmp_b = _alloc(rc, "block", m.name)
    origins = np.arange(max(rc.lo, lb), rc.T)
    ar = torch.arange(lb, device=dev)
    t0 = time.time()
    m.net.eval()
    for j in range(rc.L):
        roll = np.full((len(hs), rc.n_t), np.nan, np.float32)
        blk = np.full((len(hs), rc.n_t), np.nan, np.float32)
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
                path = torch.expm1(yn * sd + mu).clamp_min(0)       # [B, max_h]
                path[~fin] = float("nan")
            # rolling: lead h-1 of every origin -> target o + h - 1
            sel = torch.as_tensor([h - 1 for h in hs], device=dev)
            pr = path.index_select(1, sel).cpu().numpy().T          # [H, B]
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
                keep = tgt < rc.n_t                       # last window may run off the end
                blk[hi, tgt[keep]] = seg[keep]
        arr_r[j] = roll
        arr_b[j] = blk
        if (j + 1) % 25 == 0 or j + 1 == rc.L:
            el = time.time() - t0
            print(f"  [{m.name}] {j + 1}/{rc.L} links  {el:.0f}s "
                  f"(~{el / (j + 1) * (rc.L - j - 1):.0f}s left)", flush=True)
    _finish(arr_r, tmp_r)
    _finish(arr_b, tmp_b)


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
            if hasattr(m, "net"):
                _predict_torch(rc, m, args.batch)
            else:
                _predict_gbdt(rc, m)
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
    for name in meta["modes"]["rolling"]:
        rec = rc.run / "records" / f"{name}.csv"
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
    for mode in MODES:
        names = meta["modes"][mode]
        od = rc.out / mode
        od.mkdir(exist_ok=True)
        per_rows, agg_rows, pk_rows = [], [], []
        for name in names:
            P = np.load(rc.out / f"pred_{mode}_{name}.npy", mmap_mode="r")
            for hi, (hn, h) in enumerate(zip(rc.hnames, rc.hsteps)):
                p = np.asarray(P[:, hi, :], dtype=np.float64)
                ok = np.isfinite(p) & np.isfinite(Y)
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
                for j in range(rc.L):
                    if n[j] == 0:
                        continue
                    per_rows.append({"model": name, "tier": tiers.get(name, 0),
                                     "horizon": h, "horizon_name": hn, "link": links[j],
                                     "mae": mae[j], "rmse": rmse[j], "smape": smape[j],
                                     "wape": wape[j], "mase": mase[j], "n": int(n[j])})
                N = ok.sum()
                with np.errstate(invalid="ignore", divide="ignore"):
                    agg_rows.append({
                        "model": name, "tier": tiers.get(name, 0), "horizon": h,
                        "horizon_name": hn, "mae": ae.sum() / N,
                        "rmse": np.sqrt((e ** 2).sum() / N),
                        "smape": 200 * sm.sum() / N, "wape": 100 * ae.sum() / ay.sum(),
                        "n": int(N)})
                # peaks: per-link q90 of the scored actuals
                yq = np.where(ok, Y, np.nan)
                with np.errstate(all="ignore"):
                    thr = np.nanquantile(yq, 0.9, axis=1)
                pk = ok & (Y >= thr[:, None])
                off = ok & ~pk
                pe = np.where(pk, e, 0.0)
                pk_rows.append({
                    "model": name, "tier": tiers.get(name, 0), "horizon": h,
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
        s.add_argument("--run", required=True, help="finished prediction.run directory")
        s.add_argument("--out", default=None, help="default: <run>/dense")
        if name == "predict":
            s.add_argument("--models", default=None,
                           help="comma-separated (default: every model with records)")
            s.add_argument("--batch", type=int, default=2048,
                           help="origins per GPU forward pass")
    args = ap.parse_args()
    {"predict": cmd_predict, "verify": cmd_verify, "metrics": cmd_metrics}[args.cmd](args)


if __name__ == "__main__":
    main()
