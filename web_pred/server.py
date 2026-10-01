#!/usr/bin/env python3
"""Prediction viewer backend — inspect a `prediction.run` output directory.

Three views of the same fitted models (``?mode=``):

  block     window by window (default when ``<run>/dense`` exists): for horizon H
            the test block is cut into consecutive H-step windows, each forecast
            in full at 5-min resolution from its start — a prediction at every
            5-min step. Metrics average over the window's leads 1..H.
  rolling   fixed lead at every 5-min origin: the forecast for t made exactly H
            steps earlier. Same metric definition as the run's horizon-H scores.
  sampled   the harness's own records (``records/*.csv``): one point per scored
            horizon at the run's sampled origins (~18 h apart on pred_year).

block/rolling come from ``prediction.dense`` (``<run>/dense/pred_<mode>_<model>.npy``,
float32 [L, H, n_targets], plus its metric tables in ``<run>/dense/<mode>/``).

Endpoints
  GET /api/meta                          run config, links, models per mode, horizons
  GET /api/link?id=&h=&mode=             real series + every model's forecasts +
                                         per-model stats for that link
  GET /api/leaderboard?mode=             model x horizon summary table
  GET /api/linkscores?h=&mode=           per link: each model's WAPE + MAE ratio
  GET /api/tables?mode=                  the raw metric CSVs as JSON

Dense series are sent as base64 little-endian float32 (NaN = no forecast).
WAPE everywhere is sum|e| / sum|y| in percent.

Access: HTTP Basic auth on every request (the data is SWITCH network telemetry and
the box has a public IP). The password comes from $PRED_VIEWER_PASSWORD, else from
web_pred/.viewer_password (generated on first start, mode 600); any user name is
accepted. PRED_VIEWER_PASSWORD=off disables auth (only behind an SSH tunnel).

Run:  python server.py [run_dir] [port] [host]
      (defaults ../artifacts/pred_year/merged 8138 0.0.0.0)
"""
from __future__ import annotations

import base64
import gzip
import hmac
import json
import os
import secrets
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1
       else (HERE / ".." / "artifacts" / "pred_year" / "merged").resolve())
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8138
HOST = sys.argv[3] if len(sys.argv) > 3 else "0.0.0.0"

STEP = 300              # 5-min sampling
LEAD_IN = 288           # extra real context drawn before the first scored target
BASELINES = {"persistence", "seasonal_naive_day", "seasonal_naive_week",
             "historical_average"}
_HNAME = {1: "5min", 3: "15min", 6: "30min", 12: "1h", 96: "8h", 288: "day", 2016: "week"}


def _log(msg: str) -> None:
    print(f"[viewer] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Load the run: panel (real series) + sampled records
# --------------------------------------------------------------------------- #
_log(f"run dir: {RUN}")
CONFIG = json.loads((RUN / "config.json").read_text())
LINKS: list[str] = CONFIG["links"]
LINK_IDX = {name: i for i, name in enumerate(LINKS)}


def _find_panel() -> tuple[Path, np.ndarray]:
    """The panel cache that produced this run: same T, and its links a superset of
    the run's (a --min-mean-mbps / --connected-links run keeps a subset of the
    cached panel's columns). Searched from the run dir upwards (a run may sit in a
    sub-folder of artifacts/). Returns the path and the run's column indices."""
    want = set(LINKS)
    dirs = [RUN.parent, *list(RUN.parent.parents)[:3]]
    for d in dirs:
        for p in sorted(d.glob("panel_*.npz")):
            with np.load(p, allow_pickle=True) as z:
                if z["values"].shape[0] != CONFIG["n_steps"]:
                    continue
                plinks = [str(x) for x in z["links"]]
            if want <= set(plinks):
                pos = {name: i for i, name in enumerate(plinks)}
                return p, np.array([pos[name] for name in LINKS])
    raise SystemExit(f"no panel_*.npz in {[str(d) for d in dirs]} matching this run "
                     f"({CONFIG['n_steps']} steps, {CONFIG['n_links']} links)")


_panel_path, _cols = _find_panel()
_log(f"panel: {_panel_path} ({len(_cols)} of its links)")
with np.load(_panel_path, allow_pickle=True) as _d:
    VALUES = np.asarray(_d["values"], dtype=np.float32)[:, _cols]   # [T, L] raw, NaN = gap
    TS = np.asarray(_d["timestamps"], dtype=np.int64)               # [T] unix seconds
    DEVICES = [str(x) for x in np.asarray(_d["devices"])[_cols]]
T_ALL = len(TS)

_log("reading sampled records ...")
_recs = pd.concat(
    [pd.read_csv(p) for p in sorted((RUN / "records").glob("*.csv"))],
    ignore_index=True,
)
TIER = dict(_recs[["model", "tier"]].drop_duplicates().itertuples(index=False))
HORIZON_STEPS = sorted(int(h) for h in _recs["horizon"].unique())
_log(f"{len(_recs):,} records | {len(TIER)} models | horizons {HORIZON_STEPS}")

# (link, horizon, model) -> (origins, y_true, y_pred), sliced from one sorted frame.
_recs = _recs.sort_values(["link", "horizon", "model", "origin"], kind="stable")
_keys = list(zip(_recs["link"].to_numpy(), _recs["horizon"].to_numpy(),
                 _recs["model"].to_numpy()))
_orig = _recs["origin"].to_numpy()
_yt = _recs["y_true"].to_numpy(dtype=np.float64)
_yp = _recs["y_pred"].to_numpy(dtype=np.float64)
INDEX: dict[tuple, tuple] = {}
_start = 0
for _i in range(1, len(_keys) + 1):
    if _i == len(_keys) or _keys[_i] != _keys[_start]:
        INDEX[_keys[_start]] = (_orig[_start:_i], _yt[_start:_i], _yp[_start:_i])
        _start = _i
del _recs, _keys, _orig, _yt, _yp

HORIZONS = [{"steps": h, "name": _HNAME.get(h, f"{h}steps")} for h in HORIZON_STEPS]
HPOS = {h: i for i, h in enumerate(HORIZON_STEPS)}


def _models_sorted(names) -> list[dict]:
    return sorted(({"model": n, "tier": TIER.get(n, 0)} for n in names),
                  key=lambda m: (m["tier"], m["model"]))


# --------------------------------------------------------------------------- #
# Dense logs (prediction.dense)
# --------------------------------------------------------------------------- #
DENSE_DIR = RUN / "dense"
DENSE: dict[str, dict[str, np.ndarray]] = {}
DENSE_META = None
if (DENSE_DIR / "meta.json").exists():
    DENSE_META = json.loads((DENSE_DIR / "meta.json").read_text())
    if DENSE_META["links"] != LINKS:
        raise SystemExit("dense/meta.json links differ from the run's")
    if [h["steps"] for h in DENSE_META["horizons"]] != HORIZON_STEPS:
        raise SystemExit("dense horizons differ from the run's")
    for mode, names in DENSE_META["modes"].items():
        DENSE[mode] = {n: np.load(DENSE_DIR / f"pred_{mode}_{n}.npy", mmap_mode="r")
                       for n in names}
    D_LO, D_NT = DENSE_META["lo"], DENSE_META["n_targets"]
    _log(f"dense logs: " + ", ".join(f"{m}: {sorted(v)}" for m, v in DENSE.items()))

MODES = [m for m in ("block", "rolling") if m in DENSE] + ["sampled"]


def _tables_dir(mode: str) -> Path:
    return RUN if mode == "sampled" else DENSE_DIR / mode


def _csv(mode: str, name: str) -> pd.DataFrame | None:
    p = _tables_dir(mode) / name
    return pd.read_csv(p) if p.exists() else None


PER_LINK = {m: _csv(m, "per_link.csv") for m in MODES}

# Per-link traffic summary over the scored (test) span.
_test0 = int(min(v[0].min() for v in INDEX.values()))
_mean = np.nanmean(VALUES[_test0:], axis=0)
_peak = np.nanmax(VALUES[_test0:], axis=0)
LINK_META = [
    {"id": name, "device": DEVICES[i],
     "mean_mbps": None if not np.isfinite(_mean[i]) else round(float(_mean[i]), 2),
     "peak_mbps": None if not np.isfinite(_peak[i]) else round(float(_peak[i]), 2)}
    for i, name in enumerate(LINKS)
]


def _round(a, nd=4):
    out = np.round(np.asarray(a, dtype=float), nd)
    return [None if not np.isfinite(v) else float(v) for v in out]


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def _stats(yt: np.ndarray, yp: np.ndarray, naive_mae: float | None) -> dict:
    """Metrics for one model on one link+horizon (also used for a zoom window)."""
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    if yt.size == 0:
        return {}
    err = yp - yt
    ae = np.abs(err)
    mae = float(ae.mean())
    denom = float(np.abs(yt).sum())
    thr = float(np.quantile(yt, 0.9)) if yt.size >= 4 else float(yt.max())
    pk = yt >= thr
    out = {
        "n": int(yt.size),
        "mae": mae,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": float(100 * ae.sum() / denom) if denom > 0 else None,
        "bias_pct": float(100 * err.sum() / denom) if denom > 0 else None,
        "peak_bias_pct": (float(100 * err[pk].sum() / yt[pk].sum())
                          if pk.any() and yt[pk].sum() > 0 else None),
        "peak_wape": (float(100 * ae[pk].sum() / yt[pk].sum())
                      if pk.any() and yt[pk].sum() > 0 else None),
        "underpred_rate": float((err < 0).mean()),
    }
    out["skill"] = (None if not naive_mae else float(1 - mae / naive_mae))
    return out


def _mase(mode: str, link: str, h: int, model: str):
    pl = PER_LINK.get(mode)
    if pl is None:
        return None
    r = pl[(pl.link == link) & (pl.horizon == h) & (pl.model == model)]
    return None if r.empty or not np.isfinite(r.mase.iloc[0]) else float(r.mase.iloc[0])


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def api_meta() -> dict:
    models = {m: _models_sorted(DENSE[m]) for m in DENSE}
    models["sampled"] = _models_sorted(TIER)
    return {
        "run": str(RUN),
        "panel": _panel_path.name,
        "config": {k: CONFIG[k] for k in
                   ("start", "end", "train_end", "val_start", "val_end",
                    "test_start", "value_col", "min_coverage", "lookback",
                    "epochs", "max_origins", "tiers", "horizons", "metric",
                    "min_mean_mbps", "calendar", "under_weight",
                    "n_links", "n_steps", "data_source") if k in CONFIG},
        "links": LINK_META,
        "modes": MODES,
        "models": models,
        "horizons": HORIZONS,
        "step_seconds": STEP,
        "dense": None if DENSE_META is None else
                 {"t0": DENSE_META["t0"], "n_targets": D_NT, "lo": D_LO},
    }


def api_link_dense(link: str, hsteps: int, mode: str) -> dict:
    j, hi = LINK_IDX[link], HPOS[hsteps]
    a0 = max(D_LO - LEAD_IN, 0)
    real = VALUES[a0:, j]
    yt = VALUES[D_LO:, j].astype(np.float64)
    arrs = {n: np.asarray(a[j, hi], dtype=np.float64) for n, a in DENSE[mode].items()}
    naive = arrs.get("persistence")
    naive_mae = None
    if naive is not None:
        ok = np.isfinite(naive) & np.isfinite(yt)
        naive_mae = float(np.abs(naive[ok] - yt[ok]).mean()) if ok.any() else None
    models_out, stats_out = {}, []
    for name, yp in arrs.items():
        models_out[name] = {"t0": int(TS[D_LO]), "step": STEP, "b64": _b64(yp)}
        row = {"model": name, "tier": TIER.get(name, 0)}
        row.update(_stats(yt, yp, naive_mae))
        row["mase"] = _mase(mode, link, hsteps, name)
        stats_out.append(row)
    stats_out.sort(key=lambda r: (r.get("mae") is None, r.get("mae")))
    return {"link": link, "device": DEVICES[j], "horizon": hsteps, "mode": mode,
            "window_steps": hsteps if mode == "block" else None,
            "lo_t": int(TS[D_LO]),
            "real": {"t0": int(TS[a0]), "step": STEP, "b64": _b64(real)},
            "models": models_out, "stats": stats_out}


def api_link_sampled(link: str, hsteps: int) -> dict:
    j = LINK_IDX[link]
    series: dict[str, dict] = {}
    origins_all: list[np.ndarray] = []
    for name in TIER:
        key = (link, hsteps, name)
        if key not in INDEX:
            continue
        o, yt, yp = INDEX[key]
        origins_all.append(o)
        series[name] = {"o": o, "yt": yt, "yp": yp}
    if not series:
        return {"error": f"no records for {link} @ h={hsteps}"}
    # Target time of a forecast made at origin o for horizon h: index o + h - 1.
    lo = int(min(o.min() for o in origins_all))
    a0 = max(lo - LEAD_IN, 0)
    real = VALUES[a0:, j]
    naive = series.get("persistence")
    naive_mae = (float(np.abs(naive["yp"] - naive["yt"]).mean())
                 if naive is not None else None)
    models_out, stats_out = {}, []
    for name, s in series.items():
        tgt = np.minimum(s["o"] + hsteps - 1, T_ALL - 1)
        models_out[name] = {"t": [int(TS[k]) for k in tgt],
                            "y_pred": _round(s["yp"], 3), "y_true": _round(s["yt"], 3)}
        row = {"model": name, "tier": TIER.get(name, 0)}
        row.update(_stats(s["yt"], s["yp"], naive_mae))
        row["mase"] = _mase("sampled", link, hsteps, name)
        stats_out.append(row)
    stats_out.sort(key=lambda r: (r.get("mae") is None, r.get("mae")))
    return {"link": link, "device": DEVICES[j], "horizon": hsteps, "mode": "sampled",
            "window_steps": None, "lo_t": int(TS[lo]),
            "real": {"t0": int(TS[a0]), "step": STEP, "b64": _b64(real)},
            "models": models_out, "stats": stats_out}


def api_link(link: str, hsteps: int, mode: str) -> dict:
    if link not in LINK_IDX:
        return {"error": f"unknown link {link!r}"}
    if hsteps not in HPOS:
        return {"error": f"unknown horizon {hsteps}"}
    if mode in DENSE:
        return api_link_dense(link, hsteps, mode)
    return api_link_sampled(link, hsteps)


def _leaderboard(mode: str) -> dict:
    """model x horizon summary from the mode's metric tables."""
    pl, agg, pk = PER_LINK.get(mode), _csv(mode, "aggregate.csv"), _csv(mode, "peaks.csv")
    cells: dict = {}
    if pl is None or agg is None:
        return {"models": [], "horizons": HORIZONS, "cells": cells}
    base = (pl[pl.model == "persistence"].set_index(["horizon", "link"])["mae"]
            if (pl.model == "persistence").any() else None)
    for (model, h), g in pl.groupby(["model", "horizon"]):
        a = agg[(agg.model == model) & (agg.horizon == h)]
        row = {"pooled_mae": float(a.mae.iloc[0]) if len(a) else None,
               "pooled_wape": float(a.wape.iloc[0]) if len(a) else None,
               "median_wape": float(np.nanmedian(g.wape)) if g.wape.notna().any() else None,
               "median_ratio": None, "share_beats": None}
        if base is not None:
            b = base.reindex(list(zip([h] * len(g), g.link))).to_numpy()
            ok = np.isfinite(b) & (b > 1e-3) & np.isfinite(g.mae.to_numpy())
            if ok.any():
                r = g.mae.to_numpy()[ok] / b[ok]
                row["median_ratio"] = float(np.median(r))
                row["share_beats"] = float((r < 1).mean())
        cells[f"{model}|{int(h)}"] = row
    if pk is not None:
        for _, r in pk.iterrows():
            k = f"{r.model}|{int(r.horizon)}"
            if k in cells:
                cells[k]["peak_bias_pct"] = float(r.peak_bias_pct)
                if "peak_wape" in r and pd.notna(r.peak_wape):
                    cells[k]["peak_wape"] = float(r.peak_wape)
    return {"models": _models_sorted(pl.model.unique()), "horizons": HORIZONS,
            "cells": cells}


def _linkscores(mode: str) -> dict:
    """h -> link -> {model: [wape, ratio_vs_persistence]} (rounded)."""
    out: dict = {}
    pl = PER_LINK.get(mode)
    if pl is None:
        return out
    for h, g in pl.groupby("horizon"):
        base = g[g.model == "persistence"].set_index("link")["mae"]
        d: dict = {}
        for r in g.itertuples(index=False):
            b = base.get(r.link, np.nan)
            ratio = r.mae / b if np.isfinite(b) and b > 1e-3 else np.nan
            d.setdefault(r.link, {})[r.model] = [
                None if not np.isfinite(r.wape) else round(float(r.wape), 2),
                None if not np.isfinite(ratio) else round(float(ratio), 4)]
        out[int(h)] = d
    return out


LEADERBOARD = {m: _leaderboard(m) for m in MODES}
LINKSCORES = {m: _linkscores(m) for m in MODES}
_log("leaderboards + per-link scores ready")


def api_tables(mode: str) -> dict:
    out = {}
    for name in ("aggregate", "skill_mae", "ratio_vs_persistence_mae", "wape", "peaks"):
        p = _tables_dir(mode) / f"{name}.csv"
        if p.exists():
            out[name] = json.loads(pd.read_csv(p).to_json(orient="split", index=False))
    return out


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _password() -> str | None:
    env = os.environ.get("PRED_VIEWER_PASSWORD", "").strip()
    if env.lower() == "off":
        return None
    if env:
        return env
    f = HERE / ".viewer_password"
    if not f.exists():
        f.write_text(secrets.token_urlsafe(12) + "\n")
        f.chmod(0o600)
    return f.read_text().strip()


PASSWORD = _password()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def _authorized(self) -> bool:
        if PASSWORD is None:
            return True
        h = self.headers.get("Authorization", "")
        if not h.startswith("Basic "):
            return False
        try:
            _, _, pw = base64.b64decode(h[6:]).decode("utf-8").partition(":")
        except Exception:
            return False
        return hmac.compare_digest(pw.encode(), PASSWORD.encode())

    def _deny(self):
        body = b"authentication required"
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="prediction viewer"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def list_directory(self, path):            # never expose a directory listing
        self.send_error(404)
        return None

    def log_message(self, fmt, *args):        # quieter access log
        if "/api/" in (self.path or ""):
            _log(f"{self.address_string()} {fmt % args}")

    def _json(self, payload, code=200):
        body = json.dumps(payload).encode()
        gz = "gzip" in (self.headers.get("Accept-Encoding") or "") and len(body) > 4096
        if gz:
            body = gzip.compress(body, compresslevel=5)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        if not self._authorized():
            return self._deny()
        return super().do_HEAD()

    def do_GET(self):
        if not self._authorized():
            return self._deny()
        u = urlparse(self.path)
        if u.path.rsplit("/", 1)[-1].startswith(".") or u.path.endswith(".py"):
            self.send_error(404)               # the password file, server source
            return
        q = parse_qs(u.query)
        mode = (q.get("mode") or [MODES[0]])[0]
        if mode not in MODES:
            mode = MODES[0]
        try:
            if u.path == "/api/meta":
                return self._json(api_meta())
            if u.path == "/api/tables":
                return self._json(api_tables(mode))
            if u.path == "/api/leaderboard":
                return self._json(LEADERBOARD[mode])
            if u.path == "/api/linkscores":
                h = int((q.get("h") or [HORIZON_STEPS[0]])[0])
                return self._json(LINKSCORES[mode].get(h, {}))
            if u.path == "/api/link":
                link = (q.get("id") or [""])[0]
                h = int((q.get("h") or [HORIZON_STEPS[0]])[0])
                payload = api_link(link, h, mode)
                return self._json(payload, 404 if "error" in payload else 200)
        except Exception as exc:              # a bad request must not kill the server
            import traceback
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        return super().do_GET()


if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    _log(f"serving {RUN} on http://{HOST}:{PORT}  modes={MODES}  "
         f"(auth {'OFF' if PASSWORD is None else 'on'}; ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _log("stopped")
