#!/usr/bin/env python3
"""Prediction viewer — inspect a ``gr-benchmark`` run directory link by link.

Three views of the same fitted models (``?mode=``):

  block     window by window (default when ``<run>/dense`` exists): for horizon H
            the test block is cut into consecutive H-step windows, each forecast
            in full at 5-min resolution from its start — a prediction at every
            5-min step. Metrics average over the window's leads 1..H.
  rolling   fixed lead at every (s-th) 5-min origin: the forecast for t made exactly H
            steps earlier. Same metric definition as the run's horizon-H scores.
  sampled   the harness's own records (``records/*.csv``): one point per scored
            horizon at the run's sampled origins (~18 h apart on pred_year).

block/rolling come from ``gr-dense`` (``<run>/dense/pred_<mode>_<model>.npy``,
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

Access: HTTP Basic auth on every request (the data is SWITCH network telemetry and a
cloud box may have a public IP). Any user name is accepted; the password is
$GR_VIEWER_PASSWORD, else the contents of $GR_VIEWER_PASSWORD_FILE (default
~/.config/green-routing/viewer_password, generated on first start, mode 600).
GR_VIEWER_PASSWORD=off disables auth — only with --host 127.0.0.1 behind an SSH tunnel.

Run:  gr-viewer [run_dir] [--port 8138] [--host 0.0.0.0]
      (default run_dir: $GR_ARTIFACTS_DIR/pred_year/merged)
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hmac
import json
import os
import secrets
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

from ..paths import ARTIFACTS_DIR

STATIC_DIR = Path(__file__).resolve().parent / "static"
STEP = 300              # 5-min sampling
LEAD_IN = 288           # extra real context drawn before the first scored target
BASELINES = {"persistence", "seasonal_naive_day", "seasonal_naive_week",
             "historical_average"}
_HNAME = {1: "5min", 3: "15min", 6: "30min", 12: "1h", 96: "8h", 288: "day", 2016: "week"}


def _log(msg: str) -> None:
    print(f"[viewer] {msg}", flush=True)


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


# --------------------------------------------------------------------------- #
# One run, loaded once at start-up
# --------------------------------------------------------------------------- #
class RunData:
    """Everything the API serves for one run directory: the panel (real series),
    the sampled records, the dense logs and the metric tables."""

    def __init__(self, run: Path):
        self.run = Path(run).resolve()
        _log(f"run dir: {self.run}")
        self.config = json.loads((self.run / "config.json").read_text())
        self.links: list[str] = self.config["links"]
        self.link_idx = {name: i for i, name in enumerate(self.links)}
        self._load_panel()
        self._load_records()
        self._load_dense()
        self.per_link = {m: self._csv(m, "per_link.csv") for m in self.modes}
        # per-link traffic summary over the scored (test) span
        test0 = int(min(v[0].min() for v in self.index.values()))
        mean = np.nanmean(self.values[test0:], axis=0)
        peak = np.nanmax(self.values[test0:], axis=0)
        self.link_meta = [
            {"id": name, "device": self.devices[i],
             "mean_mbps": None if not np.isfinite(mean[i]) else round(float(mean[i]), 2),
             "peak_mbps": None if not np.isfinite(peak[i]) else round(float(peak[i]), 2)}
            for i, name in enumerate(self.links)
        ]
        self.leaderboards = {m: self._leaderboard(m) for m in self.modes}
        self.linkscores = {m: self._linkscores(m) for m in self.modes}
        _log("leaderboards + per-link scores ready")

    # -- loading ------------------------------------------------------------ #
    def _find_panel(self) -> tuple[Path, np.ndarray]:
        """The panel cache that produced this run: same T, and its links a superset
        of the run's (a --min-mean-mbps / --connected-links run keeps a subset of the
        cached panel's columns). Searched from the run dir upwards and in the
        artifacts dir. Returns the path and the run's column indices."""
        want = set(self.links)
        dirs = [self.run.parent, *list(self.run.parent.parents)[:3], ARTIFACTS_DIR]
        for d in dict.fromkeys(dirs):
            for p in sorted(d.glob("panel_*.npz")):
                with np.load(p, allow_pickle=True) as z:
                    if z["values"].shape[0] != self.config["n_steps"]:
                        continue
                    plinks = [str(x) for x in z["links"]]
                if want <= set(plinks):
                    pos = {name: i for i, name in enumerate(plinks)}
                    return p, np.array([pos[name] for name in self.links])
        raise SystemExit(f"no panel_*.npz in {[str(d) for d in dirs]} matching this run "
                         f"({self.config['n_steps']} steps, {self.config['n_links']} links)")

    def _load_panel(self):
        self.panel_path, cols = self._find_panel()
        _log(f"panel: {self.panel_path} ({len(cols)} of its links)")
        with np.load(self.panel_path, allow_pickle=True) as d:
            self.values = np.asarray(d["values"], dtype=np.float32)[:, cols]  # [T, L]
            self.ts = np.asarray(d["timestamps"], dtype=np.int64)            # [T] unix s
            self.devices = [str(x) for x in np.asarray(d["devices"])[cols]]

    def _load_records(self):
        _log("reading sampled records ...")
        recs = pd.concat([pd.read_csv(p) for p in sorted((self.run / "records").glob("*.csv"))],
                         ignore_index=True)
        self.tier = dict(recs[["model", "tier"]].drop_duplicates().itertuples(index=False))
        self.horizon_steps = sorted(int(h) for h in recs["horizon"].unique())
        _log(f"{len(recs):,} records | {len(self.tier)} models | horizons {self.horizon_steps}")
        # (link, horizon, model) -> (origins, y_true, y_pred), sliced from one sorted frame
        recs = recs.sort_values(["link", "horizon", "model", "origin"], kind="stable")
        keys = list(zip(recs["link"].to_numpy(), recs["horizon"].to_numpy(),
                        recs["model"].to_numpy()))
        orig = recs["origin"].to_numpy()
        yt = recs["y_true"].to_numpy(dtype=np.float64)
        yp = recs["y_pred"].to_numpy(dtype=np.float64)
        self.index: dict[tuple, tuple] = {}
        start = 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[start]:
                self.index[keys[start]] = (orig[start:i], yt[start:i], yp[start:i])
                start = i
        self.horizons = [{"steps": h, "name": _HNAME.get(h, f"{h}steps")}
                         for h in self.horizon_steps]
        self.hpos = {h: i for i, h in enumerate(self.horizon_steps)}

    def _load_dense(self):
        self.dense_dir = self.run / "dense"
        self.dense: dict[str, dict[str, np.ndarray]] = {}
        self.dense_meta = None
        if (self.dense_dir / "meta.json").exists():
            self.dense_meta = m = json.loads((self.dense_dir / "meta.json").read_text())
            if m["links"] != self.links:
                raise SystemExit("dense/meta.json links differ from the run's")
            if [h["steps"] for h in m["horizons"]] != self.horizon_steps:
                raise SystemExit("dense horizons differ from the run's")
            for mode, names in m["modes"].items():
                self.dense[mode] = {
                    n: np.load(self.dense_dir / f"pred_{mode}_{n}.npy", mmap_mode="r")
                    for n in names}
            self.d_lo, self.d_nt = m["lo"], m["n_targets"]
            _log("dense logs: " + ", ".join(f"{k}: {len(v)} models"
                                            for k, v in self.dense.items()))
        self.modes = [m for m in ("block", "rolling") if m in self.dense] + ["sampled"]

    def _tables_dir(self, mode: str) -> Path:
        return self.run if mode == "sampled" else self.dense_dir / mode

    def _csv(self, mode: str, name: str) -> pd.DataFrame | None:
        p = self._tables_dir(mode) / name
        return pd.read_csv(p) if p.exists() else None

    def _models_sorted(self, names) -> list[dict]:
        return sorted(({"model": n, "tier": self.tier.get(n, 0)} for n in names),
                      key=lambda m: (m["tier"], m["model"]))

    def _mase(self, mode: str, link: str, h: int, model: str):
        pl = self.per_link.get(mode)
        if pl is None or "mase" not in pl:
            return None
        r = pl[(pl.link == link) & (pl.horizon == h) & (pl.model == model)]
        return None if r.empty or not np.isfinite(r.mase.iloc[0]) else float(r.mase.iloc[0])

    # -- API ---------------------------------------------------------------- #
    def meta(self) -> dict:
        models = {m: self._models_sorted(self.dense[m]) for m in self.dense}
        models["sampled"] = self._models_sorted(self.tier)
        c = self.config
        return {
            "run": str(self.run),
            "panel": self.panel_path.name,
            "config": {k: c[k] for k in
                       ("start", "end", "train_end", "val_start", "val_end",
                        "test_start", "value_col", "min_coverage", "lookback",
                        "epochs", "max_origins", "tiers", "horizons", "metric",
                        "min_mean_mbps", "calendar", "under_weight",
                        "n_links", "n_steps", "data_source") if k in c},
            "links": self.link_meta,
            "modes": self.modes,
            "models": models,
            "horizons": self.horizons,
            "step_seconds": STEP,
            "dense": None if self.dense_meta is None else
                     {"t0": self.dense_meta["t0"], "n_targets": self.d_nt, "lo": self.d_lo},
        }

    def link_dense(self, link: str, hsteps: int, mode: str) -> dict:
        j, hi = self.link_idx[link], self.hpos[hsteps]
        a0 = max(self.d_lo - LEAD_IN, 0)
        real = self.values[a0:, j]
        yt = self.values[self.d_lo:, j].astype(np.float64)
        arrs = {n: np.asarray(a[j, hi], dtype=np.float64) for n, a in self.dense[mode].items()}
        naive = arrs.get("persistence")
        naive_mae = None
        if naive is not None:
            ok = np.isfinite(naive) & np.isfinite(yt)
            naive_mae = float(np.abs(naive[ok] - yt[ok]).mean()) if ok.any() else None
        models_out, stats_out = {}, []
        for name, yp in arrs.items():
            models_out[name] = {"t0": int(self.ts[self.d_lo]), "step": STEP, "b64": _b64(yp)}
            row = {"model": name, "tier": self.tier.get(name, 0)}
            row.update(_stats(yt, yp, naive_mae))
            row["mase"] = self._mase(mode, link, hsteps, name)
            stats_out.append(row)
        stats_out.sort(key=lambda r: (r.get("mae") is None, r.get("mae")))
        return {"link": link, "device": self.devices[j], "horizon": hsteps, "mode": mode,
                "window_steps": hsteps if mode == "block" else None,
                "lo_t": int(self.ts[self.d_lo]),
                "real": {"t0": int(self.ts[a0]), "step": STEP, "b64": _b64(real)},
                "models": models_out, "stats": stats_out}

    def link_sampled(self, link: str, hsteps: int) -> dict:
        j = self.link_idx[link]
        series: dict[str, dict] = {}
        origins_all: list[np.ndarray] = []
        for name in self.tier:
            key = (link, hsteps, name)
            if key not in self.index:
                continue
            o, yt, yp = self.index[key]
            origins_all.append(o)
            series[name] = {"o": o, "yt": yt, "yp": yp}
        if not series:
            return {"error": f"no records for {link} @ h={hsteps}"}
        # target time of a forecast made at origin o for horizon h: index o + h - 1
        lo = int(min(o.min() for o in origins_all))
        a0 = max(lo - LEAD_IN, 0)
        real = self.values[a0:, j]
        naive = series.get("persistence")
        naive_mae = (float(np.abs(naive["yp"] - naive["yt"]).mean())
                     if naive is not None else None)
        models_out, stats_out = {}, []
        for name, s in series.items():
            tgt = np.minimum(s["o"] + hsteps - 1, len(self.ts) - 1)
            models_out[name] = {"t": [int(self.ts[k]) for k in tgt],
                                "y_pred": _round(s["yp"], 3), "y_true": _round(s["yt"], 3)}
            row = {"model": name, "tier": self.tier.get(name, 0)}
            row.update(_stats(s["yt"], s["yp"], naive_mae))
            row["mase"] = self._mase("sampled", link, hsteps, name)
            stats_out.append(row)
        stats_out.sort(key=lambda r: (r.get("mae") is None, r.get("mae")))
        return {"link": link, "device": self.devices[j], "horizon": hsteps, "mode": "sampled",
                "window_steps": None, "lo_t": int(self.ts[lo]),
                "real": {"t0": int(self.ts[a0]), "step": STEP, "b64": _b64(real)},
                "models": models_out, "stats": stats_out}

    def link(self, link: str, hsteps: int, mode: str) -> dict:
        if link not in self.link_idx:
            return {"error": f"unknown link {link!r}"}
        if hsteps not in self.hpos:
            return {"error": f"unknown horizon {hsteps}"}
        if mode in self.dense:
            return self.link_dense(link, hsteps, mode)
        return self.link_sampled(link, hsteps)

    def _leaderboard(self, mode: str) -> dict:
        """model x horizon summary from the mode's metric tables."""
        pl, agg = self.per_link.get(mode), self._csv(mode, "aggregate.csv")
        pk = self._csv(mode, "peaks.csv")
        cells: dict = {}
        if pl is None or agg is None:
            return {"models": [], "horizons": self.horizons, "cells": cells}
        base = (pl[pl.model == "persistence"].set_index(["horizon", "link"])["mae"]
                if (pl.model == "persistence").any() else None)
        for (model, h), g in pl.groupby(["model", "horizon"]):
            a = agg[(agg.model == model) & (agg.horizon == h)]
            row = {"pooled_mae": float(a.mae.iloc[0]) if len(a) else None,
                   "pooled_wape": float(a.wape.iloc[0]) if len(a) else None,
                   "median_wape": (float(np.nanmedian(g.wape))
                                   if g.wape.notna().any() else None),
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
        return {"models": self._models_sorted(pl.model.unique()), "horizons": self.horizons,
                "cells": cells}

    def _linkscores(self, mode: str) -> dict:
        """h -> link -> {model: [wape, ratio_vs_persistence]} (rounded)."""
        out: dict = {}
        pl = self.per_link.get(mode)
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

    def tables(self, mode: str) -> dict:
        out = {}
        for name in ("aggregate", "skill_mae", "ratio_vs_persistence_mae", "wape", "peaks"):
            p = self._tables_dir(mode) / f"{name}.csv"
            if p.exists():
                out[name] = json.loads(pd.read_csv(p).to_json(orient="split", index=False))
        return out


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def load_password() -> str | None:
    env = (os.environ.get("GR_VIEWER_PASSWORD")
           or os.environ.get("PRED_VIEWER_PASSWORD") or "").strip()   # old name kept
    if env.lower() == "off":
        return None
    if env:
        return env
    f = Path(os.environ.get("GR_VIEWER_PASSWORD_FILE")
             or "~/.config/green-routing/viewer_password").expanduser()
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(secrets.token_urlsafe(12) + "\n")
        f.chmod(0o600)
        _log(f"generated a viewer password in {f}")
    return f.read_text().strip()


def make_handler(data: RunData, password: str | None):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(STATIC_DIR), **kw)

        def _authorized(self) -> bool:
            if password is None:
                return True
            h = self.headers.get("Authorization", "")
            if not h.startswith("Basic "):
                return False
            try:
                _, _, pw = base64.b64decode(h[6:]).decode("utf-8").partition(":")
            except Exception:
                return False
            return hmac.compare_digest(pw.encode(), password.encode())

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
            if u.path.rsplit("/", 1)[-1].startswith("."):
                self.send_error(404)
                return
            q = parse_qs(u.query)
            mode = (q.get("mode") or [data.modes[0]])[0]
            if mode not in data.modes:
                mode = data.modes[0]
            h = int((q.get("h") or [data.horizon_steps[0]])[0])
            try:
                if u.path == "/api/meta":
                    return self._json(data.meta())
                if u.path == "/api/tables":
                    return self._json(data.tables(mode))
                if u.path == "/api/leaderboard":
                    return self._json(data.leaderboards[mode])
                if u.path == "/api/linkscores":
                    return self._json(data.linkscores[mode].get(h, {}))
                if u.path == "/api/link":
                    payload = data.link((q.get("id") or [""])[0], h, mode)
                    return self._json(payload, 404 if "error" in payload else 200)
            except Exception as exc:              # a bad request must not kill the server
                import traceback
                traceback.print_exc()
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return super().do_GET()

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gr-viewer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=str(ARTIFACTS_DIR / "pred_year" / "merged"),
                    help="run directory (default: %(default)s)")
    ap.add_argument("port_pos", nargs="?", type=int, help=argparse.SUPPRESS)
    ap.add_argument("host_pos", nargs="?", help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, default=8138)
    ap.add_argument("--host", default="0.0.0.0",
                    help="interface to bind (127.0.0.1 = local only, e.g. behind an "
                         "SSH tunnel)")
    args = ap.parse_args(argv)
    port = args.port_pos or args.port              # old form: server.py RUN PORT HOST
    host = args.host_pos or args.host
    data = RunData(Path(args.run))
    password = load_password()
    srv = ThreadingHTTPServer((host, port), make_handler(data, password))
    _log(f"serving {data.run} on http://{host}:{port}  modes={data.modes}  "
         f"(auth {'OFF' if password is None else 'on'}; ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
