"""Build the experiment report (HTML, inline SVG charts) from results/ and the dense logs.

usage: python build_report.py [out.html]
       (default $GR_ARTIFACTS_DIR/exp/report/month-window-smoothing.html)

Reads $GR_ARTIFACTS_DIR/exp/results/ (analyze.py, cost.py) and the year run's dense
logs; the page text lives in report_template.html next to this script.
"""
from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from green_routing.paths import ARTIFACTS_DIR
from green_routing.prediction.data import load_run_panel

HERE = Path(__file__).resolve().parent
EXP = ARTIFACTS_DIR / "exp"
RES = EXP / "results"
RUN = ARTIFACTS_DIR / "pred_year" / "merged"
DENSE = RUN / "dense"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "report" / "month-window-smoothing.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

HN = ["15min", "1h", "8h", "day", "week"]
HLAB = {"15min": "15 min", "1h": "1 h", "8h": "8 h", "day": "1 day", "week": "1 week"}
HSHORT = {"15min": "15m", "1h": "1h", "8h": "8h", "day": "1d", "week": "1w"}
MODEL_LAB = {"gbdt": "GBDT", "gru": "GRU", "tcn": "TCN", "dlinear": "DLinear",
             "patchtst": "PatchTST"}
# categorical slots, fixed order (reference palette, validated adjacent order)
SLOT = {"tcn": 1, "gbdt": 2, "gru": 3, "patchtst": 4, "dlinear": 5}
VAR_LAB = {"": "raw", "+ma15m": "+ 15 min MA", "+ma1h": "+ 1 h MA", "+ma3h": "+ 3 h MA",
           "@starget": "1 h target"}
HOLIDAYS = {"2026-04-03": "Good Friday", "2026-04-06": "Easter Monday",
            "2026-05-01": "Labour Day", "2026-05-14": "Ascension",
            "2026-05-25": "Whit Monday", "2026-06-04": "Corpus Christi"}

pairs = pd.read_csv(RES / "paired_tests.csv")
summ = pd.read_csv(RES / "summary_by_model.csv")
het = pd.read_csv(RES / "e1_heterogeneity.csv") if (RES / "e1_heterogeneity.csv").exists() else pd.DataFrame()
summary_json = json.loads((RES / "summary.json").read_text())
cost = json.loads((RES / "cost.json").read_text()) if (RES / "cost.json").exists() else {}
meta = json.loads((DENSE / "meta.json").read_text())

esc = html.escape


def fmt_pct(v, d=1, sign=True):
    if v is None or not np.isfinite(v):
        return "—"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "−") + "%"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "—"
    if p < 1e-4:
        e = int(math.floor(math.log10(p)))
        return f"10<sup>{e}</sup>".replace("-", "−") if p > 0 else "0"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def pair(exp, mode, base, var, h):
    r = pairs[(pairs.exp == exp) & (pairs["mode"] == mode) & (pairs.base == base)
              & (pairs.variant == var) & (pairs.horizon == h)]
    return None if r.empty else r.iloc[0]


def srow(mode, model, h):
    r = summ[(summ["mode"] == mode) & (summ.model == model) & (summ.horizon == h)]
    return None if r.empty else r.iloc[0]


# --------------------------------------------------------------------------- #
# SVG helpers (all colours via CSS custom properties -> both themes)
# --------------------------------------------------------------------------- #
def signed(t):
    """Signed tick label with a true minus sign: +5, 0, −5."""
    return ("0" if abs(t) < 1e-12 else f"{t:+g}").replace("-", "−")


def tip_attrs(text):
    return f' tabindex="0" data-tip="{esc(text)}"'


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    t0 = math.ceil(lo / step) * step
    out, t = [], t0
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


# --- Fig 1: E1 small multiples ------------------------------------------------
def fig_e1(mode="rolling"):
    models = [("gbdt", "gbdt", "gbdt@month"), ("gru", "gru", "gru@month"),
              ("tcn", "tcn@wk20", "tcn@month"), ("dlinear", "dlinear", "dlinear@month"),
              ("patchtst", "patchtst", "patchtst@month")]
    vals = []
    for key, b, v in models:
        for h in HN:
            r = pair("E1", mode, b, v, h)
            if r is not None:
                vals.append(r.median_change_pct)
    if not vals:
        return "<p class='pending'>Month-window results are still computing.</p>"
    lim = max(4.0, math.ceil(max(abs(x) for x in vals) / 2) * 2)
    W, H = 176, 210
    pad_l, pad_r, pad_t, pad_b = 34, 8, 30, 34
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    y = lambda v: pad_t + ph / 2 - v / lim * ph / 2
    ticks = nice_ticks(-lim, lim, 4)
    panels = []
    for key, b, v in models:
        bw = pw / len(HN)
        parts = [f'<svg viewBox="0 0 {W} {H}" class="mini" role="img" '
                 f'aria-label="{esc(MODEL_LAB[key])}: per-link MAE change, 1 month vs 1 week">']
        title = MODEL_LAB[key] + (" · 20 epochs" if key == "tcn" else "")
        parts.append(f'<text x="{pad_l}" y="16" class="pt">{esc(title)}</text>')
        for t in ticks:
            parts.append(f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{y(t):.1f}" y2="{y(t):.1f}" '
                         f'class="{"base" if t == 0 else "grid"}"/>')
            parts.append(f'<text x="{pad_l - 5}" y="{y(t) + 3.5:.1f}" class="tick" '
                         f'text-anchor="end">{signed(t)}</text>')
        for i, h in enumerate(HN):
            r = pair("E1", mode, b, v, h)
            cx = pad_l + bw * (i + 0.5)
            parts.append(f'<text x="{cx:.1f}" y="{H - 14}" class="tick" text-anchor="middle">'
                         f'{HSHORT[h]}</text>')
            if r is None:
                parts.append(f'<text x="{cx:.1f}" y="{y(0) - 6:.1f}" class="tick" '
                             f'text-anchor="middle">…</text>')
                continue
            d, sig = r.median_change_pct, r.wilcoxon_p < 0.01
            y0, y1 = y(0), y(max(-lim, min(lim, d)))
            top, hgt = min(y0, y1), max(abs(y1 - y0), 1.0)
            cls = ("better" if d < 0 else "worse") + ("" if sig else " ns")
            bwid = min(18, bw - 8)
            tip = (f"{MODEL_LAB[key]} {HLAB[h]}: per-link MAE {fmt_pct(d)} with 1-month input; "
                   f"{100 * r.share_improved:.0f}% of {r.n_links} links improved; "
                   f"Wilcoxon p = {r.wilcoxon_p:.1e}")
            # rounded data end, square at the baseline
            rr = min(4, hgt / 2)
            if d < 0:   # bar grows downward from baseline: round the bottom
                path = (f"M{cx - bwid / 2:.1f},{top:.1f} h{bwid:.1f} v{hgt - rr:.1f} "
                        f"q0,{rr:.1f} -{rr:.1f},{rr:.1f} h-{bwid - 2 * rr:.1f} "
                        f"q-{rr:.1f},0 -{rr:.1f},-{rr:.1f} z")
            else:
                path = (f"M{cx - bwid / 2:.1f},{top + hgt:.1f} v-{hgt - rr:.1f} "
                        f"q0,-{rr:.1f} {rr:.1f},-{rr:.1f} h{bwid - 2 * rr:.1f} "
                        f"q{rr:.1f},0 {rr:.1f},{rr:.1f} v{hgt - rr:.1f} z")
            parts.append(f'<path d="{path}" class="bar {cls}"{tip_attrs(tip)}/>')
            ly = (top + hgt + 11) if d < 0 else (top - 4)
            lab = fmt_pct(d, 1).replace("%", "")
            parts.append(f'<text x="{cx:.1f}" y="{ly:.1f}" class="val" text-anchor="middle">'
                         f'{lab}{"" if sig else "°"}</text>')
        parts.append("</svg>")
        panels.append("".join(parts))
    return '<div class="multiples">' + "".join(panels) + "</div>"


# --- Fig 2: daily difference over the test period ------------------------------
def fig_daily(key):
    ser = summary_json.get(key)
    if not ser:
        return ""
    d = pd.DataFrame(ser)
    d["diff"] = d.var_wape - d.base_wape
    W, H = 760, 230
    pad_l, pad_r, pad_t, pad_b = 46, 12, 18, 34
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    lim = max(1.0, float(np.ceil(np.abs(d["diff"]).max())))
    y = lambda v: pad_t + ph / 2 - v / lim * ph / 2
    n = len(d)
    bw = pw / n
    parts = [f'<svg viewBox="0 0 {W} {H}" class="wide" role="img" '
             f'aria-label="Daily pooled WAPE difference, 1-month minus 1-week input">']
    for t in nice_ticks(-lim, lim, 4):
        parts.append(f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{y(t):.1f}" y2="{y(t):.1f}" '
                     f'class="{"base" if t == 0 else "grid"}"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y(t) + 3.5:.1f}" class="tick" text-anchor="end">'
                     f'{signed(t)}</text>')
    for i, r in d.iterrows():
        x = pad_l + i * bw
        hol = HOLIDAYS.get(r.date)
        if hol:
            parts.append(f'<rect x="{x:.1f}" y="{pad_t}" width="{bw:.1f}" height="{ph}" class="hol"/>')
        v = r["diff"]
        y0, y1 = y(0), y(v)
        parts.append(f'<rect x="{x + 0.6:.1f}" y="{min(y0, y1):.1f}" width="{max(bw - 1.2, 0.8):.1f}" '
                     f'height="{max(abs(y1 - y0), 0.8):.1f}" class="bar {"better" if v < 0 else "worse"}"'
                     + tip_attrs(f"{r.date}{' (' + hol + ')' if hol else ''}: WAPE {r.base_wape:.1f}% → "
                                 f"{r.var_wape:.1f}% ({v:+.2f} pp)") + "/>")
    for i, r in d.iterrows():
        if r.date.endswith("-01") or i == 0:
            x = pad_l + i * bw
            lab = pd.Timestamp(r.date).strftime("%b %-d")
            parts.append(f'<text x="{x:.1f}" y="{H - 14}" class="tick">{lab}</text>')
    short = {"2026-04-03": "Easter", "2026-05-01": "Labour Day", "2026-05-14": "Ascension",
             "2026-05-25": "Whit Mon", "2026-06-04": "Corpus Christi"}
    for date, name in short.items():
        idx = d.index[d.date == date]
        if len(idx):
            # Whit Monday and Corpus Christi are 10 days apart: anchor them outward
            anchor = {"2026-05-25": "end", "2026-06-04": "start"}.get(date, "middle")
            x = pad_l + idx[0] * bw + {"2026-04-03": bw * 2, "2026-05-25": bw,
                                       "2026-06-04": 0}.get(date, bw / 2)
            parts.append(f'<text x="{x:.1f}" y="{pad_t - 5}" class="tick hl" '
                         f'text-anchor="{anchor}">{esc(name)}</text>')
    parts.append(f'<text x="{pad_l}" y="{H - 2}" class="tick">percentage points of pooled WAPE '
                 f'(below 0 = month window better that day)</text>')
    parts.append("</svg>")
    return "".join(parts)


# --- Fig 3: smoothing trade-off ----------------------------------------------
def fig_tradeoff(mode, h):
    W, H = 380, 270
    pad_l, pad_r, pad_t, pad_b = 48, 14, 26, 40
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    pts, labels = [], []
    for m in ("tcn", "gru", "patchtst", "dlinear"):
        base = srow(mode, m, h)
        if base is None:
            continue
        seq = []
        for suf in ("", "+ma15m", "+ma1h", "+ma3h", "@starget"):
            s = srow(mode, m + suf, h)
            p = pair("E2a" if suf.startswith("+") else "E2b", mode, m, m + suf, h) if suf else None
            if s is None or (suf and p is None):
                continue
            seq.append((suf, float(s.rough), 0.0 if not suf else float(p.median_change_pct), s, p))
        pts.append((m, seq))
    allx = [x for _, s in pts for _, x, _, _, _ in s]
    ally = [v for _, s in pts for _, _, v, _, _ in s]
    if len(allx) < 3:
        return "<p class='pending'>Smoothing results are still computing.</p>"
    x0, x1 = 0.0, max(allx) * 1.12
    YCAP = 4.5                                   # outliers drawn at the top edge, labelled
    ylo, yhi = min(ally + [0]) - 0.6, min(max(ally + [0]) + 0.6, YCAP)
    X = lambda v: pad_l + (v - x0) / (x1 - x0) * pw
    Y = lambda v: pad_t + (yhi - v) / (yhi - ylo) * ph
    parts = [f'<svg viewBox="0 0 {W} {H}" class="trade" role="img" '
             f'aria-label="Smoothing trade-off at {HLAB[h]}: roughness vs MAE change">']
    parts.append(f'<text x="{pad_l}" y="15" class="pt">{HLAB[h]} ahead</text>')
    for t in nice_ticks(ylo, yhi, 5):
        parts.append(f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" '
                     f'class="{"base" if abs(t) < 1e-9 else "grid"}"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{Y(t) + 3.5:.1f}" class="tick" text-anchor="end">'
                     f'{signed(t)}</text>')
    for t in nice_ticks(x0, x1, 5):
        parts.append(f'<text x="{X(t):.1f}" y="{H - 22}" class="tick" text-anchor="middle">{t:g}</text>')
    parts.append(f'<text x="{pad_l + pw / 2:.1f}" y="{H - 6}" class="tick" text-anchor="middle">'
                 f'roughness (forecast ÷ traffic step change)</text>')
    for m, seq in pts:
        line = [(x, v) for suf, x, v, _, _ in seq if suf != "@starget"]
        if len(line) > 1:
            d = "M" + " L".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in line)
            parts.append(f'<path d="{d}" class="ln s{SLOT[m]}"/>')
        for suf, x, v, s, p in seq:
            tip = (f"{MODEL_LAB[m]} {VAR_LAB[suf]} · {HLAB[h]}: roughness {x:.2f}, "
                   + ("baseline" if not suf else
                      f"MAE {fmt_pct(v)} vs raw ({100 * p.share_improved:.0f}% of links improved)")
                   + f"; WAPE {s.pooled_wape:.1f}%, hourly WAPE {s.pooled_wape_1h:.1f}%, "
                     f"peak bias {s.peak_bias_pct:+.1f}%")
            if suf == "@starget":
                vy = min(v, yhi)
                parts.append(f'<rect x="{X(x) - 5:.1f}" y="{Y(vy) - 5:.1f}" width="10" height="10" '
                             f'class="sq s{SLOT[m]}"{tip_attrs(tip)}/>')
                if v > yhi:
                    parts.append(f'<text x="{X(x) + 9:.1f}" y="{Y(vy) + 4:.1f}" class="val">'
                                 f'↑ {fmt_pct(v)}</text>')
            else:
                parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="{5 if not suf else 4}" '
                             f'class="dot s{SLOT[m]}{" hollow" if not suf else ""}"{tip_attrs(tip)}/>')
        raw = [(x, v) for suf, x, v, _, _ in seq if not suf]
        if raw:
            labels.append([X(raw[0][0]), Y(raw[0][1]), MODEL_LAB[m]])
    # direct labels at each raw point; push a label below the line when it would
    # overlap a neighbour's, and anchor it leftward near the right edge
    placed = []
    for lx, ly, name in sorted(labels):
        w = 7.0 * len(name)
        end = lx + 8 + w > W - pad_r
        tx = lx - 8 if end else lx + 8
        box = (tx - w if end else tx, tx if end else tx + w)
        ty = ly - 8
        for (a, b, py) in placed:
            if box[0] < b and box[1] > a and abs(py - ty) < 13:
                ty = ly + 17
        placed.append((box[0], box[1], ty))
        anchor = ' text-anchor="end"' if end else ""
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" class="lab"{anchor}>{name}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --- Fig 4: one link, raw vs smoothed ----------------------------------------
def fig_trace(link, model, h_i, t_from, n_steps, mode="block"):
    lo = meta["lo"]
    j = meta["links"].index(link)
    panel = load_run_panel(RUN)                 # the run's links, in the logs' order
    ts = np.asarray(panel.timestamps)
    i0 = int(np.searchsorted(ts, pd.Timestamp(t_from, tz="UTC").timestamp())) - lo
    sl = slice(i0, i0 + n_steps)
    act = np.asarray(panel.values[lo:, j], float)[sl]
    series = [("actual", act, "act")]
    for suf, cls in (("", "s1"), ("+ma1h", "s3"), ("@starget", "s4")):
        f = DENSE / f"pred_{mode}_{model}{suf}.npy"
        if f.exists():
            series.append((VAR_LAB[suf], np.asarray(np.load(f, mmap_mode="r")[j, h_i, sl], float), cls))
    W, H = 760, 260
    pad_l, pad_r, pad_t, pad_b = 54, 120, 14, 30
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    allv = np.concatenate([s[1][np.isfinite(s[1])] for s in series])
    ylo, yhi = 0.0, float(np.nanmax(allv)) * 1.05
    X = lambda i: pad_l + i / (n_steps - 1) * pw
    Y = lambda v: pad_t + (yhi - v) / (yhi - ylo) * ph
    parts = [f'<svg viewBox="0 0 {W} {H}" class="wide" role="img" aria-label="Forecast traces">']
    for t in nice_ticks(ylo, yhi, 4):
        parts.append(f'<line x1="{pad_l}" x2="{pad_l + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{Y(t) + 3.5:.1f}" class="tick" text-anchor="end">'
                     f'{t:,.0f}</text>')
    H_steps = [3, 12, 96, 288, 2016][h_i]
    for k in range(0, n_steps, H_steps):
        tt = ts[lo + i0 + k]
        if (tt - ts[lo]) % (H_steps * 300) == 0:
            parts.append(f'<line x1="{X(k):.1f}" x2="{X(k):.1f}" y1="{pad_t}" y2="{pad_t + ph}" class="seam"/>')
    for k in range(0, n_steps, 72):
        tt = pd.Timestamp(int(ts[lo + i0 + k]), unit="s", tz="UTC").tz_convert("Europe/Zurich")
        parts.append(f'<text x="{X(k):.1f}" y="{H - 10}" class="tick" text-anchor="middle">'
                     f'{tt.strftime("%a %H:%M")}</text>')
    ends = []
    for name, v, cls in series:
        segs, cur = [], []
        for i, x in enumerate(v):
            if np.isfinite(x):
                cur.append(f"{X(i):.1f},{Y(x):.1f}")
            elif cur:
                segs.append(cur); cur = []
        if cur:
            segs.append(cur)
        d = " ".join("M" + " L".join(s) for s in segs)
        parts.append(f'<path d="{d}" class="ln {cls}"/>')
        fin = np.flatnonzero(np.isfinite(v))
        if fin.size:
            ends.append((Y(v[fin[-1]]), name, cls))
    ends.sort()
    last = -1e9
    for yv, name, cls in ends:          # end labels, nudged apart only by a few px
        yv = max(yv, last + 13)
        parts.append(f'<line x1="{pad_l + pw + 3}" x2="{pad_l + pw + 12}" y1="{yv - 3:.1f}" '
                     f'y2="{yv - 3:.1f}" class="ln {cls}"/>')
        parts.append(f'<text x="{pad_l + pw + 16}" y="{yv:.1f}" class="lab">{esc(name)}</text>')
        last = yv
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def table_e1():
    rows = []
    for key, b, v in [("gbdt", "gbdt", "gbdt@month"), ("gru", "gru", "gru@month"),
                      ("tcn", "tcn@wk20", "tcn@month"), ("dlinear", "dlinear", "dlinear@month"),
                      ("patchtst", "patchtst", "patchtst@month")]:
        for h in HN:
            r, rb = pair("E1", "rolling", b, v, h), pair("E1", "block", b, v, h)
            a, c = srow("rolling", b, h), srow("rolling", v, h)
            if r is None:
                continue
            rows.append(
                f"<tr><td>{MODEL_LAB[key]}{' (20 ep)' if key == 'tcn' else ''}</td><td>{HLAB[h]}</td>"
                f"<td>{a.median_ratio:.3f}</td><td>{c.median_ratio:.3f}</td>"
                f"<td class='{'g' if r.median_change_pct < 0 else 'b'}'>{fmt_pct(r.median_change_pct)}</td>"
                f"<td>{100 * r.share_improved:.0f}%</td><td>{fmt_p(r.wilcoxon_p)}</td>"
                f"<td>{a.pooled_wape:.1f} → {c.pooled_wape:.1f}</td>"
                f"<td>{fmt_pct(rb.median_change_pct) if rb is not None else '—'}</td></tr>")
    if not rows:
        return ""
    return ("<div class='tw'><table><thead><tr><th>Model</th><th>Horizon</th>"
            "<th>Ratio vs persistence, week</th><th>…month</th><th>Per-link MAE change</th>"
            "<th>Links improved</th><th>Wilcoxon p</th><th>Pooled WAPE % (week → month)</th>"
            "<th>Change, block mode</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def table_e2(mode, hs=("8h", "day")):
    rows = []
    for m in ("tcn", "gru", "patchtst", "dlinear"):
        for h in hs:
            for suf in ("", "+ma15m", "+ma1h", "+ma3h", "@starget"):
                s = srow(mode, m + suf, h)
                if s is None:
                    continue
                p = None if not suf else pair("E2a" if suf.startswith("+") else "E2b", mode, m, m + suf, h)
                chg = "" if p is None else fmt_pct(p.median_change_pct)
                cls = "" if p is None else ("g" if p.median_change_pct < 0 else "b")
                rows.append(
                    f"<tr class='{'first' if not suf else ''}'><td>{MODEL_LAB[m] if not suf else ''}</td>"
                    f"<td>{HLAB[h] if not suf else ''}</td><td>{VAR_LAB[suf]}</td>"
                    f"<td class='{cls}'>{chg or 'baseline'}</td>"
                    f"<td>{'' if p is None else f'{100 * p.share_improved:.0f}%'}</td>"
                    f"<td>{s.pooled_wape:.1f}</td><td>{s.pooled_wape_1h:.1f}</td><td>{s.pooled_wape_1d:.1f}</td>"
                    f"<td>{s.peak_bias_pct:+.1f}</td><td>{s.rough:.2f}</td></tr>".replace("+-", "−"))
    if not rows:
        return ""
    return ("<div class='tw'><table><thead><tr><th>Model</th><th>Horizon</th><th>Variant</th>"
            "<th>Per-link MAE change</th><th>Links improved</th><th>WAPE % (5 min)</th>"
            "<th>WAPE % (hourly volume)</th><th>WAPE % (daily volume)</th><th>Peak bias %</th>"
            "<th>Roughness</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def table_het():
    if het.empty:
        return ""
    rows = []
    for _, r in het[het.horizon == "day"].iterrows():
        key = r.variant.split("@")[0]
        rows.append(f"<tr><td>{MODEL_LAB[key]}</td>"
                    + "".join(f"<td>{fmt_pct(r[f'weekly_{q}_median_change_pct'])}</td>"
                              for q in ("low", "mid", "high"))
                    + f"<td>{r.rho_acf_week:+.2f}</td><td>{r.rho_drift:+.2f}</td>"
                      f"<td>{r.rho_mean_mbps:+.2f}</td></tr>".replace("+-", "−"))
    return ("<div class='tw'><table><thead><tr><th>Model (1 day)</th>"
            "<th>Weak weekly pattern</th><th>Medium</th><th>Strong weekly pattern</th>"
            "<th>ρ with weekly autocorrelation</th><th>ρ with level drift</th>"
            "<th>ρ with mean traffic</th></tr></thead><tbody>" + "".join(rows)
            + "</tbody></table></div>")


def table_cost():
    if not cost:
        return ""
    rows = []
    for m in ("gru", "tcn", "dlinear", "patchtst"):
        w, mo = cost.get(f"{m}@week"), cost.get(f"{m}@month")
        if not w or not mo:
            continue
        rows.append(f"<tr><td>{MODEL_LAB[m]}</td><td>{w['params_m']:.2f} → {mo['params_m']:.2f}</td>"
                    f"<td>{w['ms_step']:.0f} → {mo['ms_step']:.0f} (×{mo['ms_step'] / w['ms_step']:.1f})</td>"
                    f"<td>{w['gib']:.1f} → {mo['gib']:.1f}</td><td>{w['epochs']} → {mo['epochs']}</td></tr>")
    return ("<div class='tw'><table><thead><tr><th>Model</th><th>Parameters (M)</th>"
            "<th>Train step, ms (batch 85)</th><th>Peak GPU memory, GiB</th><th>Epochs run</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def headline():
    """Numbers quoted in the prose, pulled from the data."""
    out = {}
    for key, b, v in [("gbdt", "gbdt", "gbdt@month"), ("gru", "gru", "gru@month"),
                      ("tcn", "tcn@wk20", "tcn@month"), ("dlinear", "dlinear", "dlinear@month"),
                      ("patchtst", "patchtst", "patchtst@month")]:
        for h in HN:
            r = pair("E1", "rolling", b, v, h)
            if r is not None:
                out[f"e1_{key}_{h}"] = r
    return out


TEMPLATE = (HERE / "report_template.html").read_text()


def main():
    figs = {
        "FIG_E1": fig_e1("rolling"),
        "FIG_DAILY": fig_daily(summary_json.get("daily_key", "")),
        "FIG_T8": fig_tradeoff("block", "8h"),
        "FIG_TDAY": fig_tradeoff("block", "day"),
        "TAB_E1": table_e1(),
        "TAB_E2_BLOCK": table_e2("block"),
        "TAB_E2_ROLL": table_e2("rolling"),
        "TAB_HET": table_het(),
        "TAB_COST": table_cost(),
    }
    tr = summary_json.get("trace")
    figs["FIG_TRACE"] = fig_trace(**tr) if tr else ""
    page = TEMPLATE
    for k, v in figs.items():
        page = page.replace("{{" + k + "}}", v)
    for k, v in summary_json.get("prose", {}).items():
        page = page.replace("{{" + k + "}}", str(v))
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
