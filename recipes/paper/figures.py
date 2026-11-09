"""The evaluation's plots, as PDF, from the CSVs the other scripts of this directory write.

usage: python figures.py [OUTDIR] [--png=DIR]     default OUTDIR: <repo>/6aea077f0929c9a3a1ac5660/figures (the paper)
Inputs: $GR_ARTIFACTS_DIR/paper (facts.py, i2_rates.py, switch_rates.py, steering.py; run with the overlay root
$PAPER_COMMON/root for the common periods), $PAPER_COMMON/flat (flat_common.py: bill errors by period), and
Fig. 2's power and traffic samples from <repo>/artifacts/paper (power_traffic.py; its windows do not change).
Matplotlib's own fonts, also where LaTeX is installed (the paper's figures were made so).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.ticker                    # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402

from green_routing.paths import ARTIFACTS_DIR, REPO_ROOT   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import periods as PER                      # noqa: E402

P = ARTIFACTS_DIR / "paper"
PT = REPO_ROOT / "artifacts" / "paper"    # Fig. 2's samples (power_traffic.py, round 1's fixed windows)
plt.style.use(HERE / "paper.mplstyle")
SW, I2 = "#348ABD", "#A60628"
PNG = None
PURPLE, GREEN = "#7A68A6", "#467821"


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    if PNG:                                       # previews for checking, not for the paper
        fig.savefig(PNG / f"{name}.png", dpi=200)
    plt.close(fig)


def wcdf(v, w):
    o = np.argsort(v)
    return np.asarray(v)[o], np.cumsum(np.asarray(w)[o]) / np.sum(w)


def fig_rates(out: Path):
    """Rate a byte pays: across hours (flat rate, a year) and across city pairs (path rate)."""
    fig, ax = plt.subplots()
    for net, col in (("switch", SW), ("i2", I2)):
        h = pd.read_csv(P / f"hourly_year_{net}.csv", index_col=0)
        x, y = wcdf(h.r, h.T_gbps)
        ax.plot(x, y, color=col, ls="-")
        c = pd.read_csv(P / net / "citypairs.csv")
        x, y = wcdf(c.g_per_GB, c.GB)
        ax.plot(x, y, color=col, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 30)
    ax.set_xlabel("Carbon intensity (gCO$_2$e/GB)")
    ax.set_ylabel("Share of bytes")
    ax.set_ylim(0, 1)
    ax.grid(True, which="major")
    h = [Line2D([], [], color=SW, label="SWITCHlan"), Line2D([], [], color=I2, label="Internet2"),
         Line2D([], [], color="k", ls="-", label="over hours"), Line2D([], [], color="k", ls="--", label="over PoP pairs")]
    ax.legend(handles=h, loc="lower right", frameon=False, ncol=1, handlelength=1.5, borderaxespad=0.1)
    save(fig, out, "eval_rates")


def fig_bills(out: Path):
    """Annual bill of every entity, ranked: flat rate (line) and path rate (dots)."""
    fig, ax = plt.subplots()
    for net, col, name in (("switch", SW, "Switch LAN"), ("i2", I2, "I2 Network")):
        b = pd.read_csv(P / net / "entity_bills.csv", index_col=0).sort_values("flat_kg_per_year", ascending=False)
        b = b[b.flat_kg_per_year > 0]
        r = np.arange(1, len(b) + 1)
        ax.plot(r, b.flat_kg_per_year, color=col)
        ax.scatter(r, b.path_kg_per_year.clip(lower=1e-3), s=2, color=col, alpha=0.5, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Customer rank")
    ax.set_ylabel("Bill (kgCO$_2$e/year)")
    ax.set_ylim(1e-2, 2e5)
    ax.grid(True, which="major")
    h = [Line2D([], [], color=SW, label="SWITCHlan"), Line2D([], [], color=I2, label="Internet2"),
         Line2D([], [], color="k", label="flat rate"),
         Line2D([], [], color="k", ls="", marker="o", ms=2, alpha=0.6, label="path rate")]
    ax.legend(handles=h, loc="lower left", frameon=False, ncol=2, handlelength=1.5, borderaxespad=0.1, columnspacing=0.8)
    save(fig, out, "eval_bills")


def fig_periods(out: Path):
    """Bill error of the day-ahead rate by billing period on the test block (flat_common.py bill_periods.csv: hours,
    calendar days, 7-day blocks, calendar months), and where it comes from. The solid line is the posted bill (rate
    lowered by each network's margin m); the dotted line (actual grid) has no margin, as the margin only offsets
    the grid forecast's bias."""
    B = pd.read_csv(PER.ROOT / "flat" / "bill_periods.csv")
    B = B[B.variant.isin(["forecast", "perfect grid"])]
    per = ["hour", "day", "week", "month"]
    hours = np.array([1, 24, 168, 730])                     # period length; week = 7-day block, month = calendar month
    fig, ax = plt.subplots()
    # "perfect grid" leaves only the traffic forecast wrong: the gap to "both forecast" is the grid's
    # share of the error ("grid error only", i.e. perfect traffic, is quoted in the text, not drawn)
    styles = {"forecast": ("-", "o", "both forecast"), "perfect grid": (":", "^", "traffic error only")}
    for net, col in (("switch", SW), ("i2", I2)):
        ys = {}
        for v, (ls, mk, _) in styles.items():
            ys[v] = B[(B.network == net) & (B.variant == v)].set_index("period").wape.reindex(per).to_numpy() * 100
            ax.plot(hours, ys[v], color=col, ls=ls, marker=mk, ms=3.5, mfc=col if ls == "-" else "white", mew=0.8)
        ax.fill_between(hours, ys["perfect grid"], ys["forecast"], color=col, alpha=0.12, lw=0)
    # errors that cancel like independent days shrink as 1 / sqrt(period): a reference from SWITCHlan's day
    d0 = B[(B.network == "switch") & (B.variant == "forecast") & (B.period == "day")].wape.iloc[0] * 100
    ref = np.array([24, 720])
    ax.plot(ref, d0 * np.sqrt(24 / ref) * 1.6, color="0.6", lw=0.6)
    ax.text(150, d0 * np.sqrt(24 / 150) * 1.6, "$\\propto 1/\\sqrt{t}$", fontsize=8, color="0.4", ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 80)
    ax.set_xlim(0.6, 1300)
    ax.set_xticks(hours, ["Hour", "Day", "Week", "Month"])
    ax.minorticks_off()
    ax.set_ylabel("Bill error (%)")
    ax.grid(True, which="major")
    h = [Line2D([], [], color=SW, label="SWITCHlan"), Line2D([], [], color=I2, label="Internet2")] + \
        [Line2D([], [], color="k", ls=ls, marker=mk, ms=3.5, mfc="k" if ls == "-" else "white", mew=0.8, label=lab)
         for ls, mk, lab in styles.values()]
    ax.legend(handles=h, loc="lower left", frameon=False, ncol=2, handlelength=1.8, borderaxespad=0.1, columnspacing=0.8)
    save(fig, out, "eval_periods")


def fig_rate_error(out: Path):
    """Day-ahead error of a resource's rate against its load, with and without the cap (I2)."""
    F = pd.read_csv(P / "i2" / "rate_error_factor_by_load.csv")
    loads = list(dict.fromkeys(F.load))
    fig, ax = plt.subplots()
    for kind, col in (("uncapped", PURPLE), ("capped", GREEN)):
        d = F[F.kind == kind].set_index("load").reindex(loads)
        ax.plot(range(len(loads)), d.p90, color=col, marker="o", ms=3, label=f"{kind}, 90th pct.")
        ax.plot(range(len(loads)), d["median"], color=col, marker="o", ms=3, ls="--", label=f"{kind}, median")
    ax.set_yscale("log")
    ax.set_ylim(0.9, max(13.0, 1.4 * float(F.p90.max())))         # at least one decade; decade labels only
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    pct = "%"
    ax.set_xticks(range(len(loads)), [s.replace("%", pct).replace("-", "–") for s in loads], fontsize=8)
    ax.set_xlabel("Load of the link or router (share of capacity)")
    ax.set_ylabel("Error factor")
    ax.grid(True, which="major", axis="y")
    ax.legend(loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.1)
    save(fig, out, "eval_rate_error")


def fig_pairs(out: Path):
    """CIRo-style: share of PoP pairs whose carbon intensity falls by at least x (one pair moves)."""
    U = pd.read_csv(P / "i2" / "unilateral_pairs.csv")
    x = np.linspace(0, 90, 181)
    fig, ax = plt.subplots()
    for col, c, lab in (("reduction_forecast", I2, "chosen on day-ahead rates"),
                        ("reduction_oracle", "k", "chosen on settled rates")):
        ax.plot([(U[col] * 100 >= v).mean() for v in x], x, color=c, ls="-" if c == I2 else ":", label=lab)
    pct = "%"
    ax.set_ylabel(f"Reduction ({pct})")
    ax.set_xlabel("Share of PoP pairs")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 90)
    ax.grid(True, which="major")
    ax.legend(loc="upper right", frameon=False, handlelength=1.8, borderaxespad=0.1)
    save(fig, out, "eval_pairs")


def fig_steer(out: Path):
    """Spans left without traffic: today, hosts on day-ahead rates, a centralized controller."""
    D = pd.read_csv(P / "i2" / "everyone_hourly.csv", parse_dates=["hour"])
    C = json.load(open(P / "i2" / "central.json"))
    d = D.set_index("hour")[["spans_dark", "today_spans_dark"]].resample("D").mean()
    fig, ax = plt.subplots()
    ax.plot(d.index, d.spans_dark, color=I2, label="hosts follow day-ahead rates")
    # the number the text quotes: the mean over the hours from day 15 on
    t15 = d.index[0] + pd.Timedelta(days=14)
    m15 = D[D.hour >= t15].spans_dark.mean()
    ax.plot([t15, d.index[-1]], [m15, m15], color=I2, lw=0.6, ls=(0, (1, 1)))
    ax.axhline(C["dark"], color="k", ls=":", label="centralized link sleeping")
    ax.plot(d.index, d.today_spans_dark, color="0.5", ls="--", label="today's routing")
    ax.set_ylabel(f"Dark spans (of {C['spans']})")
    ax.set_ylim(0, 65)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, which="major", axis="y")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.14), frameon=False, handlelength=1.8, borderaxespad=0.1)  # above today's line
    save(fig, out, "eval_steer")


PT_SIZE = (3.33, 1.4)                         # Fig. 2: one column wide, exact size (no tight bbox)
PT_BINS = np.linspace(0, 1, 21)               # traffic / device maximum, 5 % bins
PT_MIN = 100                                  # device-hours per bin to draw it


def power_traffic_bins() -> dict:
    """Per network: hourly power / the device's median power, binned by hourly traffic / the device's
    maximum hourly traffic (all device-hours pooled): median and 10-90 % band per bin. SWITCHlan: 116
    routers' PSU power, 2024-02-01 -> 10-31; Internet2: 45 PoPs' router-side DC power, 2025-11-15 -> 2026-10-31
    (power_traffic.py)."""
    out = {}
    for net in ("switch", "i2"):
        d = pd.read_csv(PT / f"power_traffic_{net}.csv.gz")
        d = d[(d.W > 0) & (d.gbps > 0)]
        g = d.groupby("device")
        x, y = d.gbps / g.gbps.transform("max"), d.W / g.W.transform("median")
        b = pd.cut(x, PT_BINS, include_lowest=True)
        s = y.groupby(b, observed=False).agg(med="median", lo=lambda v: v.quantile(.1), hi=lambda v: v.quantile(.9),
                                             n="size")
        s["x"] = (PT_BINS[:-1] + PT_BINS[1:]) / 2
        out[net] = (s[s.n >= PT_MIN], int(d.device.nunique()), y)
    return out


def _pt_axes(ax, xlabel=True, ticks=(0, 0.25, 0.5, 0.75, 1)):
    ax.axhline(1.0, color="0.6", lw=0.4, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1.1)
    ax.set_xticks(list(ticks), [f"{100 * t:.0f}" for t in ticks])
    ax.set_yticks([0.9, 1.0, 1.1], ["0.9", "1.0", "1.1"])
    ax.minorticks_off()
    ax.grid(True, which="major")
    ax.tick_params(labelsize=8)
    pct = "%"
    if xlabel:
        ax.set_xlabel(f"Traffic ({pct} of device maximum)", fontsize=8, labelpad=1)


def fig_power_traffic(out: Path):
    """Fig. 2 (one panel): power relative to the device's median against traffic relative to its maximum,
    binned median (line) and 10-90 % band, SWITCHlan routers and Internet2 PoPs."""
    B = power_traffic_bins()
    fig = plt.figure(figsize=PT_SIZE)
    ax = fig.add_axes([0.13, 0.255, 0.835, 0.715])
    for net, col, lab in (("i2", I2, "Internet2 PoPs"), ("switch", SW, "SWITCHlan routers")):
        s, n, _ = B[net]
        ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=0.18, lw=0)
        ax.plot(s.x, s.med, color=col, label=f"{lab} ({n})")
    _pt_axes(ax)
    ax.set_ylabel("Power (rel. to median)", fontsize=8, labelpad=2)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper center", frameon=False, ncol=2, fontsize=7, handlelength=1.5,
              borderaxespad=0.1, columnspacing=1.0)
    with plt.rc_context({"savefig.bbox": "standard"}):          # exact PT_SIZE, as the paper places it
        fig.savefig(out / "power_vs_traffic.pdf")
        if PNG:
            fig.savefig(PNG / "power_vs_traffic.png", dpi=250)
    plt.close(fig)


def fig_power_traffic_two(out: Path):
    """Fig. 2, two-panel variant (same data and size): SWITCHlan left, Internet2 right."""
    B = power_traffic_bins()
    fig = plt.figure(figsize=PT_SIZE)
    axes = [fig.add_axes([0.13, 0.255, 0.38, 0.6]), fig.add_axes([0.585, 0.255, 0.38, 0.6])]
    for ax, net, col, lab in ((axes[0], "switch", SW, "SWITCHlan routers"), (axes[1], "i2", I2, "Internet2 PoPs")):
        s, n, _ = B[net]
        ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=0.18, lw=0)
        ax.plot(s.x, s.med, color=col)
        _pt_axes(ax, xlabel=False, ticks=(0, 0.5, 1))
        ax.set_title(f"{lab} ({n})", fontsize=7, pad=2)
    axes[1].tick_params(labelleft=False)
    axes[0].set_ylabel("Power (rel. to median)", fontsize=8, labelpad=2)
    pct = "%"
    fig.text(0.545, 0.02, f"Traffic ({pct} of device maximum)", fontsize=8, ha="center", va="bottom")
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out / "power_vs_traffic_twopanel.pdf")
        if PNG:
            fig.savefig(PNG / "power_vs_traffic_twopanel.png", dpi=250)
    plt.close(fig)


def power_traffic_numbers() -> dict:
    """Numbers a caption could state (written to paper/power_vs_traffic.json)."""
    B = power_traffic_bins()
    res = {}
    for net in ("switch", "i2"):
        s, n, y = B[net]
        res[net] = {"devices": n, "device_hours": int(len(y)), "bins_drawn": int(len(s)),
                    "median_min": float(s.med.min()), "median_max": float(s.med.max()),
                    "band_p10_min": float(s.lo.min()), "band_p90_max": float(s.hi.max()),
                    "pooled_p10_p90": [float(y.quantile(.1)), float(y.quantile(.9))],
                    "median_top_bin": float(s.med.iloc[-1]), "median_bottom_bin": float(s.med.iloc[0])}
    json.dump(res, open(PT / "power_vs_traffic.json", "w"), indent=1)
    return res


def main():
    global PNG
    args = [a for a in sys.argv[1:] if not a.startswith(("--png=", "--only="))]
    png = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--png=")]
    PNG = Path(png[0]) if png else None
    if PNG:
        PNG.mkdir(parents=True, exist_ok=True)
    out = Path(args[0]) if args else REPO_ROOT / "6aea077f0929c9a3a1ac5660" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    only = [a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")]
    figs = (fig_power_traffic, fig_power_traffic_two, fig_rates, fig_bills, fig_periods, fig_rate_error, fig_pairs, fig_steer)
    for f in figs:
        if only and f.__name__ not in only[0]:
            continue
        f(out)
        print("wrote", f.__name__, flush=True)
    print(json.dumps(power_traffic_numbers(), indent=1))


if __name__ == "__main__":
    main()
