#!/usr/bin/env python
"""
Frequency-domain analysis of the SWITCH backbone link traffic dataset
(lan-mon2021/bb-usage-logs). For every backbone link-pair we build a
continuous 5-min throughput series, then characterise it in the frequency
domain: Welch PSD, seasonal line spectra (daily/weekly harmonics),
1/f low-frequency slope (long-range dependence), spectral entropy /
flatness, and a spectrogram showing how the diurnal cycle evolves over
the ~2.5-year span.

Input: the research checkout's backbone usage logs (not in the S3 bucket) —
  $GR_BB_USAGE_DIR, default $GR_RESEARCH_ROOT/lan-mon2021/bb-usage-logs.

Outputs ($GR_ARTIFACTS_DIR/spectral):
  metrics.csv      per-link spectral metrics
  psd_<link>.npz   cached PSDs
  fig*.png         report figures (spectral_figures), spectral_report.html

Run:  python -m green_routing.analysis.spectral_analysis
      python -m green_routing.analysis.spectral_figures
      python -m green_routing.analysis.spectral_report
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy import signal

from ..dataset.config import RESEARCH_ROOT
from ..paths import ARTIFACTS_DIR

BB = os.environ.get("GR_BB_USAGE_DIR") or str(RESEARCH_ROOT / "lan-mon2021" / "bb-usage-logs")
OUT = str(ARTIFACTS_DIR / "spectral")

FS_PER_DAY = 288.0          # samples per day at 5-min sampling
NYQUIST_CPD = 144.0         # cycles/day
DT = 300.0                  # seconds

def link_names():
    names = sorted({os.path.basename(f).rsplit(".",1)[0] for f in glob.glob(f"{BB}/*")})
    return names

def load_link(link):
    """Concatenate monthly files -> regular 5-min grid. Returns DataFrame
    indexed by UTC time with columns in,out,total (Mbps). NaN where missing."""
    files = sorted(glob.glob(f"{BB}/{link}.*"))
    frames = []
    for f in files:
        try:
            a = np.loadtxt(f, usecols=(0,1,3))
        except Exception:
            continue
        if a.ndim == 1:
            a = a[None, :]
        frames.append(a)
    if not frames:
        return None
    a = np.concatenate(frames, axis=0)
    ts = a[:,0].astype(np.int64)
    df = pd.DataFrame({"in": a[:,1], "out": a[:,2]}, index=pd.to_datetime(ts, unit="s", utc=True))
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # regular 5-min grid across full observed span
    grid = pd.date_range(df.index[0], df.index[-1], freq="5min", tz="UTC")
    df = df.reindex(grid)
    df["total"] = df["in"] + df["out"]
    # --- robust cleaning of RRD counter-wrap / non-physical spikes ---
    # SWITCH backbone links are <=100G; anything above ~0.5 Tbps is corrupt.
    for c in ("in","out","total"):
        v = df[c].to_numpy(float)
        med = np.nanmedian(v)
        mad = np.nanmedian(np.abs(v-med)) * 1.4826
        # spike if grossly non-physical OR far above the robust envelope
        cap = max(med + 40*mad, 4*np.nanpercentile(v, 95) if np.isfinite(np.nanpercentile(v,95)) else 0)
        bad = (v > 500000) | (v > cap) & (v > med + 40*mad)
        v[bad] = np.nan
        df[c] = v
    return df

def spectral_metrics(x):
    """x: 1-D throughput series (may contain NaN). Returns metrics dict +
    PSD arrays. Uses Welch with a 4-week window for fine low-freq resolution."""
    x = np.asarray(x, float)
    n = len(x)
    gap_frac = np.isnan(x).mean()
    # fill gaps by time interpolation for the transform, keep record of gaps
    s = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(s).all():
        s = np.nan_to_num(s, nan=np.nanmean(s) if np.isfinite(np.nanmean(s)) else 0.0)

    mean_load = np.nanmean(x)
    std_load  = np.nanstd(x)
    cv = std_load / mean_load if mean_load > 0 else np.nan

    # Welch PSD, 4-week segments (or as large as data allows), Hann, 50% overlap
    nper = int(min(len(s), 4*7*288))
    nper = max(nper, 2*288)  # at least 2 days
    f, Pxx = signal.welch(s, fs=FS_PER_DAY, window="hann", nperseg=nper,
                          noverlap=nper//2, detrend="linear", scaling="density")
    # drop DC
    f, Pxx = f[1:], Pxx[1:]

    def band_power(fc, rel=0.06):
        lo, hi = fc*(1-rel), fc*(1+rel)
        m = (f>=lo) & (f<=hi)
        return Pxx[m].sum() if m.any() else 0.0
    total_power = Pxx.sum()

    daily   = band_power(1.0)
    d2      = band_power(2.0)
    d3      = band_power(3.0)
    weekly  = band_power(1/7)
    halfwk  = band_power(2/7)
    diurnal_family = daily+d2+d3
    seasonal_family = diurnal_family+weekly+halfwk

    frac_daily   = daily/total_power
    frac_weekly  = weekly/total_power
    frac_seasonal= seasonal_family/total_power

    # peak-to-median ratio at the daily line (spectral prominence of the cycle)
    med = np.median(Pxx)
    daily_peak_val = Pxx[(f>=0.9)&(f<=1.1)].max() if ((f>=0.9)&(f<=1.1)).any() else np.nan
    daily_snr_db = 10*np.log10(daily_peak_val/med) if med>0 and np.isfinite(daily_peak_val) else np.nan

    # spectral entropy (normalised) -> flatness/unpredictability
    p = Pxx/Pxx.sum()
    spec_entropy = -np.sum(p*np.log(p+1e-30))/np.log(len(p))
    # spectral flatness (geo mean / arith mean)
    flatness = np.exp(np.mean(np.log(Pxx+1e-30)))/np.mean(Pxx)
    # spectral centroid (cpd)
    centroid = np.sum(f*Pxx)/np.sum(Pxx)

    # 1/f low-frequency slope beta: fit log P ~ -beta log f over sub-daily
    # low frequencies (0.02 .. 0.7 cpd), excluding the weekly & daily lines
    m = (f>=0.02)&(f<=0.7)
    # remove obvious spectral lines (weekly harmonics) so slope reflects the
    # continuum / long-range dependence, not the seasonal spikes
    lines = np.zeros_like(f, bool)
    for k in range(1,6):
        lines |= (np.abs(f-k/7.0) < 0.01)
    mm = m & ~lines
    beta = hurst = np.nan
    if mm.sum() > 8:
        lf, lp = np.log(f[mm]), np.log(Pxx[mm])
        A = np.vstack([lf, np.ones_like(lf)]).T
        slope, _ = np.linalg.lstsq(A, lp, rcond=None)[0]
        beta = -slope
        # for fGn: S(f) ~ f^(1-2H) => beta = 2H-1 => H=(beta+1)/2
        hurst = (beta+1)/2

    return dict(
        n=n, gap_frac=gap_frac, mean_load=mean_load, std_load=std_load, cv=cv,
        frac_daily=frac_daily, frac_weekly=frac_weekly, frac_seasonal=frac_seasonal,
        daily_snr_db=daily_snr_db, spec_entropy=spec_entropy, flatness=flatness,
        centroid_cpd=centroid, beta=beta, hurst=hurst,
    ), (f, Pxx)

def main():
    if not os.path.isdir(BB):
        raise SystemExit(f"backbone usage logs not found at {BB} — set GR_BB_USAGE_DIR or "
                         "GR_RESEARCH_ROOT (they live in the network-energy-efficiency-"
                         "research checkout, not in the bucket)")
    os.makedirs(OUT, exist_ok=True)
    names = link_names()
    rows = []
    psd_store = {}
    for link in names:
        df = load_link(link)
        if df is None or len(df) < 4*288:
            continue
        x = df["total"].to_numpy()
        zero_frac = float(np.nanmean(x == 0))
        met, (f, Pxx) = spectral_metrics(x)
        met["link"] = link
        met["months"] = round(len(df)/288/30.4, 1)
        met["zero_frac"] = zero_frac
        # quality: need enough coverage & not mostly-idle
        met["quality"] = ("poor" if (met["months"] < 3 or zero_frac > 0.5)
                          else "marginal" if (met["months"] < 9 or zero_frac > 0.15)
                          else "good")
        rows.append(met)
        psd_store[link] = (f, Pxx)
        np.savez(f"{OUT}/psd_{link}.npz", f=f, Pxx=Pxx)
        print(f"{link:10s} months={met['months']:4} meanL={met['mean_load']:8.1f} "
              f"fD={met['frac_daily']:.3f} fW={met['frac_weekly']:.3f} "
              f"snr={met['daily_snr_db']:5.1f}dB H={met['hurst']:.2f} "
              f"Hent={met['spec_entropy']:.3f}")
    M = pd.DataFrame(rows).set_index("link").sort_values("mean_load", ascending=False)
    cols = ["quality","months","mean_load","std_load","cv","frac_daily","frac_weekly",
            "frac_seasonal","daily_snr_db","spec_entropy","flatness","centroid_cpd",
            "beta","hurst","gap_frac","zero_frac"]
    M = M[cols]
    M.to_csv(f"{OUT}/metrics.csv")
    print("\nwrote", f"{OUT}/metrics.csv", "with", len(M), "links")
    # summary
    print("\n=== summary across links ===")
    print(M[["frac_daily","frac_weekly","frac_seasonal","daily_snr_db",
             "spec_entropy","hurst","beta"]].describe().round(3).to_string())

if __name__ == "__main__":
    main()
