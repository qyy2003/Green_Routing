#!/usr/bin/env python
"""Generate the frequency-domain report figures from cached PSDs + raw links."""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import signal
from spectral_analysis import load_link, OUT, FS_PER_DAY

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.7,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
BLUE="#2a78d6"; AQUA="#1baf7a"; YEL="#eda100"; GREEN="#008300"
VIOLET="#4a3aa7"; RED="#e34948"; MAG="#e87ba4"; ORANGE="#eb6834"
MUTED="#898781"; INK="#0b0b0b"; INK2="#52514e"

def psd(link):
    d = np.load(f"{OUT}/psd_{link}.npz")
    return d["f"], d["Pxx"]

def mark_lines(ax, top=True):
    for cpd, lab, c in [(1,"daily",RED),(2,"2×/day",ORANGE),(3,"3×/day",YEL),
                        (1/7,"weekly",VIOLET),(2/7,"½-week",MAG)]:
        ax.axvline(cpd, color=c, lw=1.0, ls="--", alpha=0.65, zorder=1)
    if top:
        ax.text(1,1.012,"daily",transform=ax.get_xaxis_transform(),ha="center",
                color=RED,fontsize=8.5,fontweight="bold")
        ax.text(2,1.012,"2/d",transform=ax.get_xaxis_transform(),ha="center",color=ORANGE,fontsize=8)
        ax.text(3,1.012,"3/d",transform=ax.get_xaxis_transform(),ha="center",color=YEL,fontsize=8)
        ax.text(1/7,1.012,"weekly",transform=ax.get_xaxis_transform(),ha="center",
                color=VIOLET,fontsize=8.5,fontweight="bold")

def fmt_cpd(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("frequency  (cycles per day)")
    secax = ax.secondary_xaxis("bottom")
    # period ticks on a twin
    def per(x,_):
        if x<=0: return ""
        p=24/x
        return f"{p:.0f}h" if p>=1 else f"{p*60:.0f}m"
    ax2 = ax.twiny()
    ax2.set_xscale("log"); ax2.set_xlim(ax.get_xlim())
    ticks=[1/7,1/2,1,2,3,6,12,24,72]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{24/t:.0f}h" if 24/t>=1 else f"{24/t*60:.0f}m" for t in ticks],
                        fontsize=8.5, color=MUTED)
    ax2.set_xlabel("period", fontsize=9, color=MUTED, labelpad=2)
    ax2.grid(False)
    return ax2

# ---------------------------------------------------------------- FIG 1
def fig_hero():
    link="AG-EZ"
    f,P = psd(link)
    fig,ax = plt.subplots(figsize=(9.2,5.2))
    ax.plot(f,P,color=BLUE,lw=1.3,zorder=3)
    mark_lines(ax)
    # 1/f reference slope over the continuum
    m=(f>=0.02)&(f<=0.7)
    lf,lp=np.log(f[m]),np.log(P[m])
    b=np.polyfit(lf,lp,1)
    xx=np.array([0.015,0.8])
    ax.plot(xx,np.exp(np.polyval(b,np.log(xx))),color=INK,lw=1.6,ls=":",zorder=4,
            label=f"1/f continuum  (slope β≈{-b[0]:.2f}, H≈{(-b[0]+1)/2:.2f})")
    # noise floor
    hi=(f>6)
    floor=np.median(P[hi])
    ax.axhline(floor,color=GREEN,lw=1.3,ls="-.",alpha=0.8,
               label="high-freq noise floor (irreducible)")
    ax2=fmt_cpd(ax)
    ax.set_ylabel("power spectral density  (Mbps²·day)")
    ax.set_title(f"Canonical backbone traffic spectrum — link {link}  (32 months, 5-min)",
                 pad=26)
    ax.legend(loc="lower left",fontsize=9,framealpha=0.9,edgecolor="#e1e0d9")
    ax.set_xlim(f[1],f[-1])
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_hero_spectrum.png",bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")

# ---------------------------------------------------------------- FIG 2
def fig_archetypes():
    panels=[("TO-ZH","Access / regional — strongly diurnal",BLUE),
            ("IXG-ZH","Internet-exchange uplink — sharpest daily line",AQUA),
            ("BE-FR","Core transit — diurnal + heavy continuum",VIOLET),
            ("CE-LG","Aggregated core — nearly flat, noise-like",ORANGE)]
    fig,axes=plt.subplots(2,2,figsize=(11,7.6))
    for (lk,ttl,c),ax in zip(panels,axes.ravel()):
        f,P=psd(lk)
        ax.plot(f,P,color=c,lw=1.2,zorder=3)
        for cpd,cc in [(1,RED),(2,ORANGE),(3,YEL),(1/7,VIOLET),(2/7,MAG)]:
            ax.axvline(cpd,color=cc,lw=0.9,ls="--",alpha=0.55)
        ax.set_xscale("log");ax.set_yscale("log")
        ax.set_title(f"{lk} — {ttl}",fontsize=10.5)
        ax.set_xlim(f[1],f[-1])
        ax.tick_params(labelsize=8.5)
        if ax in axes[1]: ax.set_xlabel("cycles per day",fontsize=9.5)
        if ax in axes[:,0]: ax.set_ylabel("PSD (Mbps²·day)",fontsize=9.5)
    fig.suptitle("Four spectral archetypes across the SWITCH backbone",
                 fontsize=13,fontweight="bold",y=0.995)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_archetypes.png",bbox_inches="tight")
    plt.close(fig);print("fig2 done")

# ---------------------------------------------------------------- FIG 3
def fig_perlink_bars():
    M=pd.read_csv(f"{OUT}/metrics.csv")
    g=M[M.quality=="good"].copy().sort_values("frac_daily")
    y=np.arange(len(g))
    fig,axes=plt.subplots(1,2,figsize=(11.5,9),sharey=True)
    ax=axes[0]
    ax.barh(y,g.frac_daily*100,color=BLUE,height=0.72,zorder=3)
    ax.barh(y,g.frac_weekly*100,left=g.frac_daily*100,color=VIOLET,height=0.72,zorder=3,
            label="weekly")
    ax.set_yticks(y);ax.set_yticklabels(g.link,fontsize=8.5)
    ax.set_xlabel("share of signal variance  (%)")
    ax.set_title("Energy in the daily + weekly lines",fontsize=11)
    ax.legend([plt.Rectangle((0,0),1,1,color=BLUE),plt.Rectangle((0,0),1,1,color=VIOLET)],
              ["daily (±harmonics band)","weekly"],fontsize=9,loc="lower right")
    ax.grid(axis="y",visible=False)
    ax2=axes[1]
    colors=[RED if h>1.02 else AQUA if h<0.85 else BLUE for h in g.hurst]
    ax2.barh(y,g.hurst,color=colors,height=0.72,zorder=3)
    ax2.axvline(0.5,color=MUTED,lw=1,ls="--")
    ax2.axvline(1.0,color=INK2,lw=1,ls=":")
    ax2.text(0.5,len(g)-0.2,"H=0.5\n(no memory)",fontsize=7.5,color=MUTED,ha="center",va="top")
    ax2.text(1.0,len(g)-0.2,"H=1\n(random walk)",fontsize=7.5,color=INK2,ha="center",va="top")
    ax2.set_xlabel("Hurst exponent  H = (β+1)/2")
    ax2.set_title("Long-range dependence (spectral slope)",fontsize=11)
    ax2.set_xlim(0.4,1.4)
    ax2.grid(axis="y",visible=False)
    fig.suptitle("Per-link spectral fingerprint  (31 good-coverage backbone links, sorted by diurnal strength)",
                 fontsize=12.5,fontweight="bold",y=1.003)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_perlink.png",bbox_inches="tight")
    plt.close(fig);print("fig3 done")

# ---------------------------------------------------------------- FIG 4
def fig_map():
    M=pd.read_csv(f"{OUT}/metrics.csv")
    g=M[M.quality!="poor"].copy()
    fig,ax=plt.subplots(figsize=(9.6,6.6))
    sz=40+ (np.log10(g.mean_load.clip(1))**2)*26
    sc=ax.scatter(g.frac_daily*100,g.hurst,s=sz,c=g.spec_entropy,cmap="viridis_r",
                  edgecolor="white",lw=0.8,zorder=3,vmin=0.35,vmax=0.78)
    for _,r in g.iterrows():
        if r.mean_load>1500 or r.frac_daily>0.42 or r.hurst>1.15 or r.hurst<0.75:
            ax.annotate(r.link,(r.frac_daily*100,r.hurst),fontsize=7.5,color=INK2,
                        xytext=(3,3),textcoords="offset points")
    ax.axhline(1.0,color=INK2,lw=1,ls=":",alpha=0.7)
    ax.text(1,1.005,"H=1  random-walk limit",fontsize=8.5,color=INK2,va="bottom")
    ax.set_xlabel("diurnal energy  — share of variance in the daily band (%)")
    ax.set_ylabel("Hurst exponent  (memory / long-range dependence)")
    cb=fig.colorbar(sc,ax=ax,pad=0.015)
    cb.set_label("spectral entropy  (→ flatter / less predictable)",fontsize=9.5)
    ax.set_title("Predictability map of the backbone\nmarker size ∝ log mean load;  right = seasonal, up = persistent",
                 pad=10)
    # guidance arrows text
    ax.annotate("seasonal models win\n(HA / seasonal-naive)",(46,0.9),fontsize=8.5,
                color=BLUE,ha="right",fontweight="bold")
    ax.annotate("persistence-dominated\n(hard beyond 1 step)",(3,1.18),fontsize=8.5,
                color=ORANGE,fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_map.png",bbox_inches="tight")
    plt.close(fig);print("fig4 done")

# ---------------------------------------------------------------- FIG 5
def fig_spectrogram():
    link="EZ-ZH"
    df=load_link(link)
    x=df["total"].interpolate(limit_direction="both").to_numpy()
    idx=df.index
    # STFT: weekly windows
    nper=7*288
    f,t,Sxx=signal.spectrogram(x-np.nanmean(x),fs=FS_PER_DAY,window="hann",
                               nperseg=nper,noverlap=nper//2,detrend="linear")
    # focus on 0..3.5 cpd
    fm=(f>0)&(f<=3.6)
    S=Sxx[fm];ff=f[fm]
    tt=idx[0]+pd.to_timedelta(t,unit="D")
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,7.2),height_ratios=[2.1,1],sharex=True)
    im=ax1.pcolormesh(tt,ff,10*np.log10(S+1e-9),cmap="magma",shading="gouraud")
    ax1.axhline(1,color="white",lw=0.8,ls="--",alpha=0.6)
    ax1.axhline(1/7,color="cyan",lw=0.8,ls="--",alpha=0.6)
    ax1.text(tt[2],1.02,"daily",color="white",fontsize=9,fontweight="bold",va="bottom")
    ax1.set_ylabel("cycles per day")
    ax1.set_title(f"Spectrogram — how {link}'s rhythm evolves (weekly windows, 2021→2023)",pad=8)
    cb=fig.colorbar(im,ax=ax1,pad=0.01);cb.set_label("power (dB)",fontsize=9)
    # daily-band power fraction over time
    dband=(ff>=0.85)&(ff<=1.15)
    total=S.sum(0)
    dfrac=S[dband].sum(0)/total
    ax2.plot(tt,dfrac*100,color=BLUE,lw=1.5)
    ax2.fill_between(tt,dfrac*100,color=BLUE,alpha=0.12)
    ax2.set_ylabel("daily-band\nshare (%)")
    ax2.set_xlabel("date")
    ax2.set_title("Strength of the daily cycle over time",fontsize=10.5)
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_spectrogram.png",bbox_inches="tight")
    plt.close(fig);print("fig5 done")

# ---------------------------------------------------------------- FIG 6
def fig_duality():
    link="AG-EZ"
    df=load_link(link)
    x=df["total"].interpolate(limit_direction="both").to_numpy()
    x=x-np.nanmean(x)
    # ACF up to 14 days
    maxlag=14*288
    xf=x/np.std(x)
    ac=np.correlate(xf,xf,mode="full")/len(xf)
    ac=ac[len(xf)-1:len(xf)-1+maxlag+1]
    lags=np.arange(len(ac))/288.0
    f,P=psd(link)
    # cumulative variance vs frequency
    cum=np.cumsum(P)/P.sum()
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.9))
    ax1.plot(lags,ac,color=BLUE,lw=1.2)
    for d in range(1,8):
        ax1.axvline(d,color=RED,lw=0.7,ls="--",alpha=0.4)
    ax1.axvline(7,color=VIOLET,lw=1.1,ls="--",alpha=0.7)
    ax1.axhline(0,color=MUTED,lw=0.8)
    ax1.set_xlabel("lag (days)");ax1.set_ylabel("autocorrelation")
    ax1.set_title("Time domain: autocorrelation\n(daily combs + weekly beat)",fontsize=11)
    ax1.set_xlim(0,14)
    ax2.plot(f,cum*100,color=VIOLET,lw=1.6)
    ax2.set_xscale("log")
    for cpd,lab,c in [(1,"1 h",None)]:
        pass
    # mark horizon-relevant frequencies
    for cpd,lab in [(1/7,"1 wk"),(1,"1 day"),(24/6,"6 h"),(24/1,"1 h"),(24*12,"5 min")]:
        if cpd<=f[-1]:
            ax2.axvline(cpd,color=MUTED,lw=0.8,ls=":")
            ax2.text(cpd,3,lab,rotation=90,fontsize=8,color=INK2,va="bottom",ha="right")
    ax2.axhline(50,color=MUTED,lw=0.8,ls="--")
    ax2.set_xlabel("frequency (cycles per day)")
    ax2.set_ylabel("cumulative % of variance below f")
    ax2.set_title("Frequency domain: where the variance lives\n(slow = predictable, fast = noise)",fontsize=11)
    ax2.set_xlim(f[1],f[-1]);ax2.set_ylim(0,100)
    fig.suptitle(f"Time–frequency duality and the horizon/accuracy trade-off — {link}",
                 fontsize=12.5,fontweight="bold",y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_duality.png",bbox_inches="tight")
    plt.close(fig);print("fig6 done")

if __name__=="__main__":
    fig_hero(); fig_archetypes(); fig_perlink_bars(); fig_map()
    fig_spectrogram(); fig_duality()
    print("all figures written to",OUT)
