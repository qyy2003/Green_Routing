#!/usr/bin/env python
"""Assemble the frequency-domain report HTML with embedded figures + table.

Needs the outputs of spectral_analysis (metrics.csv) and spectral_figures (fig*.png) in
$GR_ARTIFACTS_DIR/spectral; writes spectral_report.html next to them.
"""
import base64, html, pathlib

import pandas as pd

from .spectral_analysis import OUT


def img(name):
    b=pathlib.Path(f"{OUT}/{name}").read_bytes()
    return "data:image/png;base64,"+base64.b64encode(b).decode()


def table_rows() -> str:
    """One <tr> per link from metrics.csv (sorted by mean load, as written)."""
    M = pd.read_csv(f"{OUT}/metrics.csv")
    out = []
    for _, r in M.iterrows():
        out.append(
            f"<tr class='q-{r.quality}'><td class='mono'>{html.escape(str(r.link))}</td>"
            f"<td>{r.quality}</td><td>{r.months:.1f}</td><td>{r.mean_load:,.0f}&nbsp;Mbps</td>"
            f"<td>{100 * r.frac_daily:.1f}</td><td>{100 * r.frac_weekly:.1f}</td>"
            f"<td>{r.daily_snr_db:.1f}</td><td>{r.hurst:.2f}</td><td>{r.spec_entropy:.3f}</td></tr>")
    return "\n".join(out)


F1=img("fig1_hero_spectrum.png"); F2=img("fig2_archetypes.png")
F3=img("fig3_perlink.png"); F4=img("fig4_map.png")
F5=img("fig5_spectrogram.png"); F6=img("fig6_duality.png")
ROWS=table_rows()

HTML=f"""<title>Frequency-Domain Analysis of SWITCH Backbone Traffic</title>
<style>
:root{{
  --bg:#f5f7fa; --surface:#ffffff; --fig:#fcfcfb;
  --ink:#111823; --ink2:#48525f; --muted:#79838f;
  --line:#e4e8ee; --line2:#eef1f5;
  --accent:#2760e6; --accent-soft:#eaf0fe;
  --daily:#d9433c; --weekly:#5a45c8; --good:#0c8f56;
  --shadow:0 1px 2px rgba(16,22,34,.05),0 8px 24px -12px rgba(16,22,34,.14);
  --measure:70ch;
}}
@media (prefers-color-scheme:dark){{
  :root:where(:not([data-theme=light])){{
    --bg:#0b0e13; --surface:#141a23; --fig:#fbfbfa;
    --ink:#e9edf3; --ink2:#a7b1bf; --muted:#6d7885;
    --line:#232b36; --line2:#1b222c;
    --accent:#5b8dff; --accent-soft:#182335;
    --daily:#f0665f; --weekly:#9585ee; --good:#2bb874;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 34px -14px rgba(0,0,0,.6);
  }}
}}
:root[data-theme=dark]{{
  --bg:#0b0e13; --surface:#141a23; --fig:#fbfbfa;
  --ink:#e9edf3; --ink2:#a7b1bf; --muted:#6d7885;
  --line:#232b36; --line2:#1b222c;
  --accent:#5b8dff; --accent-soft:#182335;
  --daily:#f0665f; --weekly:#9585ee; --good:#2bb874;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 34px -14px rgba(0,0,0,.6);
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased;}}
.mono{{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}}
.wrap{{max-width:960px;margin:0 auto;padding:0 24px;}}
.reading{{max-width:var(--measure);}}
h1,h2,h3{{line-height:1.18;text-wrap:balance;letter-spacing:-.015em;}}
h2{{font-size:1.62rem;font-weight:680;margin:2.9em 0 .1em;}}
h3{{font-size:1.16rem;font-weight:640;margin:2em 0 .2em;}}
p{{margin:.85em 0;color:var(--ink2);}}
p strong,li strong{{color:var(--ink);font-weight:620;}}
a{{color:var(--accent);}}
.eyebrow{{font-family:ui-monospace,monospace;font-size:.72rem;font-weight:600;
  letter-spacing:.18em;text-transform:uppercase;color:var(--accent);}}

/* hero */
.hero{{position:relative;overflow:hidden;border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,var(--surface),var(--bg));}}
.hero .wrap{{padding-top:56px;padding-bottom:34px;position:relative;z-index:2;}}
#speccanvas{{position:absolute;inset:0;width:100%;height:100%;z-index:1;
  opacity:.5;pointer-events:none;}}
.hero h1{{font-size:2.55rem;font-weight:720;margin:.24em 0 .1em;max-width:18ch;}}
.lede{{font-size:1.14rem;color:var(--ink2);max-width:60ch;margin:.5em 0 0;}}
.prov{{margin-top:16px;font-size:.83rem;color:var(--muted);}}
.prov span{{color:var(--ink2);}}

/* kpi */
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:14px;
  overflow:hidden;margin:30px 0 6px;box-shadow:var(--shadow);}}
.kpi{{background:var(--surface);padding:16px 18px;}}
.kpi .k{{font-family:ui-monospace,monospace;font-size:1.75rem;font-weight:600;
  color:var(--ink);letter-spacing:-.02em;font-variant-numeric:tabular-nums;}}
.kpi .k .u{{font-size:.9rem;color:var(--muted);font-weight:500;margin-left:2px;}}
.kpi .lab{{font-size:.78rem;color:var(--ink2);margin-top:3px;}}
@media(max-width:640px){{.kpis{{grid-template-columns:repeat(2,1fr);}}
  .hero h1{{font-size:2rem;}}}}

/* figure card */
figure{{margin:26px 0;}}
.figcard{{background:var(--fig);border:1px solid var(--line);border-radius:14px;
  padding:14px;box-shadow:var(--shadow);}}
.figcard img{{width:100%;height:auto;display:block;border-radius:6px;}}
figcaption{{font-size:.85rem;color:var(--muted);margin-top:10px;padding:0 4px;}}
figcaption b{{color:var(--ink2);font-weight:600;}}
.full{{max-width:1120px;}}

/* callout */
.note{{border-left:3px solid var(--accent);background:var(--accent-soft);
  border-radius:0 10px 10px 0;padding:14px 18px;margin:20px 0;}}
.note .t{{font-family:ui-monospace,monospace;font-size:.72rem;font-weight:600;
  letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:2px;}}
.note p{{margin:.3em 0;color:var(--ink2);}}

/* takeaway list */
ul.clean{{list-style:none;padding:0;margin:1em 0;}}
ul.clean li{{position:relative;padding:.2em 0 .2em 1.4em;color:var(--ink2);}}
ul.clean li::before{{content:"";position:absolute;left:0;top:.72em;
  width:7px;height:7px;border-radius:2px;background:var(--accent);}}

/* table */
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;
  box-shadow:var(--shadow);margin:22px 0;}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;background:var(--surface);}}
th,td{{padding:7px 12px;text-align:left;border-bottom:1px solid var(--line2);
  white-space:nowrap;}}
thead th{{position:sticky;top:0;background:var(--surface);color:var(--ink);
  font-weight:620;font-size:.74rem;letter-spacing:.02em;border-bottom:1px solid var(--line);}}
td.num{{text-align:right;font-family:ui-monospace,monospace;
  font-variant-numeric:tabular-nums;color:var(--ink2);}}
td.lk{{font-family:ui-monospace,monospace;font-weight:600;color:var(--ink);}}
tbody tr:hover{{background:var(--accent-soft);}}
tr.q-poor td{{color:var(--muted);}}
.pill{{font-family:ui-monospace,monospace;font-size:.68rem;font-weight:600;
  padding:1px 7px;border-radius:20px;text-transform:uppercase;letter-spacing:.04em;}}
.p-good{{background:rgba(12,143,86,.14);color:var(--good);}}
.p-marginal{{background:rgba(39,96,230,.14);color:var(--accent);}}
.p-poor{{background:rgba(121,131,143,.16);color:var(--muted);}}

.rule{{height:1px;background:var(--line);border:0;margin:40px 0;}}
.foot{{color:var(--muted);font-size:.82rem;padding:34px 0 60px;}}
.tag{{display:inline-block;font-family:ui-monospace,monospace;font-size:.72rem;
  padding:2px 8px;border:1px solid var(--line);border-radius:6px;color:var(--ink2);
  margin:0 4px 4px 0;}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:2px;vertical-align:middle;margin-right:5px;}}
</style>

<div class="hero">
  <canvas id="speccanvas"></canvas>
  <div class="wrap">
    <div class="eyebrow">Frequency-domain analysis &middot; SWITCH backbone</div>
    <h1>What network traffic looks like in the frequency domain</h1>
    <p class="lede">Every backbone link carries the same three ingredients &mdash; a sharp
    <b style="color:var(--daily)">daily line</b>, a softer
    <b style="color:var(--weekly)">weekly line</b>, and a
    <b>1/f &ldquo;red-noise&rdquo; continuum</b> that never flattens. Their proportions decide
    which forecaster wins, and at what horizon.</p>
    <p class="prov">Source: <span>lan-mon2021 / bb-usage-logs</span> &mdash; 48 backbone link-pairs,
    5-min <span>inMbps+outMbps</span>, Jan 2021 &rarr; Aug 2023 &middot; Welch PSD, 4-week Hann segments &middot;
    axis in <span>cycles / day (cpd)</span>.</p>

    <div class="kpis">
      <div class="kpi"><div class="k">48<span class="u">links</span></div><div class="lab">analysed &middot; 31 high-quality coverage</div></div>
      <div class="kpi"><div class="k">144<span class="u">cpd</span></div><div class="lab">Nyquist limit at 5-min sampling</div></div>
      <div class="kpi"><div class="k">~280<span class="u">k</span></div><div class="lab">samples / link over 2.5 years</div></div>
      <div class="kpi"><div class="k">30<span class="u">%</span></div><div class="lab">median variance in the daily line (up to 51%)</div></div>
      <div class="kpi"><div class="k">0.98<span class="u">H</span></div><div class="lab">median Hurst &mdash; near random-walk memory</div></div>
      <div class="kpi"><div class="k">54<span class="u">dB</span></div><div class="lab">sharpest daily-cycle prominence (IXG-ZH)</div></div>
    </div>
  </div>
</div>

<div class="wrap">

<div class="reading">
<h2>1 &nbsp;Why look at traffic in the frequency domain</h2>
<p>Our prediction work lives entirely in the <em>time domain</em> &mdash; persistence, seasonal-naive,
Historical-Average, GBDT, PatchTST, an STGNN. Each of those is, secretly, a statement about the
<em>frequency</em> content of the signal: persistence assumes all the energy is at very low frequency;
seasonal-naive assumes it is concentrated on the daily/weekly spectral lines; a noise floor sets the
error nobody can beat. Moving to the frequency domain lets us read those assumptions straight off the
data and predict <em>which method should win before running it.</em></p>
<p>We take the per-link total load (in&nbsp;+&nbsp;out, Mb/s) on a regular 5-min grid, remove
counter-wrap spikes, linearly detrend, and estimate the power spectral density with Welch's method
(4-week Hann windows, 50% overlap). We plot frequency in <b>cycles per day (cpd)</b>: the daily rhythm
sits at 1&nbsp;cpd, the working week at 1/7&nbsp;cpd, and 5-min sampling puts the Nyquist ceiling at
144&nbsp;cpd (periods down to 10&nbsp;minutes).</p>
</div>

<h2>2 &nbsp;The canonical spectrum</h2>
<div class="reading">
<p>One link tells most of the story. Link <b>AG-EZ</b> (32 months) shows the anatomy every backbone
link shares to some degree:</p>
</div>
<figure class="full">
  <div class="figcard"><img src="{F1}" alt="Log-log power spectral density of link AG-EZ showing daily peak, harmonics, weekly peak, 1/f continuum and noise floor"></div>
  <figcaption><b>Figure 1.</b> Power spectral density of AG-EZ, log&ndash;log. A tall <b style="color:var(--daily)">daily
  spike at 1&nbsp;cpd</b> with harmonics at 2&times; and 3&times;/day (the day is not a pure sine &mdash; it has a
  boxy business-hours shape); a <b style="color:var(--weekly)">weekly spike at 1/7&nbsp;cpd</b> plus its
  half-week harmonic; a straight <b>1/f&nbsp;&sup2; continuum</b> (slope &beta;&nbsp;&asymp;&nbsp;1) sloping down
  under everything; and a flat <b style="color:var(--good)">noise floor</b> past ~10&nbsp;cpd where the spectrum
  stops falling.</figcaption>
</figure>
<div class="reading">
<p>Read left to right, that is the whole forecasting problem:</p>
<ul class="clean">
<li><b>The spectral lines</b> (daily, weekly, harmonics) are perfectly periodic &mdash; a calendar-aware
model can reproduce them <em>at any horizon</em>. This is the part seasonal-naive and Historical-Average capture.</li>
<li><b>The 1/f continuum</b> is <em>memory</em>: power piling up toward zero frequency means the signal wanders
slowly and is strongly autocorrelated. This is what persistence exploits &mdash; and why it is unbeatable one step ahead.</li>
<li><b>The noise floor</b> is the irreducible part. Its height sets a hard <em>lower bound</em> on prediction error
that no model &mdash; not GBDT, not a transformer &mdash; can cross.</li>
</ul>
</div>

<h2>3 &nbsp;Four archetypes &mdash; links are not alike</h2>
<div class="reading">
<p>The <em>ratio</em> of line-energy to continuum-energy varies enormously across the network, and it tracks
where a link sits in the topology.</p>
</div>
<figure class="full">
  <div class="figcard"><img src="{F2}" alt="Four PSD panels: TO-ZH strongly diurnal, IXG-ZH sharp daily line, BE-FR diurnal plus heavy continuum, CE-LG nearly flat noise-like"></div>
  <figcaption><b>Figure 2.</b> Same axes, four links. <b>TO-ZH / IXG-ZH</b> (regional &amp; internet-exchange edges):
  a towering daily spike &mdash; human activity dominates. <b>BE-FR</b> (core transit): the daily spike is still there
  but rides on a fat continuum &mdash; aggregated flows blur the rhythm. <b>CE-LG</b> (heavily-aggregated core): the daily
  peak has almost <em>dissolved</em> into a smooth 1/f slope &mdash; this link is nearly a random walk.</figcaption>
</figure>
<div class="reading">
<p>The pattern is systematic and physical: <b>edge / access links are the most periodic</b> (users sleep, so the
link sleeps), while <b>the more traffic a core link aggregates, the flatter and noisier its spectrum becomes</b> &mdash;
independent diurnal flows with different phases average toward Gaussian-looking noise. Aggregation trades a
<em>predictable shape</em> for a <em>smoother level</em>.</p>
</div>

<h2>4 &nbsp;A fingerprint for every link</h2>
<div class="reading">
<p>Two numbers summarise a link's spectrum: how much variance sits in the <b>daily+weekly lines</b> (how seasonal
it is) and the <b>Hurst exponent</b> H&nbsp;=&nbsp;(&beta;+1)/2 from the continuum slope (how much slow memory it has).
Figure&nbsp;3 ranks all 31 good-coverage links by both.</p>
</div>
<figure class="full">
  <div class="figcard"><img src="{F3}" alt="Per-link bar chart: left daily plus weekly energy share, right Hurst exponent, for 31 links"></div>
  <figcaption><b>Figure 3.</b> Left: share of variance in the daily band (blue) + weekly line (violet). Right: Hurst
  exponent &mdash; <span class="dot" style="background:#e34948"></span>red&nbsp;&gt;&nbsp;1.02 (super-diffusive / trending),
  <span class="dot" style="background:#2a78d6"></span>blue near the random-walk line, <span class="dot" style="background:#1baf7a"></span>green&nbsp;&lt;&nbsp;0.85
  (flatter, whiter). TO-ZH, CHU-EL, KR-SG, IXG-ZH lead on seasonality; CE-LG, AG-RA, BE-LS are the flat, memory-heavy core links.</figcaption>
</figure>
<div class="reading">
<p>Plotting those two axes against each other maps the whole network into a <b>predictability space</b> that says,
per link, which forecaster to reach for.</p>
</div>
<figure class="full">
  <div class="figcard"><img src="{F4}" alt="Scatter of links: x diurnal energy, y Hurst exponent, size mean load, color spectral entropy"></div>
  <figcaption><b>Figure 4.</b> Every link placed by diurnal energy (x) and memory (y); marker size &prop; log mean load,
  colour = spectral entropy (yellow&nbsp;=&nbsp;peaky/predictable, blue&nbsp;=&nbsp;flat/hard). The <b>bottom-right cluster</b>
  (IXG-ZH, TO-ZH, KR-SG, CHU-EL, AG-EZ&hellip;) is where cheap seasonal models shine. The <b>upper-left</b> (CE-LG, CR-SLF,
  CE-LS) is persistence-dominated and genuinely hard past one step. Nearly every link sits at H&nbsp;&asymp;&nbsp;1 &mdash; the
  random-walk limit.</figcaption>
</figure>

<h3>Per-link table (all 48 links)</h3>
<div class="reading"><p>Full spectral metrics. <span class="mono">daily%</span>/<span class="mono">weekly%</span> = share of
signal variance on those lines; <span class="mono">SNR</span> = daily-peak prominence above the spectral median;
<span class="mono">H</span> = Hurst; <span class="mono">entropy</span> = normalised spectral entropy (1&nbsp;=&nbsp;white).
Poor-quality rows have &lt;3&nbsp;months or are mostly-idle links and are shown for completeness only.</p></div>
<div class="tablewrap">
<table>
<thead><tr><th>Link</th><th>Quality</th><th>Months</th><th>Mean load</th><th>Daily&nbsp;%</th>
<th>Weekly&nbsp;%</th><th>SNR&nbsp;dB</th><th>Hurst</th><th>Entropy</th></tr></thead>
<tbody>
{ROWS}
</tbody>
</table>
</div>

<h2>5 &nbsp;How the spectrum changes over time</h2>
<div class="reading">
<p>The spectrum is not frozen. Sliding a weekly window across 2.5&nbsp;years of link EZ-ZH shows the daily rhythm
<em>breathing</em> with the calendar.</p>
</div>
<figure class="full">
  <div class="figcard"><img src="{F5}" alt="Spectrogram of EZ-ZH over 2021-2023 with daily band, and a lower panel of daily-band power share over time"></div>
  <figcaption><b>Figure 5.</b> Top: spectrogram of EZ-ZH &mdash; the bright horizontal band at 1&nbsp;cpd is the daily
  cycle, persistent across the whole record. Bottom: the daily cycle's <em>share</em> of power. It collapses on a schedule
  &mdash; deep troughs every <b>August</b> (summer holidays) and around <b>Christmas/New&nbsp;Year</b>, when university and
  research traffic drains away and the diurnal shape washes out. It rebuilds each term.</figcaption>
</figure>
<div class="reading">
<div class="note"><div class="t">Why this matters for prediction</div>
<p>Seasonality is <b>non-stationary</b>: the same clock model is strong in November and weak in August. This is exactly
why the online / warm-started variants helped (GBDT+online lifted 1-h skill 0.27&nbsp;&rarr;&nbsp;0.44) &mdash; a model that
re-fits near each origin tracks the breathing; a model fit once does not. It also warns any GNN with a fixed adjacency
that link relationships shift seasonally.</p></div>
</div>

<h2>6 &nbsp;The bridge to time-domain prediction</h2>
<div class="reading">
<p>Frequency and time are the same information (Wiener&ndash;Khinchin: the PSD is the Fourier transform of the
autocorrelation). Figure&nbsp;6 shows both faces of AG-EZ side by side and turns them into a rule about horizons.</p>
</div>
<figure class="full">
  <div class="figcard"><img src="{F6}" alt="Left: autocorrelation of AG-EZ with daily comb and weekly peak. Right: cumulative variance below frequency."></div>
  <figcaption><b>Figure 6.</b> Left &mdash; the autocorrelation is the spectrum's twin: the daily comb and the tall spike at
  lag&nbsp;7&nbsp;days are the daily and weekly lines seen in time. Right &mdash; cumulative variance below a frequency. It is
  nearly <em>flat until 1&nbsp;cpd, then jumps</em>: for AG-EZ, ~12% of variance is slower than weekly, the daily line alone
  adds ~48% in one step, and only <b>~3% lives above 1&nbsp;cph</b> (faster than an hour).</figcaption>
</figure>
<div class="reading">
<p>That cumulative curve <em>is</em> the horizon/accuracy trade-off your <span class="mono">idea.md</span> notes
(&ldquo;less horizon &rarr; more accuracy&rdquo;), made quantitative. Predicting the signal below a frequency f is easy;
predicting above it is fighting the noise floor. Reading the spectrum against our validated results:</p>

<table style="font-size:.9rem;margin:18px 0;">
<thead><tr><th>Spectral fact</th><th>Time-domain consequence</th><th>Matches our result</th></tr></thead>
<tbody>
<tr><td>Steep 1/f continuum, H&asymp;1 (memory piles at f&rarr;0)</td><td>Best 1-step guess is &ldquo;same as now&rdquo;</td><td class="lk">Persistence unbeatable at 5&nbsp;min</td></tr>
<tr><td>~50% of variance concentrated on the daily line</td><td>Once you cross ~1 cycle, the clock beats inertia</td><td class="lk">GBDT overtakes persistence just after 5&nbsp;min &rarr; +0.27 @1h</td></tr>
<tr><td>Sharp, stable weekly line at 1/7 cpd</td><td>Time-of-week climatology is the thing to beat far out</td><td class="lk">Historical-Average <b>wins</b> at week-ahead (+0.24)</td></tr>
<tr><td>Energy is line-spectra, not broadband</td><td>A few seasonal components suffice; huge nets don't pay</td><td class="lk">DLinear / seasonal ties heavy PatchTST/STGNN at long horizon</td></tr>
<tr><td>Flat noise floor above ~1&ndash;6 cph</td><td>Sub-hour variation is largely irreducible</td><td class="lk">Short horizons accurate; error saturates, never &rarr;0</td></tr>
<tr><td>Aggregated core links = flat, high-entropy spectra</td><td>Nothing periodic to grab &rarr; only persistence helps</td><td class="lk">Flat cohort (CE-LG&hellip;) is where skill gains are smallest</td></tr>
</tbody>
</table>

<div class="note"><div class="t">The one-line takeaway</div>
<p>A link's <b>diurnal-energy fraction</b> tells you how far ahead a cheap seasonal model stays useful; its
<b>noise-floor height</b> tells you the error you can never beat. You can pick the forecaster per link from its
spectrum &mdash; no backtest required to know the shape of the answer.</p></div>
</div>

<h2>7 &nbsp;Implications for the green-routing predictor</h2>
<ul class="clean" style="max-width:var(--measure);">
<li><b>Route the model by spectrum.</b> Edge/access links (high daily fraction, bottom-right of Fig&nbsp;4) &rarr; seasonal-naive
or Historical-Average, essentially free. Aggregated core links (flat, top-left) &rarr; persistence + a light online correction;
don't waste a transformer on them.</li>
<li><b>Match horizon to the crossover.</b> The daily line at 1&nbsp;cpd is the frequency where clock beats inertia &mdash; below
sub-hourly lead times persistence rules, beyond that seasonal structure pays. That crossover is a link property, readable from
its PSD.</li>
<li><b>Re-fit for the breathing (Fig&nbsp;5).</b> Seasonal strength swings with holidays; online/warm-started models earn their
cost precisely because the daily line is non-stationary.</li>
<li><b>Set expectations from the noise floor.</b> Aggregation smooths the level but destroys the periodic shape &mdash; report
per-link predictability (noise-floor height / spectral entropy) so a low skill score reads as &ldquo;irreducible&rdquo;, not
&ldquo;bad model&rdquo;.</li>
<li><b>Privacy angle (idea.md).</b> The line-spectra are the information-dense part; the continuum is close to noise. A provider
could publish coarse spectral summaries (daily/weekly amplitudes) that support routing forecasts while hiding exact volumes &mdash;
worth a look.</li>
</ul>

<hr class="rule">
<div class="reading">
<h3>Method notes &amp; caveats</h3>
<p class="mono" style="font-size:.82rem;color:var(--muted);line-height:1.7;">
PSD via scipy.signal.welch, fs=288/day, 4-week Hann, 50% overlap, linear detrend, density scaling.
Line energy = &plusmn;6% band around each harmonic. &beta; fit on the log&ndash;log continuum over 0.02&ndash;0.7 cpd,
weekly harmonics masked out; H=(&beta;+1)/2. Spectral entropy normalised by log N.
Counter-wrap spikes (&gt;0.5&nbsp;Tb/s or &gt;median+40&middot;MAD) removed then interpolated; CE-LS carried ~1% such spikes.
H near/above 1 signals near-non-stationary (random-walk-ish) traffic &mdash; the fGn interpretation of H is then a
descriptor, not a strict stationarity claim. Directional in/out spectra are near-identical, so total load is shown.
</p>
<div style="margin-top:14px;">
<span class="tag">green_routing/analysis/spectral_analysis.py</span><span class="tag">green_routing/analysis/spectral_figures.py</span>
<span class="tag">artifacts/spectral/metrics.csv</span><span class="tag">bb-usage-logs &middot; 45 link-pairs</span>
</div>
</div>
<div class="foot">Frequency-domain analysis of SWITCH backbone traffic &middot; generated for the Green-Routing prediction study &middot;
data Jan 2021 &ndash; Aug 2023.</div>
</div>

<script>
// Ambient hero decoration: a stylised 1/f + daily-peak spectrum, drawn once.
(function(){{
  const c=document.getElementById('speccanvas');if(!c)return;
  const ctx=c.getContext('2d');let W,H;
  function css(v){{return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}}
  function resize(){{const r=c.getBoundingClientRect();const d=Math.min(devicePixelRatio||1,2);
    W=c.width=r.width*d;H=c.height=r.height*d;draw();}}
  function draw(){{
    ctx.clearRect(0,0,W,H);
    const acc=css('--accent')||'#2760e6';
    const n=W; ctx.lineWidth=1.4*Math.min(devicePixelRatio||1,2);
    ctx.strokeStyle=acc; ctx.globalAlpha=0.9;
    // pseudo-random but deterministic continuum + spectral lines
    function val(x){{ // x in 0..1  (log-freq)
      let base=Math.pow(1-x,1.7);           // 1/f fall
      let daily=0.62*Math.exp(-Math.pow((x-0.34)/0.012,2));
      let h2=0.26*Math.exp(-Math.pow((x-0.47)/0.012,2));
      let h3=0.16*Math.exp(-Math.pow((x-0.55)/0.012,2));
      let wk=0.34*Math.exp(-Math.pow((x-0.16)/0.011,2));
      let s=(Math.sin(x*140)*0.5+Math.sin(x*330+1)*0.5+Math.sin(x*70+2)); // grass
      let grass=0.05*(1-x)*s;
      return base*0.5+daily+h2+h3+wk+Math.max(0,grass);
    }}
    ctx.beginPath();
    for(let i=0;i<=n;i++){{const x=i/n;const y=val(x);
      const py=H-8*(devicePixelRatio||1)-(y/1.6)*(H*0.82);
      i?ctx.lineTo(i,py):ctx.moveTo(i,py);}}
    ctx.stroke();
    // soft glow fill
    const g=ctx.createLinearGradient(0,0,0,H);
    g.addColorStop(0,acc+'22');g.addColorStop(1,acc+'00');
    ctx.lineTo(W,H);ctx.lineTo(0,H);ctx.closePath();ctx.fillStyle=g;ctx.globalAlpha=0.5;ctx.fill();
  }}
  addEventListener('resize',resize);
  const mo=matchMedia('(prefers-color-scheme:dark)');mo.addEventListener&&mo.addEventListener('change',draw);
  resize();
}})();
</script>
"""
open(f"{OUT}/spectral_report.html","w").write(HTML)
print("wrote report:",f"{OUT}/spectral_report.html","chars",len(HTML))
PY_DONE=True
"""
"""
