# Per-link ISP Traffic Prediction — A Practical Roadmap

**Task:** predict per-link traffic at 5-minute granularity from a real per-link ISP dataset.

**Guiding principle:** start with the dumbest thing that could work, then escalate one tier at a time — and only keep a more complex model if it *clearly* beats the simpler one on the **same chronological split and the same metric**. Complexity is a cost; make each tier earn it.

---

## 0. Decide three things before you model anything

**1. Horizon.** Next-step (5 min ahead) is a very different problem from multi-step (e.g. the next hour = 12 steps, or next day = 288 steps). At 1-step, traffic is so autocorrelated that simple methods are hard to beat; the value of sophisticated models grows with horizon. Start with 1-step, then extend.

**2. Scope.** Three escalating framings:
- *Univariate per link* — model each link's own history in isolation.
- *Global multivariate* — one model trained across all links jointly (this is the modern default; see §2).
- *Spatiotemporal* — add the network topology as a graph so links can borrow information from their neighbours.

**3. Target transform.** Internet traffic volumes are approximately **log-normal, not Gaussian** (confirmed by long-run measurement studies). Apply `log1p` to the target, model in log space, and invert (`expm1`) before computing error metrics. This stabilises variance, tames bursts, and usually improves every model downstream.

---

## 1. The method ladder

Each tier below lists *how it works (the principle)*, *why you'd escalate to it*, and *its limits*.

### Tier 1 — Baselines (mandatory, never skip)

- **Persistence:** `ŷ(t+1) = y(t)`. Exploits the very high 5-minute autocorrelation of traffic. At 1-step this is a brutally strong baseline that many published deep models barely beat.
- **Seasonal-naive:** `ŷ(t+1) = y(t+1 − 288)` (same time yesterday) or `− 2016` (same time last week). Captures the daily/weekly cycle.
- **Historical average (HA):** mean over all past samples sharing the same (time-of-day, day-of-week) bin.

**Principle:** traffic = strong short-term autocorrelation + strong daily/weekly seasonality. These two facts alone get you most of the way with zero learning. Everything more complex must be measured as *skill relative to these*: `skill = 1 − model_error / baseline_error`. If a model can't beat seasonal-naive by a meaningful margin, it isn't learning anything useful.

### Tier 2 — Classical statistical

- **SARIMA(p,d,q)(P,D,Q)ₛ** with seasonal period `s = 288`: autoregressive (AR) terms model linear autocorrelation, differencing `d` removes trend/non-stationarity, and the seasonal terms model the daily cycle.
- **Holt-Winters (triple exponential smoothing):** three coupled update equations for level, trend, and seasonality, each smoothed exponentially. Lightweight and a good diurnal baseline.

**Principle:** linear time-series models with explicit seasonality.
**Limits:** one model per link (doesn't scale to thousands of links), `s = 288` makes seasonal ARIMA heavy and often numerically unstable, and everything is linear — misses bursts, regime shifts, and cross-link correlation. Treat this as a per-link sanity check, not a production model.

### Tier 3 — Gradient boosting (the pragmatic workhorse)

Reframe forecasting as **tabular regression**. For each `(link, time)` row:
- **Features:** recent lags (`y(t), y(t−1), …, y(t−k)`), rolling mean/std over a few windows, `sin/cos` of time-of-day, day-of-week, holiday flag, and — if you have topology — the lagged loads of neighbouring links.
- **Target:** `y(t+1)`.
- **Model:** LightGBM or XGBoost.

**Principle:** an ensemble of decision trees built by gradient boosting — nonlinear, captures feature interactions, robust to outliers, fast, and needs little tuning. Train **one global model** over all links' rows at once.
**Why escalate here:** it captures the nonlinearity and calendar structure that Tier 1–2 miss, cheaply. At 5-minute 1-step it frequently *matches or beats* deep networks. This is your real baseline-to-beat.

### Tier 4 — Deep temporal (LSTM / GRU / TCN)

- **LSTM / GRU:** a recurrent cell carries a hidden state summarising the sequence; gates control what to keep vs. forget, mitigating vanishing gradients. Learns nonlinear temporal dependencies directly from the raw series.
- **TCN (temporal convolutional network):** stacked *dilated causal* 1-D convolutions give an exponentially growing receptive field, train fully in parallel, and often match LSTM while being faster.

**Principle:** learn temporal dynamics end-to-end instead of hand-engineering lag features.
**Best practice:** train **one global model shared across all links**, with a per-link ID embedding — never one model per link. More data, better generalisation, and it handles new links. Deep temporal models pay off most on **multi-step** horizons and on **bursty** links, less so at 1-step.

### Tier 5 — Long-sequence transformers

- **PatchTST:** chops each series into patches used as tokens — efficient and strong.
- **iTransformer:** treats each series/channel as a token, so attention directly models **cross-link correlation**.
- **Informer / Autoformer / FEDformer:** long-sequence transformers adding sparse attention, trend–seasonal decomposition, or frequency-domain attention.

**Principle:** self-attention weights all past positions to capture long-range dependence.
**Caveat — important:** for short-horizon 5-minute prediction these are usually overkill, and the *DLinear* result ("Are Transformers Effective for Time Series Forecasting?") showed a single linear layer beats many of them on standard benchmarks. Only reach here for **long horizons**, and always benchmark against Tiers 1 and 3. Include a plain linear/DLinear model as a control.

### Tier 6 — Spatiotemporal GNN (the link-level frontier)

This is where your **topology** finally earns its keep.
- **Build the graph.** Since traffic lives on *edges*, use the **line graph**: each physical link becomes a node, and two link-nodes are connected if their links share a router. Alternatively, learn an adaptive adjacency.
- **Models:** DCRNN (diffusion graph-conv + GRU), STGCN, **Graph WaveNet** and **MTGNN** (which *learn* the adjacency + dilated TCN), AGCRN.

**Principle:** a link's load is physically coupled to adjacent links — they share flows and common routing — so message-passing over the topology plus a temporal module captures both the spatial and temporal axes at once. This is exactly the Andreoletti et al. (2019) DCRNN-on-backbone-links approach.
**Why escalate here:** when links are strongly spatially correlated — especially when **reroute/failure events** shift load between links together — a graph model predicts those coupled shifts that a per-link model cannot. Adaptive-graph variants (Graph WaveNet, MTGNN) usually beat fixed-topology ones. This is the natural "final" model given a real ISP topology.

### Optional frontier

- **Foundation TS models** (TimesFM, Chronos, Moirai): try **zero-shot** as a reference point, then fine-tune. Least standardised, but a fast sanity check.
- **Diffusion / probabilistic models:** if you need calibrated uncertainty or prediction intervals rather than point forecasts.

---

## 2. Things to know / remember

### Data & transforms
- **Log-transform the target** (`log1p`) — traffic is log-normal. Report metrics in the original scale.
- **Normalise per link.** Link magnitudes span orders of magnitude (core vs. access). Use per-link z-score or min-max, **fit on the training split only**. Log + reversible instance norm (RevIN) is a strong combination.
- **Counter hygiene.** If the data comes from raw SNMP interface counters, handle counter wraps/resets and convert to rates; watch for missed polls producing NaN gaps. Interpolate short gaps, flag long ones, and don't let them leak into features/labels.
- **Utilisation vs. volume.** If you have link capacities, also consider `load / capacity` — it's what congestion and energy decisions care about, and it's bounded to [0, 1].

### Temporal structure
- **Two seasonalities:** daily (288 steps) and weekly (2016 steps). Always add time-of-day (`sin/cos`) and day-of-week features. Expect weekends and holidays to differ.
- **Bursts & anomalies.** DDoS, flash crowds, and link failures cause spikes and sudden level shifts. Decide up front whether to *predict* them (hard) or be *robust* to them (Huber loss; or quantile/pinball loss if you want intervals). Remember a link can jump because a *different* link failed and traffic rerouted onto it — that's a spatial signal a GNN can exploit.
- **Concept drift / non-stationarity.** Traffic grows, capacity gets upgraded, new services shift patterns. Plan for periodic or online **retraining**, and consider RevIN to normalise each input window and de-normalise the output so the model is less sensitive to slow level drift.

### Modeling choices
- **Global > local.** One model across all links beats one-model-per-link once you're using tree/deep models — more data, cross-link transfer, and generalisation to new links. Give the model a link embedding.
- **Multi-step strategy.** *Recursive* (feed predictions back — simple but errors compound), *direct* (a separate head per horizon), or *seq2seq* (encoder–decoder — best for longer horizons). For 5-minute 1-step, single-step is fine; for "next hour," prefer direct or seq2seq.
- **Input window.** Give the model enough history to *see* a full daily cycle (≥ 288 steps) if you want it to learn diurnal shape — though with GBDT, a few hours of lags plus calendar features often suffices.

### Evaluation (this is where most projects quietly go wrong)
- **Chronological split, no shuffling.** Fit scalers on train only. No future information in any feature (no leakage).
- **Rolling-origin / walk-forward evaluation**, not a single split, for a realistic estimate.
- **Metrics:** MAE and RMSE in original units. Use **MAPE only if links stay well away from zero** — it explodes on small/idle links; prefer sMAPE or WAPE otherwise. Report **per-link and aggregated**, and **per-horizon** for multi-step.
- **Always report skill vs. persistence and seasonal-naive.** "We beat naive by X%" is the number that actually means something.
- **Report peak/burst error separately.** Accuracy during high-load or congestion periods is what matters operationally; a good average can hide bad behaviour exactly when it counts.

### Workflow discipline
- **Start simple, escalate only on a real win.** Each tier must beat the previous by a margin that justifies its cost — on the identical split and metric.
- **Plot, don't just tabulate.** Overlay predicted vs. actual for a few representative links (a busy core link, a quiet access link, one with a burst). This catches the classic failure mode where a model has merely **learned to output the last value one step late** — good-looking numbers, useless forecast.

---

## 3. Concrete first steps

1. **EDA.** Plot several links; compute ACF/PACF; confirm daily & weekly cycles; check log-normality; quantify missingness and the magnitude range across links.
2. **Baselines.** Persistence, seasonal-naive (day and week), HA. Lock in the metric and the chronological split now — everything else is measured against this.
3. **LightGBM (global)** with lag + calendar (+ neighbour-link) features. This becomes your strong baseline-to-beat.
4. **Global LSTM/GRU or TCN.** Compare against LightGBM at 1-step and at your target multi-step horizon.
5. **Spatiotemporal GNN** (Graph WaveNet or DCRNN on the line graph) — *if* topology is available and links are correlated.
6. **(Optional)** PatchTST / iTransformer for longer horizons, and a foundation model zero-shot as a reference.

---

## Quick reference

| Tier | Principle (one line) | Best for | Cost |
|---|---|---|---|
| Baselines | autocorrelation + seasonality, no learning | the bar to beat | trivial |
| SARIMA / Holt-Winters | linear autoregression + explicit seasonality | per-link sanity check | low, per-link |
| Gradient boosting | nonlinear regression on lag + calendar features | strong short-horizon baseline | low |
| LSTM / GRU / TCN | learned nonlinear temporal dynamics (global) | bursts, multi-step | medium |
| PatchTST / iTransformer | attention for long-range & cross-link context | long horizons | high |
| Spatiotemporal GNN | message-passing over topology + temporal module | correlated links, reroute events | high |

**The one rule to remember:** every tier is judged as *skill over persistence and seasonal-naive*, on the same chronological split. If it doesn't clearly beat the tier below, don't ship it.
