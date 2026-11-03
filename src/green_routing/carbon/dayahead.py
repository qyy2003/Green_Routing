"""Day-ahead forecasts of an hourly series, issued at 00:00 UTC for the 24 hours of the day:
the total traffic behind the flat carbon rate (recipes/flat_forecast).

- **Prior knowledge** (``calendar_frame``): Swiss public holidays weighted by the share of the
  population whose canton has the day off, the universities' lecture weeks (swissuniversities:
  ISO weeks 38-51 and 8-22) and the Christmas closure.
- **Features** (``design``) use only data before the issue time: the level of the last week,
  the recent hours, the same hour 1..28 days back, daily means, and the target's calendar.
  Optionally, holidays in the history are replaced by the same weekday a week earlier.
- **Models** (``backtest``): direct regressions, one row per target hour, refitted every k
  days on everything known so far (or once, on a fixed training block).
- **Scoring** (``score``, ``paired``): hourly and daily WAPE on the hours every forecast
  covers; differences with a bootstrap over whole weeks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TZ = "Europe/Zurich"

# Public holidays: share of the Swiss population (2024) whose canton has the day off. Days
# observed everywhere are 1; Good Friday is not a holiday in TI and VS, Labour Day only in
# ZH, BS, BL, SH, TG, NE, JU, TI, Corpus Christi / Assumption / Immaculate Conception only in
# the catholic cantons, the Jeune genevois in GE, the Lundi du Jeune in VD.
HOLIDAYS = {
    "2024-01-01": ("New Year", 1.0), "2024-01-02": ("Berchtoldstag", 0.7), "2024-03-29": ("Good Friday", 0.92),
    "2024-04-01": ("Easter Monday", 0.96), "2024-05-01": ("Labour Day", 0.35), "2024-05-09": ("Ascension", 1.0),
    "2024-05-20": ("Whit Monday", 0.96), "2024-05-30": ("Corpus Christi", 0.27), "2024-08-01": ("National Day", 1.0),
    "2024-08-15": ("Assumption", 0.27), "2024-09-05": ("Jeune genevois", 0.06), "2024-09-16": ("Lundi du Jeune", 0.10),
    "2024-11-01": ("All Saints", 0.33), "2024-12-25": ("Christmas", 1.0), "2024-12-26": ("St Stephen", 0.9),
    "2025-01-01": ("New Year", 1.0), "2025-01-02": ("Berchtoldstag", 0.7), "2025-04-18": ("Good Friday", 0.92),
    "2025-04-21": ("Easter Monday", 0.96), "2025-05-01": ("Labour Day", 0.35), "2025-05-29": ("Ascension", 1.0),
    "2025-06-09": ("Whit Monday", 0.96), "2025-06-19": ("Corpus Christi", 0.27), "2025-08-01": ("National Day", 1.0),
    "2025-08-15": ("Assumption", 0.27), "2025-09-11": ("Jeune genevois", 0.06), "2025-09-22": ("Lundi du Jeune", 0.10),
    "2025-12-08": ("Immaculate Conception", 0.25), "2025-12-25": ("Christmas", 1.0), "2025-12-26": ("St Stephen", 0.9),
    "2026-01-01": ("New Year", 1.0), "2026-01-02": ("Berchtoldstag", 0.7), "2026-03-19": ("St Joseph", 0.10),
    "2026-04-03": ("Good Friday", 0.92), "2026-04-06": ("Easter Monday", 0.96), "2026-05-01": ("Labour Day", 0.35),
    "2026-05-14": ("Ascension", 1.0), "2026-05-25": ("Whit Monday", 0.96), "2026-06-04": ("Corpus Christi", 0.27),
    "2026-08-01": ("National Day", 1.0), "2026-09-10": ("Jeune genevois", 0.06), "2026-09-21": ("Lundi du Jeune", 0.10),
    "2026-12-08": ("Immaculate Conception", 0.25), "2026-12-25": ("Christmas", 1.0), "2026-12-26": ("St Stephen", 0.9),
}
CAL_SPAN = ("2024-01-01", "2026-12-31")


def lectures(dates: pd.DatetimeIndex) -> np.ndarray:
    """1 in the lecture weeks of the Swiss universities (ISO weeks 38-51 and 8-22)."""
    w = dates.isocalendar().week.to_numpy()
    return (((w >= 38) & (w <= 51)) | ((w >= 8) & (w <= 22))).astype(float)


def christmas(dates: pd.DatetimeIndex) -> np.ndarray:
    """1 from 24 December to 2 January (the universities' closure)."""
    md = dates.month * 100 + dates.day
    return ((md >= 1224) | (md <= 102)).astype(float)


def calendar_frame(days: pd.DatetimeIndex, holidays: dict | None = None) -> pd.DataFrame:
    """Per local day: weekday, holiday weight, Christmas closure, lecture week."""
    holidays = HOLIDAYS if holidays is None else holidays
    days = pd.DatetimeIndex(days).normalize()
    key = days.strftime("%Y-%m-%d")
    if key.min() < CAL_SPAN[0] or key.max() > CAL_SPAN[1]:
        raise ValueError(f"holiday table covers {CAL_SPAN}, asked for {key.min()}..{key.max()}")
    hol = np.array([holidays.get(k, ("", 0.0))[1] for k in key])
    return pd.DataFrame({"dow": days.dayofweek.to_numpy(), "hol": hol, "xmas": christmas(days),
                         "lect": lectures(days)}, index=days)


# --- features ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    """What a direct day-ahead regression sees. Defaults reproduce ``total.gbdt_day_ahead``."""
    hist: int = 168                     # the last hours before the issue time, as features
    lags: tuple = ()                    # same hour k days back, k in lags
    daily: int = 0                      # mean of each of the last n days
    cal: str = "basic"                  # none | time (hour, weekday) | basic (+ holiday flag, workday) | full
    holidays: str = "old"               # old (calendar_features.HOLIDAYS, flag) | weighted | national | table
    table: tuple = ()                   # (date, weight) pairs for holidays="table" (another country)
    tz: str = TZ                        # local time of the calendar features
    clean: bool = False                 # replace holidays in the history by the same weekday a week earlier
    base: int = 168                     # hours whose mean normalises inputs and target
    model: str = "gbdt"                 # gbdt | ridge
    loss: str = "squared_error"         # gbdt loss on the log ratio: squared_error | absolute_error
    halflife: float | None = None       # sample weights halve every halflife days back (None = equal)
    params: dict = field(default_factory=dict)

    def but(self, **kw) -> "Spec":
        return replace(self, **kw)


def _holiday_weights(days: pd.DatetimeIndex, how: str, table: tuple = ()) -> np.ndarray:
    key = pd.DatetimeIndex(days).strftime("%Y-%m-%d")
    if how == "table":
        t = dict(table)
        return np.array([t.get(k, 0.0) for k in key])
    if how == "old":
        from ..prediction.calendar_features import HOLIDAYS as OLD
        return np.isin(key, sorted(OLD)).astype(float)
    w = np.array([HOLIDAYS.get(k, ("", 0.0))[1] for k in key])
    return (w >= 0.9).astype(float) if how == "national" else w


def _utc(x) -> pd.Timestamp:
    x = pd.Timestamp(x)
    return x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")


def _local_days(t: pd.DatetimeIndex, tz: str = TZ) -> pd.DatetimeIndex:
    return t.tz_convert(tz).tz_localize(None).normalize()


def cleaned(s: pd.Series, holidays: str = "weighted", table: tuple = (), tz: str = TZ) -> pd.Series:
    """The series with every holiday (weight >= 0.5) and Christmas-closure day replaced by
    the same hours a week earlier (repeatedly, so a run of such days reaches a normal week)."""
    d = _local_days(s.index, tz)
    bad = (_holiday_weights(d, holidays, table) >= 0.5) | (christmas(d) > 0)
    v = s.to_numpy(float).copy()
    for i in np.where(bad)[0]:
        if i >= 168:
            v[i] = v[i - 168]
    return pd.Series(v, index=s.index)


def design(s: pd.Series, spec: Spec, first_day=None, last_day=None):
    """Rows (issue day, target hour j = 0..23) for every 00:00 UTC issue day in
    [first_day, last_day] with enough history. ``s`` is hourly (UTC, complete grid, gaps
    filled); targets are log(s) minus the log of the base window's mean.
    Returns X (DataFrame), y, base, target times, issue days."""
    hw_of = lambda d: _holiday_weights(d, spec.holidays, spec.table)                 # noqa: E731
    hist_s = cleaned(s, spec.holidays if spec.holidays != "old" else "weighted", spec.table, spec.tz) if spec.clean else s
    v = np.log(np.maximum(s.to_numpy(float), 1e-6))
    h = np.log(np.maximum(hist_s.to_numpy(float), 1e-6))
    H = s.index
    need = max(spec.hist, spec.base, 24 * max(spec.lags, default=0), 24 * spec.daily, 1)
    pos = np.where((H.hour == 0) & (np.arange(len(H)) >= need) & (np.arange(len(H)) + 24 <= len(H)))[0]
    if first_day is not None:
        pos = pos[H[pos] >= _utc(first_day)]
    if last_day is not None:
        pos = pos[H[pos] <= _utc(last_day)]
    raw = np.exp(h)
    cs = np.r_[0, np.cumsum(raw)]
    base = np.log((cs[pos] - cs[pos - spec.base]) / spec.base)                    # [D]
    tgt = pos[:, None] + np.arange(24)[None, :]                                    # [D, 24]
    cols = {}
    if spec.hist:
        win = pos[:, None] + np.arange(-spec.hist, 0)[None, :]
        for k in range(spec.hist):
            cols[f"h{spec.hist - k}"] = np.repeat(h[win[:, k]] - base, 24)
    for k in spec.lags:
        cols[f"lag{k}d"] = (h[tgt - 24 * k] - base[:, None]).ravel()
    for k in range(1, spec.daily + 1):
        dm = np.log((cs[pos - 24 * (k - 1)] - cs[pos - 24 * k]) / 24)
        cols[f"day{k}"] = np.repeat(dm - base, 24)
    t = H[tgt.ravel()]
    cols["j"] = np.tile(np.arange(24), len(pos)).astype(float)
    if spec.cal != "none":
        lt = t.tz_convert(spec.tz)
        days = _local_days(t, spec.tz)
        cols["hour"] = lt.hour.to_numpy(float)
        cols["dow"] = lt.dayofweek.to_numpy(float)
        if spec.cal != "time":
            hw = hw_of(days)
            cols["hol"] = hw
            cols["workday"] = ((lt.dayofweek < 5) & (hw < 0.5)).astype(float)
        if spec.cal == "full":
            xm, le = christmas(days), lectures(days)
            cols["xmas"] = xm
            cols["lect"] = le
            # what changes from the history week to the target day (a holiday or term start
            # in either moves the level that normalises the target)
            issue = _local_days(H[pos], spec.tz)
            back = [issue - pd.Timedelta(days=k) for k in range(1, 8)]
            lect_hist = np.mean([lectures(b) for b in back], axis=0)
            off_hist = np.mean([np.maximum(hw_of(b), christmas(b)) for b in back], axis=0)
            cols["lect_change"] = le - np.repeat(lect_hist, 24)
            cols["off_hist"] = np.repeat(off_hist, 24)
            wk = days - pd.Timedelta(days=7)
            cols["off_lag7"] = np.maximum(hw_of(wk), christmas(wk))
    X = pd.DataFrame(cols)
    y = (v[tgt] - base[:, None]).ravel()
    return X, y, np.repeat(base, 24), t, np.repeat(H[pos], 24)


# --- models and backtests -------------------------------------------------------------------

def _model(spec: Spec, seed: int):
    if spec.model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=spec.params.get("alpha", 10.0)))
    p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)
    p.update(spec.params)
    return HistGradientBoostingRegressor(loss=spec.loss, random_state=seed, **p)


def backtest(s: pd.Series, actual: pd.Series, spec: Spec, test_days: pd.DatetimeIndex, refit_every: int | None = 7,
             train_end=None, train_start=None, seeds=(0,)) -> pd.Series:
    """Forecasts (hourly, raw scale) for every hour of ``test_days``. With ``refit_every`` = k,
    the model is refitted every k days on all issue days whose 24 targets are known by then
    (rows whose actual value is missing are skipped); with None, once on issue days before
    ``train_end``. ``s`` feeds the inputs (gap-filled); ``actual`` gives the targets."""
    test_days = pd.DatetimeIndex(test_days)
    X, y, base, t, day = design(s, spec, first_day=train_start, last_day=test_days[-1])
    known = actual.reindex(t).notna().to_numpy()
    out = np.full(len(y), np.nan)
    if refit_every is None:
        cuts = [test_days[0]]
        ends = [_utc(train_end) if train_end is not None else test_days[0]]
    else:
        cuts = list(test_days[::refit_every])
        ends = cuts
    cuts = cuts + [test_days[-1] + pd.Timedelta(days=1)]
    for a, b, e in zip(cuts[:-1], cuts[1:], ends):
        tr = (day < e) & (t < e) & known
        w = None
        if spec.halflife:
            age = (e - day[tr]).days.to_numpy()
            w = 0.5 ** (age / spec.halflife)
        te = (day >= a) & (day < b)
        if not te.any():
            continue
        pred = np.zeros(te.sum())
        for sd in seeds:
            m = _model(spec, sd)
            if w is not None:
                m.fit(X[tr], y[tr], **({"sample_weight": w} if spec.model == "gbdt" else {"ridge__sample_weight": w}))
            else:
                m.fit(X[tr], y[tr])
            pred += m.predict(X[te]) / len(seeds)
        out[te] = np.exp(pred + base[te])
    keep = (day >= test_days[0]) & (day <= test_days[-1])
    return pd.Series(out[keep], index=t[keep])


# --- rules ------------------------------------------------------------------------------------

def rules(s: pd.Series) -> dict:
    """Simple day-ahead rules on the hourly series (each value known at 00:00 UTC)."""
    wk = lambda k: sum(s.shift(168 * i) for i in range(1, k + 1)) / k        # noqa: E731
    med = pd.concat([s.shift(168 * i) for i in range(1, 5)], axis=1).median(axis=1)
    return {"same hour last week": s.shift(168), "mean of the same hour, last 4 weeks": wk(4),
            "median of the same hour, last 4 weeks": med}


# --- scoring -----------------------------------------------------------------------------------

def common_hours(F: dict, y: pd.Series, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Hours of whole ``days`` where the actual value and every forecast exist."""
    idx = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h")
    ok = y.reindex(idx).notna()
    for f in F.values():
        ok &= f.reindex(idx).notna()
    return idx[ok.to_numpy()]


def score(F: dict, y: pd.Series, idx: pd.DatetimeIndex) -> pd.DataFrame:
    yy = y.reindex(idx)
    rows = []
    for k, f in F.items():
        e = f.reindex(idx) - yy
        dd = e.groupby(idx.normalize()).sum().abs().sum() / yy.sum()
        ape = (e.abs() / yy)
        rows.append(dict(forecast=k, hourly_WAPE=float(e.abs().sum() / yy.sum()), daily_WAPE=float(dd),
                         bias=float(e.sum() / yy.sum()), p90_hour_APE=float(ape.quantile(0.9)),
                         worst_day_WAPE=float((e.abs().groupby(idx.normalize()).sum() / yy.groupby(idx.normalize()).sum()).max())))
    return pd.DataFrame(rows).sort_values("hourly_WAPE").reset_index(drop=True)


def paired(fa: pd.Series, fb: pd.Series, y: pd.Series, idx: pd.DatetimeIndex, reps: int = 4000, seed: int = 0) -> dict:
    """Hourly WAPE of b minus a, with a 95 % interval from resampling whole weeks."""
    yy = y.reindex(idx)
    ea, eb = (fa.reindex(idx) - yy).abs(), (fb.reindex(idx) - yy).abs()
    wk = idx.tz_convert(None).to_period("W")
    A, B, Y = ea.groupby(wk).sum().to_numpy(), eb.groupby(wk).sum().to_numpy(), yy.groupby(wk).sum().to_numpy()
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(Y), (reps, len(Y)))
    d = (B[i].sum(1) - A[i].sum(1)) / Y[i].sum(1)
    days = idx.normalize()
    da = ea.groupby(days).sum() / yy.groupby(days).sum()
    db = eb.groupby(days).sum() / yy.groupby(days).sum()
    return dict(diff=float((B.sum() - A.sum()) / Y.sum()), lo=float(np.quantile(d, 0.025)), hi=float(np.quantile(d, 0.975)),
                days_better=float((db < da).mean()), weeks=len(Y))


# --- pretrained (foundation) models, zero-shot ------------------------------------------------

def _hourly_covariates(t: pd.DatetimeIndex) -> dict:
    d = _local_days(t)
    return {"holiday": _holiday_weights(d, "weighted"), "christmas": christmas(d), "lectures": lectures(d)}


def foundation(s: pd.Series, test_days: pd.DatetimeIndex, model: str = "chronos-2", context: int = 2048,
               covariates: bool = False, step: str = "1h", device: str = "cuda", batch: int = 32,
               finetune: dict | None = None) -> pd.Series:
    """Zero-shot day-ahead forecasts from a pretrained model: for every issue day, the last
    ``context`` steps of ``s`` (complete grid, gaps filled) before 00:00 UTC -> the next 24
    hours (median). ``step`` = 5min feeds the 5-minute series and averages the forecast
    into hours. ``covariates`` adds the holiday weight, Christmas and lecture weeks as known
    future inputs (Chronos-2 natively; TimesFM via its in-context regression), or a function
    of the timestamps returning such a dict (another country's calendar). ``finetune``
    (Chronos-2 only) = dict(cutoff, steps, lr, mode, out): fine-tune a copy on the series
    before ``cutoff`` first."""
    test_days = pd.DatetimeIndex(test_days)
    cov_of = covariates if callable(covariates) else _hourly_covariates
    per_h = 12 if step == "5min" else 1
    horizon = 24 * per_h
    v = s.to_numpy(np.float32)
    starts = [s.index.get_loc(d) for d in test_days]
    ctx = [v[max(0, i - context):i] for i in starts]
    fut_t = [s.index[i:i + horizon] if i + horizon <= len(s) else
             pd.date_range(s.index[i], periods=horizon, freq=step) for i in starts]
    out = []
    if model.startswith("chronos"):
        import torch
        from chronos import BaseChronosPipeline
        repo = {"chronos-2": "amazon/chronos-2", "chronos-bolt": "amazon/chronos-bolt-base"}[model]
        pipe = BaseChronosPipeline.from_pretrained(repo, device_map=device, torch_dtype=torch.float32)
        if finetune:
            cut = s.index.get_loc(_utc(finetune["cutoff"]))
            mode = finetune.get("mode", "full")
            pipe = pipe.fit(inputs=[v[:cut]], prediction_length=horizon, finetune_mode=mode, context_length=context,
                            learning_rate=finetune.get("lr", 1e-5 if mode == "lora" else 1e-6),
                            num_steps=finetune.get("steps", 500), batch_size=finetune.get("batch", 64),
                            output_dir=str(finetune["out"]), remove_printer_callback=True,
                            report_to=[], save_strategy="no", logging_steps=100)
        if covariates:
            inputs = []
            for i, c, ft in zip(starts, ctx, fut_t):
                pt = s.index[max(0, i - context):i]
                pc, fc = cov_of(pt), cov_of(ft)
                inputs.append({"target": c, "past_covariates": pc, "future_covariates": fc})
            for k in range(0, len(inputs), batch):
                q, _ = pipe.predict_quantiles(inputs[k:k + batch], prediction_length=horizon, quantile_levels=[0.5])
                out += [x[0, :, 0].cpu().numpy() for x in q]
        else:
            for k in range(0, len(ctx), batch):
                q, _ = pipe.predict_quantiles([torch.tensor(c) for c in ctx[k:k + batch]], prediction_length=horizon,
                                              quantile_levels=[0.5])
                out += [x.reshape(-1, horizon, 1)[0, :, 0].cpu().numpy() for x in q]
        del pipe
    elif model == "timesfm-2.5":
        import timesfm
        import torch
        torch.set_float32_matmul_precision("high")
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=context, max_horizon=max(256, horizon), normalize_inputs=True,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=True, fix_quantile_crossing=True, per_core_batch_size=batch,
                                         return_backcast=covariates))
        if covariates:
            cov = {k: [] for k in ("holiday", "christmas", "lectures")}
            for i, c, ft in zip(starts, ctx, fut_t):
                pt = s.index[max(0, i - context):i]
                pc, fc = cov_of(pt), cov_of(ft)
                for k in cov:
                    cov[k].append(np.r_[pc[k], fc[k]])
            point, _ = m.forecast_with_covariates(inputs=[c.astype(float) for c in ctx], dynamic_numerical_covariates=cov,
                                                  xreg_mode="xreg + timesfm", ridge=1.0)
            out = [np.asarray(p)[:horizon] for p in point]
        else:
            point, _ = m.forecast(horizon=horizon, inputs=[c.astype(float) for c in ctx])
            out = [np.asarray(p)[:horizon] for p in point]
        del m
    elif model.startswith("timestral"):
        # Mistral's Timestral (2026-10): "timestral" = the 209M model, "timestral-28M" etc. the
        # smaller ones. Contexts of unequal length are left-padded with NaN (= missing).
        import torch
        from timestral import TimestralForecaster
        size = model.split("-", 1)[1] if "-" in model else "209M"
        fm = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=device)
        for k in range(0, len(ctx), batch):
            part = list(zip(starts[k:k + batch], ctx[k:k + batch], fut_t[k:k + batch]))
            n = max(len(c) for _, c, _ in part)
            X = np.full((len(part), 1, n), np.nan, np.float32)
            for r, (_, c, _) in enumerate(part):
                X[r, 0, n - len(c):] = c
            kw = {}
            if covariates:
                rows = []
                for i, c, ft in part:
                    pt = s.index[max(0, i - context):i]
                    pc, fc = cov_of(pt), cov_of(ft)
                    rows.append(np.stack([np.r_[np.full(n - len(c), np.nan), pc[key], fc[key]] for key in pc]))
                kw["known_covariates"] = np.stack(rows).astype(np.float32)
            with torch.inference_mode():
                f = fm.forecast(X, prediction_length=horizon, **kw)
            out += list(f.median[:, 0, :].float().cpu().numpy())
        del fm
    else:
        raise ValueError(model)
    vals, idx = [], []
    for d, o in zip(test_days, out):
        vals.append(np.asarray(o, float).reshape(24, per_h).mean(axis=1))
        idx.append(pd.date_range(d, periods=24, freq="h"))
    return pd.Series(np.concatenate(vals), index=pd.DatetimeIndex(np.concatenate(idx)))
