# Early runs (2024–2026-07)

Results and commands from before the year-scale benchmark, kept for the record. They used small or alphabetically truncated cohorts (`--max-links 20/30`) and older command forms (`cd code && python -m prediction.run ...` = today's `gr-benchmark ...`). Current results: [results.md](../results.md).

## Link census

**Link census** (scan 2026-07-13, details + CSVs in `artifacts/link_census/`): the raw
tree has 1 923 interface series on 208 devices; over the working window
2025-12-01 → 2026-07-11, **886 links on 141 devices are usable** (coverage ≥ 0.9 on the
5-min grid; 830 are ≥ 0.99). The panel therefore keeps **all usable links by default** —
`--max-links` is only a quick-experiment cap (it truncates in alphabetical device
order, so a capped cohort is small *and* biased). For comparison, the router-level
topology has 395 router-pair links over 85 nodes.


## The workstation environment of the time

Use the dedicated **`green-pred`** conda env (Python 3.11 + pandas/numpy/sklearn/
statsmodels/matplotlib/scipy/torch-CUDA):

```bash
/home/qyy/anaconda3/envs/green-pred/bin/python    # or: conda run -n green-pred python
```

> **Clear `PYTHONPATH`** when invoking: this box sources ROS Noetic in `~/.bashrc`,
> which injects `/opt/ros/.../dist-packages` into every env's `sys.path` and can shadow
> the env's own numpy. Prefix commands with `env -u PYTHONPATH`.

> LightGBM/XGBoost are **not** installed — Tier 3 uses scikit-learn's
> `HistGradientBoostingRegressor` (the histogram-booster equivalent). Torch is the
> **CUDA** build (RTX 3060 Laptop, CUDA 12.1); the deep/GNN tiers (4–6) run on the GPU
> automatically. Override with `PRED_DEVICE=cpu` (or `cuda`). Tiers 1–3 are CPU-only by
> nature (naive/statsmodels/sklearn).


## What the demo run shows (2024-06 → 2024-10, 30 links)

| Horizon | Winner | Skill of GBDT vs persistence |
|---|---|---|
| 5 min (1-step) | **persistence** | −0.40 (can't beat it — as the roadmap warns) |
| 1 h (12-step) | **GBDT** | +0.28 |
| day (288-step) | **GBDT** | +0.32 |

Persistence is unbeatable at 1-step; the learned model earns its cost only as the horizon
grows — the roadmap's central thesis, reproduced on real SWITCH data.

## Deep/GNN results (2024-06→2024-10, 30 links, lookback 96, 8 epochs, CPU)

Skill vs persistence (MAE), short horizons:

| model | tier | 15 min | 30 min | 1 h |
|---|---|---|---|---|
| **patchtst** | 5 | **+0.16** | **+0.22** | +0.27 |
| gbdt | 3 | +0.06 | +0.11 | **+0.27** |
| stgnn | 6 | +0.08 | +0.14 | +0.22 |
| gru | 4 | +0.01 | +0.10 | +0.14 |
| tcn | 4 | +0.05 | +0.08 | +0.06 |
| dlinear | 5 | −0.03 | −0.04 | −0.04 |

Read through the roadmap's "keep complexity only if it *clearly* beats the simpler tier"
rule: **PatchTST earns its cost at 15/30 min** (clearly > GBDT), but at **1 h it only ties
the far-cheaper GBDT** → ship GBDT there. **STGNN** beats persistence and GRU/TCN but not
GBDT on this mostly-*access*-link cohort at short horizons — the spatial/reroute signal a
graph exploits needs core links and/or longer horizons to pay off. DLinear (the linear
control) underperforming confirms PatchTST's gain is real, not an artifact.

These are small CPU nets at short horizons (where the roadmap expects deep models to be
*least* advantaged). Widen the lead by: longer horizons (`--horizons day week`), longer
`--lookback`, more `--epochs`, a core-link cohort, or bigger nets.

## Full-cohort real-data run (all usable links)

No `--max-links`: the panel keeps every usable link — 886 links / 141 devices over this
window (see *Link census* above). The first cold build reads all ~19 GB once
(~10–15 min), then the panel is cached to `artifacts/panel_<hash>.npz`.

```bash
env -u PYTHONPATH PRED_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/yuyqin/anaconda3/envs/green-pred/bin/python -m prediction.run \
    --start 2025-12-01 --end 2026-07-11 \
    --train-end 2026-05-01 --val-start 2026-05-08 \
    --val-end 2026-05-29 --test-start 2026-06-05 \
    --tiers 1 3 4 5 6 \
    --horizons 15min 1h day week \
    --max-origins 48 \
    --lookback 336 --epochs 12 \
    --out ../artifacts/pred_full
```

> Do **not** append `| tee ../artifacts/pred_full/run.log` any more: `run.py` writes
> that file itself, and a shell `tee` on the same path interleaves with it. The run
> creates `--out` too, so the old `mkdir -p` is unnecessary.

(The earlier `pred_real`/`pred_paper` results above were produced with
`--max-links 20`, i.e. ~2 % of the usable cohort — alphabetically biased. Full-cohort
numbers supersede them once this run lands.)