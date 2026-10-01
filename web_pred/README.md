# `web_pred/` — prediction viewer

Inspect a `prediction.run` output directory link by link: the **real 5-min series**
against **every model's forecasts**, zoomable, with per-link statistics beside the chart.

```bash
conda activate green
cd web_pred
python server.py                                    # ../artifacts/pred_year/merged on 0.0.0.0:8138
python server.py ../artifacts/pred_demo 8139        # any run dir, any port
python server.py ../artifacts/pred_full 8140 127.0.0.1   # localhost only
```

Then open `http://<public-ip>:8138` (the SWITCHengines security group must allow
TCP 8138) or tunnel: `ssh -N -L 8138:localhost:8138 <box>` → <http://localhost:8138>.

**Access control.** Every request needs HTTP Basic auth — the data is SWITCH network
telemetry and the box has a public IP. Any user name works; the password is
`$PRED_VIEWER_PASSWORD`, else the contents of `web_pred/.viewer_password` (generated
on first start, mode 600, gitignored). `PRED_VIEWER_PASSWORD=off` disables auth —
only do that with the server bound to 127.0.0.1 behind an SSH tunnel. Plain HTTP:
the password crosses the network unencrypted, so prefer the tunnel on untrusted
networks. Dotfiles and `*.py` are never served.

## What it reads

Everything comes from the run directory plus the panel cache that produced it —
no re-inference, no model loading:

| Source | Used for |
|---|---|
| `records/<model>.csv` | `y_true` / `y_pred` per model × link × horizon × origin |
| `panel_<hash>.npz` | the real 5-min series + timestamps — searched from the run dir upwards, matched by step count and a link-superset (so `--min-mean-mbps` subsets work) |
| `per_link.csv` | MASE, WAPE, MAE per link (link sorting, leaderboard) |
| `peaks.csv` | peak bias / peak WAPE per model × horizon (leaderboard) |
| `config.json` | window, split, horizons, link list |

A run **must** still have its `records/` — that is where the raw predictions live.

## Endpoints

    GET /api/meta                     run config, 886 links (+mean/peak Mbps), models, horizons
    GET /api/link?id=<link>&h=<steps> real series, every model's forecasts at their target
                                      times, per-model stats for that link
    GET /api/tables                   aggregate / skill / ratio / wape / peaks as JSON
    GET /api/leaderboard              model x horizon: median ratio vs persistence, share
                                      of links beaten, pooled MAE, pooled + median
                                      per-link WAPE, peak bias / peak WAPE
    GET /api/linkscores?h=<steps>     per link: each model's WAPE + MAE ratio

WAPE is sum|e| / sum|y| in percent throughout.

## Leaderboard and link sorting

**Leaderboard** (header button) shows every model × horizon for a chosen metric, best
per horizon highlighted. In the link list, *sort: WAPE* / *sort: ratio vs persistence*
rank the links worst-first for the model picked below the sort box (default: the
learned model with the best median ratio) — the quickest way to find the tail links
the global model can't fit.

## Chart

- **wheel** zoom · **drag** pan · **shift+drag** box zoom · **double-click** reset
- click a legend entry to hide a model (`real` too); hover for a tooltip with each
  model's value and its signed error
- the lower strip is `pred − real` for the visible window
- **stats over visible window** rescores MAE / RMSE / WAPE / skill / bias on exactly
  what you have zoomed into, instead of the whole scored span

## Note on zoom depth

Forecasts exist only at the run's rolling origins. `pred_full` used 48 origins spaced
**6 h** apart, so zooming below ~6 h shows the real series with no forecast markers
(the panel says so when it happens). Re-run with a smaller `--stride` for denser
origins.
