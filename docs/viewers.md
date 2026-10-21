# Viewers

Two browser tools, both plain HTTP servers with no build step:

| | Prediction viewer | Topology viewer |
|---|---|---|
| Shows | a run's forecasts against the real traffic, link by link | the backbone map with traffic, CPU, power and temperature over time |
| Start | `gr-viewer [run_dir]` | `cd web && python server.py` |
| Port | 8138 | 8137 |
| Needs | a run directory (+ its panel cache) | `web/data/frames/` built from the data |
| Auth | HTTP Basic (password) | none |

To reach them from a laptop: `ssh -N -L 8138:localhost:8138 <user>@<host>` (and/or
8137), or open the ports in the SWITCHengines security group and use the public IP.

## Prediction viewer (`gr-viewer`)

```bash
gr-viewer                                   # $GR_ARTIFACTS_DIR/pred_year/merged on 0.0.0.0:8138
gr-viewer artifacts/demo --port 8139        # any run directory
gr-viewer artifacts/demo --host 127.0.0.1   # local only (behind an SSH tunnel)
recipes/year/run.sh viewer                  # the year run
```

**Password.** Every request needs HTTP Basic auth; any user name works. The password is
`$GR_VIEWER_PASSWORD`, else the contents of `~/.config/green-routing/viewer_password`
(generated on first start, mode 600; `GR_VIEWER_PASSWORD_FILE` moves it).
`GR_VIEWER_PASSWORD=off` disables auth — only with `--host 127.0.0.1`. It is plain HTTP,
so the password crosses the network unencrypted; prefer the tunnel on untrusted networks.

### Views

| View | What each point is | Needs |
|---|---|---|
| **window by window** (default) | for horizon H, the test period in consecutive H-long windows, each forecast in full from its start: a forecast at every 5-min step. Dashed lines mark the re-forecasts. | `gr-dense` logs |
| **rolling, fixed lead** | the forecast made exactly H earlier, at every origin | `gr-dense` logs |
| **sampled origins** | the harness's own records: one point per horizon at the run's origins | nothing extra |

GBDT predicts horizons directly, so it appears in the rolling and sampled views only.
Variants logged into a run's dense directory appear as their own models
(`tcn@month`, `tcn+ma1h`, `tcn@starget`, …).

### Using it

- **Chart**: wheel to zoom, drag to pan, shift+drag to box-zoom, double-click to reset;
  click a legend entry to hide a model; hover for values and signed errors. The lower
  strip is forecast − actual. Long windows are drawn as per-pixel min/max envelopes.
- **Stats panel**: MAE, RMSE, WAPE, skill vs persistence, bias and peak bias per model
  for the link; *stats over visible window* rescores only what is zoomed in.
- **Leaderboard** (header button): model × horizon for a chosen metric — median ratio
  vs persistence, share of links beating it, median/pooled WAPE, pooled MAE, peak bias,
  peak WAPE — best per horizon highlighted.
- **Link list**: filter by name; sort by mean/peak traffic, or worst-first by WAPE or by
  ratio vs persistence for a chosen model — the quickest way to find the links the
  global model can't fit.

### What it reads

Nothing is re-computed and no model is loaded:

| Source | Used for |
|---|---|
| `config.json` | window, split, horizons, link list |
| `records/<model>.csv` | the sampled view |
| `dense/meta.json`, `dense/pred_<mode>_<model>.npy` | the dense views (memory-mapped) |
| `dense/<mode>/*.csv`, `per_link.csv`, `aggregate.csv`, `peaks.csv` | stats, leaderboard, link sorting |
| `panel_<hash>.npz` | the actual traffic — searched from the run directory upwards and in `$GR_ARTIFACTS_DIR` |

### API

```text
GET /api/meta                         run config, links (+ mean/peak Mbit/s), models per view, horizons
GET /api/link?id=<link>&h=<steps>&mode=block|rolling|sampled
                                      actual series + every model's forecasts (base64 float32)
                                      + per-model stats for that link
GET /api/leaderboard?mode=            model × horizon summary
GET /api/linkscores?h=&mode=          per link: each model's WAPE and MAE ratio
GET /api/tables?mode=                 the raw metric tables as JSON
```

## Topology viewer (`web/`)

The SWITCH optical backbone map with link utilisation or traffic and node CPU / power /
temperature at any time since late 2023: scrub time, pick a window (5 min → 1 month) and
an aggregate (mean / max / min / median), hover a link or node for its time series,
click to pin; views are shareable through URL parameters.

```bash
cd web
python build_frames.py        # data -> data/frames/<YYYYMM>.json (~2.5 min per year from the cache)
python build_frames.py 2026   # one year only
python server.py              # http://localhost:8137
```

- **Layers** (top left): links coloured by utilisation % or traffic, line width = capacity
  (1 G → 200 G); nodes by CPU %, power (W) or temperature (°C).
- **Time**: slider, date picker or ▶ play (advances by the window).
- **Share a view**: `?links=1&link=util&nodes=1&node=cpu&w=604800&agg=max&t=<unix>&inspect=BA-EZ`.

`build_frames.py` reads the same data source as the package (S3 through the object
cache, or a local mirror — see `web/_source.py`): the link interfaces in
`router-interfaces/`, CPU and temperature in `routers/`, PSU power in `router-power/`.

**The link map** (`data/links.json`, `data/node_devices.json`) is committed. Rebuilding
it with `build_topology.py` needs the real topology from the research checkout
(`GR_RESEARCH_ROOT`): the fallback topology file has no interface names, so nothing
would match a traffic CSV — the script refuses rather than overwrite the working map.
Node positions (`data/nodes.json`) come from the optical network PDF
(`extract_nodes.py`); the base map is `assets/switchlan.svg`.

API (`server.py`, stdlib only):

```text
GET /api/meta                                   links + capacities, node positions, months, span
GET /api/frame?t=<unix>                         one 5-min slot: link in/out, node cpu/power/temp
GET /api/agg?t=&w=&agg=mean|max|min|median      aggregate over [t, t+w)
GET /api/series?kind=link|node&id=&t=&w=&metric= downsampled series + min/mean/median/max
```

Caveats: 42 drawable links (`WI` and `CR` have no resolved topology link); power is
sparse (few devices report PSU current and voltage, mostly for a limited period);
temperatures above 120 are treated as tenths of a degree; links are drawn as straight
lines between node centres.
