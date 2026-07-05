# SWITCHlan Traffic Map (web viewer)

Interactive viewer that displays backbone traffic on top of the SWITCHlan
optical backbone map, at any point in time.

## Run

```bash
cd web
python3 server.py            # http://localhost:8137/   (default port 8137)
```

Open <http://localhost:8137/> and use the time slider / datetime picker / play
button. Scroll to zoom, drag to pan, hover a link for in/out values.

## How it was built (3 steps)

### 1. Base map (`assets/switchlan.svg`, `index.html`, `app.js`)
The map is the SWITCH optical backbone PDF
(`network-energy-efficiency-research/switch-network-maps/optopo-SWITCHlan-A1-redacted.pdf`)
converted straight to **vector SVG** with `pdftocairo -svg`. Keeping it vector
means the drawing scales crisply and shares one coordinate space (2328×1599)
with the traffic overlay.

### 2. Traffic index (`build_index.py` → `data/index.json`)
Scans `lan-mon2021/bb-usage-logs/` (per-link monthly files, format
`timestamp inMbps outMbps`, 5-min samples, Mbps). For each of the **48 links**
it records the endpoint node codes, parallel-link instance, and every monthly
file with its exact first/last timestamp and sample count. This is the
"where to read from" index — it lets the server jump to the right file for any
requested time without scanning.

Regenerate:
```bash
python3 build_index.py
```

### 3. Node positions + interaction (`extract_nodes.py`, `server.py`, `app.js`)
- **`extract_nodes.py` → `data/nodes.json`**: uses `pdftotext -bbox` to read
  each map label's position (same coordinate space as the SVG). Maps traffic
  endpoint codes to map labels, including aliases
  (`LZ→LUZ`, `ENSI→ENS`, `FHSG→FH`, `GL→GLI`, `GO→GOS`, `HEPVD→HEP`).
  `CHU`, `GR`, `SLF` have no label on this redacted map, so their 5 links
  (`CHU-EL`, `GR-NE`, `CR-SLF`, `SA-SLF`, `LG-SLF`) are not drawn. **43/48**
  links are drawable.
- **`server.py`**: serves the static site plus an on-demand traffic API.
  Month files are parsed lazily and cached in memory (first query for a month
  ≈120 ms, later queries ≈ few ms).
- **`app.js`**: overlays one line per link into the map SVG and colours/sizes
  it by `max(in, out)` on a log scale (see legend).

## API

| endpoint | returns |
|---|---|
| `GET /api/meta` | the traffic index (links, span, interval, units) |
| `GET /api/nodes` | node → `{x, y}` map coordinates |
| `GET /api/traffic?t=<unixts>` | `{ t, matched, links: { id: {in, out, ts} } }` — nearest sample per link |

Coverage: **2021-01-01 → 2023-08-31**, 5-minute resolution. Not every link is
monitored for the whole span, so `matched` varies with the chosen time.

## Regenerate everything

```bash
python3 build_index.py      # data/index.json
python3 extract_nodes.py    # data/nodes.json
python3 server.py           # serve
```
