# SWITCHlan Backbone Viewer — traffic & telemetry

Interactive viewer that overlays the SWITCH Cricket monitoring dataset on the
optical backbone map, at any point in time. Links show traffic / utilization;
nodes show CPU / power / temperature.

## Run

```bash
cd web
python3 server.py            # http://localhost:8137/   (default port 8137)
```

Open <http://localhost:8137/>.

- **Layers panel (top-left):** toggle **Links** (metric: Utilization % or Traffic Mbps)
  and **Nodes** (metric: CPU %, Power W, or Temperature °C).
- **Line width = link capacity** (1G → 200G; see the width key in the panel).
- **Time:** slider, datetime picker, or ▶ play (step size selectable).
- Scroll = zoom, drag = pan, hover a link/node for exact values.
- URL params for sharing a view: `?links=1&link=util&nodes=1&node=cpu&t=<unix>`.

## Data source

`/media/yuyqin/share/switch` — the parsed Cricket RRD export (see
[../dataset.md](../../dataset.md)). 5-minute samples, UTC.

Subtrees used:
- `router-interfaces/` → link traffic (ifHCInOctets/OutOctets → Mbps)
- `routers/` → node CPU % and temperature
- `router-power/` → node power (Cisco_PSU_U × Cisco_PSU_I / 1e6 → W, summed over PSUs)

Every CSV has the **two-era column quirk** (early rows in `ds0..dsN`, later rows
in the named columns); the preprocessor coalesces them by position.

## Pipeline

```bash
python3 build_topology.py   # -> data/links.json, data/node_devices.json
python3 build_frames.py     # -> data/frames/<YYYYMM>.json  (slow: reads ~250 CSVs, ~3 min)
python3 server.py           # serve
```

1. **`build_topology.py`** parses `switch-network-topology.txt`
   (`swiag2 HundredGigE0/0/0/4 <=> swira4 …` = link AG↔RA), resolves each device
   to a map node (strip `swi`, trailing digits → uppercase, with the label
   aliases in `nodes.json`), and records **42 drawable node-pairs** with their
   **capacity** (summed member speeds) and the interface CSV(s) that carry their
   traffic. It also lists the router/PSU CSVs per node.
2. **`build_frames.py`** reads only the needed CSVs, coalesces ds/named, converts
   units, and resamples onto each month's 5-minute grid → compact
   `data/frames/<YYYYMM>.json` (link in/out Mbps; node cpu/pw/temp).
3. **`server.py`** serves the static site + `/api/frame?t=<unix>` (nearest slot,
   month files cached in memory) and `/api/meta` (links, capacities, node
   positions, span).

Node positions come from `extract_nodes.py` → `data/nodes.json` (label
coordinates read from the PDF via `pdftotext -bbox`); the base map is
`assets/switchlan.svg` (`pdftocairo -svg` of the optical PDF).

## API

| endpoint | returns |
|---|---|
| `GET /api/meta` | links (with `capacity_bps`, endpoints), node positions, months, time span, units |
| `GET /api/frame?t=<unix>` | `{ t, slot_ts, links:{id:{in,out}}, nodes:{code:{cpu,pw,temp}} }` |

## Coverage & caveats

- Time span **Dec 2023 → mid-2026**, 5-min resolution (varies per series).
- **42 links** drawable; `WI` and `CR` have no resolved topology link.
- **Power is sparse:** only a subset of devices report PSU U/I, mostly for a
  limited window — expect many grey (no-data) node markers on the Power layer.
- Temperature is available for the router platforms that expose `tmp*`/temp
  datasources (raw values >120 are treated as ×10 and divided).
- Overlay links are straight lines between node centres (not the drawn cable
  routing); values are exact, geometry is simplified.
