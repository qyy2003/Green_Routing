# Tools

Scripts outside the package. None of them is needed for forecasting.

## `topology/`

| Script | What | Needs |
|---|---|---|
| `visualize_topology.py` | the backbone hero map in the README (`docs/assets/topology_map.{svg,png}`) | `pip install -e ".[topology]"`; the research checkout next to this repo (topology JSON, Observium data, backbone usage logs) |
| `web_topology.py` | the first interactive prototype (Dash): 45 backbone pairs on a 5-min timeline; superseded by `web/` | `.[topology]`, the research checkout |
| `topology.html` | a saved Plotly export of that prototype | — |
| `check_bb_usage.py` | quick check of backbone usage logs against topology capacities | the research checkout (`GR_RESEARCH_ROOT`) |

## `upstream/`

Reference copies of the jobs that run **at SWITCH**, not here:

| Script | What |
|---|---|
| `parse_rrd.py` | the nightly Cricket RRD → CSV export and upload to object storage (needs `rrdxml`, SWITCH's `cricket` module and access to `/var/lib/cricket`) |
| `update-topology.sh` | the nightly OSPF topology extraction into the research repository (Simon Leinen, SWITCH) |

They document where the data comes from — the CSV format, the bucket, the topology
files — and are not meant to be run from this repository.
