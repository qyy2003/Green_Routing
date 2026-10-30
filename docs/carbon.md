# Carbon bills

Who on SWITCHlan is responsible for which gram of CO₂, hour by hour, and how well can the
bill be known a day in advance? This is the second half of the project: the traffic
forecasts ([forecasting.md](forecasting.md)) feed a carbon-accounting pipeline that
splits the network's emissions among every customer and every external network, settles
the bill after the fact, and issues a day-ahead forecast of it.

The method follows the GHG Protocol Scope 2 location-based approach, with hourly,
flow-traced, combustion-only grid factors. It is a screening study, not an assured
inventory. Report: [SWITCHlan Forecast Carbon Bills](https://claude.ai/artifact/7QE4r5LaNaHAnvXNrzz6bk)
(private claude.ai page, built by `recipes/carbon`).

## Quick start

```bash
# prerequisites: the year run with dense logs (recipes/year), S3 access, the reference
# inputs in $GR_REFERENCE_DIR (below), EM_API for a fresh grid download
recipes/carbon/run.sh all          # ~3 min with a warm cache; or step by step:

gr-network all                     # link ends, tiers, roles, reverse-DNS far ends
gr-carbon grid [--download]        # hourly grid factors, day-ahead rules
gr-carbon inventory --year 2025    # routers, models, power, shelves, site energy, Scope 2
gr-carbon dependence               # power vs traffic of every metered router
gr-carbon allocate --year 2025     # pools, doors, parties, annual bills
gr-carbon bills                    # hourly settled + forecast bills over the test block
python recipes/carbon/build_report.py && python recipes/carbon/verify.py
```

## The method

Every hour, each party's bill has two parts:

```text
settled bill   C_k(h) = Σ_d P_acc[d,k] · I_z(d)(h)  +  B(h) · V_k(h) / Σ_j V_j(h)
backbone pool  B(h)   = Σ_d P_bb[d] · I_z(d)(h)  +  Σ_e P_e · I_z(e)(h)
forecast bill  Ĉ_k(h) : the same with V̂ (day-ahead traffic) and Î (day-ahead grid factor)
```

| Symbol | Meaning |
|---|---|
| `P_acc[d,k]` | access power of router `d` billed to party `k`: access routers wholly to their site, PoP routers by the port speed of `k`'s doors on them |
| `P_bb[d]`, `P_e` | the backbone share of router `d`; optical shelf `e` (all of it). Site power, i.e. IT power × PUE |
| `I_z(h)` | hourly grid factor of the site's zone (CH; FR for CERN), g CO₂e/kWh, combustion only |
| `V_k(h)` | bytes in + out through `k`'s doors in hour `h` |

- **Access pool, by allocated bandwidth.** Equipment that exists for one party: an access
  router at a customer site, and on PoP routers the optics of customer-facing ports plus
  the chassis base in proportion to customer-facing port capacity. Port speed stands in
  for the contracted rate until SWITCH provides it.
- **Backbone pool, by bytes.** The trunk share of PoP routers and every ECI optical shelf.
- **Doors and parties.** A door is a port where traffic enters or leaves SWITCHlan: a
  customer port, a customer site's access router, a link to an external network, or a
  trunk into a router missing from the export (`hidden:<router>`, standing for everything
  behind it). Customers *and* external networks (ISPs, IXPs, CDNs, GÉANT) are billed;
  each byte is billed half where it enters and half where it leaves.
- **Emissions are pooled, not energy**: each device's power meets its own zone's hourly
  factor before the split.
- **Energy needs no forecast.** Router power is < 1 % traffic-dependent (`gr-carbon
  dependence`), so the forecast bill differs from the settled one only through `V̂` and `Î`.
- **The day-ahead bill** is issued at 00:00 UTC: traffic from the dense **block** logs
  (one 24-hour path per link from the start of the day, [forecasting.md](forecasting.md#dense-prediction-logs-gr-dense)),
  doors outside the forecasting panel from the same hour last week, the grid factor from
  a simple rule (default: same hour yesterday).
- **Bill error** = daily WAPE over parties (Σ |daily error| / Σ daily bill). Scenarios swap
  one forecast at a time for the truth, which separates the traffic error from the grid error.

## The pipeline

| Step | Module | Reads | Writes (under `$GR_ARTIFACTS_DIR`) |
|---|---|---|---|
| `gr-network extract` | `network.traffic` | every interface CSV | `network/iface_stats.csv`, `traffic.npz` |
| `gr-network roles` | `network.matching`, `network.roles` | the above, `web/data/links.json`, `reference/sites.csv`, the year run's link list | `network/interfaces.csv`, `routers.csv`, `bundles.csv`, `matches/`, `panel_links.csv` |
| `gr-network dns` | `network.dns` | `reference/dns/ptr_*.txt` | `network/dns_links.csv`; `dns_*` columns in `interfaces.csv` |
| `gr-carbon grid` | `carbon.grid` | Electricity Maps API (`--download`) | `carbon/grid/raw/`, `em_<zone>_hourly.csv`, `rules.csv` |
| `gr-carbon inventory` | `carbon.inventory` | `router-power/`, `routers/` (headers), `eci/apollo/`, interface CSVs (Zoo week), `reference/zoo/example.zip`, `network/` | `carbon/inventory/psu_survey.csv`, `zoo_match.csv`, `devices.csv`, `routers.csv`, `routers_<year>.csv`, `shelves.csv`, `site_energy_<year>.csv`, `scope2_<year>.csv` |
| `gr-carbon dependence` | `carbon.inventory` | PSU and interface CSVs, 2024 | `carbon/inventory/power_vs_traffic.csv` |
| `gr-carbon allocate` | `carbon.allocation` | `network/interfaces.csv`, `routers.csv`, inventory, `reference/external_doors.csv` | `carbon/allocation/ports.csv`, `device_pools.csv`, `pools.json`, `doors.csv`, `annual_bills_<year>.csv` |
| `gr-carbon bills` | `carbon.bills` | the above, the run's dense block logs, door CSVs over the window | `carbon/bills/` (below) |
| `build_report.py` | `recipes/carbon` | everything above | `carbon/report/carbon-bills.html` |

`carbon/bills/`:

| File | What |
|---|---|
| `party_bills.csv` | **the bill**: per party, settled g (access + backbone), forecast g, GB, g/GB over the window |
| `daily_bills.csv` | per day and party, settled and forecast g |
| `bill_errors.csv` | every scenario's daily/hourly WAPE, median party APE, total bias |
| `errors_by_type.csv`, `settled_by_type.csv` | the same by party type |
| `scheduling_test.csv` | per party-day: g/GB of the day average, the forecast's pick, hindsight, worst hour |
| `hourly_network.csv` | network totals per hour, incl. the CH grid factor (**licensed, keep private**) |
| `summary.json`, `top_parties.csv`, `daily_examples.csv`, `hour_of_day_median.csv` | for the report |

All assumed numbers (PUE by host type, unmetered router power by model, optics W by
speed, zones, border routers, marginal energy per bit) are in one module,
[`carbon/assumptions.py`](../src/green_routing/carbon/assumptions.py).

## Inputs

### From the bucket

| Subtree | Used for | Notes |
|---|---|---|
| `router-interfaces/` | traffic, link ends, service spans, Zoo matching, door volumes | [data.md](data.md) |
| `router-power/` | measured router power | **stops on 2024-11-01**; four schemas, see `config.PSU_SCHEMAS` and [data.md](data.md#notes) |
| `routers/` | telemetry schema → router family (headers only, ranged reads) | |
| `eci/apollo/` | optical shelf power (`shelfPowerS0`) | live |

### Reference inputs (`$GR_REFERENCE_DIR`, default `artifacts/reference`)

Curated tables that are not in the bucket. They come from public sources but together
describe one operator's network in detail, so they are kept **out of the repository**, like
everything under `artifacts/` (some outputs are more sensitive still: `zoo_match.csv`
de-anonymises the Zoo's public sample). Copy the directory from the reference machine, or
rebuild it from the sources listed.

| File | Content | Source |
|---|---|---|
| `sites.csv` | site code → town, host, host type, lat/lon, `kind` (`both` = DWDM site), confidence | DNS LOC records of router names, PeeringDB (AS559), SWITCHlan map 2026-02, OpenStreetMap reverse geocoding |
| `dns/ptr_130.59.0.0_16.txt`, `dns/ptr_v6_2001-620-0.txt`, `dns/sweep_date.txt` | reverse-DNS snapshot (4,722 and 840 names, 2026-10-30) | `gr-network dns-sweep` (needs `dig`; 65,536 IPv4 queries plus the IPv6 walk) |
| `external_doors.csv` | door → external partner, type (today: zh3/bundle-ether3 → Meta) | DNS network names (`NET-BB-ACCESS-…`) |
| `external_links.csv` | 36 external connections: partner, site, capacity, BGP router | SWITCH looking glass, PeeringDB, GÉANT Compendium, SWITCH network blog |
| `overhead_contracts_embodied.csv` | PUE evidence, host electricity contracts, embodied kg CO₂e per device type, with URLs | literature and host publications |
| `zoo/example.zip`, `zoo/router-models.zip` | Network Power Zoo sample (May 2025) and model table | networkpowerzoo.ethz.ch/static-files/ |
| `fj_psu_efficiency.csv` | PSU efficiency table of the Fantastic Joules study (IMC'25) | its public repository (reference only) |

### Grid factors (Electricity Maps)

Hourly, flow-traced consumption intensity, `direct` and `lifecycle`, for CH, FR, DE, IT-NO
and AT from 2023 (10-day chunks, ~1,500 requests for a full download). The key is read
from `EM_API` or from `export EM_API=...` in `~/.bashrc` and never printed. **Academic
licence: the data must not be redistributed.** Keep `carbon/grid/` and `hourly_network.csv`
out of the repository and out of published pages; the report embeds aggregates only.

## Results (reference run, 2026-10-30)

**2025 inventory.** 192 routers in service: 254.6 MWh IT (216–336; 70 % of it from
measured PSU power), optical shelves 212.9 MWh, **738.7 MWh at the sites** (592–961 over
the PUE range). Location-based Scope 2, direct: **15.7 t CO₂e** (12.5–20.4); lifecycle
33.7 t. The Swiss factor averaged 21.6 g/kWh in 2025. On 116 metered routers traffic
explains < 1 % of power on 113 and at most 3.7 %. Pools for 2025: access 210 MWh,
backbone 529 MWh (optical 337).

**Bill window** = the year benchmark's test block, 2026-03-22 → 07-10 (2,664 h): 77.9 kW
at the sites (routers 46.1, optical 31.8; access 22.8, backbone 55.0), **2.66 t CO₂**
over 414 doors and 306 parties. Mean Swiss factor 13 g/kWh.

| Party type | Parties | Share of the bill | Access part |
|---|---|---|---|
| customer ports on PoP routers | 215 | 38 % | 18 % |
| customer sites (own access router) | 67 | 23 % | 92 % |
| external networks behind ix3, ce1, ba3 | 3 | 20 % | 3 % |
| behind other routers missing from the export | 20 | 18 % | 6 % |
| CDN (Meta, direct) | 1 | 0.2 % | 0 |

Daily bill error (WAPE across parties):

| Traffic forecast | with the actual grid factor | with the day-ahead grid forecast |
|---|---|---|
| PatchTST | **11.7 %** | **31.4 %** |
| TCN / GRU | 11.9 % / 11.9 % | 31.5 % / 31.5 % |
| DLinear | 12.8 % | 31.8 % |
| same hour yesterday / last week | 15.0 % / 21.0 % | 33.0 % / 37.7 % |
| persistence | 24.5 % | 39.4 % |
| actual traffic | 0 | 28.0 % |

- **The grid forecast is the bottleneck.** With perfect traffic it alone causes 28 % and
  biases the total by +4 %. "Same hour yesterday" was the best of five simple rules
  (MAE 6.5 g/kWh on a mean of 13). The learned traffic models halve the traffic-driven
  error relative to persistence.
- **Scheduling moves the bill, not the emissions.** 1 GB/day of flexible traffic sent in
  the hour the forecast marks cheapest: median billed 0.0040 g/GB vs 0.0098 for the day
  average (hindsight 0.0020, worst hour 0.0288), 41 % of the possible reduction. The
  marginal (physical) cost is 2.6 × 10⁻⁷ – 1.9 × 10⁻⁶ g/GB, four orders of magnitude lower.

### Against the first screening

The package was checked against the scripts it replaced (`artifacts/carbon_screening_2025`,
`artifacts/link_roles`, 2026-10-29/30): link roles, bundles, reverse-DNS far ends, grid
series and grid rules reproduce exactly. The rest moved by 1–2 % because three bugs were
fixed: the shell scan of first/last dates lost routers whose CSV ends in empty rows
(`swislf2`, +2.0 MWh in 2025); the optics share of routers in service part of the year was
computed against year-averaged power; routers commissioned in 2026 (ibm3/4, ra3/4, be2,
yv1) were missing from the bill window (router site power 45.0 → 46.1 kW). Router energy
in 2025 went from 252.6 to 254.6 MWh, the window's backbone pool from 54.1 to 55.0 kW, the
settled bill from 2.62 to 2.66 t; every bill error moved by < 0.1 pp.

## Limits and open data

- **Routers missing from the export**: 31 routers that still run (ix3, ce1, ba3, ez31, si2,
  ge3, ez11, bs1, …) are absent from the SNMP export; 38 % of the window's bill goes to
  their group doors. Per-partner bills behind ix3/ce1/ba3 need their interface data;
  routers mid-path (ce1, ez31) also carry transit, which inflates their doors.
- **Power** is carried forward from 2024 (the PSU export stopped); 30 % of router energy
  is modelled. PUE is set by host type, never measured.
- **Allocated bandwidth** is port speed, not the contracted rate.
- **DNS** shows today's configuration; a port may have been reassigned since 2025.
- **One season**: the test block is spring and summer, when the Swiss grid is cleanest and
  steadiest.
- **Embodied emissions of optics** are missing (no LCA found).

What SWITCH could provide, in order of value: the 31 missing routers back in the export
(back-filled from 2025-01), the PSU export, interface descriptions (`ifAlias`) and
contracted bandwidths, measured PUE of the main sites.

## Next steps

- Log Electricity Maps' own day-ahead/72-hour forecast every day and score it against
  the rules in `carbon.grid` as the days arrive (it cannot be tested on the past).
- Rerun on a winter window once the forecasting benchmark has one.
- Bill the parties behind ix3/ce1/ba3 individually once their SNMP data is exported.
