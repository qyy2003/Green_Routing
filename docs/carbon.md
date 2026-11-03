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
gr-carbon total                    # total traffic: counting checks, top-down vs bottom-up forecasts
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

## How bytes are counted

Bytes are measured only at doors, never on internal trunks. A byte from party A to party
B may cross six routers and five trunks, but it is counted exactly twice: as *in* on A's
door (entering SWITCHlan) and as *out* on B's door (leaving). Hence

```text
total carried   T(h) = ½ Σ_doors (in + out)        each byte once
backbone key    V_k(h) = in_k + out_k               ½ byte at each end of every byte
```

Nothing is de-duplicated after the fact because internal hops are never added up:
trunks (matched by traffic or named in reverse DNS), bundle members (the bundle is the
door) and VLAN subinterfaces (their parent is) are not doors.

**Check: the doors form a closed cut.** If no traffic leaks in or out unseen, Σ in = Σ out
over the doors every hour. Over the bill window: 261.9 Gb/s in, 261.0 Gb/s out (ratio
1.003); the median hourly imbalance is 0.5 % (p90 1.6 %). `gr-carbon total` recomputes it.

What the key does not see, and why:

- **Traffic that turns around on one router** (customer to customer, or customer to an
  external peer on the same PoP) is billed backbone though it touches no trunk. Counters
  bound it per router: `max(0, door_in − trunk_out, door_out − trunk_in) ≤ local ≤
  min(door_in, door_out)`. The lower bound is 0.13 % of door traffic; the upper bound is
  not informative. Only flow data (NetFlow/IPFIX) can measure it.
- **Routers missing from the export**: 52 % of door traffic is on trunks into them. If
  one carries transit (ce1 and ez31 sit mid-path), a byte that leaves our part of the
  network into it and comes back is counted four times instead of twice and billed to
  that router's group. Their own counters would remove this.
- **Distance does not count**: a byte crossing one trunk pays the backbone rate of a byte
  crossing six. A route-aware split needs a traffic matrix: flow data, or an estimate
  from trunk loads and the OSPF routing (network tomography).

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
| `gr-carbon total` | `carbon.total` | `network/traffic.npz`, doors, the dense block logs, grid | `carbon/total/`: `summary.json` (counting checks), `turnaround.csv`, `forecast_errors.csv`, `total_hourly.csv`, `signal_errors.csv` |
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
| PatchTST | **11.5 %** | **31.1 %** |
| TCN / GRU | 11.7 % / 11.7 % | 31.2 % / 31.3 % |
| DLinear | 12.6 % | 31.5 % |
| same hour yesterday / last week | 14.8 % / 20.8 % | 32.9 % / 37.5 % |
| persistence | 24.0 % | 38.9 % |
| actual traffic | 0 | 28.0 % |

- **The grid forecast is the bottleneck.** With perfect traffic it alone causes 28 % and
  biases the total by +4 %. "Same hour yesterday" was the best of five simple rules
  (MAE 6.5 g/kWh on a mean of 13). The learned traffic models halve the traffic-driven
  error relative to persistence.
- **Scheduling moves the bill, not the emissions.** 1 GB/day of flexible traffic sent in
  the hour the forecast marks cheapest: median billed 0.0040 g/GB vs 0.0101 for the day
  average (hindsight 0.0020, worst hour 0.0298), 41 % of the possible reduction. The
  marginal (physical) cost is 2.6 × 10⁻⁷ – 1.9 × 10⁻⁶ g/GB, four orders of magnitude lower.

### Against the first screening

The package was checked against the scripts it replaced (`artifacts/carbon_screening_2025`,
`artifacts/link_roles`, 2026-10-29/30): link roles, bundles, reverse-DNS far ends, grid
series and grid rules reproduce exactly. The rest moved by 1–2 % because four bugs were
fixed: the shell scan of first/last dates lost routers whose CSV ends in empty rows
(`swislf2`, +2.0 MWh in 2025); the optics share of routers in service part of the year was
computed against year-averaged power; routers commissioned in 2026 (ibm3/4, ra3/4, be2,
yv1) were missing from the bill window (router site power 45.0 → 46.1 kW). Router energy
in 2025 went from 252.6 to 254.6 MWh, the window's backbone pool from 54.1 to 55.0 kW, the
settled bill from 2.62 to 2.66 t. Fourth, doors outside the forecasting panel fall back to
"same hour last week", which had no history in the window's first week, so they were
forecast as zero for seven days; with a week of history the error of every traffic
forecast dropped by 0.15–0.5 pp. Against the screening: PatchTST 11.7 → 11.5 %,
persistence 24.4 → 24.0 % (actual grid factor).

## Total network traffic

Can the total be forecast instead of every link? `gr-carbon total` forecasts T(h), the
bytes carried (each once), a day ahead at 00:00 UTC over the bill window (mean 261 Gb/s):

| Forecast of the total | Hourly WAPE | Daily WAPE | Bias |
|---|---|---|---|
| GBDT trained on the total alone (top-down) | **11.1 %** | 7.9 % | −2.9 % |
| sum of TCN door forecasts (bottom-up) | **11.1 %** | **7.2 %** | +1.4 % |
| sum of GRU / PatchTST / DLinear door forecasts | 11.7 / 12.2 / 13.7 % | 7.4 / 8.6 / 10.0 % | +2–4 % |
| mean of the same hour over the last 4 weeks | 14.1 % | 10.7 % | +0.3 % |
| same hour last week / yesterday | 16.7 / 19.5 % | 12.2 / 14.8 % | ±0.5 % |
| persistence (23:00, flat) | 26.5 % | 21.5 % | −19 % |

One small GBDT on one series (trained to 2026-02-15, the last 7 days hourly plus the
calendar) matches the sum of 534 per-link deep models. But the total cannot replace the
per-door forecasts in the bill:

- **Emissions do not depend on traffic.** B(h) is static power × the grid factor, so the
  total traffic changes neither the network's emissions nor their forecast.
- **A party's backbone part depends only on its share** V_k / Σ V. Forecasting the total
  and keeping cheap shares does not work: shares from the same hour last week or the
  last 4 weeks give a daily bill WAPE of 21 % / 19 % (actual grid), against 11.5 % from
  the door forecasts. Rescaling the door forecasts to the top-down total leaves
  per-party volumes about as wrong as before (PatchTST 18.5 → 18.0 %, TCN 17.9 → 18.0 %
  daily WAPE).
- **What the total does give** is the network-wide price signal, backbone g per GB
  carried, B(h) / T(h) (mean 0.0070): day-ahead WAPE 11.7 % with the top-down total vs
  12.4 % bottom-up, with the actual grid factor. With the grid forecast every variant is
  at 50–56 %; the grid forecast alone causes 49 %.

So for billing, forecast the doors (414, of which 194 in the panel), not the internal
trunks: trunk forecasts matter for routing and capacity, not for the bill. Forecast the
total for the network signal and as a cross-check on the door forecasts.

## Carbon rates: flat or by path

Bills are settled with real traffic; the open choice is the **rate** (g CO₂ per GB) a byte
carries in a given hour. `recipes/carbon_rates` compares two answers on the same test
window, and on the second dataset (Abilene, below). Report:
[SWITCHlan Carbon Rates](https://claude.ai/artifact/7kGRETXDXdfhBVihZ4zRz5).

**Flat rate**: rate(h) = backbone emissions / bytes carried (each byte once, at the doors).
Average 0.0060 g/GB over the window (1.87 t over 313 PB), 0.0017–0.0148 g/GB from hour to
hour (10th–90th percentile). Its day-ahead forecast needs only the total traffic, so the
total was written as one 5-minute series (`carbon.total.write_total_mirror`, read through
`GR_DATA_SOURCE=local`) and the whole forecasting ladder was trained on it with the year
flags. Day-ahead (issued 00:00 UTC), hourly WAPE of the total:

| Model on the total | Hourly | Daily |
|---|---|---|
| TCN | **12.0 %** | **7.7 %** |
| GRU | 13.0 % | 8.8 % |
| PatchTST / DLinear | 13.9 % / 13.9 % | 10.4 % / 10.2 % |
| historical average | 14.3 % | 11.6 % |
| same time last week / yesterday | 16.7 % / 19.5 % | 12.2 % / 14.7 % |
| persistence | 29.2 % | 26.0 % |
| *GBDT on the hourly total (`carbon.total`)* | *11.1 %* | *7.9 %* |
| *sum of the TCN door forecasts* | *11.1 %* | *7.2 %* |

Holt-Winters and SARIMA (scored on the 160 sampled origins only) are worse than
persistence beyond 1 hour. With one series the neural models stop after 12–16 epochs.
The rate forecast is off by 12.6 % with the actual grid factor and ~50 % with the
day-ahead grid factor (28 % on daily averages); perfect traffic leaves 49 %.

**Path rate** (`carbon.path_rate`, `carbon.tomography`): every router's backbone power and
every fibre span (each site's optical shelves split over its inter-site spans, by
capacity, priced in the zone they draw from) is a resource; a byte pays the resources
on its route. Steps:

1. Router graph: 181 routers, 518 directed links with hourly loads; the 23 routers missing
   from the export are endpoints that may not relay traffic.
2. Routing: ECMP shortest paths. Link costs are not published, so the counters choose:
   predicting the loads of a fifth of the links hidden from the estimator, costs by
   capacity (1000 / Gbit/s) err 34.9 %, by distance 35.1 %, by hop 43.6 %; the gravity
   guess alone 108–115 %.
3. Traffic matrix per hour from the counters: gravity prior, then non-negative EM updates
   (Vardi 1996; the minimum-information estimate of Zhang et al. 2003) fitting link loads
   and door totals, then IPF to the measured totals. Plain tomogravity needed 32 % negative
   flows to fit, because the routing is inferred. A per-router "unseen" door absorbs
   traffic through ports the export misses (4.5 % of ingress; its cost is shared flat).
4. Pricing a resource **by usage** (emissions / bytes crossing it) or **by capacity**
   (emissions / capacity, idle remainder shared flat). Bills conserve the emissions
   exactly (unit test).

Results: priced by usage, path rates between routers are 0.00035–0.0090 g/GB (10th–90th
percentile by bytes) and 35 % of the backbone emissions change hands between parties; but
nearly idle backup spans charge their power to the few bytes on them (ce2–ge2 ≈ 1,200 g/GB),
so parties with 0.45 % of the traffic pay 15 % of the path bill. SWITCHlan runs at ~2 % of
capacity (emission-weighted), so priced by capacity only 2.9 % of the emissions follow
paths and the bills stay within 2.3 % of flat.

**Flow data vs counters.** No flow data is reachable for SWITCHlan (the bucket has none),
so the method was tested on the second dataset, Abilene (below): counters rebuilt from its
measured matrices and estimated back give a matrix error of 23.5 % (gravity alone 43 %), and
path bills within 4.0 % of the flow-based ones over 19 weeks (5.1 % hour by hour, worst
router 43 %), against 26 % for using the flat rate instead; priced by capacity the matrix
barely matters (0.1 %). Abilene is the easy case (known routing, exact counters: hidden
links predicted within 1.6 %); on SWITCHlan the gravity guess and the EM matrix differ by
32 % on usage-priced bills but only 0.9 % on capacity-priced ones.

## Second dataset: Abilene (NetFlow, 2004)

`recipes/abilene` reruns the forecasting ladder and the carbon rates on Abilene
([data.md](data.md#second-dataset-abilene-netflow-2004)), whose who-sends-to-whom matrices
are measured. The archive has traffic only; energy and grid factors were rebuilt from
public documents (`recipes/abilene/energy.csv`, `egrid2004.csv`, sources in each row):

| Input | Value | Source |
|---|---|---|
| Juniper T640 (11 PoPs) | ~1.8 kW each (1.5–2.4): base 16.9 A + 1 FPC3 + 2 FPC2 + (links + 8) PICs at −48 V | Juniper T640 Hardware Guide, Table 78; line cards assumed |
| Juniper M5 (Atlanta) | 300 W (200–434) | Juniper: M5/M10 supplies 434 W each |
| Each link | one leased OC-192 wavelength: 2 × 32.5 W transponders + 1/40 of an amplifier site (148 W) every 80 km and of the terminals | Cisco ONS 15454 DWDM manual, Table A-1 |
| Route | 20,921 km (13,000 route-miles), 255 amplifier sites | Lightwave, 4 Feb 2004 |
| PUE | 2.0 (1.6–2.5) | assumption; EPA data-centre report 2007 |
| Grid | eGRID subregion CO₂ output rate, 2004, per city (399–924 g/kWh), constant | EPA eGRID2006 v2.1 |

Results over 19 weeks: 21.8 kW of equipment (routers 19.9, optics 1.9), 44 kW at the sites
(29–78), 611 g/kWh power-weighted, about 234 t CO₂ a year. The flat rate is **18.4 g/GB**
(13–26 g/GB from hour to hour), roughly 3,000 times SWITCHlan's: 2004 routers, a
coal-heavy grid and 3.2 Gbit/s of traffic. Path rates by usage range 3–34 g/GB (10th–90th
percentile by bytes) and move 13 % of the emissions between routers; utilisation is ~2 %,
so priced by capacity the bills stay within 0.4 % of flat. One model on the total
forecasts it day-ahead with 11.9 % hourly error (PatchTST; persistence 21.4 %); with a
constant grid factor that is the rate forecast's only error source (8.2 % on the rate).
Per-link forecasting results: [results.md](results.md#6-second-dataset-abilene-netflow-2004).

## Granularity and uncertainty of the rate

`recipes/carbon_granularity` (functions in `carbon.granularity`) tests how to publish the
flat rate, on SWITCHlan (109 test days) and Abilene (20). Every bill is computed on real
5-minute traffic; the truth is the 5-minute rate. Report:
[Carbon Rate Granularity](https://claude.ai/artifact/5LojUMewPg3VyP7ncCcqbL).

| Question | Answer from the data |
|---|---|
| Publish at which grain? | **Hourly.** Keeps 98.8 % (SWITCHlan) / 97.4 % (Abilene) of the 5-minute rate's variance; daily statements move 0.4 % / 0.2 %; a forecast published hourly is as close to the 5-minute truth as a 5-minute one (13.8 % vs 13.9 % with the real grid factor; 16.8 % vs 16.7 % on Abilene). 3 h keeps 92 %, a day 49 %; scheduling with a perfect hourly rate captures 95 % of the 5-minute value, 3 h 81 %, 6 h 64 %. |
| Forecast at which grain? | Forecast 5-minute traffic and sum into hours, or train directly on hours: the hourly forecast is about the same (SWITCHlan TCN summed 13.8 % vs hourly GBDT 13.5 %; Abilene PatchTST 16.8 % vs 16.9 %). The model itself is more accurate on hours than on 5-minute values (TCN traffic WAPE on SWITCHlan's total 13.6 % at 5 min, 12.0 % hourly; 45 % → 35 % median on single links): the within-hour wiggle (5.6 % of the total's 5-minute variance, 25 % on a median link) cannot be forecast a day ahead, so averaging removes error and dropping the forecast's own 5-minute shape costs nothing (E8). Models trained on 3-hour, 6-hour or daily data never do better. |
| How often to update? | Hourly, for the hours ahead: the rate's error drops from 52 % (day-ahead) to 38 % (1 h ahead) on SWITCHlan, using the last known grid hour (better than "same hour yesterday" up to a few hours ahead, worse from 8 h); Abilene 19.7 % → 14.3 %. |
| Values or a fitted curve? | Values. Fourier curves need ≥ 4 daily harmonics (9 numbers) to match hourly values and never beat them; a curve is a compact encoding, not a better forecast. |
| One value or a distribution? | Publish quantiles with every value. From the last 28 days' joint traffic and grid errors (same past day and hour, ±3 h): 80 % band covers 78.5 %, 90 % band 88.4 %, P90 exceeded in 10.8 % of hours, CRPS 18 % better than the single value; choosing hours by the P90 lowers the realised rate by 6 % (Abilene 3 %). Traffic uncertainty alone covers 31 %: the grid factor dominates. Abilene's 20 test days are too short to calibrate (71 % coverage). |

## Forecasting the total for the flat rate

`recipes/flat_forecast` (functions in `carbon.dayahead`) asks how best to forecast the
flat rate's denominator: SWITCHlan's total traffic, hour by hour, issued at 00:00 UTC for
the day. Every choice is made on a development period (2025-11-01 … 2026-03-21, 140 days)
and reported on the benchmark's test block (2026-03-22 … 07-10, 111 days); differences
come with a 95 % interval from resampling whole weeks. Models that learn are refitted
weekly on everything known; the history starts 2025-02-01 (the export has a gap in
Dec 2024 – Jan 2025, and before that the current doors do not cover the network).
Report: [Day-Ahead Traffic Forecast](https://claude.ai/artifact/1EBDAKgqujEv3PmdyvVSEM) (not copied to
`docs/reports/`: it names operators' parties).

| Lever | Hourly WAPE, development / test |
|---|---|
| GBDT as used so far (168 h, basic calendar; trained once to 2026-02-15) | — / 11.0 % |
| 5-minute TCN as used so far (peak-aware loss) | — / 12.0 % |
| **Chronos-2 zero-shot** (pretrained, 24 weeks of hourly context) | 10.7 / 10.7 % |
| Chronos-2, 4 weeks hourly vs 4 weeks of 5-minute values | 11.2 / 11.1 vs 16.2 / 15.4 % |
| Chronos-2 + calendar covariates (holiday weight, Christmas, lecture weeks) | 10.8 / 10.6 % |
| Chronos-2 fine-tuned on the series (all weights, 1,000 steps) | 11.8 / 11.7 % (worse) |
| TimesFM 2.5 / Chronos-Bolt | 11.4 / 11.1 %, 11.0 / 11.0 % |
| GBDT weekly refit; five seeds bagged | 11.7 / 10.7 %; 11.6 / 10.7 % |
| GBDT same features without calendar / old holiday table / weighted holidays | 13.3 / 12.6, 12.6 / 11.5, 12.1 / 10.7 % |
| 0.8 × Chronos-2 + 0.2 × GBDT (weight from development) | 10.6 / 10.4 % |
| **… with the GBDT alone on holidays and the Christmas closure (final)** | **10.6 / 10.0 %** |
| … same rule with the symmetric TCN as a third member (equal weights; test only) | — / 9.8 % |

- **Pretrained beats trained** on the development period and ties on the test. On
  Abilene (same choices, no tuning) Chronos-2 scores 12.2 % against 15.4–15.6 % for the best
  trained models.
- **Hours, not 5 minutes**: at 5 minutes a context covers 12× less history and the
  5-minute detail is unforecastable a day ahead.
- **Holidays decide the rest.** On semester weekdays a national holiday cuts traffic by
  29–42 % (2025) / 12–31 % (2026), the Christmas closure by 41–56 %, Corpus Christi by
  5–8 % (catholic cantons only). Chronos-2 misses the test's four weekday holidays by 27 %
  (20 % with covariates); the GBDT, which has the 2025 holidays in its history, by 7 %.
  `dayahead.HOLIDAYS` weights each holiday by the population that has it off.
- **Training process**: a symmetric loss for the 5-minute ladder (the peak-aware
  `--under-weight 2` made the TCN +3.8 % high) helps every model, three seeds each:
  TCN 12.1 → 11.3 % (−0.8, interval −1.4 to −0.3), DLinear 13.9 → 12.9 %, PatchTST
  13.2 → 12.5 %, GRU 12.3 → 11.7 %; weekly refitting helps the GBDT a little (−0.3); its daily or monthly
  refits, history length, recency weights, loss and size move it by less than the noise;
  cleaning holidays out of the inputs hurts both GBDT and Chronos-2 on the development
  period (it hides the real January level after Christmas).
- **Not forecastable**: with the daily level known, the error would still be 7.7 %; four
  parties (ez31, ix3, CSCS's 400G port, CERN) carry 38 % of the traffic but 60 % of its
  unpredictable variation (bulk research transfers). Daily errors do not persist, so bias
  correction does not help.
- **For the rate**: with the real grid factor the day-ahead rate's error falls from 11.5 %
  to 10.6 %; with the day-ahead grid rule all forecasts stay near 52 % (49.2 % with perfect
  traffic). The grid forecast is the remaining problem.

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
- Door-to-door flow aggregates from SWITCH (sampled NetFlow is enough) would replace the
  estimated traffic matrix of the path rate and settle the routing question.
- Rerun on a winter window once the forecasting benchmark has one.
- Bill the parties behind ix3/ce1/ba3 individually once their SNMP data is exported.
