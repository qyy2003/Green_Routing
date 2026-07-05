# Green Routing

You should put the `network-energy-efficiency-research` folder along side this repo.

## Data Repo Overview

1. **`switch-device-environment/`** (~198 files)

One file per router/switch. Raw CLI output of `show env all` — temperature sensor readings (Inlet, Hotspot) per module/slot. **Use for:** thermal/power state of each device.

1. **`switch-device-environment/`** (~198 files)

One file per router/switch. Raw CLI output of `show env all` — temperature sensor readings (Inlet, Hotspot) per module/slot. **Use for:** thermal/power state of each device.

2. **`switch-device-inventory/`** (~199 files)

One file per device. Raw `show inventory` — hardware PID, serial numbers, module and optics descriptions. **Use for:** mapping hardware models to power profiles.

3. **`switch-network-topology/`**

The SWITCH network graph in three formats:

* `switch-network-topology.txt` — human-readable links: `nodeA ifaceA <=> nodeB ifaceB (speed) (OSPF-cost-v4 cost-v6)`
* `switch-network-topology.json` — IETF L3 unicast topology JSON (nodes with router-IDs + termination-points, links)
* `bundles.json` — maps Bundle-Ether logical interfaces → physical member interfaces per device

**Use for:** building the graph model for routing algorithms.

4. **`switch-network-maps/`**

Three PDF maps: ECI monitoring, L3 monitoring (redacted), optical topology. **Use for:** visual reference only.

5. **`observium-data/`**

Observium NMS exports for 3 devices (`swine2`, `swice2`, `swiel3`):

* `.json` files: full SNMP device metadata (IP, OS version, hardware, last polled, uptime)
* `swine2/`: 210 RRD binary time-series files for per-port traffic and alerts

**Use for:** per-port traffic time-series (needs `rrdtool dump` to read, see `parsing-rrd/`).

6. **`parsing-rrd/`**

Python toolchain to convert Cricket RRD → CSV:

* `parse_rrd.py` — main script; reads from `/var/lib/cricket`, outputs to `../cricket_dataset/`
* `metadata.json` — **2308 targets** across 5 categories: `cpu-usage`, `eci` (optical), `router-interfaces`, `routers`, `transceiver-monitoring`; 29 hardware target types (ASR, NCS, Arista, ECI-Apollo, etc.)

7. **`lan-mon2021/`**

SWITCHlan traffic monitoring data — two sub-datasets:

**`bb-usage-logs/`** (793 files) — Backbone link utilization:

* 45 link-pairs (e.g., `AG-EZ`, `BE-FR`, `CE-LG`) × monthly files from **Jan 2021 – Aug 2023**
* Format: `timestamp inMbps outMbps` at 5-minute intervals
* **Use for:** traffic demand matrices for routing optimization

**`lan/`** — Device-level reports:

* `rep202301...`: daily per-device interface byte-counters and uptime
* `dwdm20230912`: snapshot of all interface up/down states at a Unix timestamp
* `dwdm-lanstatus2021.log`: event log of device timeouts/recoveries since Jan 2021


## Git History Analysis

**Two automated daily pipelines**

1. "New metadata. Last update: YYYY.MM.DD" — 864 commits

   * Runs daily via `parse_rrd.py` as a cronjob (since Dec 31, 2023)
   * **What it does:** Reads RRD files from `/var/lib/cricket` on the SWITCH server, parses traffic/CPU/transceiver time-series into CSVs, then:
     * Uploads all CSVs to **SWITCHengines Object Storage** (S3 bucket: `switchlan-load-timeseries` )
     * Commits only `parsing-rrd/metadata.json` to git (the timestamp field records when the run completed)
   * **The actual CSV data lives in object storage, NOT in the git repo**
   * Coverage: daily from 2023-12-31 to 2026-05-05 (865 of 1203 days — ~72%, with gaps likely due to server downtime)
   * Metadata tracks **2308 targets**: 1135 router interfaces, 895 transceivers, 173 routers, 103 ECI optical devices, 2 CPU targets

2. "Update topology files from router configuration (DATE)" — ~50 commits

   * Updates switch-network-topology.txt, switch-network-topology.json, bundles.json
   * Reflects actual topology changes (added/removed links, OSPF cost changes)




**Take away: Most network data are on the SWITCHengines Object Storage**