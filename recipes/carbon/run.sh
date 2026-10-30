#!/usr/bin/env bash
# Carbon bills for SWITCHlan: link roles, energy inventory, grid factors, allocation,
# settled and day-ahead forecast bills over the year benchmark's test block, the report.
#
#   recipes/carbon/run.sh network     link ends, tiers, roles, reverse-DNS far ends
#   recipes/carbon/run.sh grid        Electricity Maps history -> hourly CSVs (needs EM_API
#                                     unless the raw cache is there; DOWNLOAD=1 to refresh)
#   recipes/carbon/run.sh inventory   PSU survey, router models, power, shelves, Scope 2
#   recipes/carbon/run.sh dependence  power vs traffic of every metered router
#   recipes/carbon/run.sh allocate    pools, doors, parties, annual bills
#   recipes/carbon/run.sh bills       hourly settled + forecast bills
#   recipes/carbon/run.sh report      $GR_ARTIFACTS_DIR/carbon/report/carbon-bills.html
#   recipes/carbon/run.sh verify      compare with expected.json
#   recipes/carbon/run.sh all         every step in this order (~3 min with a warm data cache;
#                                     the first run also downloads ~4 GB of power data)
#
# Needs: the year run with dense logs ($GR_ARTIFACTS_DIR/pred_year/merged, recipes/year),
# the reference inputs in $GR_REFERENCE_DIR (default $GR_ARTIFACTS_DIR/reference, see
# docs/carbon.md), and the Electricity Maps key EM_API for a fresh grid download.
# Outputs: $GR_ARTIFACTS_DIR/network and $GR_ARTIFACTS_DIR/carbon/. The grid/ directory
# holds licensed Electricity Maps data: never commit or publish it.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
need gr-network
need gr-carbon
: "${WORKERS:=16}"
: "${YEAR:=2025}"
: "${DOWNLOAD:=0}"

step_network()    { log "network"; gr-network all --workers "$WORKERS"; }
step_grid()       { log "grid"; if [[ "$DOWNLOAD" == 1 ]]; then gr-carbon grid --download; else gr-carbon grid; fi; }
step_inventory()  { log "inventory $YEAR"; gr-carbon inventory --year "$YEAR" --workers "$WORKERS"; }
step_dependence() { log "power vs traffic"; gr-carbon dependence --workers "$WORKERS"; }
step_allocate()   { log "allocation $YEAR"; gr-carbon allocate --year "$YEAR"; }
step_bills()      { log "bills"; gr-carbon bills --year "$YEAR"; }
step_report()     { log "report"; "$PYTHON" "$RECIPES/carbon/build_report.py"; }
step_verify()     { log "verify"; "$PYTHON" "$RECIPES/carbon/verify.py"; }

case "${1:-}" in
  network|grid|inventory|dependence|allocate|bills|report|verify) "step_$1" ;;
  all) step_network; step_grid; step_inventory; step_dependence; step_allocate; step_bills; step_report; step_verify ;;
  *) sed -n '2,21p' "$0"; exit 1 ;;
esac
