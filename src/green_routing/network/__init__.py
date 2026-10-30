"""What connects to what in SWITCHlan, inferred from the telemetry and public DNS.

  traffic    one pass over every interface CSV: statistics + the series for matching
  matching   link ends (a.in == b.out across routers) and bundle members
  roles      router tiers (core / regional / access) and interface roles
             (backbone trunk, access uplink, edge port, ...)
  dns        far ends from SWITCH's reverse DNS, incl. trunks into routers that are
             missing from the export

``gr-network all`` runs them in order into ``$GR_ARTIFACTS_DIR/network``; the carbon
pipeline (``green_routing.carbon``) reads its ``interfaces.csv`` and ``routers.csv``.
"""
