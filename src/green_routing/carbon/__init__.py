"""Carbon accounting for SWITCHlan: who is billed for which gram, hour by hour, and how
well the bill can be known a day ahead.

  assumptions  every assumed number (PUE, unmetered router power, optics, zones)
  inventory    routers and optical shelves: model, power, service span, site; annual
               energy and location-based Scope 2
  grid         hourly grid carbon intensity (Electricity Maps) and its day-ahead forecast
  allocation   access pool by allocated bandwidth, backbone pool by bytes; doors and
               parties (customers, external networks, routers missing from the export)
  bills        hourly settled and forecast bills per party, error decomposition,
               scheduling test

Method: the GHG Protocol Scope 2 location-based approach with hourly, flow-traced,
combustion-only factors. ``gr-carbon all`` runs the steps; docs/carbon.md explains them.
"""
