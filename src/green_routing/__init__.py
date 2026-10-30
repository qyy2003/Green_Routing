"""Green Routing — per-link traffic forecasting and per-party carbon bills on the SWITCH backbone.

Subpackages:
  dataset      read the SWITCH Cricket telemetry (S3 or a local mirror), coalesce the
               two CSV schema eras, select cohorts, build tensors and splits
  prediction   the forecasting ladder (baselines -> GBDT -> GRU/TCN -> DLinear/PatchTST
               -> Graph WaveNet), the rolling-origin harness, dense prediction logs,
               metrics and the model zoo
  viewer       web viewer for a run's predictions, link by link
  analysis     frequency-domain study of backbone traffic
  network      link ends, router tiers and interface roles, from traffic and reverse DNS
  carbon       energy inventory, hourly grid factors, allocation to customers and external
               networks, settled and day-ahead forecast carbon bills

Command-line tools (see pyproject.toml): gr-data, gr-benchmark, gr-dense, gr-zoo,
gr-viewer, gr-build-tensor, gr-network, gr-carbon. Documentation: docs/ in the repository.
"""
__version__ = "0.3.0"
