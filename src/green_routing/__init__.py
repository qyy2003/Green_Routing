"""Green Routing — per-link traffic forecasting on the SWITCH backbone.

Subpackages:
  dataset      read the SWITCH Cricket telemetry (S3 or a local mirror), coalesce the
               two CSV schema eras, select cohorts, build tensors and splits
  prediction   the forecasting ladder (baselines -> GBDT -> GRU/TCN -> DLinear/PatchTST
               -> Graph WaveNet), the rolling-origin harness, dense prediction logs,
               metrics and the model zoo
  viewer       web viewer for a run's predictions, link by link
  analysis     frequency-domain study of backbone traffic

Command-line tools (see pyproject.toml): gr-data, gr-benchmark, gr-dense, gr-zoo,
gr-viewer, gr-build-tensor. Documentation: docs/ in the repository.
"""
__version__ = "0.3.0"
