"""SWITCH Cricket dataset processing for network-prediction benchmarks.

Modules:
  config     data source (S3 / local mirror), sampling constants, target columns
  storage    uniform reads over s3:// and local paths, with a parallel object cache
  loaders    coalesce the two schema eras; per-series & per-target readers
  topology   parse switch-network-topology.txt -> nodes/edges/adjacency
  cohort     coverage scan + cohort selection on a common window/grid
  tensor     build node-aligned X[T, N, F] + adjacency
  splits     chronological split w/ purge, rolling-origin backtest, federated partition
  build      end-to-end CLI that writes benchmark artifacts

See ../../dataset.md and ../../benchmark_design.md.
"""
from . import config, storage, loaders, topology, cohort, tensor, splits  # noqa: F401
