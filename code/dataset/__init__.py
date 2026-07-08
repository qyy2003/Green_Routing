"""SWITCH Cricket dataset processing for network-prediction benchmarks.

Modules:
  config     paths, sampling constants, target column names
  loaders    coalesce the two schema eras; per-series & per-target readers
  topology   parse switch-network-topology.txt -> nodes/edges/adjacency
  cohort     coverage scan + cohort selection on a common window/grid
  tensor     build node-aligned X[T, N, F] + adjacency
  splits     chronological split w/ purge, rolling-origin backtest, federated partition
  build      end-to-end CLI that writes benchmark artifacts

See ../../dataset.md and ../../benchmark_design.md.
"""
from . import config, loaders, topology, cohort, tensor, splits  # noqa: F401
