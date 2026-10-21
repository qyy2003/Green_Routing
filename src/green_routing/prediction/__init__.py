"""Per-link ISP traffic prediction — the method ladder from
docs/roadmap.md in the repository.

Layout (each tier earns its place only by beating the tier below on the *same*
chronological split and metric — see the roadmap):

    data.py             build/cache a per-link traffic panel (uses ``dataset``)
    metrics.py          MAE/RMSE/sMAPE/WAPE/MASE + skill vs a baseline
    harness.py          the testing loop: rolling-origin backtest
    baselines.py        Tier 1 — persistence, seasonal-naive, historical average
    models_classical.py Tier 2 — Holt-Winters, SARIMA
    models_gbdt.py      Tier 3 — global histogram gradient boosting
    run.py              CLI orchestrator: run tiers, tabulate skill, plot

Run benchmarks with ``gr-benchmark`` (= ``python -m green_routing.prediction.run``) and
dense prediction logs with ``gr-dense``. Documentation: docs/forecasting.md.
"""
