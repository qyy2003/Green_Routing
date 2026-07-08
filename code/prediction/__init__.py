"""Per-link ISP traffic prediction — the method ladder from
``per-link-traffic-prediction-roadmap.md`` (in this package).

Layout (each tier earns its place only by beating the tier below on the *same*
chronological split and metric — see the roadmap):

    data.py             build/cache a per-link traffic panel (uses ``dataset``)
    metrics.py          MAE/RMSE/sMAPE/WAPE/MASE + skill vs a baseline
    harness.py          the testing loop: rolling-origin backtest
    baselines.py        Tier 1 — persistence, seasonal-naive, historical average
    models_classical.py Tier 2 — Holt-Winters, SARIMA
    models_gbdt.py      Tier 3 — global histogram gradient boosting
    run.py              CLI orchestrator: run tiers, tabulate skill, plot

Run everything through ``python -m prediction.run`` from
``/home/yuyqin/ETH_Master_Study/Green_Routing/Green_Routing/code`` with the base conda python
(``/home/yuyqin/anaconda3/bin/python`` — pandas/numpy work there; RRDparsing does not).
"""
