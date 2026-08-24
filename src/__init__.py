"""Project FORESIGHT — shared source code package.

This package hosts the reusable implementation of the official Zidio
Project FORESIGHT deliverables:

    src/paths.py     repository-root path conventions (no hard-coded absolute paths)
    src/pipeline.py  D1 — reproducible pipeline: raw extracts -> analysis-ready
    src/forecast.py  D3 — weekly SKU-level forecasting, baseline, backtest, metrics
    src/risk.py      D4 — stockout / overstock risk scoring + decision grid

Convention: always import from the repository root, e.g.::

    from src.paths import DATA_RAW, DATA_PROCESSED
    from src.pipeline import run_pipeline

NOTE: These modules are intentionally architectural stubs (STEP 2).
The official internship dataset has not been provided/loaded yet and no
algorithm, metric, or business result has been produced.
"""
