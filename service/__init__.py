"""FORESIGHT scoring service (D6).

FastAPI layer that exposes the official Project FORESIGHT forecasting (D3)
and inventory-risk (D4) engines as an HTTP API.

Design rules
------------
* API layer ONLY: every computation is delegated to ``src.forecast`` /
  ``src.risk``; no forecasting or risk logic lives here.
* Data-gated: the service consumes only the official D1 analysis-ready
  outputs under ``data/processed/``. When those are absent it responds with
  machine-readable ``data_not_available`` states and NEVER fabricates
  forecasts, risk scores, decisions, or rupee values.
* Legacy Mini-FORESIGHT artifacts are never read.
"""

__version__ = "1.0.0"
