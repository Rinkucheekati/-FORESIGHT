# Project FORESIGHT — Demand & Inventory Intelligence

**Client:** NorthBay Living  ·  **Role:** Data Scientist & Analytics  ·  **Engagement:** Zidio Development — Data Science & Analytics (4-week client project)

> **Status: D1–D7 complete.** All seven deliverables are implemented and
> verified end-to-end on the official dataset (deterministic 25,000-transaction
> selection, seed 42).## Live Deployment

- **FORESIGHT Dashboard:** [Open Dashboard](https://foresight-dashboard-augk.onrender.com)
- **FORESIGHT Scoring API:** [Open API](https://foresight-scoring-api-7n0w.onrender.com)
- **API Documentation:** [Open API Docs](https://foresight-scoring-api-7n0w.onrender.com/docs)

---

## 1. Objective

Retailers must hold enough inventory to satisfy demand without tying up
capital in surplus stock. Project FORESIGHT provides an end-to-end capability:

    clean extracts -> profile demand -> forecast weekly demand ->
    score stockout / overstock risk -> prioritise rupee-based decisions

delivered through a planning dashboard, a scoring service and an executive
readout.

## 2. Official input data & deterministic 25,000-row selection

The official retail dataset is stored locally at
`data/raw/retail_contaminated_dataset/` (git-ignored), containing the
customer, SKU and store masters, promotions, inventory snapshots and the
`sales_transactions.csv` extract (9,945,511 rows). From it, a deterministic
25,000-row selection was produced **once** by `src/official_selection.py`
(seed 42, `pandas.DataFrame.sample(n=25000, replace=False, random_state=42)`)
and written to `data/raw/official_selected/sales_transactions_25000.csv`
(git-ignored; 25,000 rows, 4,616 SKUs, 30 stores, 8,808 customers). The
SHA-256 hashes of the source file and the selection are recorded in
`data/raw/official_selected/selection_manifest.json`; every pipeline stage
reads this exact file and re-verifies its 25,000-row count and column
contract. No other data source is used and no source records were fabricated.

## 3. Deliverables D1–D7 (current state)

| # | Deliverable | Implementation | Key outputs |
|---|---|---|---|
| D1 | Reproducible data pipeline (official inputs -> analysis-ready) | `src/pipeline.py` | `data/processed/`: `sales_daily_clean.csv`, `sku_master_clean.csv`, `calendar_clean.csv`, `inventory_snapshots_clean.csv`, `sales_analysis_ready.csv`, `d1_data_quality_report.json` |
| D2 | Data-quality & EDA insight memo | `src/eda.py` + `reports/generate_d2_report.py` | `reports/d2_eda_memo.md`, `reports/d2_eda_report.json`, `reports/d2_charts/` (8 charts) |
| D3 | Weekly SKU-level forecast; seasonal-naive baseline; WAPE (primary) + bias (secondary); rolling-origin CV; no leakage | `src/forecast.py` + `reports/generate_d3_report.py` | `data/processed/forecast_results.csv`, `d3_backtest_results.csv`, `d3_model_comparison.json`, `reports/d3_forecast_report.json` |
| D4 | Stockout / overstock risk; recommended action; rupee value at stake; four-cell decision grid | `src/risk.py` + `reports/generate_d4_report.py` | `data/processed/inventory_risk.csv`, `recommendations.csv`, `reports/d4_risk_report.json` |
| D5 | Streamlit planning dashboard (category/SKU filters, forecast-vs-actual, risk flags, prioritised list, loading/empty states) | `app/app.py` | Run with Streamlit (Section 6) |
| D6 | Scoring service (single SKU + batch, documented I/O, bad-input handling) | `service/main.py`, `service/schemas.py` | FastAPI app — I/O contract in `service/README.md` |
| D7 | Executive readout (rupee impact first, honest accuracy) | `reports/generate_d7_executive_readout.py` | `reports/d7_executive_readout.pptx` (8 slides) |

## 4. Results

### D3 — weekly demand forecast (primary result)

Model-selection rule: the ML model is kept **only** if it beats the
seasonal-naive baseline on WAPE on identical rolling-origin folds — it did.

| Metric | Model — `hist_gradient_boosting` | Baseline — `seasonal_naive` (52-week) |
|---|---|---|
| **WAPE (primary)** | **0.5073** | 0.6389 |
| Bias (secondary; + = over-forecast, − = under-forecast) | +0.0902 | −0.1224 |
| MAPE (secondary) | 0.6669 | 0.7602 |

**Improvement vs baseline: 20.6% WAPE reduction** (model 0.5073 vs baseline
0.6389; source: `data/processed/d3_model_comparison.json`,
`reports/d3_forecast_report.json`).

Methodology (configured in `reports/generate_d3_report.py` / `ForecastConfig`):

- Grain: weekly SKU-level demand; horizon: 8 weeks; random seed 42.
- Evaluation: 3-fold rolling-origin backtest, chronological splits only
  (train strictly before test; no random/shuffled 70-30 split); minimum
  12 training weeks per fold.
- Model: scikit-learn `HistGradientBoostingRegressor` (`max_iter=60`,
  `max_leaf_nodes=15`); baseline: seasonal-naive with a 52-week period.
- Leakage prevention: explicit `LeakageError` checks in feature preparation.
- Coverage: all 4,616 SKUs forecast for 8 weeks (36,928 forecast rows).
  4,301 SKUs with fewer than 12 weeks of history use the documented
  `category_historical` fallback and are flagged low-confidence
  (34,408 fallback rows vs 2,520 model rows) rather than dropped or
  padded with zeros.

The seasonal-naive baseline is evaluable only where a full 52-week seasonal
period exists (237 SKU-week test rows across the three folds); model and
baseline are compared on identical rolling-origin test windows
(`same_windows: true` in `d3_model_comparison.json`).

### D4 — inventory risk & decisions

- 4,495 SKUs scored (status `READY`; official D3 weekly forecast integrated).
- Four-cell decision grid: **REORDER_NOW 118 · MARKDOWN_CLEAR 1,516 ·
  WATCH_VOLATILE 2,792 · HEALTHY 69**.
- Total rupee value at stake: **₹1,555,838,458.90** — cost-basis valuation on
  official `unit_cost` (stockout-shortfall component: ₹240,959.03; total
  on-hand inventory position: 3,750,907 units).
- Insufficient-history SKUs are handled conservatively (placed in
  `WATCH_VOLATILE` without precise scores; low-history SKUs are flagged and
  reported without fabricated precision).
- Decision thresholds (coverage targets, volatility cut-offs, matrix
  cut-offs) are documented implementation assumptions — see
  `configuration_assumptions` in `reports/d4_risk_report.json`. Rupee
  exposure uses cost-basis valuation; lost revenue is not inferred.

### D2 — data quality & EDA

- Analysis-ready coverage: 24,109 SKU-day rows, 4,616 SKUs, 1,461 days
  (2022-01-01 → 2025-12-31).
- Memo (`reports/d2_eda_memo.md`), machine-readable report
  (`reports/d2_eda_report.json`) and 8 charts (`reports/d2_charts/`):
  daily demand, weekly demand, weekly revenue, weekly seasonality, category
  contribution, demand variability, promotion impact, top-SKU concentration.

## 5. Repository structure

```text
mini-foresight/
├── data/
│   ├── raw/
│   │   ├── retail_contaminated_dataset/   # Official inputs (git-ignored)
│   │   └── official_selected/             # sales_transactions_25000.csv +
│   │                                      # selection_manifest.json (git-ignored)
│   └── processed/                         # D1 outputs, D3 forecasts, D4 risk tables
├── notebooks/                             # Legacy Mini-FORESIGHT notebooks (reference only)
├── src/                                   # Reusable source code
│   ├── paths.py                           # Root-relative path conventions
│   ├── official_selection.py              # Deterministic 25,000-row selection (seed 42)
│   ├── pipeline.py                        # D1 — ingest, clean, validate, feature build
│   ├── retail_adapter.py                  # Shared retail feature engineering
│   ├── eda.py                             # D2 — profiling helpers
│   ├── forecast.py                        # D3 — forecasting models + backtest
│   └── risk.py                            # D4 — risk scoring + decision grid
├── app/
│   └── app.py                             # D5 — Streamlit planning dashboard
├── service/                               # D6 — FastAPI scoring service
├── reports/                               # D2–D4 reports + D7 executive readout (.pptx)
├── tests/                                 # verify_d1.py … verify_d4.py checks
├── requirements.txt
├── render.yaml                            # Render deployment blueprint
├── DEPLOYMENT.md                          # Deployment guide
└── README.md
```

## 6. Running the project

All commands run from the `mini-foresight/` repository root with Python 3.13
and the dependencies from `requirements.txt` installed
(`pip install -r requirements.txt`).

1. Deterministic official data selection (run once; already done):
   `python -m src.official_selection`
2. D1 — data pipeline: `python -m src.pipeline`
3. D2 — EDA report & memo: `python reports/generate_d2_report.py`
4. D3 — forecast, backtest and report: `python reports/generate_d3_report.py`
5. D4 — risk scoring and report: `python reports/generate_d4_report.py`
6. D5 — planning dashboard: `streamlit run app/app.py`
7. D6 — scoring service: `uvicorn service.main:app --reload`
   (interactive API docs at `/docs`; I/O contract in `service/README.md`)
8. D7 — executive readout: `python reports/generate_d7_executive_readout.py`
9. Verifications: `python tests/verify_d1.py` … `python tests/verify_d4.py`

Steps 2–5 and 8 regenerate every artefact deterministically; the selection
and D1 fail fast if the official input files are missing.

## 7. Deployment status

The D5 dashboard and D6 scoring service are deployment-ready (FastAPI app,
Streamlit app, `render.yaml` blueprint, `DEPLOYMENT.md` guide) but are not
deployed yet, so **no public URLs exist**. Deployment URLs will be added to
this section after deployment; do not rely on any URL not listed here.

## 8. Honest reporting guard

- Every metric in this README is reproduced from committed artefacts
  (`d3_model_comparison.json`, `d3_forecast_report.json`,
  `d4_risk_report.json`, `d1_data_quality_report.json`).
- Risk thresholds and coverage targets are documented implementation
  assumptions (`configuration_assumptions` in `reports/d4_risk_report.json`).
- No record is fabricated to fill calendar gaps; insufficient-history SKUs
  are handled via the documented category fallback and flagged instead.
- The legacy `notebooks/` Mini-FORESIGHT demos are kept for reference only
  and are not part of any official result.
