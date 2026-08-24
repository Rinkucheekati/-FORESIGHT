# FORESIGHT Scoring Service (D6)

Official **Project FORESIGHT** scoring API (Zidio internship project, D6).
Exposes the already-implemented **D3 weekly demand forecasting engine**
(`src/forecast.py`) and **D4 inventory-risk / four-cell decision engine**
(`src/risk.py`) over HTTP for a single SKU or a batch of SKUs.

> ⚠️ **SERVICE RUNNING ≠ OFFICIAL DATA AVAILABLE.**
> A `200` from `/health` only means this API process is alive. Whether real
> FORESIGHT results can be returned is governed exclusively by official
> data readiness — check `GET /data-status`.

---

## 1. Purpose

| Requirement (Zidio D6) | Where satisfied |
|---|---|
| Deployed scoring service | FastAPI app in `service/main.py` |
| Forecast + risk for one SKU or batch | `POST /score/sku`, `POST /score/batch` |
| Documented inputs/outputs | Pydantic models in `service/schemas.py` + `/docs` |
| Graceful bad-input handling | Validation + structured error handlers |

## 2. Architecture

```
HTTP client ──► service/main.py        (API layer ONLY: routing/validation)
                   │  delegates to
                   ├─► src/forecast.py   D3: readiness, weekly SKU forecast,
                   │                      horizon config, honest history checks
                   └─► src/risk.py       D4: inventory position, stockout/
                                          overstock risk, rupee at stake,
                                          four-cell decision
                                  │ reads only
                                  ▼
                       data/processed/*_clean.csv   (official D1 outputs)
```

No forecasting or risk logic exists in this package — it wires the official
engines (`check_forecast_data_readiness`, `forecast_weekly_sku`,
`create_forecast_report`, `check_risk_data_readiness`, `score_all_skus`,
`create_risk_report`).

## 3. Install & run (local development)

From the repository root:

```bash
pip install -r requirements.txt      # includes fastapi/pydantic/uvicorn/httpx
uvicorn service.main:app --reload    # http://127.0.0.1:8000
```

Interactive docs: <http://127.0.0.1:8000/docs>
Machine contract: <http://127.0.0.1:8000/openapi.json>

## 4. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness probe → `{"status":"ok","service":"FORESIGHT scoring service"}` |
| GET | `/data-status` | Official D1/D3/D4 readiness snapshot (+ D4 report summary when READY) |
| POST | `/score/sku` | Forecast + risk + decision + rupee-at-stake for ONE SKU |
| POST | `/score/batch` | Same, for a list of SKUs (duplicates de-duplicated) |
| GET | `/` | Service description and endpoint index |
| GET | `/docs`, `/openapi.json` | Interactive docs / machine contract |

## 5. Request examples

Single SKU (default official horizon):

```json
POST /score/sku
{ "sku_id": "SKU001" }
```

With an explicit horizon override (weeks):

```json
POST /score/sku
{ "sku_id": "SKU001", "horizon_weeks": 12 }
```

Batch:

```json
POST /score/batch
{ "sku_ids": ["SKU001", "SKU002"], "horizon_weeks": null }
```

## 6. Response examples

**A. Official data ready (shape illustrated; values come only from real data):**

```json
{
  "sku_id": "SKU001",
  "status": "ok",
  "data_status": "ready",
  "forecast": {
    "status": "ok",
    "forecast_run_id": "fc_YYYYMMDDTHHMMSSZ",
    "model_name": "hist_gradient_boosting",
    "horizon_weeks": 8,
    "fallback_used": false,
    "low_history_flagged": false,
    "rows": [
      {"period": "2025-W20", "sku_id": "SKU001",
       "forecast_units": 12.4, "model_name": "hist_gradient_boosting"}
    ]
  },
  "risk": {
    "status": "ok",
    "inventory_position": 340,
    "coverage_weeks": 6.2,
    "stockout_risk": 0.12,
    "overstock_risk": 0.0,
    "volatility_cv": 0.31,
    "source_reorder_point": 250,
    "calculated_risk_threshold": 180.5
  },
  "decision": "HEALTHY",
  "rupee_value_at_stake": 0.0
}
```

**B. Official row-level data unavailable (current repository state):**

```json
{
  "status": "data_not_available",
  "message": "No official FORESIGHT row-level observations are currently available ...",
  "missing_d1_outputs": ["calendar_clean.csv"],
  "reasons": ["..."],
  "sku_id": "SKU001",
  "forecast": null,
  "risk": null,
  "decision": null,
  "rupee_value_at_stake": null
}
```

Other per-SKU statuses: `insufficient_history`, `sku_not_found`,
`invalid_request`, `error`. Batch responses add `sku_ids`,
`duplicates_ignored`, and a `results[]` list carrying each item's own status.

## 7. Data-gated behavior

* Inputs are ONLY the official D1 outputs:
  `data/processed/{sales_daily,sku_master,calendar,inventory_snapshots}_clean.csv`.
* If any are missing → every scoring endpoint returns
  `status="data_not_available"` with the exact wording supplied by the
  official D3/D4 gates and `null` result blocks.
* Insufficient history → `status="insufficient_history"` (never padded with zeros).
* The service NEVER reads legacy Mini-FORESIGHT files
  (`data/raw/*`, legacy `forecast_results.csv`, `inventory_risk.csv`,
  `recommendations.csv`) and never generates replacement data.

**No fabricated data or results are ever returned.** Unavailable computed
values are always `null`.

## 8. Error handling

| Case | Result |
|---|---|
| Missing/blank `sku_id`, bad types | HTTP 422 `invalid_request` with field-level errors |
| Empty batch | HTTP 422 |
| Invalid `horizon_weeks` | HTTP 422 structured message |
| Duplicate batch IDs | De-duplicated; `duplicates_ignored` reported |
| Missing D1 outputs | HTTP 200 `data_not_available` (documented) |
| Insufficient history | HTTP 200 `insufficient_history` per SKU |
| Unexpected internal error | HTTP 500 generic message — tracebacks never exposed |
