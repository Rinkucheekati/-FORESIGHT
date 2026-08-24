"""FORESIGHT scoring service — FastAPI application (D6, data-gated).

Purpose
-------
Expose the OFFICIAL Project FORESIGHT demand-forecasting (D3) and inventory-
risk/decision (D4) engines over HTTP for a single SKU or a batch of SKUs.

This module is an API layer ONLY: forecasting and risk computations are
delegated to ``src.forecast`` and ``src.risk``; nothing is recomputed here and
no numbers are invented.

Data gating
-----------
Official source chain (required)::

    data/raw/    sales_daily.csv · sku_master.csv · calendar.csv ·
                 inventory_snapshots.csv      (official internship extracts)
        ↓  official D1 pipeline (src.pipeline)
    data/processed/  *_clean.csv                  (analysis-ready inputs)

If the D1 outputs are absent, scoring endpoints return HTTP 200 with
``status = "data_not_available"`` and ``null`` result blocks — using the exact
readiness wording supplied by the D3/D4 modules. The service NEVER falls back
to legacy Mini-FORESIGHT artifacts and never fabricates values.

Run from the repository root::

    uvicorn service.main:app --reload

Interactive docs: ``/docs`` · OpenAPI schema: ``/openapi.json``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# --- Repository-relative import bootstrap (no hard-coded absolute paths) ----
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import forecast as fc  # noqa: E402
from src import risk as risk_mod  # noqa: E402
from src.risk import NO_OBSERVATIONS_MESSAGE  # noqa: E402

from service.schemas import (  # noqa: E402
    SERVICE_NAME,
    STATUS_DATA_NOT_AVAILABLE,
    STATUS_INSUFFICIENT_DATA,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_INVALID_REQUEST,
    STATUS_OK,
    STATUS_SKU_NOT_FOUND,
    ScoreBatchRequest,
    ScoreSKURequest,
    build_data_not_available,
    build_error,
    build_insufficient_data,
    build_invalid_request,
)

logger = logging.getLogger("foresight.service")

DEFAULT_FCST_CFG = fc.ForecastConfig()
DEFAULT_RISK_CFG = risk_mod.RiskConfig()

app = FastAPI(
    title="Project FORESIGHT Scoring Service",
    description=(
        "Official Zidio FORESIGHT D6 API.\n\n"
        "Returns **weekly demand forecasts** (D3) and **inventory risk / "
        "four-cell decisions** (D4) for one SKU or a batch of SKUs.\n\n"
        "**Data availability:** responses are computed exclusively from the "
        "official D1 analysis-ready outputs (`data/processed/*_clean.csv`). "
        "When those are absent the service responds with "
        "`status = \"data_not_available\"` and `null` result blocks — it never "
        "fabricates forecasts, risk scores, or monetary values, and never "
        "falls back to legacy demo artifacts."
    ),
    version="1.0.0",
)


# --------------------------------------------------------------------------- #
# Internal helpers (wiring only — no business logic)
# --------------------------------------------------------------------------- #


def _fcst_cfg(horizon_weeks: Optional[int]) -> fc.ForecastConfig:
    """Build the official D3 config, applying an optional horizon override."""
    if horizon_weeks is None:
        return DEFAULT_FCST_CFG
    try:
        cfg = fc.ForecastConfig(horizon_weeks=int(horizon_weeks))
        cfg.validate()
        return cfg
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=build_invalid_request(
                "Invalid horizon_weeks override.", [str(exc)]
            ),
        )


def _combined_readiness(
    fcfg: fc.ForecastConfig,
    rcfg: risk_mod.RiskConfig,
) -> Dict[str, Any]:
    """Run BOTH official gates verbatim and combine into one snapshot."""
    fc_ready = fc.check_forecast_data_readiness(None, fcfg)
    rk_ready = risk_mod.check_risk_data_readiness(None, rcfg)

    statuses = {fc_ready["status"], rk_ready["status"]}
    if "DATA_NOT_AVAILABLE" in statuses:
        combined = "data_not_available"
        message = NO_OBSERVATIONS_MESSAGE
    elif "INSUFFICIENT_DATA" in statuses:
        combined = "insufficient_data"
        fc_reasons = fc_ready.get("reasons", []) or []
        rk_reasons = rk_ready.get("reasons", []) or []
        message = "; ".join([*rk_reasons, *fc_reasons][:3]) or (
            "Official data present but insufficient for scoring."
        )
    else:
        combined = "ready"
        message = "Official FORESIGHT data is ready for scoring."

    return {
        "combined_status": combined,
        "message": message,
        "forecast_readiness": {
            "status": fc_ready["status"],
            "reasons": fc_ready.get("reasons", []),
            "missing_d1_outputs": fc_ready.get("missing_d1_outputs"),
            "details": fc_ready.get("details"),
        },
        "risk_readiness": {
            "status": rk_ready["status"],
            "reasons": rk_ready.get("reasons", []),
            "missing_d1_outputs": rk_ready.get("missing_d1_outputs"),
        },
    }

def _score_one_sku(sku_id: str, fcfg: fc.ForecastConfig,
                   rcfg: risk_mod.RiskConfig) -> Dict[str, Any]:
    """Score ONE SKU through the official D3 + D4 interfaces (no logic here).

    ``status`` is one of: ok | insufficient_history | sku_not_found | error.
    ``forecast`` / ``risk`` blocks are populated only when the official data
    actually supports them.
    """
    payload: Dict[str, Any] = {
        "sku_id": sku_id,
        "status": STATUS_OK,
        "message": "",
        "forecast": None,
        "risk": None,
        "decision": None,
        "decision_reason": None,
        "rupee_value_at_stake": None,
    }

    # ---- Official D3 forecast -------------------------------------------
    try:
        fcres = fc.forecast_weekly_sku(sku_id, None, fcfg)
    except fc.InsufficientHistoryError as exc:
        payload["status"] = STATUS_INSUFFICIENT_HISTORY
        payload["message"] = str(exc)
        return payload
    except Exception as exc:  # noqa: BLE001 - structured, no traceback
        logger.exception("D3 forecast failed for %s", sku_id)
        payload["status"] = "error"
        payload["message"] = (
            "Official D3 engine could not produce a forecast for this SKU."
        )
        payload["detail"] = str(exc)[:200]
        return payload

    payload["forecast"] = {
        "status": STATUS_OK,
        "forecast_run_id": fcres.get("forecast_run_id"),
        "model_name": fcres.get("model_name"),
        "horizon_weeks": fcres.get("horizon_weeks"),
        "fallback_used": fcres.get("fallback_used"),
        "low_history_flagged": fcres.get("low_history_flagged"),
        "category": fcres.get("category"),
        "subcategory": fcres.get("subcategory"),
        "history_summary": fcres.get("history_summary"),
        "rows": fcres.get("forecast_rows") or [],
    }

    # ---- Official D4 risk / decision ------------------------------------
    try:
        scored = risk_mod.score_all_skus(None, rcfg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("D4 scoring failed for %s", sku_id)
        payload["risk"] = {
            "status": "error",
            "note": "Official D4 scoring failed.",
            "detail": str(exc)[:200],
        }
        return payload

    if scored.get("status") != risk_mod.STATUS_READY:
        payload["risk"] = {
            "status": scored.get("status"),
            "reasons": scored.get("reasons", []),
            "missing_d1_outputs": scored.get("missing_d1_outputs"),
            "note": "D4 produced no SKU records for the current official data.",
        }
        payload["status"] = (
            STATUS_DATA_NOT_AVAILABLE
            if scored.get("status") == risk_mod.STATUS_DATA_NOT_AVAILABLE
            else STATUS_INSUFFICIENT_DATA
        )
        return payload

    record = next(
        (r for r in scored.get("scored_skus", [])
         if str(r.get("sku_id")) == str(sku_id)),
        None,
    )
    if record is None:
        payload["status"] = STATUS_SKU_NOT_FOUND
        payload["message"] = (
            "SKU has a forecast but no official D4 inventory record."
        )
        return payload

    payload["risk"] = {
        "status": STATUS_OK,
        "category": record.get("category"),
        "subcategory": record.get("subcategory"),
        "snapshot_date": record.get("snapshot_date"),
        "on_hand_units": record.get("on_hand_units"),
        "on_order_units": record.get("on_order_units"),
        "inventory_position": record.get("inventory_position"),
        "lead_time_days": record.get("lead_time_days"),
        "source_reorder_point": record.get("source_reorder_point"),
        "calculated_risk_threshold": record.get("calculated_risk_threshold"),
        "demand_rate_weekly": record.get("demand_rate_weekly"),
        "forecast_rate_weekly": record.get("forecast_rate_weekly"),
        "preferred_demand_source": record.get("preferred_demand_source"),
        "coverage_weeks": record.get("coverage_weeks"),
        "shortfall_units": record.get("shortfall_units"),
        "stockout_risk": record.get("stockout_risk"),
        "overstock_risk": record.get("overstock_risk"),
        "volatility_cv": record.get("volatility_cv"),
        "unit_cost": record.get("unit_cost"),
        "excess_units_over_healthy": record.get("excess_units_over_healthy"),
        "inventory_value_exposure_rupees":
            record.get("inventory_value_exposure_rupees"),
        "stockout_shortfall_rupees": record.get("stockout_shortfall_rupees"),
        "data_quality_flag": record.get("data_quality_flag"),
        "weeks_observed": record.get("weeks_observed"),
    }
    payload["decision"] = record.get("decision")
    payload["decision_reason"] = record.get("decision_reason")
    payload["rupee_value_at_stake"] = record.get("rupee_value_at_stake")
    return payload

def _gated_response_if_needed(readiness: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a clean gated payload when official data is not ready."""
    combined = readiness["combined_status"]
    if combined == "data_not_available":
        rk = readiness["risk_readiness"]
        fcr = readiness["forecast_readiness"]
        missing = sorted(set(rk.get("missing_d1_outputs") or [])
                         | set(fcr.get("missing_d1_outputs") or []))
        reasons: List[str] = [*(rk.get("reasons") or [])]
        reasons += [r for r in (fcr.get("reasons") or []) if r not in reasons]
        return build_data_not_available(
            message=readiness["message"],
            missing_d1_outputs=missing,
            reasons=reasons,
            extra={
                "forecast_readiness_status": fcr.get("status"),
                "risk_readiness_status": rk.get("status"),
            },
        )
    if combined == "insufficient_data":
        rk = readiness["risk_readiness"]
        fcr = readiness["forecast_readiness"]
        seen, uniq = set(), []
        for r in [*(rk.get("reasons") or []), *(fcr.get("reasons") or [])]:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return build_insufficient_data(readiness["message"], uniq)
    return None


@app.get("/", tags=["meta"], summary="Service description")
def root() -> Dict[str, Any]:
    """Short service description and available endpoints (no business metrics)."""
    return {
        "service": SERVICE_NAME,
        "project": "Project FORESIGHT — D6 scoring service",
        "version": app.version,
        "data_policy": (
            "Responses are computed only from official D1 analysis-ready "
            "outputs. When they are absent, scoring endpoints return "
            "status='data_not_available' with null result blocks."
        ),
        "endpoints": {
            "GET /health": "Liveness probe (service running ≠ data available).",
            "GET /data-status": "Official D1/D3/D4 data-readiness snapshot.",
            "POST /score/sku": "Forecast + risk + decision for ONE SKU.",
            "POST /score/batch": "Forecast + risk for a list of SKUs.",
            "GET /docs": "Interactive OpenAPI documentation.",
            "GET /openapi.json": "Machine-readable API contract.",
        },
    }


@app.get("/health", tags=["meta"], summary="Liveness probe")
def health() -> Dict[str, Any]:
    """Verify the service itself is running.

    NOTE: a healthy service does NOT imply official data is available —
    use ``GET /data-status`` for that.
    """
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/data-status", tags=["meta"],
         summary="Official data-readiness snapshot")
def data_status() -> Dict[str, Any]:
    """Combined official-data readiness from the D3 and D4 gates.

    Always HTTP 200; the ``status`` field carries the gate outcome.
    """
    try:
        readiness = _combined_readiness(DEFAULT_FCST_CFG, DEFAULT_RISK_CFG)
    except Exception:  # noqa: BLE001
        logger.exception("data-status failed")
        return JSONResponse(status_code=500, content=build_error())

    payload: Dict[str, Any] = {
        "status": readiness["combined_status"],
        "message": readiness["message"],
        "forecast_readiness": readiness["forecast_readiness"],
        "risk_readiness": readiness["risk_readiness"],
    }
    gated = _gated_response_if_needed(readiness)
    if gated is not None:
        payload["missing_d1_outputs"] = gated.get("missing_d1_outputs")
        payload["reasons"] = gated.get("reasons")
    else:
        scored = risk_mod.score_all_skus(None, DEFAULT_RISK_CFG)
        report = risk_mod.create_risk_report(scored, DEFAULT_RISK_CFG)
        payload["risk_report_summary"] = {
            k: report.get(k) for k in (
                "inventory_summary", "stockout_risk_summary",
                "overstock_risk_summary", "rupee_value_summary",
                "decision_summary", "low_history_summary",
                "configuration_assumptions", "limitations",
            )
        }
    return payload

@app.post("/score/sku", tags=["scoring"], summary="Score ONE SKU",
          responses={
              200: {"description": "Scored payload, or a clean "
                                   "data_not_available state (no fabricated "
                                   "numbers)."},
              422: {"description": "Invalid request body or horizon."},
          })
def score_sku(body: ScoreSKURequest) -> Dict[str, Any]:
    """Forecast + risk + decision for a single SKU via the official engines.

    Behavior
    --------
    * Official data absent  → 200 with ``status="data_not_available"``,
      ``forecast=null``, ``risk=null`` (module wording reused verbatim).
    * Data ready            → D3 forecast rows + D4 risk/decision/rupee.
    * SKU lacks history     → ``status="insufficient_history"`` (honest).
    """
    try:
        fcfg = _fcst_cfg(body.horizon_weeks)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {
            "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=detail)

    try:
        readiness = _combined_readiness(fcfg, DEFAULT_RISK_CFG)
    except Exception:  # noqa: BLE001
        logger.exception("readiness failed on /score/sku")
        return JSONResponse(status_code=500, content=build_error())

    gated = _gated_response_if_needed(readiness)
    if gated is not None:
        gated["sku_id"] = body.sku_id
        return gated

    result = _score_one_sku(body.sku_id, fcfg, DEFAULT_RISK_CFG)
    result["data_status"] = "ready"
    return result


@app.post("/score/batch", tags=["scoring"], summary="Score a batch of SKUs",
          responses={
              200: {"description": "Per-SKU results, or a clean gated state."},
              422: {"description": "Invalid request body or horizon."},
          })
def score_batch(body: ScoreBatchRequest) -> Dict[str, Any]:
    """Forecast + risk for each SKU in the batch.

    Duplicate IDs are de-duplicated (order preserved) and counted in
    ``duplicates_ignored``. Each SKU carries its own per-item status; a single
    bad SKU never fails the whole batch.
    """
    try:
        fcfg = _fcst_cfg(body.horizon_weeks)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {
            "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=detail)

    unique_ids: List[str] = []
    for s in body.sku_ids:
        if s not in unique_ids:
            unique_ids.append(s)
    duplicates_ignored = len(body.sku_ids) - len(unique_ids)

    try:
        readiness = _combined_readiness(fcfg, DEFAULT_RISK_CFG)
    except Exception:  # noqa: BLE001
        logger.exception("readiness failed on /score/batch")
        return JSONResponse(status_code=500, content=build_error())

    gated = _gated_response_if_needed(readiness)
    if gated is not None:
        gated["sku_ids"] = unique_ids
        gated["duplicates_ignored"] = duplicates_ignored
        return gated

    results = [_score_one_sku(s, fcfg, DEFAULT_RISK_CFG) for s in unique_ids]
    return {
        "data_status": "ready",
        "sku_ids": unique_ids,
        "duplicates_ignored": duplicates_ignored,
        "results": results,
    }


# --------------------------------------------------------------------------- #
# Graceful error handling (never expose Python tracebacks)
# --------------------------------------------------------------------------- #

from fastapi.exceptions import RequestValidationError  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_req, exc: RequestValidationError):
    errors: List[str] = []
    for err in exc.errors()[:10]:
        loc = ".".join(str(p) for p in err.get("loc", []))
        errors.append(f"{loc}: {err.get('msg')}")
    return JSONResponse(
        status_code=422,
        content=build_invalid_request("Request body failed validation.", errors),
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_http_handler(_req, exc: StarletteHTTPException):
    detail = exc.detail
    content = detail if isinstance(detail, dict) else {"message": str(detail)}
    content.setdefault("status",
                       "error" if exc.status_code >= 500 else STATUS_INVALID_REQUEST)
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_req, exc: Exception):
    logger.exception("Unhandled service error")
    return JSONResponse(status_code=500, content=build_error())