"""Documented request/response schemas for the FORESIGHT scoring service.

Request models are strict Pydantic models (validated by FastAPI). Response
shapes are produced by the builder helpers below so that every endpoint emits
a consistent, machine-readable contract; unavailable computed values are
always ``None`` — never fabricated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class ScoreSKURequest(BaseModel):
    """Request body for ``POST /score/sku``.

    Attributes
    ----------
    sku_id: Official SKU identifier from sku_master. Required, non-empty
        after trimming whitespace.
    horizon_weeks: Optional forecast-horizon override in weeks. Must be an
        integer between 1 and 52 when supplied. When omitted, the service uses
        the official D3 default configuration (8 weeks).
    """

    sku_id: str = Field(
        ...,
        min_length=1,
        description="Official FORESIGHT SKU identifier (e.g. from sku_master).",
        examples=["SKU001"],
    )
    horizon_weeks: Optional[int] = Field(
        default=None,
        ge=1,
        le=52,
        description=(
            "Optional forecast horizon override in whole weeks (1-52). "
            "Omit to use the official D3 default horizon."
        ),
    )

    @field_validator("sku_id")
    @classmethod
    def _sku_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("sku_id must contain a non-whitespace value.")
        return v


class ScoreBatchRequest(BaseModel):
    """Request body for ``POST /score/batch``.

    A list of one or more official SKU identifiers. Duplicate identifiers are
    de-duplicated (order preserved) and reported in the response rather than
    scored twice.
    """

    sku_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Non-empty list of official FORESIGHT SKU identifiers.",
    )
    horizon_weeks: Optional[int] = Field(
        default=None,
        ge=1,
        le=52,
        description="Optional forecast-horizon override in weeks (applies to "
        "every SKU in the batch). Omit for the official D3 default.",
    )

    @field_validator("sku_ids")
    @classmethod
    def _ids_not_blank(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("sku_ids must not contain empty or blank entries.")
        return cleaned


# --------------------------------------------------------------------------- #
# Response payload builders (single source of response shape)
# --------------------------------------------------------------------------- #

STATUS_OK = "ok"
STATUS_DATA_NOT_AVAILABLE = "data_not_available"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
STATUS_SKU_NOT_FOUND = "sku_not_found"
STATUS_INVALID_REQUEST = "invalid_request"
STATUS_ERROR = "error"

SERVICE_NAME = "FORESIGHT scoring service"


def build_data_not_available(
    message: str,
    missing_d1_outputs: Optional[List[str]] = None,
    reasons: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Machine-readable gated response used by every scoring endpoint.

    ``message`` reuses the exact readiness wording provided by the D3/D4
    modules wherever those modules already supply it. No numeric results are
    included by design.
    """
    payload: Dict[str, Any] = {
        "status": STATUS_DATA_NOT_AVAILABLE,
        "message": message,
        "missing_d1_outputs": missing_d1_outputs or [],
        "reasons": reasons or [],
        "forecast": None,
        "risk": None,
        "decision": None,
        "rupee_value_at_stake": None,
    }
    if extra:
        payload.update(extra)
    return payload


def build_insufficient_data(
    message: str, reasons: Optional[List[str]] = None
) -> Dict[str, Any]:
    return {
        "status": STATUS_INSUFFICIENT_DATA,
        "message": message,
        "reasons": reasons or [],
        "forecast": None,
        "risk": None,
        "decision": None,
        "rupee_value_at_stake": None,
    }


def build_invalid_request(message: str, errors: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "status": STATUS_INVALID_REQUEST,
        "message": message,
        "errors": errors or [],
    }


def build_error(message: str = "Unexpected internal error.") -> Dict[str, Any]:
    """Generic structured error (never exposes Python tracebacks)."""
    return {"status": STATUS_ERROR, "message": message}
