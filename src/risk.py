"""D4 — Inventory risk & decision engine for Project FORESIGHT.

Purpose
-------
Converts official demand/inventory information into actionable SKU-level
decisions across TWO independent risk axes (stockout and overstock),
quantifies the RUPEE VALUE AT STAKE using official monetary fields, and
assigns the official FOUR-CELL decision grid:

    REORDER_NOW | MARKDOWN_CLEAR | WATCH_VOLATILE | HEALTHY

The old Mini-FORESIGHT "High/Medium/Low reorder-only" approach is NOT used.

Data gating (critical)
----------------------
Consumes ONLY the official D1 analysis-ready outputs under ``data/processed/``::

    sales_daily_clean.csv
    sku_master_clean.csv
    calendar_clean.csv
    inventory_snapshots_clean.csv

plus the official D3 forecast output WHEN AVAILABLE. Legacy raw files under
``data/raw/`` are never read. When official inputs are absent, every public
entry point returns a structured ``DATA_NOT_AVAILABLE`` status — no risk
scores, monetary values, or decisions are fabricated.

IMPORTANT NOTE ON ``data/processed/forecast_results.csv``
---------------------------------------------------------
A LEGACY Mini-FORESIGHT file of that name (daily grain, ``date`` column)
exists in this repository. The official D3 weekly output uses a ``period``
column (ISO ``YYYY-Www``) plus ``sku_id``, ``forecast_units``, ``model_name``.
D4 therefore validates the forecast-file schema strictly: a file lacking the
official weekly ``period`` column is treated as NOT an official D3 output and
is ignored (forecast status reported as unavailable). This makes it impossible
for the legacy demo forecast to leak into official risk scoring.

Official requirements vs implementation assumptions
----------------------------------------------------
OFFICIAL (from the Zidio brief): two-axis risk, rupee value at stake,
four-cell decision grid, official fields (on_hand_units, on_order_units,
lead_time_days, reorder_point, unit_cost, list_price), new-SKU/low-history
handling, honest data-gating.

IMPLEMENTATION CONFIGURATION ASSUMPTIONS (NOT supplied by Zidio — all
configurable and labelled): coverage targets, overstock coverage threshold,
volatility CV threshold, minimum-history requirements, and the numeric cut-offs
inside the decision matrix. See ``RiskConfig``.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.paths import DATA_PROCESSED

# --------------------------------------------------------------------------- #
# Logging / exception types
# --------------------------------------------------------------------------- #

logger = logging.getLogger("foresight.risk")


class D4Error(Exception):
    """Base class for D4 risk-engine errors."""


class MissingOfficialInputsError(D4Error):
    """Raised when required official D1 analysis-ready outputs are absent."""

    def __init__(self, missing: List[str], processed_dir: Path) -> None:
        self.missing = missing
        self.processed_dir = Path(processed_dir)
        super().__init__(
            "Official FORESIGHT D1 analysis-ready outputs are missing: "
            + ", ".join(missing)
            + f". Expected under: {self.processed_dir}. D4 refuses to run on "
            "raw/legacy data (e.g. the Mini-FORESIGHT demo files). Run the "
            "official D1 pipeline first; this engine never creates, downloads, "
            "or fabricates data."
        )


class InsufficientDataError(D4Error):
    """Raised/returned when official data exists but cannot support risk scoring."""


class InvalidDataError(D4Error):
    """Raised when official inputs violate basic validity rules irreparably."""


# --------------------------------------------------------------------------- #
# Contracts & constants
# --------------------------------------------------------------------------- #

D1_OUTPUT_FILES = {
    "sales_daily": "sales_daily_clean.csv",
    "sku_master": "sku_master_clean.csv",
    "calendar": "calendar_clean.csv",
    "inventory_snapshots": "inventory_snapshots_clean.csv",
}

# Official D3 weekly forecast output (schema-validated; legacy file rejected).
D3_FORECAST_FILE = "forecast_results.csv"

# Official four-cell decision grid (exact business actions).
DECISION_REORDER_NOW = "REORDER_NOW"
DECISION_MARKDOWN_CLEAR = "MARKDOWN_CLEAR"
DECISION_WATCH_VOLATILE = "WATCH_VOLATILE"
DECISION_HEALTHY = "HEALTHY"

# Data-quality / confidence flags.
FLAG_OK = "ok"
FLAG_LOW_HISTORY = "low_history"
FLAG_INSUFFICIENT_DATA = "insufficient_data"

# Demand-source labels (historical vs official D3 forecast).
SOURCE_HISTORICAL = "historical"
SOURCE_D3_FORECAST = "d3_forecast"

STATUS_READY = "READY"
STATUS_DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

NO_OBSERVATIONS_MESSAGE = (
    "No official FORESIGHT row-level observations are currently available for "
    "inventory-risk scoring. No risk scores, monetary exposure, or SKU "
    "decisions have been fabricated."
)


# --------------------------------------------------------------------------- #
# Configuration — IMPLEMENTATION ASSUMPTIONS (not supplied by Zidio)
# --------------------------------------------------------------------------- #


@dataclass
class RiskConfig:
    """Operational thresholds for D4.

    OFFICIAL vs CONFIGURATION: the Zidio brief requires two-axis risk, rupee
    value at stake and the four-cell grid, but does NOT supply numeric
    thresholds. Every number below is a documented CONFIGURATION ASSUMPTION —
    configurable per run, reported as such, never presented as a Zidio rule.

    coverage_target_weeks : healthy weeks-of-cover target. ASSUMPTION: 4.0.
    overstock_coverage_weeks : cover above which an SKU is overstocked.
        ASSUMPTION: 8.0 (risk scales linearly to 1.0 at double this cover).
    stockout_ratio_threshold : inventory-position / lead-time-demand ratio
        below which stockout risk begins (ratio >= threshold -> 0; ratio 0 -> 1).
        ASSUMPTION: 1.0.
    volatility_cv_high : weekly-demand CV above which demand is volatile.
        ASSUMPTION: 0.5.
    min_history_weeks : minimum weekly observations for confident scoring.
        ASSUMPTION: 8.
    low_history_weeks : below this many weeks the SKU is flagged low-history.
        ASSUMPTION: 4.
    reorder_now_stockout_min : stockout score selecting REORDER_NOW. ASSUMPTION: 0.5.
    markdown_clear_overstock_min : overstock score selecting MARKDOWN_CLEAR. ASSUMPTION: 0.5.
    """

    coverage_target_weeks: float = 4.0
    overstock_coverage_weeks: float = 8.0
    stockout_ratio_threshold: float = 1.0
    volatility_cv_high: float = 0.5
    min_history_weeks: int = 8
    low_history_weeks: int = 4
    reorder_now_stockout_min: float = 0.5
    markdown_clear_overstock_min: float = 0.5

    def validate(self) -> None:
        """Reject invalid thresholds (never silently repair configuration)."""
        positive_floats = {
            "coverage_target_weeks": self.coverage_target_weeks,
            "overstock_coverage_weeks": self.overstock_coverage_weeks,
            "stockout_ratio_threshold": self.stockout_ratio_threshold,
            "volatility_cv_high": self.volatility_cv_high,
            "reorder_now_stockout_min": self.reorder_now_stockout_min,
            "markdown_clear_overstock_min": self.markdown_clear_overstock_min,
        }
        for name, val in positive_floats.items():
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"RiskConfig.{name} must be a positive finite number.")
        for name, val in {
            "min_history_weeks": self.min_history_weeks,
            "low_history_weeks": self.low_history_weeks,
        }.items():
            if not isinstance(val, int) or val < 1:
                raise ValueError(f"RiskConfig.{name} must be a positive integer.")
        if self.low_history_weeks > self.min_history_weeks:
            raise ValueError(
                "RiskConfig.low_history_weeks must be <= min_history_weeks."
            )


DEFAULT_RISK_CONFIG = RiskConfig()


# --------------------------------------------------------------------------- #
# Official input loading (data-gated)
# --------------------------------------------------------------------------- #


def _guard_d1_outputs(processed_dir: Path) -> List[str]:
    """Return missing official D1 output filenames ([] if all present)."""
    processed_dir = Path(processed_dir)
    return [
        fname
        for fname in D1_OUTPUT_FILES.values()
        if not (processed_dir / fname).is_file()
    ]


def load_official_inputs(
    processed_dir: Path = DATA_PROCESSED,
    include_forecast: bool = True,
) -> Dict[str, Any]:
    """Load ONLY official D1 analysis-ready outputs (+ official D3 forecast).

    Never reads ``data/raw/``. Raises ``MissingOfficialInputsError`` when any
    of the four D1 files is absent. The returned dict additionally carries the
    D3 forecast block (``forecast``, ``forecast_status``, ``forecast_note``).
    """
    processed_dir = Path(processed_dir)
    missing = _guard_d1_outputs(processed_dir)
    if missing:
        raise MissingOfficialInputsError(missing, processed_dir)

    tables: Dict[str, Any] = {}
    for name, fname in D1_OUTPUT_FILES.items():
        try:
            tables[name] = pd.read_csv(processed_dir / fname)
        except Exception as exc:  # noqa: BLE001 - surface a clear D4 error
            raise D4Error(f"Failed to read official D1 output '{fname}': {exc}") from exc

    if include_forecast:
        tables.update(_load_official_forecast(processed_dir))
    else:
        tables.update(
            {
                "forecast": None,
                "forecast_status": "NOT_REQUESTED",
                "forecast_note": "Forecast integration not requested.",
            }
        )
    return tables


def _load_official_forecast(processed_dir: Path) -> Dict[str, Any]:
    """Consume the official D3 forecast ONLY through a schema-validated file.

    Anti-legacy guard: the file must contain the OFFICIAL weekly columns
    {period, sku_id, forecast_units, model_name} with ISO ``YYYY-Www`` period
    labels. A legacy daily-grain file (``date`` column) is rejected and
    reported unavailable rather than consumed — legacy forecasts can never
    leak into official risk scoring.
    """
    path = Path(processed_dir) / D3_FORECAST_FILE
    required_cols = {"period", "sku_id", "forecast_units", "model_name"}

    if not path.is_file():
        return {
            "forecast": None,
            "forecast_status": "FILE_NOT_PRESENT",
            "forecast_note": (
                f"No official D3 forecast output found ('{D3_FORECAST_FILE}' "
                "absent). Demand rate will use official historical sales only."
            ),
        }
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "forecast": None,
            "forecast_status": "UNREADABLE",
            "forecast_note": f"Official D3 forecast file could not be read: {exc}",
        }

    cols = set(str(c) for c in df.columns)
    if not required_cols.issubset(cols):
        return {
            "forecast": None,
            "forecast_status": "SCHEMA_REJECTED",
            "forecast_note": (
                f"'{D3_FORECAST_FILE}' does not match the official D3 weekly "
                f"schema (requires {sorted(required_cols)}); file ignored. "
                "Legacy/demo forecasts are never consumed by D4."
            ),
        }

    if not df["period"].astype(str).str.match(r"^\d{4}-W\d{2}$").all():
        return {
            "forecast": None,
            "forecast_status": "SCHEMA_REJECTED",
            "forecast_note": (
                "'period' values are not official ISO week labels "
                "(YYYY-Www); file ignored."
            ),
        }

    return {
        "forecast": df,
        "forecast_status": "AVAILABLE",
        "forecast_note": "Official D3 weekly forecast loaded.",
    }

# --------------------------------------------------------------------------- #
# Data readiness
# --------------------------------------------------------------------------- #


def check_risk_data_readiness(
    tables: Optional[Dict[str, Any]] = None,
    config: RiskConfig = DEFAULT_RISK_CONFIG,
    processed_dir: Path = DATA_PROCESSED,
) -> Dict[str, Any]:
    """Structured D4 data-readiness status.

    Verifies: official D1 outputs exist; required official columns exist;
    inventory observations exist; SKU references resolve against sku_master;
    demand-history sufficiency vs config; D3 forecast availability (reported,
    not required — the historical rate is the documented fallback).

    Returns ``status`` in {READY, DATA_NOT_AVAILABLE, INSUFFICIENT_DATA} with
    explicit ``reasons``. No risk numbers are produced when not READY.
    """
    details: Dict[str, Any] = {"config_assumptions": {
        "coverage_target_weeks": config.coverage_target_weeks,
        "overstock_coverage_weeks": config.overstock_coverage_weeks,
        "stockout_ratio_threshold": config.stockout_ratio_threshold,
        "volatility_cv_high": config.volatility_cv_high,
        "min_history_weeks": config.min_history_weeks,
        "low_history_weeks": config.low_history_weeks,
    }}
    reasons: List[str] = []

    missing_files = _guard_d1_outputs(processed_dir)
    if missing_files:
        return {
            "status": STATUS_DATA_NOT_AVAILABLE,
            "reasons": [NO_OBSERVATIONS_MESSAGE],
            "missing_d1_outputs": missing_files,
            "details": details,
        }

    if tables is None:
        try:
            tables = load_official_inputs(processed_dir)
        except MissingOfficialInputsError as exc:
            return {
                "status": STATUS_DATA_NOT_AVAILABLE,
                "reasons": [str(exc)],
                "missing_d1_outputs": exc.missing,
                "details": details,
            }

    # Required official columns per extract.
    required_cols = {
        "sales_daily": {"date", "sku_id", "units_sold"},
        "sku_master": {"sku_id", "unit_cost"},
        "inventory_snapshots": {
            "date", "sku_id", "on_hand_units", "on_order_units",
            "lead_time_days", "reorder_point",
        },
    }
    for name, cols in required_cols.items():
        df = tables.get(name)
        if df is None:
            reasons.append(f"Official table '{name}' is missing.")
            continue
        absent = sorted(cols - set(str(c) for c in df.columns))
        if absent:
            reasons.append(f"'{name}' is missing required official columns: {absent}.")

    if reasons:
        return {"status": STATUS_INSUFFICIENT_DATA, "reasons": reasons, "details": details}

    inv = tables["inventory_snapshots"]
    master = tables["sku_master"]
    sales = tables["sales_daily"]

    if len(inv) == 0:
        reasons.append("No inventory observations present.")

    unknown_skus_inv = sorted(
        set(inv["sku_id"].astype(str)) - set(master["sku_id"].astype(str))
    )
    if unknown_skus_inv:
        reasons.append(
            f"{len(unknown_skus_inv)} inventory SKU(s) missing from sku_master."
        )

    weekly_counts = _weekly_demand_counts(sales)
    skus_with_history = int((weekly_counts >= config.min_history_weeks).sum())
    skus_low = int((weekly_counts < config.min_history_weeks).sum())
    details.update(
        {
            "total_skus_in_master": int(master["sku_id"].nunique()),
            "skus_with_sufficient_history": skus_with_history,
            "skus_low_history": skus_low,
            "forecast_status": tables.get("forecast_status"),
        }
    )
    if skus_with_history == 0 and master["sku_id"].nunique() > 0:
        reasons.append(
            f"No SKU has at least {config.min_history_weeks} weeks of demand "
            "history for confident risk scoring."
        )

    status = STATUS_READY if not reasons else STATUS_INSUFFICIENT_DATA
    return {
        "status": status,
        "reasons": reasons or ["All official inputs verified."],
        "details": details,
        "weekly_observations_per_sku": {str(k): int(v) for k, v in weekly_counts.items()},
    }

# --------------------------------------------------------------------------- #
# Input validation (flag-only; never silently repairs business values)
# --------------------------------------------------------------------------- #


def validate_inventory_inputs(
    inventory: pd.DataFrame,
    sku_master: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Flag invalid official values per D1 data-quality conventions.

    Checks: negative on-hand/on-order, negative/invalid unit cost, negative or
    unparseable lead time, negative reorder point, duplicate SKU/date records,
    SKUs missing from master. Issues are REPORTED — business values are never
    repaired.
    """
    issues: List[Dict[str, Any]] = []

    def add(issue: str, count: int, detail: str = "") -> None:
        issues.append({"issue": issue, "affected_rows": int(count), "detail": detail})

    for col in ("on_hand_units", "on_order_units"):
        if col in inventory.columns:
            vals = pd.to_numeric(inventory[col], errors="coerce")
            add(f"negative_{col}", int((vals < 0).sum()), f"{col} < 0")

    if "lead_time_days" in inventory.columns:
        lt = pd.to_numeric(inventory["lead_time_days"], errors="coerce")
        add("negative_lead_time_days", int((lt < 0).sum()))
        add("invalid_lead_time_days", int(lt.isna().sum()))

    if "reorder_point" in inventory.columns:
        rp = pd.to_numeric(inventory["reorder_point"], errors="coerce")
        add("negative_reorder_point", int((rp < 0).sum()))
        add("invalid_reorder_point", int(rp.isna().sum()))

    if {"sku_id", "date"} <= set(str(c) for c in inventory.columns):
        add(
            "duplicate_sku_date_records",
            int(inventory.duplicated(subset=["sku_id", "date"]).sum()),
        )

    if "unit_cost" in sku_master.columns:
        uc = pd.to_numeric(sku_master["unit_cost"], errors="coerce")
        add("negative_unit_cost", int((uc < 0).sum()))
        add("invalid_unit_cost", int(uc.isna().sum()))

    unknown = sorted(
        set(inventory["sku_id"].astype(str)) - set(sku_master["sku_id"].astype(str))
    )
    add("skus_missing_from_master", len(unknown), ", ".join(unknown)[:200])

    # Keep zero-count bookkeeping rows out of the actionable list except the
    # two structural checks where zero is itself meaningful context.
    return [
        i
        for i in issues
        if i["affected_rows"] > 0
        or i["issue"] in ("duplicate_sku_date_records", "skus_missing_from_master")
    ]


# --------------------------------------------------------------------------- #
# Weekly demand helper (consistent with the D3 weekly grain)
# --------------------------------------------------------------------------- #


def _weekly_demand_from_sales(sales_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate official daily sales into ``sku_id x ISO-week`` demand.

    Mirrors the D3 weekly grain (ISO year + week) so risk math and forecasting
    always share one definition of a week. Returns columns:
      sku_id, period, iso_year, iso_week, units_sold
    """
    s = sales_daily.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    iso = s["date"].dt.isocalendar()
    s["iso_year"] = iso["year"].astype("Int64")
    s["iso_week"] = iso["week"].astype("Int64")
    s["period"] = (
        s["iso_year"].astype(str) + "-W" + s["iso_week"].astype(str).str.zfill(2)
    )
    weekly = (
        s.groupby(["sku_id", "period", "iso_year", "iso_week"], as_index=False)
        .agg(units_sold=("units_sold", "sum"))
        .sort_values(["sku_id", "iso_year", "iso_week"])
        .reset_index(drop=True)
    )
    return weekly


def _weekly_demand_counts(sales_daily: pd.DataFrame) -> pd.Series:
    """Observed week counts per SKU from official daily sales."""
    weekly = _weekly_demand_from_sales(sales_daily)
    if weekly.empty:
        return pd.Series(dtype="int64")
    return weekly.groupby("sku_id")["period"].nunique().sort_index()

# --------------------------------------------------------------------------- #
# 3. INVENTORY POSITION (official formula, documented)
# --------------------------------------------------------------------------- #


def compute_inventory_position(
    inventory_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Latest per-SKU inventory position from official snapshot fields.

    Formula (explicit, using ONLY official fields)::

        inventory_position = on_hand_units + on_order_units

    ``on_hand_units`` is stock physically available; ``on_order_units`` is
    confirmed incoming supply. No additional inventory fields are invented.
    The latest snapshot per SKU (by ``date``) is used as the current position.

    Returns one row per SKU:
      sku_id, snapshot_date, on_hand_units, on_order_units,
      inventory_position, lead_time_days, source_reorder_point

    The company's official ``reorder_point`` is preserved verbatim under the
    name ``source_reorder_point`` — D4 never overwrites it. Any independent
    risk threshold this engine computes is kept separately (see
    ``calculated_risk_threshold`` in ``score_all_skus``).
    """
    required = {"sku_id", "date", "on_hand_units", "on_order_units",
                "lead_time_days", "reorder_point"}
    missing = required - set(str(c) for c in inventory_snapshots.columns)
    if missing:
        raise InvalidDataError(
            f"compute_inventory_position(): inventory_snapshots is missing "
            f"official columns {sorted(missing)}."
        )

    inv = inventory_snapshots.copy()
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    for col in ("on_hand_units", "on_order_units", "lead_time_days", "reorder_point"):
        inv[col] = pd.to_numeric(inv[col], errors="coerce")

    latest = (
        inv.sort_values(["sku_id", "date"])
        .groupby("sku_id", as_index=False)
        .tail(1)
        .loc[:, ["sku_id", "date", "on_hand_units", "on_order_units",
                 "lead_time_days", "reorder_point"]]
        .rename(columns={
            "date": "snapshot_date",
            "reorder_point": "source_reorder_point",
        })
        .reset_index(drop=True)
    )
    latest["inventory_position"] = (
        latest["on_hand_units"].fillna(0) + latest["on_order_units"].fillna(0)
    )
    return latest


# --------------------------------------------------------------------------- #
# 4. DEMAND RATE (weekly; historical vs official D3 forecast distinguished)
# --------------------------------------------------------------------------- #


def compute_demand_rate(
    sales_daily: pd.DataFrame,
    forecast: Optional[pd.DataFrame] = None,
    config: RiskConfig = DEFAULT_RISK_CONFIG,
) -> pd.DataFrame:
    """Per-SKU weekly demand rate from official data.

    * HISTORICAL rate: mean weekly units_sold over each SKU's observed weeks
      (weekly grain identical to D3). No daily Mini-FORESIGHT methodology.
    * FORECAST rate: mean of the official D3 weekly ``forecast_units`` over its
      horizon, when an official schema-valid D3 output is supplied.

    Both are returned separately and never mixed silently;
    ``preferred_demand_source`` records which rate risk scoring uses
    (official forecast when available, otherwise historical).

    Future actual demand is NEVER used: only past observations or the D3
    forecast enter this calculation.
    """
    weekly = _weekly_demand_from_sales(sales_daily)
    if weekly.empty:
        raise InsufficientDataError("No weekly demand could be derived from official sales.")

    hist = (
        weekly.groupby("sku_id")
        .agg(
            demand_rate_weekly_hist=("units_sold", "mean"),
            weeks_observed=("period", "nunique"),
            demand_std_weekly=("units_sold", "std"),
        )
        .reset_index()
    )

    fc_rate: Dict[str, float] = {}
    if forecast is not None and len(forecast):
        f = forecast.copy()
        f["forecast_units"] = pd.to_numeric(f["forecast_units"], errors="coerce")
        fc_rate = (
            f.groupby("sku_id")["forecast_units"].mean().dropna().to_dict()
        )

    hist["forecast_rate_weekly"] = hist["sku_id"].map(fc_rate)

    def _prefer(row: pd.Series) -> str:
        return SOURCE_D3_FORECAST if pd.notna(row["forecast_rate_weekly"]) else SOURCE_HISTORICAL

    hist["preferred_demand_source"] = hist.apply(_prefer, axis=1)

    def _flag(row: pd.Series) -> str:
        if int(row["weeks_observed"]) < config.low_history_weeks:
            return FLAG_INSUFFICIENT_DATA
        if int(row["weeks_observed"]) < config.min_history_weeks:
            return FLAG_LOW_HISTORY
        return FLAG_OK

    hist["data_quality_flag"] = hist.apply(_flag, axis=1)
    return hist


# --------------------------------------------------------------------------- #
# 5. LEAD-TIME DEMAND (documented weekly->daily conversion)
# --------------------------------------------------------------------------- #


def compute_lead_time_demand(
    lead_time_days: float,
    weekly_demand_rate: float,
) -> Optional[float]:
    """Demand expected to occur during one supplier lead time.

    Conversion (documented): the demand rate is WEEKLY (consistent with D3);
    lead time is OFFICIAL days, so::

        lead_time_weeks = lead_time_days / 7.0
        lead_time_demand = weekly_demand_rate * lead_time_weeks

    Returns ``None`` when either input is not a valid finite number (no value
    invented). The lead time always comes from the official
    ``inventory_snapshots.lead_time_days`` field — never hard-coded.
    """
    try:
        ltd_days = float(lead_time_days)
        rate = float(weekly_demand_rate)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(ltd_days) and np.isfinite(rate)) or ltd_days < 0:
        return None
    return float(rate * (ltd_days / 7.0))

# --------------------------------------------------------------------------- #
# 6. STOCKOUT RISK
# --------------------------------------------------------------------------- #


def compute_stockout_risk(
    inventory_position: float,
    lead_time_demand: Optional[float],
    weekly_demand_rate: float,
    config: RiskConfig = DEFAULT_RISK_CONFIG,
) -> Dict[str, Any]:
    """Stockout-risk score on a 0..1 scale (higher = more risk).

    Indicators returned:
      coverage_weeks   — inventory_position / weekly_demand_rate (None when the
                         rate is zero/invalid; zero demand = no exposure)
      shortfall_units  — max(0, lead_time_demand - inventory_position)
      stockout_risk    — linear in the position/lead-time-demand ratio:

                             ratio = inventory_position / lead_time_demand
                             risk  = clamp01((threshold - ratio) / threshold)

                         threshold = config.stockout_ratio_threshold (ASSUMPTION).

    Zero-demand handling: if the weekly rate <= 0 or lead-time demand is
    None/0 there is nothing to run out of during the lead time -> risk 0.0.
    """
    out: Dict[str, Any] = {
        "coverage_weeks": None,
        "shortfall_units": None,
        "stockout_risk": None,
        "basis": "lead_time_demand",
    }

    try:
        ip = float(inventory_position)
    except (TypeError, ValueError):
        return {**out, "note": "invalid inventory_position"}

    rate = float(weekly_demand_rate) if weekly_demand_rate is not None else np.nan

    if np.isfinite(rate) and rate > 0:
        out["coverage_weeks"] = float(ip / rate)

    if lead_time_demand is None:
        ltd = np.nan
    else:
        try:
            ltd = float(lead_time_demand)
        except (TypeError, ValueError):
            ltd = np.nan

    if not np.isfinite(ltd) or ltd <= 0 or not np.isfinite(rate) or rate <= 0:
        out["stockout_risk"] = 0.0
        out["shortfall_units"] = 0.0
        out["note"] = "no lead-time demand exposure (zero/invalid demand)"
        return out

    shortfall = max(0.0, ltd - ip)
    threshold = float(config.stockout_ratio_threshold)
    ratio = ip / ltd
    risk = (threshold - ratio) / threshold if threshold > 0 else 1.0
    out["shortfall_units"] = float(shortfall)
    out["stockout_risk"] = float(min(1.0, max(0.0, risk)))
    return out


# --------------------------------------------------------------------------- #
# 7. OVERSTOCK RISK (independent axis; coverage-based, NOT reorder-point based)
# --------------------------------------------------------------------------- #


def compute_overstock_risk(
    inventory_position: float,
    weekly_demand_rate: float,
    config: RiskConfig = DEFAULT_RISK_CONFIG,
) -> Dict[str, Any]:
    """Overstock-risk score on a 0..1 scale (higher = more overstock).

    Coverage-based::

        coverage_weeks = inventory_position / weekly_demand_rate
        excess_weeks   = coverage_weeks - config.overstock_coverage_weeks
        overstock_risk = clamp01(excess_weeks / overstock_coverage_weeks)

    Cover at the ASSUMED threshold scores 0; double the threshold scores 1.
    ``excess_units_over_healthy`` is returned for rupee valuation. With a
    zero/invalid demand rate the test is undefined -> None + explicit note.
    """
    try:
        ip = float(inventory_position)
    except (TypeError, ValueError):
        return {"overstock_risk": None, "excess_units_over_healthy": None,
                "coverage_weeks": None, "note": "invalid inventory_position"}

    rate = float(weekly_demand_rate) if weekly_demand_rate is not None else np.nan
    if not np.isfinite(rate) or rate <= 0:
        return {
            "overstock_risk": None,
            "excess_units_over_healthy": None,
            "coverage_weeks": None,
            "note": "undefined: zero/invalid demand rate",
        }

    coverage_weeks = ip / rate
    thr = float(config.overstock_coverage_weeks)
    excess_weeks = max(0.0, coverage_weeks - thr)
    risk = min(1.0, max(0.0, excess_weeks / thr)) if thr > 0 else 0.0
    healthy_units = rate * float(config.coverage_target_weeks)

    return {
        "overstock_risk": float(risk),
        "excess_units_over_healthy": float(max(0.0, ip - healthy_units)),
        "coverage_weeks": float(coverage_weeks),
    }

# --------------------------------------------------------------------------- #
# 10. VOLATILITY (transparent CV; insufficient data flagged honestly)
# --------------------------------------------------------------------------- #


def compute_volatility(
    sku_weekly_history: pd.Series,
    config: RiskConfig = DEFAULT_RISK_CONFIG,
) -> Dict[str, Any]:
    """Coefficient of variation of weekly demand (sample std / mean).

    Flags: ok (>= min_history_weeks), low_history (< min but >=
    low_history_weeks), insufficient_data (< low_history_weeks — volatility
    NOT computed). Never fabricates a volatility number.
    """
    vals = pd.to_numeric(pd.Series(sku_weekly_history), errors="coerce").dropna()
    n = int(len(vals))
    if n < config.low_history_weeks:
        return {
            "volatility_cv": None,
            "n_weeks_used": n,
            "flag": FLAG_INSUFFICIENT_DATA,
            "note": (
                f"only {n} observed weeks (<{config.low_history_weeks}); "
                "volatility intentionally not computed"
            ),
        }
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n >= 2 else 0.0
    cv = (std / mean) if mean > 0 else None
    flag = FLAG_OK if n >= config.min_history_weeks else FLAG_LOW_HISTORY
    note = "" if cv is not None else "mean weekly demand is zero; CV undefined"
    return {"volatility_cv": cv, "n_weeks_used": n, "flag": flag, "note": note}


# --------------------------------------------------------------------------- #
# 8. RUPEE VALUE AT STAKE (official monetary fields only)
# --------------------------------------------------------------------------- #


def compute_rupee_at_stake(
    unit_cost: Optional[float],
    shortfall_units: Optional[float],
    excess_units: Optional[float],
) -> Dict[str, Any]:
    """Rupee exposure valued at the OFFICIAL ``unit_cost`` from sku_master.

    Two clearly-labelled components:

      inventory_value_exposure_rupees
          capital tied up in units held beyond the healthy-cover target
          (= excess_units_over_healthy x unit_cost); cost basis because it is
          cash already spent on inventory.
      stockout_shortfall_rupees
          lead-time shortfall units x unit_cost — a conservative COST floor.

    A lost-REVENUE valuation would require a service-level / margin assumption
    Zidio does not supply, so it is deliberately NOT applied. Invalid inputs
    yield ``None`` components — no fabricated rupee values.
    """
    result: Dict[str, Any] = {
        "unit_cost_basis": None,
        "inventory_value_exposure_rupees": None,
        "stockout_shortfall_rupees": None,
        "rupee_value_at_stake": None,
        "valuation_note": (
            "cost-basis valuation using official unit_cost; lost-revenue "
            "valuation not applied (requires service-level assumption not "
            "provided by Zidio)"
        ),
    }
    if unit_cost is None:
        return result
    try:
        cost = float(unit_cost)
    except (TypeError, ValueError):
        return result
    if not np.isfinite(cost) or cost < 0:
        return result

    def _f(x: Optional[float]) -> Optional[float]:
        try:
            v = float(x)
            return v if np.isfinite(v) and v >= 0 else None
        except (TypeError, ValueError):
            return None

    excess = _f(excess_units)
    short = _f(shortfall_units)

    inv_exp = excess * cost if excess is not None else None
    short_exp = short * cost if short is not None else None
    total_vals = [v for v in (inv_exp, short_exp) if v is not None]

    result["unit_cost_basis"] = float(cost)
    result["inventory_value_exposure_rupees"] = inv_exp
    result["stockout_shortfall_rupees"] = short_exp
    result["rupee_value_at_stake"] = sum(total_vals) if total_vals else None
    return result

# --------------------------------------------------------------------------- #
# 9. FOUR-CELL DECISION GRID (both axes + volatility)
# --------------------------------------------------------------------------- #


def assign_decision(
    stockout_risk: Optional[float],
    overstock_risk: Optional[float],
    volatility_cv: Optional[float],
    config: RiskConfig = DEFAULT_RISK_CONFIG,
    data_quality_flag: str = FLAG_OK,
) -> Dict[str, Any]:
    """Map the two risk axes (+ volatility, + confidence) to ONE business cell.

    Decision matrix (documented; numeric cut-offs are CONFIGURATION ASSUMPTIONS):

      1. data_quality_flag == 'insufficient_data'
             -> WATCH_VOLATILE   (a precise cell cannot be justified without
                                  enough official observations)
      2. stockout_risk >= reorder_now_stockout_min  AND
         overstock below its markdown threshold
             -> REORDER_NOW      (high stockout pressure, no overstock conflict)
      3. overstock_risk >= markdown_clear_overstock_min AND
         stockout below the reorder threshold
             -> MARKDOWN_CLEAR   (excess cover, little stockout pressure)
      4. BOTH axes elevated simultaneously, OR volatility >=
         volatility_cv_high with at least one axis non-trivially elevated
             -> WATCH_VOLATILE   (conflicting/uncertain signals)
      5. otherwise
             -> HEALTHY

    Stockout is evaluated before markdown: protecting availability outranks
    clearing excess when both pressures co-exist at similar strength.
    Returns {decision, reason} — never High/Medium/Low.
    """
    so = float(stockout_risk) if stockout_risk is not None else None
    ov = float(overstock_risk) if overstock_risk is not None else None
    cv = float(volatility_cv) if volatility_cv is not None else None

    reorder_min = float(config.reorder_now_stockout_min)
    markdown_min = float(config.markdown_clear_overstock_min)

    def elevated(v: Optional[float]) -> bool:
        return v is not None and v > 0.0

    # 1) insufficient official evidence -> refuse a precise business cell.
    if data_quality_flag == FLAG_INSUFFICIENT_DATA:
        return {
            "decision": DECISION_WATCH_VOLATILE,
            "reason": (
                "insufficient official history to justify a precise decision; "
                "placed in Watch/Volatile pending more observations"
            ),
        }

    high_so = so is not None and so >= reorder_min
    high_ov = ov is not None and ov >= markdown_min

    # 2) stockout priority (no conflicting overstock signal).
    if high_so and not high_ov:
        return {
            "decision": DECISION_REORDER_NOW,
            "reason": (
                f"stockout risk {so:.2f} >= {reorder_min:.2f} while overstock "
                f"is below {markdown_min:.2f}: protect availability"
            ),
        }

    # 3) clear excess when stockout pressure is low.
    if high_ov and not high_so:
        return {
            "decision": DECISION_MARKDOWN_CLEAR,
            "reason": (
                f"overstock risk {ov:.2f} >= {markdown_min:.2f} with stockout "
                f"below {reorder_min:.2f}: reduce excess inventory"
            ),
        }

    # 4) conflicting or volatile signals.
    both_elevated = high_so and high_ov
    volatile = cv is not None and cv >= float(config.volatility_cv_high) and (
        elevated(so) or elevated(ov)
    )
    if both_elevated or volatile:
        why = []
        if both_elevated:
            why.append("both risk axes elevated simultaneously")
        if volatile:
            why.append(f"demand volatility CV {cv:.2f} >= "
                       f"{float(config.volatility_cv_high):.2f}")
        return {
            "decision": DECISION_WATCH_VOLATILE,
            "reason": "; ".join(why),
        }

    # 5) calm on both axes.
    return {
        "decision": DECISION_HEALTHY,
        "reason": (
            f"stockout {so if so is not None else 'n/a'} and overstock "
            f"{ov if ov is not None else 'n/a'} both below their thresholds "
            "with no volatility trigger"
        ),
    }

# --------------------------------------------------------------------------- #
# 11. SKU-LEVEL SCORING (one structured record per official SKU)
# --------------------------------------------------------------------------- #


def score_all_skus(
    tables: Optional[Dict[str, Any]] = None,
    config: RiskConfig = DEFAULT_RISK_CONFIG,
    processed_dir: Path = DATA_PROCESSED,
) -> Dict[str, Any]:
    """Score every official SKU when data permits; otherwise return a gated status.

    Per-SKU fields (all derived from official inputs; none fabricated):
      sku_id, category, subcategory, snapshot_date, on_hand_units,
      on_order_units, inventory_position, lead_time_days,
      source_reorder_point (official, untouched), calculated_risk_threshold
      (= lead-time demand; kept SEPARATE from the company's reorder point),
      demand_rate_weekly, forecast_rate_weekly, preferred_demand_source,
      coverage_weeks, shortfall_units, stockout_risk, overstock_risk,
      volatility_cv, unit_cost, excess_units_over_healthy,
      inventory_value_exposure_rupees, stockout_shortfall_rupees,
      rupee_value_at_stake, decision, decision_reason, data_quality_flag.

    Data-gated: returns DATA_NOT_AVAILABLE / INSUFFICIENT_DATA states instead
    of computing anything from legacy or partial inputs.
    """
    config.validate()

    readiness = check_risk_data_readiness(tables, config, processed_dir)
    if readiness["status"] != STATUS_READY:
        return {
            "status": readiness["status"],
            "reasons": readiness.get("reasons", []),
            "missing_d1_outputs": readiness.get("missing_d1_outputs"),
            "scored_skus": [],
            "issues": [],
            "forecast": {
                "status": (readiness.get("details") or {}).get("forecast_status"),
                "note": (
                    None
                    if readiness["status"] == STATUS_READY
                    else "D3 forecast not integrated (official data unavailable)."
                ),
            },
        }

    if tables is None:
        tables = load_official_inputs(processed_dir)

    sales = tables["sales_daily"]
    master = tables["sku_master"]
    inv = tables["inventory_snapshots"]
    forecast = tables.get("forecast")
    forecast_status = tables.get("forecast_status")

    issues = validate_inventory_inputs(inv, master)
    positions = compute_inventory_position(inv)
    rates = compute_demand_rate(sales, forecast, config)
    weekly = _weekly_demand_from_sales(sales)

    records: List[Dict[str, Any]] = []
    for _, pos in positions.iterrows():
        sku_id = str(pos["sku_id"])
        rate_row = rates[rates["sku_id"].astype(str) == sku_id]
        if rate_row.empty:
            records.append({
                "sku_id": sku_id,
                "decision": DECISION_WATCH_VOLATILE,
                "decision_reason": "no official demand history for this SKU",
                "data_quality_flag": FLAG_INSUFFICIENT_DATA,
            })
            continue
        rate_row = rate_row.iloc[0]

        flag = str(rate_row["data_quality_flag"])
        use_forecast = (
            pd.notna(rate_row["forecast_rate_weekly"])
            and flag != FLAG_INSUFFICIENT_DATA
        )
        rate = (
            float(rate_row["forecast_rate_weekly"])
            if use_forecast
            else float(rate_row["demand_rate_weekly_hist"])
        )

        ltd = compute_lead_time_demand(pos["lead_time_days"], rate)
        so = compute_stockout_risk(pos["inventory_position"], ltd, rate, config)
        ov = compute_overstock_risk(pos["inventory_position"], rate, config)

        hist_vals = weekly.loc[weekly["sku_id"].astype(str) == sku_id, "units_sold"]
        vol = compute_volatility(hist_vals, config)

        mrow = master[master["sku_id"].astype(str) == sku_id]
        has_master = not mrow.empty
        unit_cost = None
        if has_master and pd.notna(mrow.iloc[0].get("unit_cost")):
            try:
                candidate_cost = float(mrow.iloc[0]["unit_cost"])
                if np.isfinite(candidate_cost) and candidate_cost >= 0:
                    unit_cost = candidate_cost
            except (TypeError, ValueError):
                unit_cost = None

        rupee = compute_rupee_at_stake(
            unit_cost, so.get("shortfall_units"),
            ov.get("excess_units_over_healthy"),
        )

        # Confidence: insufficient-history overrides everything; low-history on
        # either axis is surfaced but does not by itself block scoring.
        if flag == FLAG_INSUFFICIENT_DATA or vol["flag"] == FLAG_INSUFFICIENT_DATA:
            eff_flag = FLAG_INSUFFICIENT_DATA
        elif FLAG_LOW_HISTORY in (flag, vol["flag"]):
            eff_flag = FLAG_LOW_HISTORY
        else:
            eff_flag = FLAG_OK

        decision = assign_decision(
            so.get("stockout_risk"), ov.get("overstock_risk"),
            vol.get("volatility_cv"), config, eff_flag,
        )

        records.append({
            "sku_id": sku_id,
            "category": mrow.iloc[0].get("category") if has_master else None,
            "subcategory": mrow.iloc[0].get("subcategory") if has_master else None,
            "snapshot_date": str(pos["snapshot_date"]),
            "on_hand_units": pos["on_hand_units"],
            "on_order_units": pos["on_order_units"],
            "inventory_position": pos["inventory_position"],
            "lead_time_days": pos["lead_time_days"],
            "source_reorder_point": pos["source_reorder_point"],
            "calculated_risk_threshold": ltd,
            "demand_rate_weekly": float(rate_row["demand_rate_weekly_hist"]),
            "forecast_rate_weekly": (
                float(rate_row["forecast_rate_weekly"])
                if pd.notna(rate_row["forecast_rate_weekly"]) else None
            ),
            "preferred_demand_source": str(rate_row["preferred_demand_source"]),
            "coverage_weeks": so.get("coverage_weeks", ov.get("coverage_weeks")),
            "shortfall_units": so.get("shortfall_units"),
            "stockout_risk": so.get("stockout_risk"),
            "overstock_risk": ov.get("overstock_risk"),
            "volatility_cv": vol.get("volatility_cv"),
            "unit_cost": unit_cost,
            "excess_units_over_healthy": ov.get("excess_units_over_healthy"),
            "inventory_value_exposure_rupees": rupee["inventory_value_exposure_rupees"],
            "stockout_shortfall_rupees": rupee["stockout_shortfall_rupees"],
            "rupee_value_at_stake": rupee["rupee_value_at_stake"],
            "decision": decision["decision"],
            "decision_reason": decision["reason"],
            "data_quality_flag": eff_flag,
            "weeks_observed": int(rate_row["weeks_observed"]),
        })

    return {
        "status": STATUS_READY,
        "scored_skus": records,
        "issues": issues,
        "config_assumptions": readiness["details"]["config_assumptions"],
        "forecast": {
            "status": forecast_status,
            "note": tables.get("forecast_note"),
        },
    }

# --------------------------------------------------------------------------- #
# 18. RISK REPORT (honest empty sections when official data is unavailable)
# --------------------------------------------------------------------------- #


def _config_assumption_block(config: RiskConfig) -> Dict[str, Any]:
    keys = (
        "coverage_target_weeks", "overstock_coverage_weeks",
        "stockout_ratio_threshold", "volatility_cv_high",
        "min_history_weeks", "low_history_weeks",
        "reorder_now_stockout_min", "markdown_clear_overstock_min",
    )
    return {
        "note": (
            "numeric thresholds below are implementation configuration "
            "assumptions, NOT rules supplied by Zidio"
        ),
        **{k: getattr(config, k) for k in keys},
    }


def create_risk_report(
    scored: Optional[Dict[str, Any]] = None,
    config: RiskConfig = DEFAULT_RISK_CONFIG,
) -> Dict[str, Any]:
    """Structured D4 report.

    When ``scored`` is missing / not READY, every numerical section stays
    empty/None and NO_OBSERVATIONS_MESSAGE is included. Numeric sections are
    populated ONLY from actual computed scoring results.
    """
    base: Dict[str, Any] = {
        "project": "Project FORESIGHT — D4 Inventory Risk & Decision Engine",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "data_status": STATUS_DATA_NOT_AVAILABLE,
        "official_requirements": [
            "two-axis risk (stockout AND overstock)",
            "rupee value at stake",
            "four-cell decision grid: REORDER_NOW / MARKDOWN_CLEAR / "
            "WATCH_VOLATILE / HEALTHY",
            "official fields only; new-SKU/low-history confidence flags",
        ],
        "configuration_assumptions": _config_assumption_block(config),
        "inventory_summary": None,
        "stockout_risk_summary": None,
        "overstock_risk_summary": None,
        "rupee_value_summary": None,
        "decision_summary": None,
        "low_history_summary": None,
        "limitations": [NO_OBSERVATIONS_MESSAGE],
        "missing_d1_outputs": None,
    }

    if not scored or scored.get("status") != STATUS_READY:
        if scored:
            base["data_status"] = scored.get("status", base["data_status"])
            base["limitations"].extend(scored.get("reasons", [])[:5])
            base["missing_d1_outputs"] = scored.get("missing_d1_outputs")
        return base

    rows = scored.get("scored_skus", [])
    decisions = [r.get("decision") for r in rows]
    decision_counts = {d: int(decisions.count(d)) for d in sorted(set(decisions))}

    def _sum(key: str) -> Optional[float]:
        vals = [
            float(r[key]) for r in rows
            if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])
        ]
        return float(sum(vals)) if vals else None

    def _mean(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    low_hist = [r["sku_id"] for r in rows if r.get("data_quality_flag") != FLAG_OK]

    base.update(
        {
            "data_status": STATUS_READY,
            "inventory_summary": {
                "skus_scored": len(rows),
                "total_inventory_position_units": _sum("inventory_position"),
                "total_on_hand_units": _sum("on_hand_units"),
                "total_on_order_units": _sum("on_order_units"),
            },
            "stockout_risk_summary": {
                "mean_stockout_risk": _mean("stockout_risk"),
                "skus_with_shortfall": int(sum(
                    1 for r in rows if (r.get("shortfall_units") or 0) > 0
                )),
            },
            "overstock_risk_summary": {
                "mean_overstock_risk": _mean("overstock_risk"),
                "skus_with_excess": int(sum(
                    1 for r in rows if (r.get("excess_units_over_healthy") or 0) > 0
                )),
            },
            "rupee_value_summary": {
                "valuation_basis": "official unit_cost (cost basis)",
                "total_rupee_value_at_stake": _sum("rupee_value_at_stake"),
                "total_inventory_value_exposure_rupees":
                    _sum("inventory_value_exposure_rupees"),
                "total_stockout_shortfall_rupees":
                    _sum("stockout_shortfall_rupees"),
            },
            "decision_summary": {
                "grid": [
                    DECISION_REORDER_NOW, DECISION_MARKDOWN_CLEAR,
                    DECISION_WATCH_VOLATILE, DECISION_HEALTHY,
                ],
                "counts": decision_counts,
            },
            "low_history_summary": {
                "count": len(low_hist),
                "sku_ids": low_hist,
                "handling": (
                    "insufficient-data SKUs are placed in WATCH_VOLATILE with "
                    "no precise scores; low-history SKUs are flagged and "
                    "reported without fabricated precision"
                ),
            },
            "forecast_integration": scored.get("forecast"),
            "input_quality_issues": scored.get("issues"),
            "limitations": [
                "Risk scores reflect configured thresholds documented in "
                "'configuration_assumptions'; they are operational defaults, "
                "not Zidio-supplied business rules.",
                "Rupee exposure uses cost-basis valuation; lost-revenue "
                "valuation would need a service-level assumption Zidio has "
                "not provided.",
                "Coverage/correlation observations are not causal claims.",
            ],
        }
    )
    return base
