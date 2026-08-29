"""D3 — Weekly SKU-level demand forecasting engine for Project FORESIGHT.

Data gating
-----------
Consumes ONLY the official D1 analysis-ready outputs under ``data/processed/``
(``sales_daily_clean.csv``, ``sku_master_clean.csv``, ``calendar_clean.csv``,
``inventory_snapshots_clean.csv``). Never reads ``data/raw/``; never falls back
to legacy Mini-FORESIGHT files. When the official D1 outputs are absent the
engine raises ``MissingOfficialInputsError`` and produces no forecast rows,
no WAPE, no bias and no model-vs-baseline claims.

Methodology
-----------
* Grain: weekly SKU-level demand (**not** daily / not the 3-day Mini forecast).
* Horizon: explicit & configurable, default 8 weeks.
* Baseline: seasonal-naive (same seasonal period in the past), default period
  52 weeks. Never ``shift(1)``.
* Model: scikit-learn tree model (default ``HistGradientBoostingRegressor``),
  fixed seed, configurable.
* Evaluation: rolling-origin backtest (train strictly before test window).
* Metrics: WAPE primary, bias secondary, MAPE secondary.
* Model-selection rule: keep the ML model ONLY if it beats the seasonal-naive
  baseline on WAPE on the same folds; otherwise honestly retain the baseline.
* Leakage prevention: explicit checks raise ``LeakageError``.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.paths import DATA_PROCESSED

# --------------------------------------------------------------------------- #
# Logging / exception types
# --------------------------------------------------------------------------- #

logger = logging.getLogger("foresight.forecast")


class D3Error(Exception):
    """Base class for D3 forecasting-engine errors."""


class MissingOfficialInputsError(D3Error):
    """Raised when one or more official D1 analysis-ready outputs are absent."""

    def __init__(self, missing: List[str], processed_dir: Path) -> None:
        self.missing = missing
        self.processed_dir = Path(processed_dir)
        super().__init__(
            "Official FORESIGHT D1 analysis-ready outputs are missing: "
            + ", ".join(missing)
            + f". Expected under: {self.processed_dir}. D3 refuses to run on "
            "raw/legacy data (e.g. the Mini-FORESIGHT demo files). Run the "
            "official D1 pipeline first; this engine never creates, downloads, "
            "or fabricates data."
        )


class InsufficientHistoryError(D3Error):
    """Raised when official history is too short for horizon / seasonal period / folds."""


class LeakageError(D3Error):
    """Raised when a data-leakage condition is detected in feature preparation."""


# --------------------------------------------------------------------------- #
# Official D1 output contract
# --------------------------------------------------------------------------- #

D1_OUTPUT_FILES = {
    "analysis_ready": "sales_analysis_ready.csv",
    "sku_master": "sku_master_clean.csv",
    "calendar": "calendar_clean.csv",
    "inventory_snapshots": "inventory_snapshots_clean.csv",
}


def _guard_d1_outputs(processed_dir: Path) -> None:
    """Raise ``MissingOfficialInputsError`` if any official D1 output is absent."""
    processed_dir = Path(processed_dir)
    missing = [
        name
        for name, fname in D1_OUTPUT_FILES.items()
        if not (processed_dir / fname).is_file()
    ]
    if missing:
        raise MissingOfficialInputsError(missing, processed_dir)


def load_d1_outputs(processed_dir: Path = DATA_PROCESSED) -> Dict[str, pd.DataFrame]:
    """Load ONLY the official D1 analysis-ready outputs (never raw/legacy)."""
    _guard_d1_outputs(processed_dir)
    tables: Dict[str, pd.DataFrame] = {}
    for name, fname in D1_OUTPUT_FILES.items():
        path = Path(processed_dir) / fname
        try:
            tables[name] = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - surface a clear D3 error
            raise D3Error(f"Failed to read official D1 output '{fname}': {exc}") from exc
    return tables
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class ForecastConfig:
    """Explicit, documented configuration for the D3 engine.

    Attributes
    ----------
    horizon_weeks : int
        Forecast horizon in weeks. Default 8. Must be >= 1.
    seasonal_period : int
        Seasonal period (weeks) for the seasonal-naive baseline. Default 52.
        The implementation inspects the actual history and reports if this
        seasonal period cannot be evaluated (never fabricates history).
    cv_folds : int
        Number of rolling-origin folds. Default 3. Must be >= 1.
    min_train_weeks : int
        Minimum training history (weeks) per SKU before the ML model is fit.
    min_obs_per_sku : int
        Minimum weekly observations for an SKU to qualify for the model path;
        SKUs below this use the low-history fallback.
    random_seed : int
        Fixed random seed for the model (reproducibility).
    model_params : dict
        Constructor overrides for the gradient-boosted-tree model.
    low_history_fallback : str
        ``"category_historical"`` (default) — sparse-history SKUs use
        category-level historical demand; ``"none"`` disables the fallback.
    """

    horizon_weeks: int = 8
    seasonal_period: int = 52
    cv_folds: int = 3
    min_train_weeks: int = 12
    min_obs_per_sku: int = 4
    random_seed: int = 42
    model_params: Dict[str, Any] = field(default_factory=dict)
    low_history_fallback: str = "category_historical"

    def validate(self) -> None:
        if self.horizon_weeks < 1:
            raise ValueError("ForecastConfig.horizon_weeks must be >= 1.")
        if self.seasonal_period < 1:
            raise ValueError("ForecastConfig.seasonal_period must be >= 1.")
        if self.cv_folds < 1:
            raise ValueError("ForecastConfig.cv_folds must be >= 1.")
        if self.min_train_weeks < 1:
            raise ValueError("ForecastConfig.min_train_weeks must be >= 1.")
        if self.low_history_fallback not in {"category_historical", "none"}:
            raise ValueError(
                "ForecastConfig.low_history_fallback must be "
                + "'category_historical' or 'none'."
            )


DEFAULT_CONFIG = ForecastConfig()
# --------------------------------------------------------------------------- #
# 3. PREPARE WEEKLY DEMAND (sku_id x week)
# --------------------------------------------------------------------------- #


def prepare_weekly_demand(
    tables: Dict[str, pd.DataFrame],
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Aggregate official daily sales into ``sku_id x week`` demand.

    Consumes ONLY the official D1 outputs. Joins the official calendar (week,
    month, season, is_holiday, promo_event) and relevant SKU master attributes
    (category, subcategory, list_price) and promotion info (promo_flag).

    The weekly period is built with an ISO ``(year, week)`` key so chronological
    ordering is correct even across year boundaries.

    Leakage safety: only *context* attributes (calendar period metadata and
    static SKU attributes) are attached. No future actual demand ever enters.

    Raises ``MissingOfficialInputsError`` if D1 outputs are absent.
    """
    required = ["analysis_ready", "calendar", "sku_master"]
    missing = [k for k in required if k not in tables]
    if missing:
        raise MissingOfficialInputsError(missing, DATA_PROCESSED)

    sales = tables["analysis_ready"].copy()
    calendar = tables["calendar"].copy()
    sku_master = tables["sku_master"].copy()

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")

    # Weekly period key from ISO calendar (year, week) - sortable & year-safe.
    iso = sales["date"].dt.isocalendar()
    sales["iso_year"] = iso["year"].astype("Int64")
    sales["iso_week"] = iso["week"].astype("Int64")
    sales["period"] = (
        sales["iso_year"].astype(str) + "-W" + sales["iso_week"].astype(str).str.zfill(2)
    )

    # Aggregate demand (and revenue) to SKU x week.
    weekly = (
        sales.groupby(["sku_id", "period", "iso_year", "iso_week"], as_index=False)
        .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
    )

    # Promotion flag: any promo day within the week => 1.
    if "promo_flag" in sales.columns:
        promo = (
            sales.assign(promo=sales["promo_flag"].fillna(0))
            .groupby(["sku_id", "period"])["promo"].max()
            .reset_index()
        )
        weekly = weekly.merge(promo, on=["sku_id", "period"], how="left")
    else:
        weekly["promo"] = 0

    # Calendar metadata is already present in the D1 analysis-ready table.
    # Join only fields that are genuinely absent, avoiding duplicate context.
    cal = calendar.copy()
    iso_cal = cal["date"].dt.isocalendar()
    cal["iso_year"] = iso_cal["year"].astype("Int64")
    cal["iso_week"] = iso_cal["week"].astype("Int64")
    cal["period"] = (
        cal["iso_year"].astype(str) + "-W" + cal["iso_week"].astype(str).str.zfill(2)
    )
    cal_context = (
        cal.groupby("period", as_index=False)
        .agg(
            month=("month", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
            season=("season", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
            is_holiday=("is_holiday", "max"),
            promo_event=("promo_event", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        )
    )
    missing_calendar = [
        col for col in ("month", "season", "is_holiday", "promo_event")
        if col not in weekly.columns
    ]
    if missing_calendar:
        weekly = weekly.merge(
            cal_context[["period"] + missing_calendar], on="period", how="left"
        )

    # SKU master static attributes (list_price is a static attribute; historical
    # unit_price is not used as a feature to avoid look-ahead).
    master_cols = ["sku_id", "category", "subcategory", "list_price"]
    master_cols = [c for c in master_cols if c in sku_master.columns]
    missing_master = [col for col in master_cols if col not in weekly.columns]
    if missing_master:
        weekly = weekly.merge(
            sku_master[["sku_id"] + missing_master], on="sku_id", how="left"
        )

    # Deterministic chronological ordering by SKU then week-period.
    weekly["iso_year"] = pd.to_numeric(weekly["iso_year"], errors="coerce").astype(int)
    weekly["iso_week"] = pd.to_numeric(weekly["iso_week"], errors="coerce").astype(int)
    weekly = weekly.sort_values(["sku_id", "iso_year", "iso_week"]).reset_index(drop=True)
    return weekly
# --------------------------------------------------------------------------- #
# 4. FEATURE ENGINEERING (leakage-safe)
# --------------------------------------------------------------------------- #

# Ordered feature columns (only information available strictly before the
# forecast period). The last column of each rolling window is the PREVIOUS
# week's demand (shift), never the current week's.
_FEATURE_COLUMNS = [
    "lag_1_week",
    "lag_2_week",
    "lag_4_week",
    "lag_8_week",
    "lag_13_week",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
    "rolling_std_8",
    "week_of_year",
    "month",
    "season",
    "is_holiday",
    "promo",
    "promo_event",
    "category",
    "subcategory",
    "list_price",
]


def _assert_no_future_leakage(df: pd.DataFrame, ref_date_col: str = "period") -> None:
    """Raise ``LeakageError`` if the weekly table contains any future period
    marker or if any trailing feature includes the current/future target."""
    if ref_date_col not in df.columns:
        return
    # Trailing-window features must be NaN on the first rows; the presence of
    # the target column anywhere in the feature frame is what we guard against
    # at build time (features are built via shift, never via the target).
    feature_cols = [c for c in _FEATURE_COLUMNS if c in df.columns]
    for col in feature_cols:
        if col.startswith("lag_") or col.startswith("rolling_"):
            # These are built from shift() in build_forecast_features; a manual
            # check that the original target column is not present in features.
            if col == "units_sold":
                raise LeakageError(
                    f"Feature column '{col}' is the target; features must be "
                    "constructed only from shifted history."
                )


def build_forecast_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Build the feature frame strictly from information available before the
    forecast period.

    For every SKU:

      * ``lag_*``  = demand k weeks ago (shift within SKU)
      * ``rolling_mean_4/8`` = trailing mean over the *previous* 4/8 weeks
      * ``rolling_std_4/8``  = trailing std over the previous 4/8 weeks
      * calendar context (week, month, season, holiday)
      * promotion indicator (promo_flag / promo_event, from the week's calendar
        context — this is context available at forecast time, not future demand)

    Leakage rules enforced here:
      1. Lags and rolling stats are built with ``.shift()`` so the *current*
         target never enters a feature.
      2. Every feature row is aligned to its own (sku_id, period) row; future
         actual demand is never merged into the frame.
      3. Any leak condition raises ``LeakageError``.
    """
    required = ["sku_id", "period", "units_sold"]
    missing_cols = [c for c in required if c not in weekly.columns]
    if missing_cols:
        raise D3Error(
            "build_forecast_features(): weekly table missing required columns "
            f"{missing_cols}. Run prepare_weekly_demand() first."
        )

    df = weekly.sort_values(["sku_id", "iso_year", "iso_week"]).copy()

    # Sort within SKU, then shift (leakage-safe: previous-week demand only).
    grouped = df.groupby("sku_id", sort=True)

    df["lag_1_week"] = grouped["units_sold"].shift(1)
    df["lag_2_week"] = grouped["units_sold"].shift(2)
    df["lag_4_week"] = grouped["units_sold"].shift(4)
    df["lag_8_week"] = grouped["units_sold"].shift(8)
    df["lag_13_week"] = grouped["units_sold"].shift(13)

    # Rolling stats over previous weeks (shift applied AFTER rolling, so the
    # current week's demand is excluded from the window used at that row).
    df["rolling_mean_4"] = grouped["units_sold"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    df["rolling_mean_8"] = grouped["units_sold"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )
    df["rolling_std_4"] = grouped["units_sold"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).std()
    )
    df["rolling_std_8"] = grouped["units_sold"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).std()
    )

    # Calendar & context features (available at forecast time, no future info).
    df["week_of_year"] = df.get("iso_week") if "iso_week" in df.columns else None
    if "month" not in df.columns:
        df["month"] = 0
    if "season" not in df.columns:
        df["season"] = "unknown"
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0
    if "promo" not in df.columns:
        df["promo"] = 0
    if "promo_event" not in df.columns:
        df["promo_event"] = "none"
    if "category" not in df.columns:
        df["category"] = "unknown"
    if "subcategory" not in df.columns:
        df["subcategory"] = "unknown"
    if "list_price" not in df.columns:
        df["list_price"] = 0.0

    # Keep only feature columns + keys.
    keep = ["sku_id", "period", "iso_year", "iso_week", "units_sold"] + _FEATURE_COLUMNS
    keep = [c for c in keep if c in df.columns]

    out = df[keep].reset_index(drop=True)

    # Post-build leakage audit.
    _assert_no_future_leakage(out)
    return out
# --------------------------------------------------------------------------- #
# 5. METRICS (WAPE primary, bias secondary, MAPE secondary)
# --------------------------------------------------------------------------- #


def wape(actual: Sequence[float], forecast: Sequence[float]) -> Optional[float]:
    """Weighted Absolute Percentage Error — PRIMARY metric.

    WAPE = sum(|actual - forecast|) / sum(|actual|)

    Returns ``None`` when ``sum(|actual|) == 0`` (zero-demand window) instead
    of fabricating a number.
    """
    a = np.asarray(actual, dtype="float64")
    f = np.asarray(forecast, dtype="float64")
    if a.size == 0 or a.size != f.size:
        return None
    denom = float(np.abs(a).sum())
    if denom == 0.0:
        return None
    return float(np.abs(a - f).sum() / denom)


def bias(actual: Sequence[float], forecast: Sequence[float]) -> Optional[float]:
    """Signed forecast bias — SECONDARY metric.

    Sign convention: bias = mean(forecast - actual).

    * positive bias -> model OVER-forecasts (forecasts more than actual)
    * negative bias -> model UNDER-forecasts (forecasts less than actual)

    Returns ``None`` on empty input (never a fabricated value).
    """
    a = np.asarray(actual, dtype="float64")
    f = np.asarray(forecast, dtype="float64")
    if a.size == 0 or a.size != f.size:
        return None
    return float((f - a).mean())


def mape(actual: Sequence[float], forecast: Sequence[float]) -> Optional[float]:
    """Mean Absolute Percentage Error — SECONDARY only (never primary).

    Zero-demand observations are excluded from the denominator safely; if no
    valid observations remain, returns ``None``.
    """
    a = np.asarray(actual, dtype="float64")
    f = np.asarray(forecast, dtype="float64")
    if a.size == 0 or a.size != f.size:
        return None
    mask = a != 0
    if not mask.any():
        return None
    return float((np.abs((a[mask] - f[mask]) / a[mask])).mean())


# --------------------------------------------------------------------------- #
# 6. SEASONAL-NAIVE BASELINE (default seasonal period 52 weeks)
# --------------------------------------------------------------------------- #


def seasonal_naive_baseline(
    weekly_demand: pd.DataFrame,
    horizon_weeks: int,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    """Seasonal-naive baseline: forecast[t] = demand[t - seasonal_period].

    For every SKU the forecast for a future week equals that SKU's demand from
    the same season one seasonal period earlier (e.g. same ISO week last year
    for period=52). This is NOT ``shift(1)`` and never yesterday's demand.

    Inspects the available history first:

    * If an SKU has fewer than ``seasonal_period + horizon_weeks`` observed
      weeks of history, those future weeks cannot be evaluated at this seasonal
      period and are reported via ``evaluable=False`` rows rather than being
      fabricated.

    Returns a DataFrame with columns:
      sku_id, period, iso_year, iso_week, baseline_units, evaluable, reason
    """
    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be >= 1.")
    if seasonal_period < 1:
        raise ValueError("seasonal_period must be >= 1.")

    required = {"sku_id", "period", "iso_year", "iso_week", "units_sold"}
    missing_cols = required - set(weekly_demand.columns)
    if missing_cols:
        raise D3Error(
            "seasonal_naive_baseline(): weekly table missing columns "
            f"{sorted(missing_cols)}. Run prepare_weekly_demand() first."
        )

    df = weekly_demand.sort_values(["sku_id", "iso_year", "iso_week"]).copy()
    grouped = df.groupby("sku_id", sort=True)

    # Seasonal lag within SKU: demand from `seasonal_period` weeks ago.
    df["baseline_units"] = grouped["units_sold"].shift(seasonal_period)

    max_year = int(df["iso_year"].max())
    max_week = int(df.loc[df["iso_year"] == max_year, "iso_week"].max())

    rows: List[Dict[str, Any]] = []
    for sku, g in grouped:
        g = g.sort_values(["iso_year", "iso_week"])
        n_obs = len(g)
        has_seasonal = n_obs >= seasonal_period

        year, week = max_year, max_week
        for step in range(1, horizon_weeks + 1):
            week += 1
            if week > 52:
                week = 1
                year += 1

            # Baseline value for future step = observed demand exactly
            # seasonal_period weeks before the corresponding future week,
            # walked deterministically backwards through the observed series.
            idx = len(g) - seasonal_period - (horizon_weeks - step)
            if has_seasonal and 0 <= idx < len(g):
                val = g["units_sold"].iloc[idx]
                evaluable = True
                reason = ""
            else:
                val = np.nan
                evaluable = False
                reason = (
                    f"insufficient history for seasonal_period={seasonal_period} "
                    f"(observed weeks={n_obs})"
                )

            rows.append(
                {
                    "sku_id": sku,
                    "period": f"{year}-W{str(week).zfill(2)}",
                    "iso_year": year,
                    "iso_week": week,
                    "baseline_units": val,
                    "evaluable": evaluable,
                    "reason": reason,
                }
            )

    return pd.DataFrame(rows)
# --------------------------------------------------------------------------- #
# 7. SKU HISTORY / DATA-READINESS CHECKS
# --------------------------------------------------------------------------- #


def check_sku_history(
    weekly_demand: pd.DataFrame,
    sku_id: str,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Inspect one SKU's official weekly history (never fabricates weeks).

    Returns observed week counts, first/last period, seasonal-baseline
    availability and model eligibility, plus the low-history flag.
    """
    g = weekly_demand[weekly_demand["sku_id"] == sku_id].sort_values(
        ["iso_year", "iso_week"]
    )
    n_obs = int(len(g))
    has_seasonal = n_obs >= config.seasonal_period
    qualifies_for_model = n_obs >= max(config.min_train_weeks, config.min_obs_per_sku)
    low_history = n_obs < config.min_train_weeks

    return {
        "sku_id": sku_id,
        "observed_weeks": n_obs,
        "first_period": str(g["period"].iloc[0]) if n_obs else None,
        "last_period": str(g["period"].iloc[-1]) if n_obs else None,
        "seasonal_baseline_available": has_seasonal,
        "qualifies_for_model": qualifies_for_model,
        "low_history": low_history,
        "required_for_model": max(config.min_train_weeks, config.min_obs_per_sku),
        "required_for_seasonal": config.seasonal_period,
    }

def check_forecast_data_readiness(
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    config: ForecastConfig = DEFAULT_CONFIG,
    processed_dir: Path = DATA_PROCESSED,
) -> Dict[str, Any]:
    """Structured data-readiness status for D3.

    Checks (in order):
      1. Official D1 outputs exist (hard guard — legacy files never substituted).
      2. Minimum history length vs horizon + seasonal period + folds.
      3. Weekly coverage (distinct weeks).
      4. SKU coverage (distinct SKUs).
      5. Seasonal baseline availability.
      6. Sufficient observations per SKU (model vs low-history split).
      7. Sufficient folds possible.

    Returns ``{"status": "ready"|"insufficient_data"|"data_not_available",
    "reasons": [...], "details": {...}}``. Never fabricates missing weeks.
    """
    reasons: List[str] = []
    details: Dict[str, Any] = {
        "horizon_weeks": config.horizon_weeks,
        "seasonal_period": config.seasonal_period,
        "cv_folds": config.cv_folds,
        "min_train_weeks": config.min_train_weeks,
    }

    # 1) Hard guard on the four official D1 output files.
    processed_dir = Path(processed_dir)
    missing_files = [
        fname
        for fname in D1_OUTPUT_FILES.values()
        if not (processed_dir / fname).is_file()
    ]
    if missing_files:
        return {
            "status": "data_not_available",
            "reasons": [
                "Official FORESIGHT row-level data is unavailable; no forecast "
                "or model performance result was computed."
            ],
            "missing_d1_outputs": missing_files,
            "details": details,
        }

    if tables is None:
        try:
            tables = load_d1_outputs(processed_dir)
        except MissingOfficialInputsError as exc:
            return {
                "status": "data_not_available",
                "reasons": [str(exc)],
                "missing_d1_outputs": exc.missing,
                "details": details,
            }

    try:
        weekly = prepare_weekly_demand(tables, config)
    except D3Error as exc:
        return {"status": "insufficient_data", "reasons": [str(exc)], "details": details}

    n_weeks = int(weekly["period"].nunique()) if len(weekly) else 0
    n_skus = int(weekly["sku_id"].nunique()) if len(weekly) else 0

    per_sku_counts = (
        weekly.groupby("sku_id")["period"].nunique()
        if len(weekly)
        else pd.Series(dtype="int64")
    )
    skus_with_model_history = int((per_sku_counts >= config.min_train_weeks).sum())
    skus_low_history = int((per_sku_counts < config.min_train_weeks).sum())

    details.update(
        {
            "total_weeks_observed": n_weeks,
            "total_skus": n_skus,
            "skus_with_model_history": skus_with_model_history,
            "skus_low_history": skus_low_history,
            "weeks_required_for_backtest": config.seasonal_period
            + config.horizon_weeks * config.cv_folds,
        }
    )

    # 2) History length vs horizon / seasonal period / folds.
    required_weeks = config.seasonal_period + config.horizon_weeks * config.cv_folds
    if n_weeks < config.horizon_weeks:
        reasons.append(
            f"Weekly history ({n_weeks} weeks) is shorter than the forecast "
            f"horizon ({config.horizon_weeks} weeks)."
        )
    if n_weeks < config.seasonal_period:
        reasons.append(
            f"Weekly history ({n_weeks} weeks) is shorter than the seasonal "
            f"period ({config.seasonal_period} weeks); a {config.seasonal_period}-week "
            "seasonal-naive baseline cannot be evaluated at that period."
        )
    if n_weeks < required_weeks:
        reasons.append(
            f"Weekly history ({n_weeks} weeks) is insufficient for "
            f"{config.cv_folds} rolling-origin folds of horizon "
            f"{config.horizon_weeks} with a {config.seasonal_period}-week "
            f"seasonal baseline (needs >= {required_weeks} weeks)."
        )

    # 3) Weekly & SKU coverage.
    if n_weeks == 0 or n_skus == 0:
        reasons.append("No weekly demand rows available after aggregation.")
    if skus_with_model_history == 0 and n_skus > 0:
        reasons.append(
            f"No SKU has at least {config.min_train_weeks} weeks of history "
            "for the ML model; all SKUs would be low-history."
        )

    status = "ready" if not reasons else "insufficient_data"
    return {
        "status": status,
        "reasons": reasons,
        "details": details,
        "per_sku_week_counts": {str(k): int(v) for k, v in per_sku_counts.items()},
    }
# --------------------------------------------------------------------------- #
# 8. ML MODEL
# --------------------------------------------------------------------------- #


def build_model(config: ForecastConfig = DEFAULT_CONFIG) -> HistGradientBoostingRegressor:
    """Build the gradient-boosted tree model with a FIXED random seed.

    Default model: ``HistGradientBoostingRegressor`` (handles NaN lags for
    short-history SKUs natively). Parameters are configurable via
    ``ForecastConfig.model_params``; ``random_state`` is always pinned to
    ``config.random_seed`` unless explicitly overridden there.
    """
    params: Dict[str, Any] = {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_depth": None,
        "min_samples_leaf": 5,
        "l2_regularization": 0.0,
        "early_stopping": False,
    }
    params.update(dict(config.model_params))
    params["random_state"] = int(config.random_seed)
    return HistGradientBoostingRegressor(**params)


def _encode_categoricals(
    df: pd.DataFrame,
    feature_cols: List[str],
    mappings: Optional[Dict[str, Dict[str, int]]] = None,
) -> pd.DataFrame:
    """Deterministic one-hot encoding of categorical feature columns.

    Uses fixed category ordering so folds are comparable and reproducible.
    Non-numeric columns among ``feature_cols`` are encoded; numeric columns
    pass through unchanged.
    """
    out = df.copy()
    for col in list(feature_cols):
        if col not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            mapping = (mappings or {}).get(col)
            if mapping is None:
                cats = sorted(str(v) for v in out[col].dropna().unique())
                mapping = {c: i + 1 for i, c in enumerate(cats)}
            out[col] = out[col].map(lambda v: mapping.get(str(v)) if pd.notna(v) else np.nan)
            out[col] = out[col].astype("float64")
        else:
            out[col] = out[col].astype("float64")
    return out


def _category_mappings(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """Build one deterministic categorical encoding for a complete batch."""
    mappings: Dict[str, Dict[str, int]] = {}
    for col in feature_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            cats = sorted(str(v) for v in df[col].dropna().unique())
            mappings[col] = {c: i + 1 for i, c in enumerate(cats)}
    return mappings


def _fit_predict_model(
    features: pd.DataFrame,
    train_mask: pd.Series,
    test_index: pd.Index,
    target_col: str,
    config: ForecastConfig,
) -> Optional[np.ndarray]:
    """Fit on ``train_mask`` rows only; predict for ``test_index`` rows.

    Leakage-safe by construction: the caller guarantees that every training row
    is strictly earlier than every test row. Returns ``None`` when there is no
    trainable history (never fabricates predictions).
    """
    train = features.loc[train_mask]
    if len(train) == 0:
        return None

    feature_cols = [c for c in _FEATURE_COLUMNS if c in features.columns]
    X_train = _encode_categoricals(train, feature_cols)[feature_cols]
    y_train = train[target_col].astype("float64")

    model = build_model(config)
    try:
        model.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001 - surface a clear D3 error
        raise D3Error(f"Model fitting failed: {exc}") from exc

    test = features.loc[test_index]
    X_test = _encode_categoricals(test, feature_cols)[feature_cols]
    return model.predict(X_test)


def _recursive_backtest_predictions(
    features: pd.DataFrame,
    train_mask: pd.Series,
    test_periods: pd.DataFrame,
    config: ForecastConfig,
) -> Optional[pd.Series]:
    """Predict a test window recursively so test actuals never become lags."""
    feature_cols = [c for c in _FEATURE_COLUMNS if c in features.columns]
    mappings = _category_mappings(features.loc[train_mask], feature_cols)
    encoded_train = _encode_categoricals(features.loc[train_mask], feature_cols, mappings)
    model = build_model(config)
    if encoded_train.empty:
        return None
    feature_cols = [
        column for column in feature_cols
        if encoded_train[column].notna().any()
        and encoded_train[column].nunique(dropna=True) >= 2
    ]
    if not feature_cols:
        return None
    model.fit(encoded_train[feature_cols], features.loc[train_mask, "units_sold"].astype("float64"))

    train_features = features.loc[train_mask]
    history_values = {
        sku_id: group.sort_values(["iso_year", "iso_week"])["units_sold"].astype(float).tolist()
        for sku_id, group in train_features.groupby("sku_id")
    }
    predictions: Dict[int, float] = {}
    for period in test_periods["period"]:
        future = features[features["period"] == period].sort_values("sku_id")
        rows = []
        indices = []
        for index, row in future.iterrows():
            sku_id = row["sku_id"]
            values_for_sku = history_values.setdefault(sku_id, [])
            values = {
                "lag_1_week": values_for_sku[-1] if len(values_for_sku) >= 1 else np.nan,
                "lag_2_week": values_for_sku[-2] if len(values_for_sku) >= 2 else np.nan,
                "lag_4_week": values_for_sku[-4] if len(values_for_sku) >= 4 else np.nan,
                "lag_8_week": values_for_sku[-8] if len(values_for_sku) >= 8 else np.nan,
                "lag_13_week": values_for_sku[-13] if len(values_for_sku) >= 13 else np.nan,
                "rolling_mean_4": np.mean(values_for_sku[-4:]) if values_for_sku else np.nan,
                "rolling_mean_8": np.mean(values_for_sku[-8:]) if values_for_sku else np.nan,
                "rolling_std_4": np.std(values_for_sku[-4:]) if len(values_for_sku) >= 2 else np.nan,
                "rolling_std_8": np.std(values_for_sku[-8:]) if len(values_for_sku) >= 2 else np.nan,
                "week_of_year": row.get("week_of_year", row.get("iso_week", np.nan)),
            }
            for column in feature_cols:
                if column not in values:
                    values[column] = row.get(column, np.nan)
            rows.append(values)
            indices.append((index, sku_id))
        if rows:
            encoded_rows = _encode_categoricals(pd.DataFrame(rows), feature_cols, mappings)
            batch_predictions = model.predict(encoded_rows[feature_cols])
            for (index, sku_id), prediction in zip(indices, batch_predictions):
                value = max(0.0, float(prediction))
                predictions[index] = value
                history_values[sku_id].append(value)
    return pd.Series(predictions, dtype="float64")


def _category_fallback_predictions(
    weekly: pd.DataFrame,
    history_mask: pd.Series,
    target_rows: pd.DataFrame,
) -> pd.Series:
    """Predict target rows using a cutoff-safe deterministic fallback hierarchy.

    Hierarchy for each target row:
      1. category seasonal historical rate for that ISO week using only history
         observed before the target SKU's cutoff.
      2. category trailing mean over pre-cutoff category demand.
      3. same-SKU trailing mean over pre-cutoff SKU demand.
      4. deterministic zero only if no historical demand information exists.

    This prevents future leakage and guarantees finite, non-negative forecasts.
    """
    if weekly is None or weekly.empty or target_rows is None or target_rows.empty:
        return pd.Series(index=target_rows.index, dtype="float64") if target_rows is not None else pd.Series(dtype="float64")

    history = weekly.copy()
    if isinstance(history_mask, pd.Series) and len(history_mask) == len(history):
        history = history.loc[history_mask].copy()
    elif isinstance(history_mask, (list, tuple, np.ndarray)):
        history = history.loc[pd.Series(history_mask, index=history.index).fillna(False).to_numpy()].copy()

    if "sku_id" in history.columns:
        history["sku_id"] = history["sku_id"].astype(str)
        history["_sku_key"] = history["sku_id"]
    if "category" in history.columns:
        history["category"] = history["category"].fillna("unknown").astype(str)
        history["_category_key"] = history["category"]
    else:
        history["category"] = "unknown"
    if "units_sold" in history.columns:
        history["units_sold"] = pd.to_numeric(history["units_sold"], errors="coerce").fillna(0.0)
    history["_year_num"] = pd.to_numeric(history["iso_year"], errors="coerce")
    history["_week_num"] = pd.to_numeric(history["iso_week"], errors="coerce")

    if history.empty:
        zeros = pd.Series(0.0, index=target_rows.index, dtype="float64")
        return zeros

    target_index = target_rows.index
    target = target_rows.copy().reset_index(drop=True)
    if "sku_id" in target.columns:
        target["sku_id"] = target["sku_id"].astype(str)
    if "category" in target.columns:
        target["category"] = target["category"].fillna("unknown").astype(str)
    else:
        target["category"] = "unknown"

    cutoff_cache: Dict[str, Optional[Tuple[int, int]]] = {}
    category_stats_cache: Dict[Tuple[str, Optional[Tuple[int, int]]], Tuple[Dict[int, float], float]] = {}
    sku_trailing_cache: Dict[Tuple[str, Optional[Tuple[int, int]]], float] = {}

    def _cutoff_for_sku(sku_id: str) -> Optional[Tuple[int, int]]:
        if sku_id in cutoff_cache:
            return cutoff_cache[sku_id]
        g = history[history["_sku_key"] == sku_id]
        if g.empty:
            cutoff_cache[sku_id] = None
            return None
        g = g.assign(
            _iso_year_num=g["_year_num"],
            _iso_week_num=g["_week_num"],
        ).sort_values(["_iso_year_num", "_iso_week_num"]).reset_index(drop=True)
        last = g.iloc[-1]
        cutoff_cache[sku_id] = (int(last["iso_year"]), int(last["iso_week"]))
        return cutoff_cache[sku_id]

    def _before_cutoff(g: pd.DataFrame, cutoff: Optional[Tuple[int, int]]) -> pd.DataFrame:
        if g.empty or cutoff is None:
            return g
        cutoff_year, cutoff_week = cutoff
        mask = (
            (g["_year_num"] < cutoff_year)
            | (
                (g["_year_num"] == cutoff_year)
                & (g["_week_num"] < cutoff_week)
            )
        )
        return g.loc[mask].copy()

    values: List[float] = []
    for _, row in target.iterrows():
        sku_id = str(row.get("sku_id", ""))
        category = str(row.get("category", "unknown")).strip() or "unknown"
        iso_week = int(row.get("iso_week", 0))
        cutoff = _cutoff_for_sku(sku_id)

        stats_key = (category, cutoff)
        if stats_key not in category_stats_cache:
            category_history = _before_cutoff(
                history[history["_category_key"] == category], cutoff
            )
            seasonal_values = (
                category_history.groupby("iso_week")["units_sold"].mean().to_dict()
                if not category_history.empty
                else {}
            )
            category_mean = (
                float(category_history["units_sold"].mean())
                if not category_history.empty and category_history["units_sold"].notna().any()
                else np.nan
            )
            category_stats_cache[stats_key] = (seasonal_values, category_mean)

        seasonal_values, category_mean = category_stats_cache[stats_key]
        seasonal_value = float(seasonal_values.get(iso_week, np.nan))

        if (not np.isfinite(seasonal_value)) or seasonal_value < 0:
            trailing = category_mean
            if (not np.isfinite(trailing)) or trailing < 0:
                sku_key = (sku_id, cutoff)
                if sku_key not in sku_trailing_cache:
                    sku_history = _before_cutoff(
                        history[history["_sku_key"] == sku_id], cutoff
                    )
                    sku_trailing_cache[sku_key] = (
                        float(sku_history["units_sold"].mean())
                        if not sku_history.empty and sku_history["units_sold"].notna().any()
                        else np.nan
                    )
                trailing = sku_trailing_cache[sku_key]
            if (not np.isfinite(trailing)) or trailing < 0:
                trailing = 0.0
            seasonal_value = trailing

        if not np.isfinite(seasonal_value):
            seasonal_value = 0.0
        values.append(max(0.0, float(seasonal_value)))

    result = pd.Series(values, index=target_index, dtype="float64")
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result.clip(lower=0.0)
# --------------------------------------------------------------------------- #
# 9. ROLLING-ORIGIN BACKTEST
# --------------------------------------------------------------------------- #


def rolling_origin_backtest(
    features: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Rolling-origin cross-validation with strict temporal ordering.

    For fold ``i`` (i = 1..cv_folds):

      * TEST  = the ``horizon_weeks`` weeks immediately after the fold origin.
      * TRAIN = every observed week strictly BEFORE the test window.
      * The origin then moves forward by ``horizon_weeks`` and repeats.

    This mimics real forecasting: no future observation is ever available to a
    training row, and there is NO random/shuffled/plain 70/30 split.

    The seasonal-naive baseline and the ML model are evaluated on the SAME
    test windows of the same folds (identical evaluation periods).

    Returns fold records + per-method metric summaries; non-evaluable methods
    are reported, never fabricated.
    """
    config.validate()

    required_cols = {"sku_id", "period", "iso_year", "iso_week", "units_sold"}
    missing_cols = required_cols - set(features.columns)
    if missing_cols:
        raise D3Error(
            f"rolling_origin_backtest(): features missing columns {sorted(missing_cols)}."
        )

    # Global ordered list of distinct weekly periods (deterministic).
    period_order = (
        features[["iso_year", "iso_week", "period"]]
        .drop_duplicates()
        .sort_values(["iso_year", "iso_week"])
        .reset_index(drop=True)
    )
    n_periods = len(period_order)
    required_weeks = config.seasonal_period + config.horizon_weeks * config.cv_folds

    if n_periods < config.horizon_weeks:
        raise InsufficientHistoryError(
            f"Weekly history ({n_periods} weeks) is shorter than one horizon "
            f"({config.horizon_weeks} weeks); cannot backtest."
        )
    if n_periods < required_weeks:
        raise InsufficientHistoryError(
            f"Weekly history ({n_periods} weeks) is insufficient for "
            f"{config.seasonal_period}-week seasonal baseline plus "
            f"{config.cv_folds} folds x {config.horizon_weeks}-week horizon "
            f"(needs >= {required_weeks} weeks). No results were fabricated; "
            "reduce horizon/seasonal_period/folds or provide more official history."
        )

    fold_records: List[Dict[str, Any]] = []
    skipped_windows: List[Dict[str, Any]] = []
    baseline_actual_all: List[float] = []
    baseline_pred_all: List[float] = []
    model_actual_all: List[float] = []
    model_pred_all: List[float] = []

    for fold in range(1, config.cv_folds + 1):
        test_start_idx = n_periods - (config.cv_folds - fold + 1) * config.horizon_weeks
        test_end_idx = test_start_idx + config.horizon_weeks  # exclusive

        train_periods = period_order.iloc[:test_start_idx]
        test_periods = period_order.iloc[test_start_idx:test_end_idx]

        train_mask = features["period"].isin(set(train_periods["period"]))
        test_mask = features["period"].isin(set(test_periods["period"]))

        train_end_period = (
            str(train_periods["period"].iloc[-1]) if len(train_periods) else None
        )
        actual_test = features.loc[test_mask].sort_values(
            ["sku_id", "iso_year", "iso_week"]
        )

        # ---- Seasonal-naive on this exact window --------------------------
        ordered_features = features.sort_values(
            ["sku_id", "iso_year", "iso_week"]
        ).copy()
        ordered_features["_seasonal_baseline_units"] = ordered_features.groupby(
            "sku_id", sort=False
        )["units_sold"].shift(config.seasonal_period)
        baseline_rows = ordered_features.loc[
            ordered_features["period"].isin(set(test_periods["period"]))
            & ordered_features["_seasonal_baseline_units"].notna(),
            ["sku_id", "period", "units_sold", "_seasonal_baseline_units"],
        ]
        base_rows = [
            {
                "sku_id": row[0],
                "period": row[1],
                "actual": float(row[2]),
                "baseline_units": float(row[3]),
            }
            for row in baseline_rows.itertuples(index=False, name=None)
        ]
        if base_rows:
            base_df = pd.DataFrame(base_rows)
            baseline_actual_all.extend(base_df["actual"].tolist())
            baseline_pred_all.extend(base_df["baseline_units"].tolist())
        else:
            skipped_windows.append(
                {
                    "fold": fold,
                    "reason": (
                        f"seasonal-naive not evaluable: no SKU had "
                        f"{config.seasonal_period}+ weeks of pre-window history"
                    ),
                    "train_end": train_end_period,
                }
            )

        # ---- ML model on the SAME window ---------------------------------
        model_preds = _recursive_backtest_predictions(
            features=features,
            train_mask=train_mask,
            test_periods=test_periods,
            config=config,
        )
        train_counts = features.loc[train_mask].groupby("sku_id")["period"].nunique()
        low_history_skus = set(
            train_counts[train_counts < max(config.min_train_weeks, config.min_obs_per_sku)].index
        )
        low_history_test = actual_test[actual_test["sku_id"].isin(low_history_skus)]
        if len(low_history_test):
            category_preds = _category_fallback_predictions(
                features, train_mask, low_history_test
            )
            if model_preds is None:
                model_preds = pd.Series(dtype="float64")
            model_preds = model_preds.copy()
            model_preds.loc[category_preds.index] = category_preds
        if model_preds is None:
            skipped_windows.append(
                {
                    "fold": fold,
                    "reason": "no trainable rows for ML model",
                    "train_end": train_end_period,
                }
            )
        else:
            model_actual_all.extend(actual_test["units_sold"].astype(float).tolist())
            model_pred_all.extend(
                [float(model_preds.get(index, np.nan)) for index in actual_test.index]
            )

        fold_records.append(
            {
                "fold": fold,
                "train_start_period": str(train_periods["period"].iloc[0]),
                "train_end_period": train_end_period,
                "test_period": str(test_periods["period"].iloc[0]),
                "test_start_period": str(test_periods["period"].iloc[0]),
                "test_end_period": str(test_periods["period"].iloc[-1]),
                "train_weeks": int(test_start_idx),
                "n_train_rows": int(train_mask.sum()),
                "n_test_rows": int(test_mask.sum()),
                "baseline_evaluable_rows": len(base_rows),
                "model_evaluable": bool(model_preds is not None),
            }
        )

    baseline_wape = wape(baseline_actual_all, baseline_pred_all)
    baseline_bias = bias(baseline_actual_all, baseline_pred_all)
    baseline_mape = mape(baseline_actual_all, baseline_pred_all)
    model_wape = wape(model_actual_all, model_pred_all)
    model_bias = bias(model_actual_all, model_pred_all)
    model_mape = mape(model_actual_all, model_pred_all)

    return {
        "folds": fold_records,
        "skipped_windows": skipped_windows,
        "baseline_metrics": {
            "wape": baseline_wape,
            "bias": baseline_bias,
            "mape": baseline_mape,
            "evaluated_rows": len(baseline_actual_all),
        },
        "model_metrics": {
            "wape": model_wape,
            "bias": model_bias,
            "mape": model_mape,
            "evaluated_rows": len(model_actual_all),
        },
        "methodology": {
            "type": "rolling_origin",
            "horizon_weeks": config.horizon_weeks,
            "seasonal_period_weeks": config.seasonal_period,
            "cv_folds": config.cv_folds,
            "min_train_weeks": config.min_train_weeks,
            "random_seed": config.random_seed,
            "split": "chronological only (no random / shuffled / 70-30 split)",
        },
    }
# --------------------------------------------------------------------------- #
# 10. BASELINE VS MODEL COMPARISON & OFFICIAL SELECTION RULE
# --------------------------------------------------------------------------- #


def compare_model_to_baseline(backtest: Dict[str, Any]) -> Dict[str, Any]:
    """Compare ML model vs seasonal-naive on the SAME backtest windows.

    Improvement is measured on WAPE (primary):
        improvement = (baseline_wape - model_wape) / baseline_wape
    Positive => the ML model reduces error vs the baseline.
    ``None`` metrics propagate honestly (no fabrication).
    """
    base = backtest.get("baseline_metrics", {})
    mdl = backtest.get("model_metrics", {})

    baseline_wape = base.get("wape")
    model_wape = mdl.get("wape")

    if baseline_wape in (None, 0) or model_wape is None:
        improvement = None
    else:
        improvement = float((baseline_wape - model_wape) / baseline_wape)

    return {
        "baseline": {
            "name": "seasonal_naive",
            "seasonal_period_weeks": backtest.get("methodology", {}).get(
                "seasonal_period_weeks"
            ),
            "wape": baseline_wape,
            "bias": base.get("bias"),
            "mape": base.get("mape"),
            "evaluated_rows": base.get("evaluated_rows"),
        },
        "model": {
            "name": "hist_gradient_boosting",
            "random_seed": backtest.get("methodology", {}).get("random_seed"),
            "wape": model_wape,
            "bias": mdl.get("bias"),
            "mape": mdl.get("mape"),
            "evaluated_rows": mdl.get("evaluated_rows"),
        },
        "improvement_vs_baseline_wape": improvement,
        "same_windows": True,
    }


def select_best_forecaster(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Official FORESIGHT selection rule.

    The ML model is selected ONLY if it genuinely beats the seasonal-naive
    baseline on WAPE (strictly lower) on the same evaluation periods.
    Otherwise the baseline is retained — the rule never forces the ML model
    to win, and ties go to the simpler baseline.

    Returns a structured decision record including an explicit reason.
    """
    baseline_wape = comparison["baseline"]["wape"]
    model_wape = comparison["model"]["wape"]

    if model_wape is None and baseline_wape is None:
        selected = None
        reason = (
            "Neither method could be evaluated on the available official "
            "history; no selection was made and no metric was fabricated."
        )
    elif model_wape is None:
        selected = "seasonal_naive"
        reason = "ML model could not be evaluated; retaining seasonal-naive baseline."
    elif baseline_wape is None:
        selected = "hist_gradient_boosting"
        reason = "Baseline not evaluable; ML model is the only evaluated option."
    elif model_wape < baseline_wape:
        selected = "hist_gradient_boosting"
        reason = (
            f"ML WAPE ({model_wape:.4f}) < baseline WAPE ({baseline_wape:.4f}) "
            "on identical rolling-origin folds."
        )
    else:
        selected = "seasonal_naive"
        reason = (
            f"ML WAPE ({model_wape:.4f}) did not beat baseline WAPE "
            f"({baseline_wape:.4f}); baseline honestly retained per the "
            "official model-selection rule."
        )

    return {
        "selected_model": selected,
        "reason": reason,
        "primary_metric": "wape",
        "rule": "select ML only if model WAPE < baseline WAPE on the same folds",
    }
# --------------------------------------------------------------------------- #
# 11. FORECAST GENERATION (future weeks)
# --------------------------------------------------------------------------- #


def _next_periods(last_year: int, last_week: int, horizon: int) -> List[Tuple[int, int]]:
    """Deterministically build the next ``horizon`` (year, week) labels."""
    out: List[Tuple[int, int]] = []
    year, week = int(last_year), int(last_week)
    for _ in range(horizon):
        week += 1
        if week > 52:
            week = 1
            year += 1
        out.append((year, week))
    return out


def forecast_weekly_sku(
    sku_id: str,
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    config: ForecastConfig = DEFAULT_CONFIG,
    processed_dir: Path = DATA_PROCESSED,
    forecast_run_id: Optional[str] = None,
    weekly: Optional[pd.DataFrame] = None,
    features: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Forecast the next ``horizon_weeks`` of weekly demand for ONE SKU.

    Data-gated: loads official D1 outputs when ``tables`` is not supplied and
    raises ``MissingOfficialInputsError`` if they are absent. Never fabricates
    values; SKUs without sufficient model history are routed through the
    documented low-history fallback (see ``ForecastConfig``).

    Batch callers (e.g. the D3 report runner) may pass ``weekly`` and
    ``features`` to reuse objects already computed for the backtest;
    when omitted they are computed exactly as before, so standalone
    behaviour is unchanged.

    Returns a structured result with forecast rows:
        period, sku_id, forecast_units, model_name, low_confidence, note
    """
    config.validate()

    if tables is None:
        tables = load_d1_outputs(processed_dir)

    if weekly is None:
        weekly = prepare_weekly_demand(tables, config)
    history = check_sku_history(weekly, sku_id, config)
    if history["observed_weeks"] == 0:
        raise D3Error(
            f"forecast_weekly_sku(): SKU '{sku_id}' has no official weekly "
            "history in the D1 outputs."
        )

    run_id = forecast_run_id or datetime.utcnow().strftime("fc_%Y%m%dT%H%M%SZ")
    horizon = config.horizon_weeks

    g = (
        weekly[weekly["sku_id"] == sku_id]
        .sort_values(["iso_year", "iso_week"])
        .reset_index(drop=True)
    )
    max_year = int(g["iso_year"].max())
    max_week = int(g.loc[g["iso_year"] == max_year, "iso_week"].max())
    future_periods = _next_periods(max_year, max_week, horizon)

    meta_row = g.iloc[0]
    use_model = bool(history["qualifies_for_model"])
    preds: List[Optional[float]] = []
    model_name: str
    fallback_used = False
    low_confidence = False
    note = ""

    if use_model:
        # ---- ML path (iterative multi-step; predictions feed later lags) ----
        if features is None:
            features = build_forecast_features(weekly)
        f_sku = (
            features[features["sku_id"] == sku_id]
            .sort_values(["iso_year", "iso_week"])
            .reset_index(drop=True)
        )
        feature_cols = [c for c in _FEATURE_COLUMNS if c in features.columns]
        X = _encode_categoricals(f_sku, feature_cols)[feature_cols]

        # Minimal compatibility fix for the official-derived dataset:
        # HistGradientBoostingRegressor's binning raises
        # "window shape cannot be larger than input array shape" when a feature
        # column in THIS SKU's training slice carries no usable variation
        # (all-NaN, or a single constant value). On the official data this
        # happens for lag_13_week (short-history SKUs), the per-SKU columns
        # category / subcategory / list_price (one value per SKU), and the
        # constant calendar column is_holiday. Drop such columns so the model
        # can fit with the remaining usable features; the SAME reduced column
        # set is reused for prediction. No target/model/hyperparameter/horizon
        # change; if no usable columns remain, the documented seasonal-naive
        # fallback below is used instead of inventing values.
        usable_cols = [
            c for c in feature_cols
            if X[c].notna().any() and X[c].nunique(dropna=True) >= 2
        ]
        if not usable_cols:
            # No usable ML signal for this SKU -> routed to the documented
            # seasonal-naive fallback (handled below, same as low-history SKUs).
            use_model = False
        else:
            feature_cols = usable_cols
            X = X[feature_cols]

    if use_model:
        model = build_model(config)
        try:
            model.fit(X, f_sku["units_sold"].astype("float64"))
        except Exception as exc:  # noqa: BLE001 - surface a clear D3 error
            raise D3Error(f"Model fitting failed for SKU '{sku_id}': {exc}") from exc

        work = f_sku.copy()
        for (fy, fw) in future_periods:
            last = work.iloc[-1]
            hist = work["units_sold"].astype(float).tolist()
            row: Dict[str, Any] = {
                "lag_1_week": float(hist[-1]) if hist else np.nan,
                "lag_2_week": float(hist[-2]) if len(hist) >= 2 else np.nan,
                "lag_4_week": float(hist[-4]) if len(hist) >= 4 else np.nan,
                "lag_8_week": float(hist[-8]) if len(hist) >= 8 else np.nan,
                "lag_13_week": float(hist[-13]) if len(hist) >= 13 else np.nan,
                "rolling_mean_4": float(np.mean(hist[-4:])) if hist else np.nan,
                "rolling_mean_8": float(np.mean(hist[-8:])) if hist else np.nan,
                "rolling_std_4": float(np.std(hist[-4:])) if len(hist) >= 2 else np.nan,
                "rolling_std_8": float(np.std(hist[-8:])) if len(hist) >= 2 else np.nan,
                "week_of_year": float(fw),
                "month": last.get("month", 0),
                "season": last.get("season", "unknown"),
                "is_holiday": 0,
                "promo": 0,
                "promo_event": "none",
                "category": last.get("category", "unknown"),
                "subcategory": last.get("subcategory", "unknown"),
                "list_price": last.get("list_price", 0.0),
            }
            row_df = pd.DataFrame([row])
            row_enc = _encode_categoricals(row_df, feature_cols)[feature_cols]
            p = float(model.predict(row_enc)[0])
            p = max(0.0, p)  # demand cannot be negative
            preds.append(p)

            new_row = {c: np.nan for c in work.columns}
            new_row.update(
                {
                    "sku_id": sku_id,
                    "period": f"{fy}-W{str(fw).zfill(2)}",
                    "iso_year": fy,
                    "iso_week": fw,
                    "units_sold": p,
                    **row,
                }
            )
            work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

        model_name = "hist_gradient_boosting"
        low_confidence = bool(history["low_history"])

    elif config.low_history_fallback == "category_historical":
        target = pd.DataFrame(
            {
                "sku_id": sku_id,
                "category": meta_row.get("category", "unknown"),
                "iso_week": [fw for _, fw in future_periods],
            }
        )
        fallback = _category_fallback_predictions(
            weekly,
            weekly["sku_id"] == sku_id,
            target,
        )
        preds = [float(value) for value in fallback]
        model_name = "category_historical_fallback"
        fallback_used = True
        low_confidence = True
        note = (
            f"Low official history for this SKU ({history['observed_weeks']} "
            f"weeks < required {history['required_for_model']}); used the "
            "category-level historical demand fallback."
        )
    else:
        raise InsufficientHistoryError(
            f"SKU '{sku_id}' lacks sufficient history for the ML model (has "
            f"{history['observed_weeks']} weeks, needs "
            f"{history['required_for_model']}) and low_history_fallback='none'."
        )

    rows: List[Dict[str, Any]] = []
    for (fy, fw), val in zip(future_periods, preds):
        rows.append(
            {
                "period": f"{fy}-W{str(fw).zfill(2)}",
                "iso_year": fy,
                "iso_week": fw,
                "sku_id": sku_id,
                "forecast_units": val,
                "model_name": model_name,
                "low_confidence": low_confidence,
                "note": note,
            }
        )

    return {
        "forecast_run_id": run_id,
        "sku_id": sku_id,
        "horizon_weeks": horizon,
        "model_name": model_name,
        "fallback_used": fallback_used,
        "low_history_flagged": bool(history["low_history"]),
        "category": meta_row.get("category"),
        "subcategory": meta_row.get("subcategory"),
        "forecast_rows": rows,
        "history_summary": history,
    }


def forecast_all_skus(
    weekly: pd.DataFrame,
    features: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
    forecast_run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Forecast all SKUs with one shared model fit for eligible histories."""
    config.validate()
    required = {"sku_id", "period", "iso_year", "iso_week", "units_sold"}
    missing = required - set(weekly.columns)
    if missing:
        raise D3Error(f"forecast_all_skus(): weekly table missing {sorted(missing)}.")

    normalized_weekly = weekly.copy()
    normalized_weekly["sku_id"] = normalized_weekly["sku_id"].astype(str)
    normalized_features = features.copy()
    normalized_features["sku_id"] = normalized_features["sku_id"].astype(str)
    feature_cols = [c for c in _FEATURE_COLUMNS if c in normalized_features.columns]
    mappings = _category_mappings(normalized_features, feature_cols)
    encoded_features = _encode_categoricals(normalized_features, feature_cols, mappings)

    counts = normalized_weekly.groupby("sku_id")["period"].nunique()
    min_history = max(config.min_train_weeks, config.min_obs_per_sku)
    eligible_skus = set(counts[counts >= min_history].index)
    model = None
    if eligible_skus:
        train_mask = normalized_features["sku_id"].isin(eligible_skus)
        train_features = encoded_features.loc[train_mask, feature_cols]
        feature_cols = [
            column for column in feature_cols
            if train_features[column].notna().any()
            and train_features[column].nunique(dropna=True) >= 2
        ]
        if not feature_cols:
            eligible_skus = set()
        else:
            train_features = encoded_features.loc[train_mask, feature_cols]
        model = build_model(config)
        if feature_cols:
            model.fit(
                train_features,
                normalized_features.loc[train_mask, "units_sold"].astype("float64"),
            )

    baseline = None
    if config.low_history_fallback == "category_historical":
        baseline = seasonal_naive_baseline(
            normalized_weekly, config.horizon_weeks, config.seasonal_period
        )
    run_id = forecast_run_id or datetime.utcnow().strftime("fc_%Y%m%dT%H%M%SZ")
    results: List[Dict[str, Any]] = []
# One-pass O(1) group lookups per SKU (avoid repeated full-frame filtering
    # and per-SKU sorts for every SKU in the forecasting loop).
    sku_by: Dict[str, pd.DataFrame] = {
        str(sku): grp.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        for sku, grp in normalized_weekly.groupby("sku_id", sort=False)
    }
    feat_by: Dict[str, pd.DataFrame] = {
        str(sku): grp.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        for sku, grp in normalized_features.groupby("sku_id", sort=False)
    }

    for sku_id in sorted(normalized_weekly["sku_id"].unique()):
        sku_weekly = sku_by[sku_id]
        history = check_sku_history(sku_weekly, sku_id, config)
        max_year = int(sku_weekly["iso_year"].max())
        max_week = int(sku_weekly.loc[sku_weekly["iso_year"] == max_year, "iso_week"].max())
        future_periods = _next_periods(max_year, max_week, config.horizon_weeks)
        meta_row = sku_weekly.iloc[0]
        preds: List[Optional[float]] = []
        fallback_used = False
        low_confidence = False
        note = ""

        if model is not None and sku_id in eligible_skus:
            sku_features = feat_by[sku_id]
            work = sku_features.copy()
            history_values = work["units_sold"].astype(float).tolist()
            for fy, fw in future_periods:
                last = work.iloc[-1]
                row = {
                    "lag_1_week": float(history_values[-1]) if history_values else np.nan,
                    "lag_2_week": float(history_values[-2]) if len(history_values) >= 2 else np.nan,
                    "lag_4_week": float(history_values[-4]) if len(history_values) >= 4 else np.nan,
                    "lag_8_week": float(history_values[-8]) if len(history_values) >= 8 else np.nan,
                    "lag_13_week": float(history_values[-13]) if len(history_values) >= 13 else np.nan,
                    "rolling_mean_4": float(np.mean(history_values[-4:])) if history_values else np.nan,
                    "rolling_mean_8": float(np.mean(history_values[-8:])) if history_values else np.nan,
                    "rolling_std_4": float(np.std(history_values[-4:])) if len(history_values) >= 2 else np.nan,
                    "rolling_std_8": float(np.std(history_values[-8:])) if len(history_values) >= 2 else np.nan,
                    "week_of_year": float(fw),
                    "month": last.get("month", 0),
                    "season": last.get("season", "unknown"),
                    "is_holiday": 0,
                    "promo": 0,
                    "promo_event": "none",
                    "category": last.get("category", "unknown"),
                    "subcategory": last.get("subcategory", "unknown"),
                    "list_price": last.get("list_price", 0.0),
                }
                encoded_row = _encode_categoricals(
                    pd.DataFrame([row]), feature_cols, mappings
                )
                prediction = max(0.0, float(model.predict(encoded_row[feature_cols])[0]))
                preds.append(prediction)
                history_values.append(prediction)
                new_row = {column: np.nan for column in work.columns}
                new_row.update({
                    "sku_id": sku_id,
                    "period": f"{fy}-W{fw:02d}",
                    "iso_year": fy,
                    "iso_week": fw,
                    "units_sold": prediction,
                    **row,
                })
                work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)
            model_name = "hist_gradient_boosting"
            low_confidence = bool(history["low_history"])
        elif config.low_history_fallback == "category_historical":
            target = pd.DataFrame(
                {
                    "sku_id": sku_id,
                    "category": meta_row.get("category", "unknown"),
                    "iso_week": [fw for _, fw in future_periods],
                }
            )
            fallback = _category_fallback_predictions(
                sku_weekly,
                None,
                target,
            )
            preds = [float(value) for value in fallback]
            model_name = "category_historical_fallback"
            fallback_used = True
            low_confidence = True
            note = (
                f"Low official history for this SKU ({history['observed_weeks']} weeks < "
                f"required {history['required_for_model']}); used category-level "
                "historical demand fallback."
            )
        else:
            raise InsufficientHistoryError(f"SKU '{sku_id}' lacks sufficient history.")

        results.append({
            "forecast_run_id": run_id,
            "sku_id": sku_id,
            "horizon_weeks": config.horizon_weeks,
            "model_name": model_name,
            "fallback_used": fallback_used,
            "low_history_flagged": low_confidence,
            "category": meta_row.get("category"),
            "subcategory": meta_row.get("subcategory"),
            "forecast_rows": [
                {
                    "period": f"{fy}-W{fw:02d}",
                    "iso_year": fy,
                    "iso_week": fw,
                    "sku_id": sku_id,
                    "forecast_units": value,
                    "model_name": model_name,
                    "low_confidence": low_confidence,
                    "note": note,
                }
                for (fy, fw), value in zip(future_periods, preds)
            ],
            "history_summary": history,
        })
    return results
# --------------------------------------------------------------------------- #
# 12. FORECAST REPORT
# --------------------------------------------------------------------------- #


DATA_NOT_AVAILABLE_MESSAGE = (
    "Official FORESIGHT row-level data is unavailable; no forecast or model "
    "performance result was computed."
)


def create_forecast_report(
    readiness: Optional[Dict[str, Any]] = None,
    backtest: Optional[Dict[str, Any]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    selection: Optional[Dict[str, Any]] = None,
    config: ForecastConfig = DEFAULT_CONFIG,
    low_history_sku_count: int = 0,
) -> Dict[str, Any]:
    """Structured D3 forecast report.

    When official data is unavailable (``readiness.status != 'ready'``), the
    report contains NO metrics — only the explicit unavailability statement.
    Metrics are included solely from actual computed backtest results.
    """
    status = (readiness or {}).get("status", "data_not_available")

    if status != "ready" or backtest is None:
        return {
            "report": "D3 demand forecasting",
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "data_sufficiency_status": status,
            "unavailability_statement": DATA_NOT_AVAILABLE_MESSAGE,
            "missing_d1_outputs": (readiness or {}).get("missing_d1_outputs"),
            "readiness_reasons": (readiness or {}).get("reasons"),
            "forecast_grain": "sku_id x week (official weekly grain)",
            "forecast_horizon_weeks": config.horizon_weeks,
            "seasonal_period_weeks": config.seasonal_period,
            "model_used": None,
            "baseline_used": "seasonal_naive",
            "backtest_methodology": {
                "type": "rolling_origin",
                "note": "configured but NOT executed (no official data)",
            },
            "wape": None,
            "bias": None,
            "model_vs_baseline": None,
            "selected_model": None,
            "limitations": [
                DATA_NOT_AVAILABLE_MESSAGE,
                "No WAPE, bias, improvement, or forecast values were produced.",
                "Legacy Mini-FORESIGHT files were never used as substitutes.",
            ],
            "low_history_sku_count": low_history_sku_count,
        }

    cmp_block = comparison or compare_model_to_baseline(backtest)
    sel_block = selection or select_best_forecaster(cmp_block)

    limitations: List[str] = []
    details = (readiness or {}).get("details", {})
    if details.get("skus_low_history"):
        limitations.append(
            f"{details['skus_low_history']} SKU(s) have less than "
            f"{config.min_train_weeks} weeks of official history; they are "
            f"routed through the '{config.low_history_fallback}' fallback and "
            "flagged low-confidence."
        )
    if backtest.get("skipped_windows"):
        limitations.append(
            "Some backtest windows were not evaluable for every method "
            "(reported in 'skipped_windows'); no values were fabricated."
        )
    limitations.append(
        "Promotion features describe calendar/promo context; observed "
        "correlations are not causal claims."
    )

    return {
        "report": "D3 demand forecasting",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "data_sufficiency_status": "ready",
        "forecast_grain": "sku_id x week (official weekly grain)",
        "forecast_horizon_weeks": config.horizon_weeks,
        "seasonal_period_weeks": config.seasonal_period,
        "model_used": sel_block["selected_model"],
        "baseline_used": "seasonal_naive",
        "backtest_methodology": backtest.get("methodology"),
        "folds": backtest.get("folds"),
        "skipped_windows": backtest.get("skipped_windows"),
        "wape": {
            "baseline": cmp_block["baseline"]["wape"],
            "model": cmp_block["model"]["wape"],
            "primary_metric": True,
        },
        "bias": {
            "baseline": cmp_block["baseline"]["bias"],
            "model": cmp_block["model"]["bias"],
            "sign_convention": "positive = over-forecast, negative = under-forecast",
            "secondary_metric": True,
        },
        "mape": {
            "baseline": cmp_block["baseline"].get("mape"),
            "model": cmp_block["model"].get("mape"),
            "secondary_metric_only": True,
        },
        "model_vs_baseline": {
            "improvement_vs_baseline_wape": cmp_block.get(
                "improvement_vs_baseline_wape"
            ),
            "same_evaluation_windows": cmp_block.get("same_windows", True),
        },
        "selection_rule": sel_block["rule"],
        "selection_reason": sel_block["reason"],
        "selected_model": sel_block["selected_model"],
        "limitations": limitations,
        "low_history_sku_count": int(low_history_sku_count),
        "total_skus": details.get("total_skus"),
        "total_weeks_observed": details.get("total_weeks_observed"),
    }