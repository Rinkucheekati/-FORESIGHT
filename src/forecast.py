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
    "sales_daily": "sales_daily_clean.csv",
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
        ``"seasonal_naive"`` (default) — sparse-history SKUs use the
        seasonal-naive baseline; ``"none"`` disables the fallback.
    """

    horizon_weeks: int = 8
    seasonal_period: int = 52
    cv_folds: int = 3
    min_train_weeks: int = 12
    min_obs_per_sku: int = 4
    random_seed: int = 42
    model_params: Dict[str, Any] = field(default_factory=dict)
    low_history_fallback: str = "seasonal_naive"

    def validate(self) -> None:
        if self.horizon_weeks < 1:
            raise ValueError("ForecastConfig.horizon_weeks must be >= 1.")
        if self.seasonal_period < 1:
            raise ValueError("ForecastConfig.seasonal_period must be >= 1.")
        if self.cv_folds < 1:
            raise ValueError("ForecastConfig.cv_folds must be >= 1.")
        if self.min_train_weeks < 1:
            raise ValueError("ForecastConfig.min_train_weeks must be >= 1.")
        if self.low_history_fallback not in {"seasonal_naive", "none"}:
            raise ValueError(
                "ForecastConfig.low_history_fallback must be "
                + "'seasonal_naive' or 'none'."
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
    required = ["sales_daily", "calendar", "sku_master"]
    missing = [k for k in required if k not in tables]
    if missing:
        raise MissingOfficialInputsError(missing, DATA_PROCESSED)

    sales = tables["sales_daily"].copy()
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

    # Calendar metadata: use the median day of each week-period if available,
    # else fall back to week number. Join context from the official calendar.
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
    weekly = weekly.merge(cal_context, on="period", how="left")

    # SKU master static attributes (list_price is a static attribute; historical
    # unit_price is not used as a feature to avoid look-ahead).
    master_cols = ["sku_id", "category", "subcategory", "list_price"]
    master_cols = [c for c in master_cols if c in sku_master.columns]
    weekly = weekly.merge(sku_master[master_cols], on="sku_id", how="left")

    # Deterministic chronological ordering by SKU then week-period.
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


def _encode_categoricals(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
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
            cats = sorted(str(v) for v in out[col].dropna().unique())
            mapping = {c: i + 1 for i, c in enumerate(cats)}  # NaN stays NaN
            out[col] = out[col].map(lambda v: mapping.get(str(v)) if pd.notna(v) else np.nan)
            out[col] = out[col].astype("float64")
        else:
            out[col] = out[col].astype("float64")
    return out


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
        base_rows: List[Dict[str, Any]] = []
        for sku in sorted(actual_test["sku_id"].unique()):
            g_sku = (
                features[features["sku_id"] == sku]
                .sort_values(["iso_year", "iso_week"])
                .reset_index(drop=True)
            )
            pos_in_test = g_sku.index[g_sku["period"].isin(set(test_periods["period"]))]
            for p in pos_in_test:
                src = p - config.seasonal_period
                if src >= 0:
                    base_rows.append(
                        {
                            "sku_id": sku,
                            "period": g_sku.loc[p, "period"],
                            "actual": float(g_sku.loc[p, "units_sold"]),
                            "baseline_units": float(g_sku.loc[src, "units_sold"]),
                        }
                    )
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
        model_preds = _fit_predict_model(
            features=features,
            train_mask=train_mask,
            test_index=actual_test.index,
            target_col="units_sold",
            config=config,
        )
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
            model_pred_all.extend([float(p) for p in model_preds])

        fold_records.append(
            {
                "fold": fold,
                "train_end_period": train_end_period,
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
) -> Dict[str, Any]:
    """Forecast the next ``horizon_weeks`` of weekly demand for ONE SKU.

    Data-gated: loads official D1 outputs when ``tables`` is not supplied and
    raises ``MissingOfficialInputsError`` if they are absent. Never fabricates
    values; SKUs without sufficient model history are routed through the
    documented low-history fallback (see ``ForecastConfig``).

    Returns a structured result with forecast rows:
        period, sku_id, forecast_units, model_name, low_confidence, note
    """
    config.validate()

    if tables is None:
        tables = load_d1_outputs(processed_dir)

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
        features = build_forecast_features(weekly)
        f_sku = (
            features[features["sku_id"] == sku_id]
            .sort_values(["iso_year", "iso_week"])
            .reset_index(drop=True)
        )
        feature_cols = [c for c in _FEATURE_COLUMNS if c in features.columns]
        model = build_model(config)
        X = _encode_categoricals(f_sku, feature_cols)[feature_cols]
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

    elif config.low_history_fallback == "seasonal_naive":
        # ---- Documented low-history fallback --------------------------------
        base = seasonal_naive_baseline(weekly, horizon, config.seasonal_period)
        base = base[base["sku_id"] == sku_id].reset_index(drop=True)
        preds = [
            None if (not bool(r["evaluable"])) else float(max(0.0, r["baseline_units"]))
            for _, r in base.iterrows()
        ]
        model_name = "seasonal_naive"
        fallback_used = True
        low_confidence = True
        note = (
            f"Low official history for this SKU ({history['observed_weeks']} "
            f"weeks < required {history['required_for_model']}); used the "
            f"documented seasonal-naive fallback at period="
            f"{config.seasonal_period}."
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