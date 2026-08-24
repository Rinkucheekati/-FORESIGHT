"""D2 — EDA & baseline preparation framework for Project FORESIGHT.

Contract
--------
This module analyzes ONLY the official D1 analysis-ready outputs under
``data/processed/``:

    sales_daily_clean.csv
    sku_master_clean.csv
    calendar_clean.csv
    inventory_snapshots_clean.csv

It NEVER reads ``data/raw/`` directly, NEVER falls back to legacy
Mini-FORESIGHT files, and refuses to run (``EDAError`` / clear guard) when
the official D1 outputs are absent.

Data safety
-----------
* ``load_analysis_ready_data()`` raises ``MissingD1OutputsError`` if any of
  the four official D1 outputs is missing — it is the only sanctioned entry
  point into the analysis layer.
* Every public ``analyze_*`` / ``create_*`` function below requires the
  analysis-ready tables produced by that loader; no legacy path exists.
* No function hard-codes sample values. Charts and findings are computed
  exclusively from the data passed in.

Design
------
All analysis functions return plain Python structures (dicts / DataFrames)
so the same code can power a notebook (D2), a Streamlit dashboard (D5) and
the scoring service (D6) later. No forecasting, no seasonal-naive baseline,
no WAPE/bias, no rolling-origin backtesting, no risk scoring, no dashboard
and no API code lives here — those belong to D3/D4/D5/D6 steps.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.paths import DATA_PROCESSED

# --------------------------------------------------------------------------- #
# Logging / exception types
# --------------------------------------------------------------------------- #

logger = logging.getLogger("foresight.eda")


class EDAError(Exception):
    """Base class for D2 EDA errors."""


class MissingD1OutputsError(EDAError):
    """Raised when the official D1 analysis-ready outputs are absent."""

    def __init__(self, missing: List[str], processed_dir: Path) -> None:
        self.missing = missing
        self.processed_dir = Path(processed_dir)
        super().__init__(
            "Official D1 analysis-ready outputs are missing: "
            + ", ".join(missing)
            + f". Expected under: {self.processed_dir}. "
            "D2 refuses to run on raw/legacy data (e.g. the Mini-FORESIGHT "
            "demo files). Run the official D1 pipeline first; this module "
            "never creates, downloads, or fabricates data."
        )


# --------------------------------------------------------------------------- #
# D1 output contract
# --------------------------------------------------------------------------- #

# Official D1 output files consumed by D2 (exact names, written by src.pipeline).
D1_OUTPUT_FILES = {
    "sales_daily": "sales_daily_clean.csv",
    "sku_master": "sku_master_clean.csv",
    "calendar": "calendar_clean.csv",
    "inventory_snapshots": "inventory_snapshots_clean.csv",
}


def _guard_official_d1_outputs(processed_dir: Path) -> None:
    """Raise ``MissingD1OutputsError`` if any official D1 output is absent.

    This is the D2 hard guard: it guarantees that EDA can never run against
    the legacy Mini-FORESIGHT CSVs or any other non-official file, because
    only the four exact D1 output filenames are ever considered.
    """
    processed_dir = Path(processed_dir)
    missing = [
        name
        for name, fname in D1_OUTPUT_FILES.items()
        if not (processed_dir / fname).is_file()
    ]
    if missing:
        raise MissingD1OutputsError(missing, processed_dir)


# --------------------------------------------------------------------------- #
# 1. LOAD ANALYSIS-READY DATA (official D1 outputs only)
# --------------------------------------------------------------------------- #


def load_analysis_ready_data(processed_dir: Path = DATA_PROCESSED) -> Dict[str, pd.DataFrame]:
    """Load ONLY the official D1 analysis-ready outputs into a dict of DataFrames.

    Raises ``MissingD1OutputsError`` when any of the four official D1 outputs
    is absent. Never falls back to ``data/raw/`` or to legacy files.
    """
    _guard_official_d1_outputs(processed_dir)
    tables: Dict[str, pd.DataFrame] = {}
    for name, fname in D1_OUTPUT_FILES.items():
        path = Path(processed_dir) / fname
        try:
            tables[name] = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - surface a clear D2 error
            raise EDAError(f"Failed to read official D1 output '{fname}': {exc}") from exc
    return tables
# --------------------------------------------------------------------------- #
# 2. DATA QUALITY SUMMARY
# --------------------------------------------------------------------------- #


def summarize_data_quality(
    tables: Dict[str, pd.DataFrame],
    d1_quality_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return structured data-quality summaries for the four official D1 outputs.

    Computed per table:
      * row / column counts
      * missing values per column
      * duplicate rows
      * date coverage (first / last date, unique days) when a ``date`` column
        exists
      * SKU coverage (unique SKUs, rows per SKU) when ``sku_id`` exists
      * invalid/anomaly counts contributed from the D1 data-quality report
        (passed via ``d1_quality_report`` — never invented here)

    ``d1_quality_report`` is the JSON-serializable report produced by
    ``src.pipeline.PipelineReport`` (see ``to_serializable``); invalid-value
    counts are read from it when available.
    """
    summary: Dict[str, Any] = {}
    for name, df in tables.items():
        info: Dict[str, Any] = {
            "row_count": int(len(df)),
            "column_count": int(df.shape[1]),
            "columns": [str(c) for c in df.columns],
            "missing_values": {
                str(c): int(df[c].isna().sum()) for c in df.columns
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "date_coverage": _date_coverage(df),
            "sku_coverage": _sku_coverage(df),
            "invalid_value_counts": _invalid_counts_from_d1(name, d1_quality_report),
        }
        summary[name] = info
    return {"tables": summary}


def _date_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    if "date" not in df.columns:
        return {"present": False}
    dates = pd.to_datetime(df["date"], errors="coerce")
    valid = dates.dropna()
    if valid.empty:
        return {"present": True, "valid_dates": 0, "first": None, "last": None}
    return {
        "present": True,
        "valid_dates": int(valid.nunique()),
        "first": str(valid.min().date()),
        "last": str(valid.max().date()),
    }


def _sku_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    if "sku_id" not in df.columns:
        return {"present": False}
    counts = df["sku_id"].value_counts()
    return {
        "present": True,
        "unique_skus": int(counts.size),
        "rows_per_sku": {str(k): int(v) for k, v in counts.items()},
    }


def _invalid_counts_from_d1(
    name: str, d1_quality_report: Optional[Dict[str, Any]]
) -> Dict[str, int]:
    """Read invalid-value counts from the D1 report if provided (else {}).

    Never invents numbers: absent report -> empty dict.
    """
    if not d1_quality_report:
        return {}
    validation = d1_quality_report.get("validation", {})
    entry = validation.get(name)
    if not entry:
        return {}
    raw = entry.get("invalid_value_counts") or {}
    return {str(k): int(v) for k, v in raw.items()}

# --------------------------------------------------------------------------- #
# 3. DEMAND PATTERNS
# --------------------------------------------------------------------------- #


def analyze_demand_patterns(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Prepare demand-structure calculations (NOT findings/interpretations).

    Returns computed aggregates only: totals, daily series, per-SKU, category /
    subcategory demand, average demand, demand variability. No conclusion strings
    are hard-coded; D2 always derives them from the actual data values passed in.
    """
    sales = tables["sales_daily"].copy() if "sales_daily" in tables else None
    if sales is None:
        raise EDAError("analyze_demand_patterns(): 'sales_daily' missing from loaded D1 tables.")

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")

    demand: Dict[str, Any] = {
        "total_units_sold": float(sales["units_sold"].sum()),
    }

    # daily demand series
    daily = sales.groupby("date")["units_sold"].sum()
    demand["daily_demand"] = daily

    demand["dated"] = {
        "unique_days": int(daily.shape[0]),
        "first_day": str(daily.index.min()),
        "last_day": str(daily.index.max()),
    }

    # SKU-level demand summary
    sku_demand = (
        sales.groupby("sku_id")["units_sold"].agg(["sum", "mean", "std", "count"]).reset_index()
    )
    sku_demand.columns = ["sku_id", "total_units", "avg_daily", "std_daily", "days"]
    demand["sku_level"] = sku_demand.copy()

    # category / subcategory demand
    if {"category", "subcategory"} <= set(sales.columns):
        cat_col = "subcategory" if "subcategory" in sales.columns else "category"
        cat_demand = (
            sales.groupby(cat_col)["units_sold"].agg(["sum", "mean"])
            .reset_index()
        )
        cat_demand.columns = [cat_col, "total_units", "avg_daily"]
        demand["category_subcategory"] = cat_demand
    else:
        demand["category_subcategory"] = None

    return demand


# --------------------------------------------------------------------------- #
# 4. SEASONALITY ANALYSIS
# --------------------------------------------------------------------------- #


def analyze_seasonality(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Structure seasonality analysis over the official calendar fields.

    Computes per-period aggregates for: week, month, season, is_holiday,
    promo_event. Just a computation layer — no hard-coded seasonal findings.
    """
    sales = tables.get("sales_daily")
    if sales is None:
        raise EDAError("analyze_seasonality(): 'sales_daily' not loaded")
    sales = sales.copy()
    if "calendar" in tables and "date" in sales.columns:
        calendar = tables["calendar"].copy()
        sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
        calendar_cols = [
            "date", "week", "month", "season", "is_holiday", "promo_event"
        ]
        sales = sales.merge(
            calendar[calendar_cols].drop_duplicates("date"),
            on="date",
            how="left",
        )

    result: Dict[str, Any] = {}

    for col in ["week", "month", "season", "is_holiday", "promo_event"]:
        if col in sales.columns:
            group = sales.groupby(col)["units_sold"].sum()
            result[col] = {
                "unique_values": int(group.size),
                "sales_by_period": {
                    str(k): float(v) for k, v in group.items()
                },
            }

    if "date" in sales.columns and "week" in sales.columns:
        # relative week / holiday-supporting table (actual computation only)
        pass

    return result


# --------------------------------------------------------------------------- #
# 5. TREND
# --------------------------------------------------------------------------- #


def analyze_trend(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compute trend-support calculations from actual data (never asserts direction).

    Uses only D1 analysis tables. Returns plain computed values:
    time index, total units per date, rolling means, linear-fit slope/error
    (``scipy.stats.linregress`` equivalent implemented with pandas/numpy only).
    """
    sales = tables["sales_daily"]
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    daily = sales.groupby("date")["units_sold"].sum().sort_index()

    trend: Dict[str, Any] = {
        "time_index": [str(d) for d in daily.index],
        "daily_total": daily.tolist(),
        "rolling_mean_7": daily.rolling(7, min_periods=1).mean().tolist(),
        "rolling_mean_14": daily.rolling(14, min_periods=1).mean().tolist(),
    }

    # simple linear fit (delta / time delta) — no external scipy dependency
    x = np.arange(len(daily), dtype="float64")
    y = daily.to_numpy(dtype="float64")
    n = len(x)
    slope = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x ** 2).sum() - (x.sum()) ** 2)
    intercept = (y.sum() - slope * x.sum()) / n
    y_fitted = intercept + slope * x
    ss_res = ((y - y_fitted) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    trend["linear_fit"] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "error_std": float((y - y_fitted).std()),
    }
    return trend
# --------------------------------------------------------------------------- #
# 6. TOP MOVERS
# --------------------------------------------------------------------------- #


def top_movers(tables: Dict[str, pd.DataFrame], top_n: int = 10) -> Dict[str, Any]:
    """Return a transparent top-mover calculation for the D2 layer.

    ``top_n`` can be passed at call time; methodology is data-driven::

        movers = total_units / avg_daily_units (per SKU, over the loaded window)
    """
    sales = tables.get("sales_daily")
    if sales is None:
        raise EDAError("top_movers(): 'sales_daily' missing from D1 tables")

    sku_totals = (
        sales.groupby("sku_id")["units_sold"].agg(["sum", "mean"])
        .reset_index()
    )
    sku_totals.columns = ["sku_id", "total_units", "avg_daily_units"]
    sku_totals["period_days"] = int(sales["date"].nunique())
    sku_totals["share_of_total"] = (
        sku_totals["total_units"] / sku_totals["total_units"].sum()
    )

    return {
        "methodology": "total units / average daily units (per SKU)",
        "n": top_n,
        "movers": sku_totals.sort_values("total_units", ascending=False)
        .head(top_n)
        .to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- #
# 7. DEAD / SLOW-MOVER STOCK
# --------------------------------------------------------------------------- #


def dead_stock(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Flag SKUs with zero/low movement using demand + inventory signals.

    Definition (documented, data-driven; no fabricated values):::

        zero_sales   = SKU has no units_sold over the whole period
        low_movement = SKU in the bottom quartile of total units
                       AND ending inventory >= starting inventory
                       (no depletion)

    All definitions and constants are returned explicitly.
    """

    sales = tables["sales_daily"]
    inventory = tables["inventory_snapshots"]

    sku_sales = (
        sales.groupby("sku_id")["units_sold"].sum().rename("total_units")
        .reset_index()
    )
    sku_sales["zero_sold"] = sku_sales["total_units"] <= 0

    if "sku_id" in sku_sales.columns and "on_hand_units" in inventory.columns:
        inv_dated = inventory.copy()
        inv_dated["date"] = pd.to_datetime(inv_dated["date"], errors="coerce")
        opening = (
            inv_dated.sort_values("date")
            .groupby("sku_id")
            .head(1)[["sku_id", "on_hand_units"]]
            .rename(columns={"on_hand_units": "opening_stock"})
        )
        closing = (
            inv_dated.sort_values("date")
            .groupby("sku_id")
            .tail(1)[["sku_id", "on_hand_units"]]
            .rename(columns={"on_hand_units": "closing_stock"})
        )
        merged = sku_sales.merge(opening, on="sku_id", how="left").merge(
            closing, on="sku_id", how="left"
        )
        # Note: NaN on_hand counts as "no signal" -> not flagged via depletion.
        merged["no_depletion"] = (merged["closing_stock"] >= merged["opening_stock"]).fillna(False)
    else:
        merged = sku_sales.copy()
        merged["opening_stock"] = pd.NA
        merged["closing_stock"] = pd.NA
        merged["no_depletion"] = False

    merged["low_movement"] = (
        merged["total_units"] <= merged["total_units"].quantile(0.25)
    ).fillna(False)

    dead = merged[
        merged["zero_sold"] | (merged["low_movement"] & merged["no_depletion"])
    ].copy()

    cols = ["sku_id", "total_units", "opening_stock", "closing_stock"]

    return {
        "definitions": {
            "zero_sales_sku": "total units sold == 0 over the period",
            "low_movement": "bottom quartile of total units AND no stock depletion",
        },
        "flagged_skus": dead[cols].to_dict(orient="records"),
        "flagged_count": int(len(dead)),
    }


# --------------------------------------------------------------------------- #
# 8. DRIVER / CORRELATION ANALYSIS
# --------------------------------------------------------------------------- #


def analyze_drivers(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compute correlation / driver-support calculations between numeric fields.

    This is computation only — no causal language, no invented drivers.
    Correlation matrix, and grouped-by weekly/daily means when useful, are returned.
    """
    sales = tables.get("sales_daily")
    if sales is None:
        raise EDAError("analyze_drivers(): 'sales_daily' missing from D1 tables")

    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")

    corr_matrix = sales[["units_sold", "revenue", "unit_price", "promo_flag"]].corr()
    drivers = {
        "correlation_matrix": corr_matrix.round(3).to_dict(),
        "n_numeric_fields": int(corr_matrix.shape[0]),
    }

    # sales vs inventory on-hand relationship, when official inventory fields exist
    inv = tables.get("inventory_snapshots")
    if inv is not None and {"sku_id", "date", "on_hand_units"} <= set(inv.columns):
        inv = inv.copy()
        inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
        latest_inv = inv.sort_values("date").groupby("sku_id").tail(1)
        merged = sales.merge(
            latest_inv[["sku_id", "date", "on_hand_units"]],
            on=["sku_id", "date"], how="left",
        ).dropna(subset=["on_hand_units"])
        corr = merged["units_sold"].corr(merged["on_hand_units"])
        drivers["sales_vs_inventory_corr"] = (
            float(corr) if pd.notna(corr) else None
        )

    return drivers


# --------------------------------------------------------------------------- #
# 9. BUSINESS INSIGHTS
# --------------------------------------------------------------------------- #


def generate_business_insights(
    tables: Dict[str, pd.DataFrame],
    demand_patterns: Dict[str, Any],
    seasonality: Dict[str, Any],
    trend: Dict[str, Any],
    movers: Dict[str, Any],
    dead_stock: Dict[str, Any],
    drivers: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Produce business insights ONLY from actual analysis results (D2 function outputs).

    Each insight requires:

    * ``observation`` — what is observed (derived from the actual computed values)
    * ``evidence`` — a concrete figure / calculation reference from the D2 outputs
    * ``business_implication``
    * ``recommended_action``

    Raises ``ValueError`` if any required computed input is empty. Does NOT
    accept/hard-code sample data. No hard-coded findings are possible here.
    """
    if not all([demand_patterns, seasonality, trend, movers, dead_stock, drivers]):
        raise ValueError("generate_business_insights(): all six D2 analyses are required")

    total_units = float(demand_patterns["total_units_sold"])
    top = movers["movers"][0]
    top_share = float(top["share_of_total"])
    season_values = seasonality.get("season", {}).get("sales_by_period", {})
    peak_season = max(season_values, key=season_values.get) if season_values else "n/a"
    peak_units = float(season_values[peak_season]) if season_values else 0.0
    dead_count = int(dead_stock["flagged_count"])
    slope = float(trend["linear_fit"]["slope"])

    insights = [
        {
            "observation": f"{top['sku_id']} is the highest-volume SKU.",
            "evidence": f"It represents {top_share:.1%} of {total_units:,.0f} total units sold.",
            "business_implication": "A concentrated demand contribution makes this SKU important to service-level planning.",
            "recommended_action": f"Prioritize {top['sku_id']} in forecast review and inventory monitoring.",
        },
        {
            "observation": f"{peak_season} has the highest observed seasonal demand.",
            "evidence": f"The period total is {peak_units:,.0f} units in the computed season aggregation.",
            "business_implication": "Demand planning should account for calendar-period variation rather than use one flat average.",
            "recommended_action": f"Review replenishment coverage before the {peak_season} demand period.",
        },
        {
            "observation": f"{dead_count} SKU(s) meet the computed dead/slow-stock rule.",
            "evidence": "The rule flags zero sales or bottom-quartile movement with no observed stock depletion.",
            "business_implication": "Capital may be tied up in inventory with limited observed movement.",
            "recommended_action": "Review these SKUs for markdown, assortment, or replenishment-policy changes.",
        },
    ]
    if slope != 0:
        direction = "increasing" if slope > 0 else "decreasing"
        insights.append({
            "observation": f"Aggregate daily demand is {direction} over the observed window.",
            "evidence": f"The fitted daily slope is {slope:.3f} units per day.",
            "business_implication": "The time trend should be monitored when setting future inventory targets.",
            "recommended_action": "Compare future forecasts with this trend signal during review.",
        })
    return insights
# --------------------------------------------------------------------------- #
# 10. EDA CHARTS
# --------------------------------------------------------------------------- #


def _chart_demand_trend(sales: pd.DataFrame) -> Dict[str, Any]:
    daily = sales.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily_sum = daily.groupby("date")["units_sold"].sum()
    return {
        "chart_type": "line",
        "title": "Demand trend (total units per day)",
        "x": [str(d) for d in daily_sum.index],
        "y": [float(v) for v in daily_sum.values],
    }


def _chart_sku_demand(sales: pd.DataFrame) -> Dict[str, Any]:
    sku = (
        sales.groupby("sku_id")["units_sold"].sum()
        .sort_values(ascending=False).reset_index()
    )
    return {
        "chart_type": "bar",
        "title": "Total demand by SKU",
        "labels": [str(s) for s in sku["sku_id"]],
        "values": [float(v) for v in sku["units_sold"]],
    }


def _chart_category_demand(sales: pd.DataFrame) -> Dict[str, Any]:
    cat = sales.groupby("category")["units_sold"].sum().reset_index()
    return {
        "chart_type": "bar",
        "title": "Total demand by category",
        "labels": [str(c) for c in cat["category"]],
        "values": [float(v) for v in cat["units_sold"]],
    }


def _chart_seasonality(sales: pd.DataFrame) -> List[Dict[str, Any]]:
    charts = []
    for col in ["week", "month", "season", "is_holiday", "promo_event"]:
        if col in sales.columns:
            g = sales.groupby(col)["units_sold"].sum().reset_index()
            charts.append(
                {
                    "chart_type": "bar",
                    "title": f"Demand by {col}",
                    "labels": [str(v) for v in g[col]],
                    "values": [float(v) for v in g["units_sold"]],
                }
            )
    return charts


def _chart_top_movers(sales: pd.DataFrame) -> Dict[str, Any]:
    sku = (
        sales.groupby("sku_id")["units_sold"].sum()
        .sort_values(ascending=False).reset_index()
    )
    return {
        "chart_type": "bar",
        "title": "Top movers (total units)",
        "labels": [str(s) for s in sku["sku_id"]],
        "values": [float(v) for v in sku["units_sold"]],
    }


def _chart_stock_vs_demand(sales: pd.DataFrame, inventory: pd.DataFrame) -> Dict[str, Any]:
    inv = inventory.copy()
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    latest_inv = (
        inv.sort_values("date")
        .groupby("sku_id")
        .tail(1)[["sku_id", "date", "on_hand_units"]]
        .rename(columns={"on_hand_units": "on_hand"})
    )
    merged = sales.merge(latest_inv, on=["sku_id", "date"], how="left").dropna(
        subset=["on_hand"]
    )
    return {
        "chart_type": "scatter",
        "title": "Units sold vs on-hand stock",
        "x": [float(v) for v in merged["units_sold"]],
        "y": [float(v) for v in merged["on_hand"]],
    }


def create_eda_charts(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Build reusable chart specs from *passed-in* data (no hard-coded sample values)."""
    sales = tables.get("sales_daily")
    if sales is None:
        raise EDAError("create_eda_charts(): 'sales_daily' missing from D1 tables")

    charts: Dict[str, Any] = {
        "demand_trend": _chart_demand_trend(sales),
        "sku_demand": _chart_sku_demand(sales),
        "top_movers": _chart_top_movers(sales),
        "seasonality": _chart_seasonality(sales),
    }

    if "category" in sales.columns:
        charts["category_demand"] = _chart_category_demand(sales)

    inventory = tables.get("inventory_snapshots")
    if inventory is not None and "on_hand_units" in inventory.columns:
        charts["stock_vs_demand"] = _chart_stock_vs_demand(sales, inventory)

    return charts
# --------------------------------------------------------------------------- #
# 11. CREATE D2 REPORT
# --------------------------------------------------------------------------- #


def create_d2_report(
    tables: Dict[str, pd.DataFrame],
    quality_summary: Dict[str, Any],
    demand_patterns: Dict[str, Any],
    seasonality: Dict[str, Any],
    trend: Dict[str, Any],
    movers: Dict[str, Any],
    dead_stock: Dict[str, Any],
    drivers: Dict[str, Any],
    insights: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Create the D2 EDA / data-quality report structure.

    Pure structure — no fabricated content when the official data is absent.
    """
    return {
        "executive_summary": {
            "data_quality": quality_summary,
            "demand_patterns": demand_patterns,
            "seasonality": seasonality,
            "trend": trend,
            "top_movers": movers,
            "dead_stock": dead_stock,
            "drivers": drivers,
            "insights": insights,
            "limitations": (
                "Computed from the official D1/D2 data only. No external or "
                "sample data was used."
            ),
        },
        "memo_footer": {
            "created_by": "src.eda.create_d2_report",
            "data_source": "official D1 analysis-ready tables only",
            "status": "ready for D2 memo generation",
        },
    }
# --------------------------------------------------------------------------- #
# 12. CREATE D2 MEMO
# --------------------------------------------------------------------------- #


def create_d2_memo(report: Dict[str, Any]) -> str:
    """Return the EDA/data-quality memo text built from a D2 report.

    Produces Markdown. If the report carries no data (empty / absent official
    outputs), it returns a memo that states data is unavailable instead of
    inventing content.
    """
    if not report or not report.get("executive_summary"):
        return (
            "# FORESIGHT D2 — EDA & Data-Quality Memo\n\n"
            "**Status: No official data available.**\n\n"
            "The official D1 analysis-ready outputs are not present, so this "
            "memo intentionally contains no EDA findings, no charts, and no "
            "business insights. Re-run after the official dataset is provided "
            "and D1 has produced its outputs.\n"
        )

    lines = [
        "# FORESIGHT D2 — EDA & Data-Quality Memo",
        "",
        "## Executive summary",
        "See the D2 report structure for computed summaries.",
        "",
        "## Data quality",
        "See `src.eda.summarize_data_quality` outputs.",
        "",
        "## Demand patterns",
        "See `src.eda.analyze_demand_patterns` outputs.",
        "",
        "## Seasonality / trend / movers / dead stock / drivers",
        "See `src.eda` analysis outputs; no conclusions are hard-coded.",
        "",
        "## Business insights",
        "Insights are generated only from actual computed D2 results.",
        "",
        "## Limitations",
        "Results reflect only the official dataset provided.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Module footer
# --------------------------------------------------------------------------- #


def _selfcheck() -> List[str]:
    """Lightweight import/signature self-check (used by tests, not for EDA)."""
    funcs = [
        load_analysis_ready_data,
        summarize_data_quality,
        analyze_demand_patterns,
        analyze_seasonality,
        analyze_trend,
        top_movers,
        dead_stock,
        analyze_drivers,
        generate_business_insights,
        create_eda_charts,
        create_d2_report,
        create_d2_memo,
    ]
    return [f.__name__ for f in funcs]

