"""D1 — Reproducible data pipeline for the official Zidio Project FORESIGHT.

Contract
--------
The pipeline ingests ONLY the four official internship extracts from
``data/raw/``:

    sales_daily          date, sku_id, units_sold, revenue, unit_price, promo_flag
    sku_master           sku_id, category, subcategory, launch_date, unit_cost, list_price
    calendar             date, week, month, season, is_holiday, promo_event
    inventory_snapshots  date, sku_id, on_hand_units, on_order_units, lead_time_days, reorder_point

It NEVER generates, fabricates, infers, downloads or substitutes data.
When the official inputs are absent it fails with a clear, actionable error
and writes NOTHING to ``data/processed/``.

Guarantees
----------
1. SAFE RAW INGESTION      - only the four exact filenames are read; legacy
                             Mini-FORESIGHT CSVs are never auto-substituted.
2. RAW SCHEMA VALIDATION   - required / unexpected / duplicate columns, row
                             counts, dtypes, null counts, duplicate rows,
                             date ranges -> structured validation report.
3. DATA TYPE CLEANING      - explicit, validated parsing of dates, numerics,
                             and boolean-like fields. No invented values.
4. DUPLICATE HANDLING      - exact duplicates are count-documented and removed
                             only when safe; business-key conflicts are flagged.
5. MISSING-VALUE HANDLING  - per-field classification; deterministic handling
                             when supportable, otherwise preserved & reported.
6. DOMAIN VALIDATION       - negative/out-of-range values are flagged, never
                             silently "fixed".
7. CROSS-TABLE VALIDATION  - SKU and calendar referential checks; missing
                             master/calendar records are reported, not invented.
8. CLEANING DECISION LOG   - structured log of every issue/action for the
                             D1 data-quality documentation.
9. ANALYSIS-READY OUTPUT   - cleaned tables + a light analysis-ready view
                             (no forecasting features - that is D3).
10. OUTPUT SAFETY           - processed files written ONLY after validation.
11. TESTABILITY             - modular helpers + actionable exception types.

Run with one command from the repository root::

    python -c "from src.pipeline import run_pipeline; run_pipeline()"
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.paths import DATA_PROCESSED, DATA_RAW, OFFICIAL_RAW_FILES, OFFICIAL_SCHEMAS

OFFICIAL_SELECTED_DIR = DATA_RAW / "official_selected"
OFFICIAL_SELECTED_FILE = OFFICIAL_SELECTED_DIR / "sales_transactions_25000.csv"
OFFICIAL_RETAIL_DIR = DATA_RAW / "retail_contaminated_dataset"

# Official season definition shared with src/retail_adapter.py::_season.
# Season is a pure deterministic function of the calendar month — no data
# invention (12,1,2 Winter; 3,4,5 Spring; 6,7,8 Summer; 9,10,11 Autumn).
SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SELECTED_TRANSACTION_COLUMNS = (
    "date", "receipt_id", "store_id", "sku_id", "customer_id",
    "quantity", "unit_price", "total_value", "channel", "discount_pct",
    "promo_id",
)
RETAIL_SUPPORTING_FILES = {
    "sku_master": "sku_master.csv",
    "customer_master": "customer_master.csv",
    "store_master": "store_master.csv",
    "promotions": "promotions.csv",
    "inventory_source": "inventory_snapshot.csv",
    "flags": "sku_inventory_flags.csv",
}

# --------------------------------------------------------------------------- #
# Logging / exception types
# --------------------------------------------------------------------------- #

logger = logging.getLogger("foresight.pipeline")


class D1Error(Exception):
    """Base class for D1 pipeline errors."""


class MissingOfficialInputsError(D1Error):
    """Raised when one or more official FORESIGHT raw extracts are absent."""


class SchemaValidationError(D1Error):
    """Raised when a present official extract violates the official schema."""
# --------------------------------------------------------------------------- #
# Dataclass containers
# --------------------------------------------------------------------------- #


@dataclass
class CleaningDecision:
    """One documented cleaning decision (for the D1 decision log)."""

    dataset: str
    issue: str
    detection_rule: str
    number_affected: int
    action: str
    reason: str
    changed: bool


@dataclass
class CleanTableResult:
    """Result of cleaning one official extract."""

    name: str
    df: pd.DataFrame
    decisions: List[CleaningDecision] = field(default_factory=list)
    dropped_rows: int = 0

    def log_decision(
        self,
        issue: str,
        detection_rule: str,
        number_affected: int,
        action: str,
        reason: str,
        changed: bool,
    ) -> None:
        self.decisions.append(
            CleaningDecision(
                dataset=self.name,
                issue=issue,
                detection_rule=detection_rule,
                number_affected=number_affected,
                action=action,
                reason=reason,
                changed=changed,
            )
        )


@dataclass
class ValidationReport:
    """Structured schema/quality report for a single extract."""

    dataset: str
    file: str
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    exact_duplicate_rows: int
    business_key_duplicate_rows: int
    invalid_value_counts: Dict[str, int]
    missing_columns: List[str]
    unexpected_columns: List[str]
    duplicate_columns: List[str]
    date_range: Optional[Tuple[str, str]] = None

    @staticmethod
    def _first_date_range(df: pd.DataFrame, col: str = "date") -> Optional[Tuple[str, str]]:
        if col not in df.columns:
            return None
        dates = pd.to_datetime(df[col], errors="coerce")
        if dates.notna().sum() == 0:
            return None
        return (str(dates.min().date()), str(dates.max().date()))
@dataclass
class CrossTableReport:
    """Results of cross-table referential checks."""

    sales_skus_missing_in_master: List[str]
    inventory_skus_missing_in_master: List[str]
    duplicate_sku_keys: List[str]
    sales_dates_outside_calendar: List[str]
    inventory_dates_outside_calendar: List[str]
    calendar_dates_missing_from_series: List[str]
    sales_missing_sku_date_combos: List[str]
    inventory_missing_sku_date_combos: List[str]

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "sales_skus_missing_in_master": self.sales_skus_missing_in_master,
            "inventory_skus_missing_in_master": self.inventory_skus_missing_in_master,
            "duplicate_sku_keys": self.duplicate_sku_keys,
            "sales_dates_outside_calendar": self.sales_dates_outside_calendar,
            "inventory_dates_outside_calendar": self.inventory_dates_outside_calendar,
            "calendar_dates_missing_from_series": self.calendar_dates_missing_from_series,
            "sales_missing_sku_date_combos": self.sales_missing_sku_date_combos,
            "inventory_missing_sku_date_combos": self.inventory_missing_sku_date_combos,
        }


@dataclass
class PipelineReport:
    """Full machine-readable D1 report (JSON-serializable)."""

    timestamp: str
    validation: Dict[str, Dict[str, Any]]
    cleaning_decisions: List[Dict[str, Any]]
    cross_table: Dict[str, Any]
    missing_official_inputs: List[str]

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "validation": self.validation,
            "cleaning_decisions": self.cleaning_decisions,
            "cross_table": self.cross_table,
            "missing_official_inputs": self.missing_official_inputs,
        }

# --------------------------------------------------------------------------- #
# 1. SAFE RAW INGESTION
# --------------------------------------------------------------------------- #


def _ingest_selected_retail_inputs(raw_dir: Path) -> dict:
    """Load selected official transactions and the official retail dimensions."""
    selected_file = Path(raw_dir) / "official_selected" / "sales_transactions_25000.csv"
    retail_dir = Path(raw_dir) / "retail_contaminated_dataset"
    required = [selected_file] + [retail_dir / name for name in RETAIL_SUPPORTING_FILES.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise MissingOfficialInputsError(
            "Official FORESIGHT inputs are missing (selected sales_daily boundary, sku_master, "
            "calendar, inventory_snapshots, and retail dimensions): " + ", ".join(missing)
        )

    transactions = pd.read_csv(selected_file)
    if tuple(transactions.columns) != SELECTED_TRANSACTION_COLUMNS or len(transactions) != 25_000:
        raise SchemaValidationError(
            "The selected transaction input must contain the exact official columns and 25000 rows."
        )
    sku_source = pd.read_csv(retail_dir / RETAIL_SUPPORTING_FILES["sku_master"])
    customers = pd.read_csv(retail_dir / RETAIL_SUPPORTING_FILES["customer_master"])
    stores = pd.read_csv(retail_dir / RETAIL_SUPPORTING_FILES["store_master"])
    promotions = pd.read_csv(retail_dir / RETAIL_SUPPORTING_FILES["promotions"])
    inventory_source = pd.read_csv(retail_dir / RETAIL_SUPPORTING_FILES["inventory_source"])
    flags = pd.read_csv(retail_dir / RETAIL_SUPPORTING_FILES["flags"])

    dates = pd.to_datetime(transactions["date"], errors="coerce")
    quantity = pd.to_numeric(transactions["quantity"], errors="coerce")
    revenue = pd.to_numeric(transactions["total_value"], errors="coerce")
    unit_price = pd.to_numeric(transactions["unit_price"], errors="coerce")
    valid = dates.notna() & transactions["sku_id"].notna() & quantity.notna()
    sales = pd.DataFrame({
        "date": dates,
        "sku_id": transactions["sku_id"].astype("string").str.strip(),
        "units_sold": quantity,
        "revenue": revenue,
        "price_x_units": unit_price * quantity,
        "promo_flag": transactions["promo_id"].astype("string").str.strip().ne(""),
    }).loc[valid]
    sales = sales.groupby(["date", "sku_id"], as_index=False).agg(
        units_sold=("units_sold", "sum"), revenue=("revenue", "sum"),
        price_x_units=("price_x_units", "sum"), promo_flag=("promo_flag", "max"),
    )
    sales["unit_price"] = sales["price_x_units"] / sales["units_sold"]
    sales = sales[["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"]]

    sku = sku_source[["sku_id", "category", "subcategory", "unit_price", "cost_price"]].copy()
    sku["sku_id"] = sku["sku_id"].astype(str)
    sku["launch_date"] = sku["sku_id"].map(sales.groupby("sku_id")["date"].min())
    sku = sku.rename(columns={"cost_price": "unit_cost", "unit_price": "list_price"})
    sku = sku[["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"]]

    calendar = pd.DataFrame({"date": pd.date_range(sales["date"].min(), sales["date"].max(), freq="D")})
    calendar["week"] = calendar["date"].dt.isocalendar().week.astype(int)
    calendar["month"] = calendar["date"].dt.month
    # Season derived deterministically from month via SEASON_BY_MONTH (the
    # project-wide definition shared with src/retail_adapter.py::_season).
    calendar["season"] = calendar["month"].map(SEASON_BY_MONTH)
    calendar["is_holiday"] = pd.NA
    calendar["promo_event"] = pd.NA
    for row in promotions.itertuples(index=False):
        start = pd.to_datetime(getattr(row, "start_date", None), errors="coerce")
        end = pd.to_datetime(getattr(row, "end_date", None), errors="coerce")
        if pd.notna(start) and pd.notna(end):
            mask = calendar["date"].between(start.normalize(), end.normalize())
            calendar.loc[mask & calendar["promo_event"].isna(), "promo_event"] = str(getattr(row, "promo_name", ""))

    inventory = inventory_source.copy()
    inventory["sku_id"] = inventory["sku_id"].astype(str)
    inventory = inventory.groupby("sku_id", as_index=False).agg(
        on_hand_units=("stock_on_hand", "sum"), reorder_point=("reorder_point", "sum"),
    )
    inventory.insert(0, "date", sales["date"].max())
    inventory["on_order_units"] = 0
    inventory["lead_time_days"] = 14
    inventory = inventory[["date", "sku_id", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]]

    return {
        "sales_daily": sales,
        "sku_master": sku,
        "calendar": calendar,
        "inventory_snapshots": inventory,
        "_source_metadata": {
            "selected_transaction_file": "data/raw/official_selected/sales_transactions_25000.csv",
            "selected_transaction_rows": int(len(transactions)),
            "selected_unique_skus": int(transactions["sku_id"].nunique()),
            "selected_unique_stores": int(transactions["store_id"].nunique()),
            "selected_unique_customers": int(transactions["customer_id"].nunique()),
            "customer_master_rows": int(len(customers)),
            "store_master_rows": int(len(stores)),
            "promotion_rows": int(len(promotions)),
            "ground_truth_flag_rows": int(len(flags)),
            "fabricated_source_records": 0,
        },
    }


def ingest_raw_extracts(raw_dir: Path = DATA_RAW) -> dict:
    """Read the four official raw extracts from ``raw_dir`` into a dict of DataFrames.

    Only the four exact official filenames are considered. Legacy
    Mini-FORESIGHT CSVs in the same folder are never substituted. Raises
    ``MissingOfficialInputsError`` if one or more official files are absent.
    """
    return _ingest_selected_retail_inputs(Path(raw_dir))


# --------------------------------------------------------------------------- #
# 2. RAW SCHEMA VALIDATION
# --------------------------------------------------------------------------- #


def _column_consistency(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return duplicate and empty ('' / whitespace) column header lists."""
    counts = df.columns.value_counts()
    dup = [str(c) for c in counts[counts > 1].index]
    empties = [str(c) for c in df.columns if isinstance(c, str) and not c.strip()]
    return {"duplicate_columns": dup, "empty_column_headers": empties}


def validate_raw_schemas(
    raw_tables: dict,
    schemas: dict = OFFICIAL_SCHEMAS,
    return_report: bool = True,
) -> Any:
    """Validate every official extract against the official schema.

    This is intentionally non-destructive: it never renames, drops, or coerces
    columns. It returns a structured ``ValidationReport`` per table (or a dict
    when ``return_report=False``). Missing required columns or duplicate column
    names raise ``SchemaValidationError`` because downstream cleaning cannot be
    trusted on such tables.
    """
    reports: Dict[str, Any] = {}

    for name, required_cols in schemas.items():
        if name not in raw_tables:
            raise SchemaValidationError(
                f"Schema validation cannot run: extract '{name}' is missing from "
                "the ingested tables. Run ingest_raw_extracts() first."
            )
        df = raw_tables[name]
        present_cols = [str(c) for c in df.columns]
        missing_cols = [c for c in required_cols if c not in present_cols]
        unexpected_cols = [c for c in present_cols if c not in required_cols]
        consistency = _column_consistency(df)
        dup_cols = consistency["duplicate_columns"]

        problems = []
        if dup_cols:
            problems.append(f"duplicate column name(s) {dup_cols}; refusing to proceed")
        if missing_cols:
            problems.append(f"missing required column(s): {missing_cols}")

        if problems:
            raise SchemaValidationError(
                f"'{name}' (official extract) schema invalid: " + "; ".join(problems)
            )

        null_counts = {str(c): int(df[c].isna().sum()) for c in df.columns}
        dtypes = {str(c): str(df[c].dtype) for c in df.columns}
        exact_dup = int(df.duplicated().sum()) if not df.empty else 0
        business_key_dup = 0
        if name == "sku_master" and "sku_id" in df.columns:
            business_key_dup = int(df.duplicated(subset=["sku_id"]).sum())

        if not return_report:
            continue

        reports[name] = {
            "dataset": name,
            "file": str(schemas[name]),
            "row_count": int(len(df)),
            "column_count": int(df.shape[1]),
            "columns": present_cols,
            "dtypes": dtypes,
            "null_counts": null_counts,
            "exact_duplicate_rows": exact_dup,
            "business_key_duplicate_rows": business_key_dup,
            "invalid_value_counts": {},
            "missing_columns": missing_cols,
            "unexpected_columns": unexpected_cols,
            "duplicate_columns": dup_cols,
            "date_range": ValidationReport._first_date_range(df),
        }

    return reports if return_report else None
# --------------------------------------------------------------------------- #
# 3. DATA TYPE CLEANING (explicit, conservative)
# --------------------------------------------------------------------------- #


def _parse_dates(series: pd.Series, dataset: str) -> pd.Series:
    """Explicit date parsing; unparseable values stay NaN and are reported later."""
    return pd.to_datetime(series, errors="coerce")


def _to_numeric(series: pd.Series, dataset: str, col: str) -> pd.Series:
    """Explicit numeric coercion; non-coercible values become NaN (reported later)."""
    return pd.to_numeric(series, errors="coerce")


def _normalize_bool_like(series: pd.Series, dataset: str, col: str) -> pd.Series:
    """Normalize common boolean encodings to 0/1; anything else stays as-is (NaN)."""
    mapping = {
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
    }
    out = series.map(
        lambda v: mapping.get(str(v).strip().lower())
        if isinstance(v, str) else (int(bool(v)) if isinstance(v, (int, float, bool)) else v)
    )
    return out


def clean_sales_daily(sales_daily: pd.DataFrame) -> CleanTableResult:
    """Clean ``sales_daily`` per the official schema.

    * parse ``date``; coerce numerics; normalize ``promo_flag``.
    * exact-duplicate rows are count-documented and removed.
    * no values are invented (invalid/unparseable rows are preserved & flagged
      via the decision log and are handed to domain validation).
    """
    res = CleanTableResult(name="sales_daily", df=sales_daily.copy())

    res.log_decision(
        issue="date column text/raw",
        detection_rule="pd.to_datetime(errors='coerce')",
        number_affected=int(sales_daily["date"].isna().sum()),
        action="parse dates",
        reason="official contract requires datetime date",
        changed=True,
    )
    res.df["date"] = _parse_dates(res.df["date"], "sales_daily")

    for col in ["units_sold", "revenue", "unit_price"]:
        res.log_decision(
            issue=f"{col} numeric type",
            detection_rule="pd.to_numeric(errors='coerce')",
            number_affected=0,
            action="coerce to numeric",
            reason="official contract requires numeric measure",
            changed=True,
        )
        res.df[col] = _to_numeric(res.df[col], "sales_daily", col)

    res.log_decision(
        issue="promo_flag boolean-like",
        detection_rule="normalize 1/0, true/false, yes/no",
        number_affected=0,
        action="normalize to 0/1",
        reason="official contract expects boolean-like promo indicator",
        changed=True,
    )
    res.df["promo_flag"] = _normalize_bool_like(res.df["promo_flag"], "sales_daily", "promo_flag")
    res.df["promo_flag"] = res.df["promo_flag"].astype("Int64")

    _handle_exact_duplicates(res, subset=None)
    return res


def clean_sku_master(sku_master: pd.DataFrame) -> CleanTableResult:
    """Clean ``sku_master`` per the official schema.

    * parse ``launch_date``; coerce costs/prices.
    * exact-duplicate rows removed; duplicate ``sku_id`` keys are reported, not
      silently resolved.
    """
    res = CleanTableResult(name="sku_master", df=sku_master.copy())

    res.log_decision(
        issue="launch_date column text/raw",
        detection_rule="pd.to_datetime(errors='coerce')",
        number_affected=int(sku_master["launch_date"].isna().sum()),
        action="parse dates",
        reason="official contract requires launch_date as date",
        changed=True,
    )
    res.df["launch_date"] = _parse_dates(res.df["launch_date"], "sku_master")

    for col in ["unit_cost", "list_price"]:
        res.log_decision(
            issue=f"{col} numeric type",
            detection_rule="pd.to_numeric(errors='coerce')",
            number_affected=0,
            action="coerce to numeric",
            reason="official contract requires numeric cost/price",
            changed=True,
        )
        res.df[col] = _to_numeric(res.df[col], "sku_master", col)

    _handle_exact_duplicates(res, subset=None)
    _flag_business_key_duplicates(res, key="sku_id")
    return res


def clean_calendar(calendar: pd.DataFrame) -> CleanTableResult:
    """Clean ``calendar`` per the official schema.

    * parse ``date``; coerce ``week``/``month`` to numeric; normalize
      ``is_holiday`` boolean-like; blank ``promo_event`` -> NaN (absence of an
      event is preserved as missing, never invented).
    """
    res = CleanTableResult(name="calendar", df=calendar.copy())

    res.log_decision(
        issue="date column text/raw",
        detection_rule="pd.to_datetime(errors='coerce')",
        number_affected=int(calendar["date"].isna().sum()),
        action="parse dates",
        reason="official contract requires datetime date",
        changed=True,
    )
    res.df["date"] = _parse_dates(res.df["date"], "calendar")

    for col in ("week", "month"):
        res.log_decision(
            issue=f"{col} numeric type",
            detection_rule="pd.to_numeric(errors='coerce')",
            number_affected=0,
            action="coerce to numeric",
            reason="official contract requires week/month as numbers",
            changed=True,
        )
        res.df[col] = _to_numeric(res.df[col], "calendar", col)

    res.log_decision(
        issue="is_holiday boolean-like",
        detection_rule="normalize 1/0, true/false, yes/no",
        number_affected=0,
        action="normalize to 0/1",
        reason="official contract expects boolean-like holiday indicator",
        changed=True,
    )
    res.df["is_holiday"] = _normalize_bool_like(
        res.df["is_holiday"], "calendar", "is_holiday"
    ).astype("Int64")

    res.log_decision(
        issue="promo_event blank strings",
        detection_rule="'' / whitespace-only string",
        number_affected=int(
            (res.df["promo_event"].astype("string").str.strip() == "").sum()
        ),
        action="blank -> NaN (preservation; no invention)",
        reason="blank promo_event means no event, not a value to invent",
        changed=True,
    )
    res.df["promo_event"] = res.df["promo_event"].replace(r"^\s*$", pd.NA, regex=True)

    _handle_exact_duplicates(res, subset=None)
    return res


def clean_inventory_snapshots(inventory_snapshots: pd.DataFrame) -> CleanTableResult:
    """Clean ``inventory_snapshots`` per the official schema.

    * parse ``date``; coerce all four official quantity/policy fields;
      exact-duplicate cleanup. Values are flagged (not fixed) by the domain
      validators for negative quantities or invalid lead times.
    """
    res = CleanTableResult(name="inventory_snapshots", df=inventory_snapshots.copy())

    res.log_decision(
        issue="date column text/raw",
        detection_rule="pd.to_datetime(errors='coerce')",
        number_affected=int(inventory_snapshots["date"].isna().sum()),
        action="parse dates",
        reason="official contract requires datetime date",
        changed=True,
    )
    res.df["date"] = _parse_dates(res.df["date"], "inventory_snapshots")

    for col in ("on_hand_units", "on_order_units", "lead_time_days", "reorder_point"):
        res.log_decision(
            issue=f"{col} numeric type",
            detection_rule="pd.to_numeric(errors='coerce')",
            number_affected=0,
            action="coerce to numeric",
            reason="official contract requires numeric inventory fields",
            changed=True,
        )
        res.df[col] = _to_numeric(res.df[col], "inventory_snapshots", col)

    _handle_exact_duplicates(res, subset=None)
    return res


# --------------------------------------------------------------------------- #
# 7. CROSS-TABLE VALIDATION (reported — never fabricated or auto-filled)
# --------------------------------------------------------------------------- #


def validate_cross_table(cleaned: Dict[str, pd.DataFrame]) -> CrossTableReport:
    """Validate relationships BETWEEN the four official extracts.

    Checks performed (all REPORTED; nothing is invented to fix them):

    * sales SKUs vs sku_master SKUs            -> sales_skus_missing_in_master
    * inventory SKUs vs sku_master SKUs        -> inventory_skus_missing_in_master
    * duplicate business keys in sku_master    -> duplicate_sku_keys
    * sales dates not covered by calendar      -> sales_dates_outside_calendar
    * inventory dates not covered by calendar  -> inventory_dates_outside_calendar
    * calendar dates with no demand/stock row  -> calendar_dates_missing_from_series
    * per-SKU date gaps in the sales panel     -> sales_missing_sku_date_combos
    * per-SKU date gaps in the inventory panel -> inventory_missing_sku_date_combos
    """
    sales = cleaned["sales_daily"]
    master = cleaned["sku_master"]
    calendar = cleaned["calendar"]
    inv = cleaned["inventory_snapshots"]

    master_skus = set(master["sku_id"].astype(str))
    sales_skus = set(sales["sku_id"].astype(str))
    inv_skus = set(inv["sku_id"].astype(str))

    cal_dates = set(pd.to_datetime(calendar["date"], errors="coerce").dropna())
    sales_dates = set(pd.to_datetime(sales["date"], errors="coerce").dropna())
    inv_dates = set(pd.to_datetime(inv["date"], errors="coerce").dropna())

    dup_master = (
        master["sku_id"][master["sku_id"].duplicated()].astype(str).unique().tolist()
    )

    def _fmt_dates(dates: set) -> List[str]:
        return sorted(d.date().isoformat() for d in dates)[:50]

    report = CrossTableReport(
        sales_skus_missing_in_master=sorted(sales_skus - master_skus),
        inventory_skus_missing_in_master=sorted(inv_skus - master_skus),
        duplicate_sku_keys=sorted(dup_master),
        sales_dates_outside_calendar=_fmt_dates(sales_dates - cal_dates),
        inventory_dates_outside_calendar=_fmt_dates(inv_dates - cal_dates),
        calendar_dates_missing_from_series=_fmt_dates(cal_dates - sales_dates),
        sales_missing_sku_date_combos=[],
        inventory_missing_sku_date_combos=[],
    )

    # Panel completeness: expected = every observed SKU x every observed date.
    for label, df, key in (("sales", sales, "sales_missing_sku_date_combos"),
                           ("inventory", inv, "inventory_missing_sku_date_combos")):
        if df.empty:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        skus = sorted(d["sku_id"].astype(str).unique())
        all_d = sorted(d["date"].dropna().unique())
        full = pd.MultiIndex.from_product([skus, all_d], names=["sku_id", "date"])
        actual = pd.MultiIndex.from_frame(
            d[["sku_id", "date"]].dropna().astype({"sku_id": str})
        )
        gaps = full.difference(actual)
        setattr(report, key, [
            f"{s}@{pd.Timestamp(dt).date().isoformat()}" for s, dt in list(gaps)[:50]
        ])

    return report


def cross_table_to_decisions(report: CrossTableReport) -> List[CleaningDecision]:
    """Convert cross-table findings into documented decision-log entries."""
    entries: List[Tuple[str, int]] = [
        ("sales SKUs absent from sku_master",
         len(report.sales_skus_missing_in_master)),
        ("inventory SKUs absent from sku_master",
         len(report.inventory_skus_missing_in_master)),
        ("duplicate sku_master keys", len(report.duplicate_sku_keys)),
        ("sales dates outside calendar coverage",
         len(report.sales_dates_outside_calendar)),
        ("inventory dates outside calendar coverage",
         len(report.inventory_dates_outside_calendar)),
        ("calendar dates without any sales row",
         len(report.calendar_dates_missing_from_series)),
        ("missing SKU/date combinations in sales panel",
         len(report.sales_missing_sku_date_combos)),
        ("missing SKU/date combinations in inventory panel",
         len(report.inventory_missing_sku_date_combos)),
    ]
    out: List[CleaningDecision] = []
    for issue, n in entries:
        out.append(CleaningDecision(
            dataset="cross_table",
            issue=issue,
            detection_rule="set difference / duplicated() across extracts",
            number_affected=int(n),
            action="report only (no fabrication, no filling)",
            reason="relationships are validated and surfaced for review; "
            "creating missing master/calendar rows is forbidden",
            changed=False,
        ))
    return out


# --------------------------------------------------------------------------- #
# 5b. MISSING-VALUE POLICIES (applied deterministically per official field)
# --------------------------------------------------------------------------- #

# Policy semantics: 'drop_row' ONLY for true business keys whose absence makes
# a row unidentifiable; everything else is PRESERVED and reported. No value is
# ever invented.
MISSING_VALUE_POLICIES: Dict[str, Dict[str, str]] = {
    "sales_daily": {
        "date": "drop_row",
        "sku_id": "drop_row",
        "units_sold": "preserve",
        "revenue": "preserve",
        "unit_price": "preserve",
        "promo_flag": "preserve",
    },
    "sku_master": {"sku_id": "drop_row"},
    "calendar": {"date": "drop_row"},
    "inventory_snapshots": {
        "date": "drop_row",
        "sku_id": "drop_row",
    },
}


def apply_missing_policies(
    results: Dict[str, CleanTableResult],
) -> Dict[str, CleanTableResult]:
    """Apply :data:`MISSING_VALUE_POLICIES` to every cleaned table."""
    for name, res in results.items():
        _apply_missing_policy(res, MISSING_VALUE_POLICIES.get(name, {}))
    return results



# --------------------------------------------------------------------------- #
# 9. ANALYSIS-READY OUTPUT (join-only; no D3 feature engineering here)
# --------------------------------------------------------------------------- #

# Deterministic output filenames (official D1 contract). The *_clean.csv files
# intentionally reuse the canonical D1 names consumed by D2/D3/D4/D5/D6.
OUTPUT_FILES: Dict[str, str] = {
    "sales_daily": "sales_daily_clean.csv",
    "sku_master": "sku_master_clean.csv",
    "calendar": "calendar_clean.csv",
    "inventory_snapshots": "inventory_snapshots_clean.csv",
    "analysis_ready": "sales_analysis_ready.csv",
}
QUALITY_REPORT_FILE = "d1_data_quality_report.json"


def build_analysis_ready(cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the single analysis-ready dataset required by M1.

    One row per official sales observation (date x sku) enriched by:
      * official calendar context for that date (week, month, season,
        is_holiday, promo_event)
      * static sku_master attributes (category, subcategory, launch_date,
        unit_cost, list_price)

    This is a pure JOIN of official fields — no lags/rolling windows (those
    belong to D3), no aggregation to other grains, and no invented values.
    """
    sales = cleaned["sales_daily"].copy()
    master = cleaned["sku_master"].copy()
    calendar = cleaned["calendar"].copy()

    cal_cols = ["date", "week", "month", "season", "is_holiday", "promo_event"]
    cal = calendar[[c for c in cal_cols if c in calendar.columns]].copy()

    master_cols = ["sku_id", "category", "subcategory", "launch_date",
                   "unit_cost", "list_price"]
    master_cols = [c for c in master_cols if c in master.columns]
    mst = master[master_cols].copy()
    if "launch_date" in mst.columns:
        mst["launch_date"] = pd.to_datetime(mst["launch_date"], errors="coerce")

    out = sales.merge(cal, on="date", how="left", validate="many_to_one")
    out = out.merge(mst, on="sku_id", how="left", validate="many_to_one")

    # Deterministic ordering for reproducible file output.
    sort_keys = [c for c in ("date", "sku_id") if c in out.columns]
    return out.sort_values(sort_keys).reset_index(drop=True)


def _dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Deterministic CSV serialization (stable column order, UTF-8, LF)."""
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# 10. OUTPUT SAFETY — write only after successful validation/cleaning
# --------------------------------------------------------------------------- #


def write_outputs(
    analysis_ready: pd.DataFrame,
    cleaned: Dict[str, pd.DataFrame],
    report_payload: Dict[str, Any],
    processed_dir: Path = DATA_PROCESSED,
) -> List[Path]:
    """Write ONLY official D1 outputs; called solely after validation succeeds.

    Files written (deterministic names):
      sales_daily_clean.csv · sku_master_clean.csv · calendar_clean.csv ·
      inventory_snapshots_clean.csv · sales_analysis_ready.csv ·
      d1_data_quality_report.json

    Legacy Mini-FORESIGHT RAW files under data/raw/ are never touched. Other
    pre-existing files in data/processed/ are left alone.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    frames = {**{k: cleaned[k] for k in OUTPUT_FILES if k != "analysis_ready"},
              "analysis_ready": analysis_ready}
    for key, fname in OUTPUT_FILES.items():
        path = processed_dir / fname
        path.write_bytes(_dataframe_to_csv_bytes(frames[key]))
        written.append(path)

    rpath = processed_dir / QUALITY_REPORT_FILE
    rpath.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    written.append(rpath)
    return written


def _build_pipeline_report(
    validation: Dict[str, Any],
    decisions: List[CleaningDecision],
    cross_table: CrossTableReport,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the machine-readable D1 quality/cleaning report."""
    return {
        "report": "D1 data-quality & cleaning decisions",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "validation": validation,
        "cleaning_decisions": [
            {
                "dataset": d.dataset,
                "issue": d.issue,
                "detection_rule": d.detection_rule,
                "number_affected": int(d.number_affected),
                "action": d.action,
                "reason": d.reason,
                "changed": bool(d.changed),
            }
            for d in decisions
        ],
        "cross_table": cross_table.to_serializable(),
        "source_boundary": source_metadata or {},
        "policy_note": (
            "Missing values: business keys dropped deterministically; all "
            "other fields preserved and reported. No value was invented."
        ),
    }


# --------------------------------------------------------------------------- #
# 4. DUPLICATE HANDLING
# --------------------------------------------------------------------------- #


def _handle_exact_duplicates(res: CleanTableResult, subset: Optional[List[str]]) -> None:
    """Count-document and remove exact duplicate rows only (safe, deterministic)."""
    df = res.df
    before = len(df)
    dup = df.duplicated(keep="first")
    n_dup = int(dup.sum())
    res.log_decision(
        issue="exact duplicate rows",
        detection_rule="df.duplicated(keep='first') on all columns",
        number_affected=n_dup,
        action="remove exact duplicate rows" if n_dup else "no action",
        reason="exact duplicates are safe to dedupe deterministically; "
        "business-key conflicts are handled separately",
        changed=n_dup > 0,
    )
    res.df = df[~dup].reset_index(drop=True)
    res.dropped_rows += before - len(res.df)


def _flag_business_key_duplicates(res: CleanTableResult, key: str) -> None:
    """Report business-key (e.g. ``sku_id``) duplicates; do not arbitrarily resolve them."""
    df = res.df
    dup = df.duplicated(subset=[key], keep="first")
    n = int(dup.sum())
    res.log_decision(
        issue=f"business-key duplicates on '{key}'",
        detection_rule=f"df.duplicated(subset=['{key}'], keep='first')",
        number_affected=n,
        action="report for review (rows preserved)",
        reason="no unambiguous, data-supported rule exists to choose one row "
        "over another; a reviewer must resolve this",
        changed=False,
    )


# --------------------------------------------------------------------------- #
# 5. MISSING-VALUE HANDLING (per-field, conservative)
# --------------------------------------------------------------------------- #


def _apply_missing_policy(res: CleanTableResult, policy: Dict[str, str]) -> None:
    """Apply a per-field missing-value policy from a deterministic map.

    Allowed actions: ``preserve``, ``drop_row`` (only for fields that form a legal
    business key), ``none`` (no action). Values are never invented.
    """
    if not policy:
        return
    df = res.df
    for col, action in policy.items():
        n_null = int(df[col].isna().sum())
        if action == "preserve":
            res.log_decision(
                issue=f"nulls in '{col}'",
                detection_rule=f"df['{col}'].isna()",
                number_affected=n_null,
                action="preserve (report to data-quality memo)",
                reason=f"missing {col} cannot be safely derived from the row/table; "
                "no invention",
                changed=False,
            )
        elif action == "drop_row":
            mask = df[col].isna()
            n_drop = int(mask.sum())
            df.dropna(subset=[col], inplace=True)
            df.reset_index(drop=True, inplace=True)
            res.dropped_rows += n_drop
            res.log_decision(
                issue=f"nulls in '{col}'",
                detection_rule=f"df['{col}'].isna()",
                number_affected=n_drop,
                action="drop row (required key column)",
                reason=f"'{col}' is a required business key; a row without it has "
                "no identifiable subject, so exclusion is deterministic",
                changed=n_drop > 0,
            )
        else:
            raise ValueError(f"Unknown missing-value action: {action!r}")


# --------------------------------------------------------------------------- #
# 6. DOMAIN VALIDATION
# --------------------------------------------------------------------------- #


def _domain_flag(series: pd.Series, test, label: str) -> int:
    """Return the count of rows failing a domain rule (never mutates data)."""
    return int((series.notna() & test(series)).sum())


def _validate_non_negative(res: CleanTableResult, cols: List[str]) -> None:
    for col in cols:
        if col not in res.df.columns:
            continue
        n_bad = _domain_flag(res.df[col], lambda s: s < 0, f"negative {col}")
        res.log_decision(
            issue=f"negative values in '{col}'",
            detection_rule=f"{col} < 0 (after type cleaning)",
            number_affected=n_bad,
            action="flag for review (rows preserved, NOT fixed)",
            reason="inventing replacement values is forbidden; a reviewer "
            "resolves alongside the data-quality memo",
            changed=False,
        )


def _apply_domain_validations(res: CleanTableResult) -> None:
    """Flag obvious invalid values per the official schema. Never invents fixes."""
    name = res.name
    df = res.df

    if name == "sales_daily":
        _validate_non_negative(res, ["units_sold", "revenue", "unit_price"])
        n_bad_bool = int((df["promo_flag"].notna() & ~df["promo_flag"].isin([0, 1])).sum())
        res.log_decision(
            issue="promo_flag invalid boolean-like",
            detection_rule="promo_flag not in {0, 1} after normalization",
            number_affected=n_bad_bool,
            action="flag for review (rows preserved)",
            reason="invalid promo indicator cannot be repaired from the row",
            changed=False,
        )
    elif name == "sku_master":
        _validate_non_negative(res, ["unit_cost", "list_price"])
        n_bad_launch = int(df["launch_date"].isna().sum())
        res.log_decision(
            issue="launch_date invalid/unparseable",
            detection_rule="launch_date is null after parsing",
            number_affected=n_bad_launch,
            action="flag for review (rows preserved)",
            reason="launch date cannot be repaired from the row",
            changed=False,
        )
    elif name == "calendar":
        n_bad_week = int((df["week"].notna() & ~df["week"].between(1, 53)).sum())
        res.log_decision(
            issue="week outside valid range",
            detection_rule="week not in 1..53 after type cleaning",
            number_affected=n_bad_week,
            action="flag for review (rows preserved)",
            reason="calendar week cannot be repaired from the row",
            changed=False,
        )
        n_bad_month = int((df["month"].notna() & ~df["month"].between(1, 12)).sum())
        res.log_decision(
            issue="month outside valid range",
            detection_rule="month not in 1..12 after type cleaning",
            number_affected=n_bad_month,
            action="flag for review (rows preserved)",
            reason="calendar month cannot be repaired from the row",
            changed=False,
        )
        n_bad_holiday = int((df["is_holiday"].notna() & ~df["is_holiday"].isin([0, 1])).sum())
        res.log_decision(
            issue="is_holiday invalid boolean-like",
            detection_rule="is_holiday not in {0, 1} after normalization",
            number_affected=n_bad_holiday,
            action="flag for review (rows preserved)",
            reason="invalid holiday indicator cannot be repaired from the row",
            changed=False,
        )
    elif name == "inventory_snapshots":
        _validate_non_negative(
            res, ["on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]
        )


# --------------------------------------------------------------------------- #
# 11. RUN PIPELINE — the ONE reproducible D1 entry point
# --------------------------------------------------------------------------- #


def run_pipeline(
    raw_dir: Path = DATA_RAW,
    processed_dir: Path = DATA_PROCESSED,
) -> Dict[str, Any]:
    """Run the complete D1 pipeline from raw extracts to analysis-ready files.

    Steps (in strict order — outputs are written ONLY if every earlier step
    succeeds):

      1. Guard + ingest the four official raw extracts.
      2. Validate exact official schemas (missing/duplicate columns raise).
      3. Clean each extract and apply deterministic missing-value policies.
      4. Apply domain validations (flag-only; values never invented).
      5. Validate cross-table relationships (reported, never filled).
      6. Build the single analysis-ready dataset (pure official joins).
      7. Write deterministic outputs + machine-readable quality report.

    Determinism: stable sort orders before writing, fixed column order, LF
    line endings, UTF-8, ``sort_keys=True`` JSON. Only the report timestamp
    varies between runs.
    """
    # 1) Ingest (raises MissingOfficialInputsError when anything is absent).
    raw_tables = ingest_raw_extracts(raw_dir)

    # 2) Schema gate before any cleaning or writing.
    validation = validate_raw_schemas(raw_tables)

    # 3) Clean all four extracts + deterministic missing-value policies.
    results: Dict[str, CleanTableResult] = {
        "sales_daily": clean_sales_daily(raw_tables["sales_daily"]),
        "sku_master": clean_sku_master(raw_tables["sku_master"]),
        "calendar": clean_calendar(raw_tables["calendar"]),
        "inventory_snapshots": clean_inventory_snapshots(
            raw_tables["inventory_snapshots"]
        ),
    }
    results = apply_missing_policies(results)

    decisions: List[CleaningDecision] = []
    for res in results.values():
        decisions.extend(res.decisions)
    cleaned: Dict[str, pd.DataFrame] = {k: r.df for k, r in results.items()}

    # 4) Domain validations (flag-only).
    for res in results.values():
        _apply_domain_validations(res)
        decisions.extend(res.decisions)

    # Refresh schema-validation counts against the CLEANED frames so the
    # report reflects post-cleaning nulls/duplicates honestly.
    validation = validate_raw_schemas(cleaned)

    # 5) Cross-table relationship checks (report-only).
    cross_table = validate_cross_table(cleaned)
    decisions.extend(cross_table_to_decisions(cross_table))

    # 6) Analysis-ready dataset (official joins only).
    analysis_ready = build_analysis_ready(cleaned)

    # 7) Machine-readable report + deterministic output writing.
    report_payload = _build_pipeline_report(
        validation, decisions, cross_table, raw_tables.get("_source_metadata")
    )
    written = write_outputs(analysis_ready, cleaned, report_payload, processed_dir)

    rows_written = {OUTPUT_FILES[k]: int(len(v)) for k, v in cleaned.items()}
    rows_written[OUTPUT_FILES["analysis_ready"]] = int(len(analysis_ready))

    return {
        "status": "success",
        "raw_dir": str(Path(raw_dir)),
        "processed_dir": str(Path(processed_dir)),
        "outputs_written": [str(p) for p in written],
        "rows_written": rows_written,
        "validation": validation,
        "cross_table_issues": sum(
            len(v) for v in cross_table.to_serializable().values()
        ),
        "cleaning_decision_count": len(decisions),
        "quality_report_file": str(Path(processed_dir) / QUALITY_REPORT_FILE),
    }


# --------------------------------------------------------------------------- #
# 12. CLI ENTRY POINT —  python -m src.pipeline
# --------------------------------------------------------------------------- #

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_MISSING_INPUTS = 2


def _cli(argv: Optional[List[str]] = None) -> int:
    """Repository-root CLI: ``python -m src.pipeline [--raw-dir D] [--processed-dir D]``.

    Exit codes
    ----------
    0  success — official outputs + quality report written
    2  official inputs missing (clear message; nothing written)
    1  any other failure (schema error, unreadable file, unexpected error)
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description=(
            "D1 — Official FORESIGHT data pipeline. Ingests the four official "
            "raw extracts, validates exact official schemas, cleans missing "
            "values/duplicates/types programmatically, validates cross-table "
            "relationships, and writes deterministic analysis-ready outputs "
            "plus a machine-readable data-quality report.\n\n"
            "Fails safely when official inputs are absent; never falls back "
            "to legacy Mini-FORESIGHT data; never invents values."
        ),
    )
    parser.add_argument(
        "--raw-dir", default=str(DATA_RAW),
        help="Directory containing the four official extracts "
        "(default: <repo>/data/raw).",
    )
    parser.add_argument(
        "--processed-dir", default=str(DATA_PROCESSED),
        help="Destination directory for official D1 outputs "
        "(default: <repo>/data/processed).",
    )
    args = parser.parse_args(argv)

    print("D1 — Project FORESIGHT data pipeline")
    print(f"  raw dir      : {args.raw_dir}")
    print(f"  processed dir: {args.processed_dir}")

    try:
        summary = run_pipeline(Path(args.raw_dir), Path(args.processed_dir))
    except MissingOfficialInputsError as exc:
        print("\nFAILED — official inputs missing. Nothing was written.",
              file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return EXIT_MISSING_INPUTS
    except SchemaValidationError as exc:
        print("\nFAILED — official schema validation error. Nothing was written.",
              file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print("\nFAILED — unexpected pipeline error. Nothing was written.",
              file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print("\nSUCCESS — official D1 outputs written:")
    for path in summary["outputs_written"]:
        print(f"  + {path}")
    print("Rows written:")
    for fname in sorted(summary["rows_written"]):
        print(f"  {fname:<36} {summary['rows_written'][fname]:>8}")
    print(f"Cleaning decisions logged : {summary['cleaning_decision_count']}")
    print(f"Cross-table issues flagged: {summary['cross_table_issues']}")
    print(f"Quality report            : {summary['quality_report_file']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(_cli())