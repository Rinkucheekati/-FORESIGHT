"""Prepare FORESIGHT inputs from the mentor-shared retail transaction dataset.

The source dataset lives in ``data/raw/retail_contaminated_dataset`` and has a
retail-star schema rather than the four-file FORESIGHT schema.  This adapter is
the documented, reproducible bridge between the two.

Because the official ``sales_transactions.csv`` holds ~9.95M rows, the adapter
reduces it to a deterministic project subset of exactly 25,000 rows (fixed seed) while
preserving a per-SKU floor for the ground-truth-flagged SKUs in
``sku_inventory_flags.csv``.  Transaction rows are read in chunks (never loaded
fully) and the sampled rows are aggregated to the SKU-day planning grain.

Derived assumptions are deliberately explicit:
* sales are rolled up across stores to the SKU-day planning grain;
* the source has no inbound-order field, so ``on_order_units`` is 0;
* the source has no supplier lead time, so a 14-day planning assumption is
  used and recorded in the manifest; and
* ``launch_date`` is the first observed transaction date for a SKU, not a
  claimed product-launch date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.paths import DATA_RAW, REPO_ROOT


SOURCE_DIR = DATA_RAW / "retail_contaminated_dataset"
STAGING_DIR = REPO_ROOT / "data" / "staging" / "official_retail_daily_chunks"
MANIFEST_FILE = "official_retail_adapter_manifest.json"
CHUNK_SIZE = 250_000
LEAD_TIME_DAYS_ASSUMPTION = 14

CUSTOMER_MASTER_FILE = "customer_master.csv"
STORE_MASTER_FILE = "store_master.csv"
GROUND_TRUTH_FLAGS_FILE = "sku_inventory_flags.csv"

# ---- Project subset bounds (deterministic 25k rule) -------------------------
# The mentor requires a project-sized subset of exactly 25,000 transaction records
# rather than processing the full ~10M-row sales_transactions.csv. Sampling is
# deterministic (fixed seed) so the same subset is produced on every run.
SUBSET_TARGET_ROWS = 25_000
SUBSET_SEED = 42
# Ground-truth-flagged SKUs (from sku_inventory_flags.csv) get a deterministic
# floor of up to this many rows each so the D4 answer-key SKUs can never be
# accidentally eliminated by the general sample. 10 x 600 flagged SKUs -> at
# These rows are included within the exact target, not added on top of it.
SUBSET_FLAGGED_PER_SKU = 10

REQUIRED_SOURCE_FILES = {
    "sales_transactions": "sales_transactions.csv",
    "sku_master": "sku_master.csv",
    "customer_master": "customer_master.csv",
    "store_master": "store_master.csv",
    "promotions": "promotions.csv",
    "inventory_snapshot": "inventory_snapshot.csv",
    "ground_truth_flags": "sku_inventory_flags.csv",
}

REQUIRED_TRANSACTION_COLUMNS = {
    "date", "sku_id", "quantity", "unit_price", "total_value",
    "discount_pct", "promo_id",
}


class RetailAdapterError(Exception):
    """Raised when the official retail source cannot be mapped safely."""


def _require_source_files(source_dir: Path) -> Dict[str, Path]:
    paths = {name: source_dir / filename for name, filename in REQUIRED_SOURCE_FILES.items()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RetailAdapterError(
            "Official retail source is incomplete. Missing: " + ", ".join(missing)
            + f". Expected under {source_dir}."
        )
    return paths


def _flagged_sku_ids(flags_path: Path) -> set[str]:
    """Return the set of SKU ids that carry ground-truth risk flags.

    These SKUs are deterministically force-included so the D4 risk engine can
    still be scored against the ground-truth answer key. The flags file is the
    official ''sku_inventory_flags.csv'' (anomalies version) and is read
    read-only; it is never modified.
    """
    flags = pd.read_csv(flags_path)
    if "sku_id" not in flags.columns:
        raise RetailAdapterError(
            f"{flags_path.name} is missing the required 'sku_id' column."
        )
    return set(flags["sku_id"].astype("string").str.strip())


def _deterministic_subset(
    reader: Iterable[pd.DataFrame],
    flagged_skus: set[str],
    target: int,
    per_sku_cap: int,
    seed: int,
) -> tuple[pd.DataFrame, int, int, Dict[str, int]]:
    """Deterministic exact-target subset that preserves ground-truth SKUs.

    Runs a single streaming pass over ``sales_transactions.csv`` (never loading
    the full ~10M rows). Two disjoint reservoirs are built:

    * *General* — rows whose SKU is not ground-truth-flagged are reservoir
      sampled (fixed seed) to ``target`` rows, giving a uniform, date/entity
    diverse sample.
    * *Flagged floor* — rows belonging to the ground-truth-flagged SKUs are
      kept up to ``per_sku_cap`` rows per SKU (in file order) so the D4
      answer-key SKUs are guaranteed representation.

    Only valid rows (parseable date, non-empty sku_id, positive quantity) are
    ever retained. Every value returned is a verbatim official source value;
    nothing is fabricated, re-rolled, or shifted.

    Returns ``(subset_df, total_scanned, general_kept, flagged_kept)`` where
    ``flagged_kept`` is a ``{sku_id: rows}`` summary of the floor.
    """
    rng = np.random.default_rng(seed)

    def _valid_mask(chunk: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(chunk["date"], errors="coerce")
        quantity = pd.to_numeric(chunk["quantity"], errors="coerce")
        sku = chunk["sku_id"].astype("string").str.strip()
        return dates.notna() & sku.notna() & sku.ne("") & quantity.gt(0)

    general_parts: List[pd.DataFrame] = []
    general_res: Optional[pd.DataFrame] = None
    general_seen = 0
    flagged_rows: List[pd.DataFrame] = []
    flagged_kept: Dict[str, int] = defaultdict(int)
    total_scanned = 0

    for chunk in reader:
        chunk = chunk.reset_index(drop=True)
        total_scanned += len(chunk)
        valid = _valid_mask(chunk)
        if not valid.any():
            continue
        v = chunk.loc[valid].copy()
        is_flag = v["sku_id"].isin(flagged_skus)

        # --- General reservoir (non-flagged rows only) -----------------------
        nf = v.loc[~is_flag]
        if len(nf):
            general_seen += len(nf)
            if general_res is None:
                general_parts.append(nf)
                if sum(len(part) for part in general_parts) >= target:
                    combined = pd.concat(general_parts, ignore_index=True)
                    idx = rng.choice(len(combined), size=target, replace=False)
                    general_res = combined.iloc[np.sort(idx)].reset_index(drop=True)
                    general_parts = []
            else:
                # Vectorised reservoir replacement (Algorithm R).
                m = len(nf)
                global_idx = (np.arange(general_seen - m + 1, general_seen + 1)).astype(np.float64)
                keep_mask = rng.random(m) < (target / global_idx)
                if keep_mask.any():
                    rows = keep_mask.nonzero()[0]
                    slots = rng.integers(0, target, size=len(rows))
                    general_res.iloc[slots] = nf.iloc[rows].to_numpy()

        # --- Flagged per-SKU floor (deterministic, file order) ---------------
        f = v.loc[is_flag]
        if len(f):
            for sku_id, sub in f.groupby("sku_id"):
                remaining = per_sku_cap - flagged_kept[sku_id]
                if remaining <= 0:
                    continue
                take = min(remaining, len(sub))
                if take > 0:
                    flagged_rows.append(sub.iloc[:take])
                    flagged_kept[sku_id] += take

    general_frame = (
        general_res
        if general_res is not None
        else pd.concat(general_parts, ignore_index=True) if general_parts else pd.DataFrame()
    )
    flagged_frame = pd.concat(flagged_rows, ignore_index=True) if flagged_rows else pd.DataFrame()
    general_target = max(0, target - len(flagged_frame))
    if len(general_frame) > general_target:
        selected = rng.choice(len(general_frame), size=general_target, replace=False)
        general_frame = general_frame.iloc[np.sort(selected)].reset_index(drop=True)
    subset_df = pd.concat([general_frame, flagged_frame], ignore_index=True).reset_index(drop=True)

    general_kept = int(len(general_frame))
    return subset_df, total_scanned, general_kept, dict(flagged_kept)


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _write_partition(frame: pd.DataFrame, staging_dir: Path) -> None:
    """Append one chunk's partial SKU-day totals into month partitions."""
    frame = frame.copy()
    frame["month_key"] = frame["date"].dt.strftime("%Y-%m")
    for month_key, part in frame.groupby("month_key", sort=False):
        target = staging_dir / f"{month_key}.csv"
        part.drop(columns="month_key").to_csv(
            target, mode="a", index=False, header=not target.exists()
        )


def _build_sales_daily(
    source_path: Path,
    flags_path: Path,
    staging_dir: Path,
    destination: Path,
) -> Dict[str, Any]:
    """Build FORESIGHT's SKU-day fact table from a deterministic 25k subset.

    Official ``sales_transactions.csv`` (~9.95M rows, ~764 MB) is read in
    chunks and reduced by :func:`_deterministic_subset` to a reproducible ~50k
    project subset (fixed seed, ground-truth-flagged SKU floor). Only the
    sampled rows are aggregated to SKU-day and written through the existing
    month-partitioned staging mechanism — the full source is never held in
    memory.
    """
    if staging_dir.exists() and any(staging_dir.iterdir()):
        raise RetailAdapterError(
            f"Staging directory is not empty: {staging_dir}. Remove only this "
            "generated staging directory, then rerun the adapter."
        )
    staging_dir.mkdir(parents=True, exist_ok=True)

    flagged_skus = _flagged_sku_ids(flags_path)

    usecols = sorted(REQUIRED_TRANSACTION_COLUMNS)
    reader = pd.read_csv(source_path, usecols=usecols, chunksize=CHUNK_SIZE)
    subset, total_scanned, general_kept, flagged_kept = _deterministic_subset(
        reader, flagged_skus, SUBSET_TARGET_ROWS, SUBSET_FLAGGED_PER_SKU, SUBSET_SEED
    )
    if subset.empty:
        raise RetailAdapterError("No valid transaction rows were found in the official source.")

    # ---- Validate / coerce the sampled rows, then aggregate to SKU-day ------
    dates = pd.to_datetime(subset["date"], errors="coerce")
    quantity = pd.to_numeric(subset["quantity"], errors="coerce")
    total_value = pd.to_numeric(subset["total_value"], errors="coerce")
    unit_price = pd.to_numeric(subset["unit_price"], errors="coerce")
    sku = subset["sku_id"].astype("string").str.strip()
    valid = dates.notna() & sku.notna() & sku.ne("") & quantity.gt(0)
    valid_rows = int(valid.sum())
    invalid_rows = int((~valid).sum())

    part = pd.DataFrame({
        "date": dates[valid].dt.normalize(),
        "sku_id": sku[valid].astype(str),
        "units_sold": quantity[valid],
        "revenue": total_value[valid],
        "price_x_units": unit_price[valid] * quantity[valid],
        "promo_flag": (
            subset.loc[valid, "promo_id"].astype("string").str.strip().notna()
            & subset.loc[valid, "promo_id"].astype("string").str.strip().ne("")
        ).astype(int),
    })
    grouped = (
        part.groupby(["date", "sku_id"], as_index=False)
        .agg(
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            price_x_units=("price_x_units", "sum"),
            promo_flag=("promo_flag", "max"),
        )
    )
    _write_partition(grouped, staging_dir)
    min_date, max_date = part["date"].min(), part["date"].max()
    source_skus: set[str] = set(part["sku_id"].unique())

    first_seen: Dict[str, pd.Timestamp] = {}
    output_rows = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        header = True
        for partition in sorted(staging_dir.glob("*.csv")):
            partial = pd.read_csv(partition, parse_dates=["date"])
            daily = (
                partial.groupby(["date", "sku_id"], as_index=False)
                .agg(
                    units_sold=("units_sold", "sum"),
                    revenue=("revenue", "sum"),
                    price_x_units=("price_x_units", "sum"),
                    promo_flag=("promo_flag", "max"),
                )
                .sort_values(["date", "sku_id"])
            )
            daily["unit_price"] = daily["price_x_units"] / daily["units_sold"]
            daily = daily[["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"]]
            daily.to_csv(stream, index=False, header=header)
            header = False
            output_rows += len(daily)
            starts = daily.groupby("sku_id")["date"].min()
            for sku_id, first_date in starts.items():
                if sku_id not in first_seen or first_date < first_seen[sku_id]:
                    first_seen[sku_id] = first_date

    return {
        "source_rows": total_scanned,
        "source_scanned_rows": total_scanned,
        "sampled_rows": int(len(subset)),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "general_reservoir_rows": general_kept,
        "flagged_floor_rows": int(sum(flagged_kept.values())),
        "flagged_skus_represented": len(flagged_kept),
        "sampling_method": (
            "deterministic reservoir (non-flagged) + per-SKU floor for "
            "ground-truth-flagged SKUs; fixed seed"
        ),
        "sample_seed": SUBSET_SEED,
        "sample_target_rows": SUBSET_TARGET_ROWS,
        "output_rows": output_rows,
        "source_sku_count": len(source_skus),
        "date_min": str(min_date.date()),
        "date_max": str(max_date.date()),
        "first_seen": {sku_id: str(value.date()) for sku_id, value in first_seen.items()},
    }


def _build_sku_master(source_path: Path, first_seen: Dict[str, str], destination: Path) -> int:
    source = pd.read_csv(source_path)
    required = {"sku_id", "category", "subcategory", "cost_price", "unit_price"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise RetailAdapterError(f"sku_master.csv missing required columns: {missing}")
    out = pd.DataFrame({
        "sku_id": source["sku_id"].astype(str),
        "category": source["category"],
        "subcategory": source["subcategory"],
        "launch_date": source["sku_id"].astype(str).map(first_seen),
        "unit_cost": pd.to_numeric(source["cost_price"], errors="coerce"),
        "list_price": pd.to_numeric(source["unit_price"], errors="coerce"),
    }).drop_duplicates(subset=["sku_id"], keep="first").sort_values("sku_id")
    # Referential integrity: keep ONLY the SKUs referenced by the sampled sales
    # transactions (first_seen is derived solely from the sampled fact table).
    keep_skus = set(first_seen.keys())
    out = out[out["sku_id"].isin(keep_skus)]
    out.to_csv(destination, index=False)
    return len(out)


def _build_calendar(
    promotion_path: Path, date_min: str, date_max: str, destination: Path
) -> int:
    promotions = pd.read_csv(promotion_path)
    dates = pd.DataFrame({"date": pd.date_range(date_min, date_max, freq="D")})
    dates["week"] = dates["date"].dt.isocalendar().week.astype(int)
    dates["month"] = dates["date"].dt.month
    dates["season"] = dates["month"].map(_season)
    # The source has no verified holiday calendar. Keep this unavailable rather
    # than inventing national-holiday labels.
    dates["is_holiday"] = pd.NA
    dates["promo_event"] = pd.NA
    if {"promo_name", "start_date", "end_date"}.issubset(promotions.columns):
        for row in promotions[["promo_name", "start_date", "end_date"]].itertuples(index=False):
            start, end = pd.to_datetime(row.start_date, errors="coerce"), pd.to_datetime(row.end_date, errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            mask = dates["date"].between(start.normalize(), end.normalize())
            dates.loc[mask & dates["promo_event"].isna(), "promo_event"] = str(row.promo_name)
    dates.to_csv(destination, index=False)
    return len(dates)


def _build_inventory(source_path: Path, snapshot_date: str, keep_skus: set, destination: Path) -> int:
    source = pd.read_csv(source_path)
    required = {"sku_id", "stock_on_hand", "reorder_point", "safety_stock"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise RetailAdapterError(f"inventory_snapshot.csv missing required columns: {missing}")
    src = source.copy()
    src["sku_id"] = src["sku_id"].astype(str)
    # Referential integrity: keep inventory only for SKUs that appear in the
    # sampled sales transactions.
    if keep_skus:
        src = src[src["sku_id"].isin(set(keep_skus))]
    out = (
        src.assign(
            stock_on_hand=pd.to_numeric(src["stock_on_hand"], errors="coerce"),
            reorder_point=pd.to_numeric(src["reorder_point"], errors="coerce"),
            safety_stock=pd.to_numeric(src["safety_stock"], errors="coerce"),
        )
        .groupby("sku_id", as_index=False)
        .agg(
            on_hand_units=("stock_on_hand", "sum"),
            reorder_point=("reorder_point", "sum"),
            safety_stock_units=("safety_stock", "sum"),
        )
    )
    out.insert(0, "date", snapshot_date)
    out["on_order_units"] = 0
    out["lead_time_days"] = LEAD_TIME_DAYS_ASSUMPTION
    out = out[["date", "sku_id", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]]
    out.sort_values("sku_id").to_csv(destination, index=False)
    return len(out)


def prepare_official_retail_inputs(
    source_dir: Path = SOURCE_DIR,
    raw_dir: Path = DATA_RAW,
    staging_dir: Path = STAGING_DIR,
) -> Dict[str, Any]:
    """Create the four canonical FORESIGHT raw CSVs from the official retail ZIP extract."""
    source_dir, raw_dir, staging_dir = Path(source_dir), Path(raw_dir), Path(staging_dir)
    paths = _require_source_files(source_dir)
    for filename in ("sales_daily.csv", "sku_master.csv", "calendar.csv", "inventory_snapshots.csv"):
        if (raw_dir / filename).exists():
            raise RetailAdapterError(
                f"Refusing to overwrite existing canonical input: {raw_dir / filename}. "
                "Archive or move it deliberately before running the adapter."
            )

    sales_summary = _build_sales_daily(
        paths["sales_transactions"],
        source_dir / GROUND_TRUTH_FLAGS_FILE,
        staging_dir,
        raw_dir / "sales_daily.csv",
    )
    sampled_first_seen = sales_summary.get("first_seen") or {}
    sku_rows = _build_sku_master(paths["sku_master"], sales_summary.pop("first_seen"), raw_dir / "sku_master.csv")
    calendar_rows = _build_calendar(
        paths["promotions"], sales_summary["date_min"], sales_summary["date_max"], raw_dir / "calendar.csv"
    )
    inventory_rows = _build_inventory(
        paths["inventory_snapshot"],
        sales_summary["date_max"],
        set(sampled_first_seen.keys()),
        raw_dir / "inventory_snapshots.csv",
    )

    manifest = {
        "report": "Official retail source adapter",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "directory": str(source_dir),
            "version": "retail_contaminated_dataset",
            "sales_source_file": "sales_transactions.csv",
            "ground_truth_file": "sku_inventory_flags.csv",
        },
        "sampling": {
            "target_rows": SUBSET_TARGET_ROWS,
            "seed": SUBSET_SEED,
            "flagged_per_sku_floor": SUBSET_FLAGGED_PER_SKU,
            "sampled_rows": int(sales_summary.get("sampled_rows", 0)),
            "general_reservoir_rows": int(sales_summary.get("general_reservoir_rows", 0)),
            "flagged_floor_rows": int(sales_summary.get("flagged_floor_rows", 0)),
            "flagged_skus_represented": int(sales_summary.get("flagged_skus_represented", 0)),
            "method": "deterministic reservoir (non-flagged) + per-SKU floor for "
                      "ground-truth-flagged SKUs (sku_inventory_flags.csv); official source only",
        },
        "sales_aggregation": {
            "grain": "date x sku_id, aggregated across all stores",
            **sales_summary,
        },
        "canonical_output_rows": {
            "sku_master": sku_rows,
            "calendar": calendar_rows,
            "inventory_snapshots": inventory_rows,
        },
        "derived_fields_and_assumptions": {
            "promo_flag": "1 when a source transaction has a nonblank promo_id; daily SKU values use max.",
            "unit_price": "quantity-weighted source unit_price within each SKU-day.",
            "launch_date": "first observed transaction date per SKU; source does not supply launch dates.",
            "calendar": "week, month and season derived from transaction dates; is_holiday left unavailable because the source has no holiday calendar.",
            "inventory_grain": "source store x SKU inventory aggregated across stores to SKU for chain-level planning.",
            "on_order_units": "0 because the source contains no incoming-order field.",
            "lead_time_days": f"{LEAD_TIME_DAYS_ASSUMPTION}, a documented planning assumption because the source contains no lead-time field.",
        },
    }
    (raw_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare canonical FORESIGHT inputs from the official retail source.")
    parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    parser.add_argument("--raw-dir", default=str(DATA_RAW))
    parser.add_argument("--staging-dir", default=str(STAGING_DIR))
    args = parser.parse_args(argv)
    try:
        result = prepare_official_retail_inputs(args.source_dir, args.raw_dir, args.staging_dir)
    except RetailAdapterError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    print("Official retail inputs prepared successfully")
    print(json.dumps(result["sales_aggregation"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
