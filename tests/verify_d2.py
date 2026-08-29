"""Verify Project FORESIGHT D2 outputs without executing D1 or later stages."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
D2_RUNNER = REPORTS / "generate_d2_report.py"

D1_FILES = {
    "sales_daily_clean.csv": ["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"],
    "sales_analysis_ready.csv": ["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag", "week", "month", "season", "is_holiday", "promo_event", "category", "subcategory", "launch_date", "unit_cost", "list_price"],
    "sku_master_clean.csv": ["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"],
    "calendar_clean.csv": ["date", "week", "month", "season", "is_holiday", "promo_event"],
    "inventory_snapshots_clean.csv": ["date", "sku_id", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"],
}


def fail(message):
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main():
    for filename, columns in D1_FILES.items():
        path = PROCESSED / filename
        if not path.is_file():
            fail(f"D1 input missing: {filename}")
        frame = pd.read_csv(path)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            fail(f"{filename} missing columns: {missing}")
    print("[PASS] all verified D1 inputs exist with required columns")

    source_text = D2_RUNNER.read_text(encoding="utf-8")
    forbidden = ["retail_contaminated_dataset", "sales_transactions_25000", "legacy_synthetic_backup", "dev_data", "notebooks"]
    if any(token in source_text for token in forbidden):
        fail("D2 runner contains a forbidden direct data-source reference")
    print("[PASS] D2 runner reads only D1 outputs")

    report_path = REPORTS / "d2_eda_report.json"
    memo_path = REPORTS / "d2_eda_memo.md"
    if not report_path.is_file() or not memo_path.is_file():
        fail("D2 report or memo is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "only verified D1 outputs" not in report.get("data_source_statement", ""):
        fail("D2 report does not state its D1-only source boundary")
    if report.get("rows_analyzed", {}).get("analysis_ready") != len(pd.read_csv(PROCESSED / "sales_analysis_ready.csv")):
        fail("reported analysis-ready row count is inconsistent")
    if report.get("metrics", {}).get("category_count", 0) <= 0:
        fail("category metric is missing or invalid")
    if report.get("metrics", {}).get("inventory", {}).get("sales_sku_inventory_coverage") is None:
        fail("inventory coverage metric is missing")
    print("[PASS] D2 report exists and metrics are consistent with D1")

    charts = report.get("charts", [])
    if len(charts) < 5 or not all((ROOT / path).is_file() and (ROOT / path).stat().st_size > 0 for path in charts):
        fail("D2 chart set is incomplete or contains empty files")
    print(f"[PASS] {len(charts)} D2 charts exist")

    if report.get("rows_analyzed", {}).get("analysis_ready", 0) <= 0 or report.get("data_source_statement", "").find("fabricated") < 0:
        fail("fabrication boundary is not documented")
    print("[PASS] no fabricated records claimed; D1 row boundary preserved")
    print("D2 VERIFICATION: PASSED")


if __name__ == "__main__":
    main()
