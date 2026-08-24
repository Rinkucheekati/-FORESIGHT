"""D1 reproducibility verification (STEP 9).

Run from the repository root:  python tests/verify_d1.py

Checks the seven STEP 9 requirements WITHOUT fabricating data and WITHOUT
executing D2/D3/D4/D5/D6. The repository may contain synthetic development
inputs, so missing-input checks always use isolated temporary directories.
"""
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.pipeline import (  # noqa: E402
    OUTPUT_FILES, QUALITY_REPORT_FILE, MissingOfficialInputsError,
    SchemaValidationError, run_pipeline, _cli,
)
from src.paths import DATA_RAW, DATA_PROCESSED  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# 5) imports -----------------------------------------------------------------
import src.pipeline as pl  # noqa: E402
check("5. pipeline imports successfully", True, f"module={pl.__name__}")

OFFICIAL_COLS = {
    "sales_daily": ["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"],
    "sku_master": ["sku_id", "category", "subcategory", "launch_date",
                   "unit_cost", "list_price"],
    "calendar": ["date", "week", "month", "season", "is_holiday", "promo_event"],
    "inventory_snapshots": ["date", "sku_id", "on_hand_units", "on_order_units",
                            "lead_time_days", "reorder_point"],
}

# 3) official schemas enforced ----------------------------------------------
wrong = pd.DataFrame({"foo": [1, 2]})
try:
    pl.validate_raw_schemas({"sales_daily": wrong})
    check("3a. schema enforcement raises on wrong/missing schema", False)
except SchemaValidationError as e:
    check("3a. schema enforcement raises on wrong/missing schema", True, str(e)[:70])

empty_frames = {k: pd.DataFrame(columns=v) for k, v in OFFICIAL_COLS.items()}
rep = pl.validate_raw_schemas(empty_frames)
exact = all(sorted(rep[k]["columns"]) == sorted(v) for k, v in OFFICIAL_COLS.items())
check("3b. exact official column names validated", exact)

extra_frames = {k: df.copy() for k, df in empty_frames.items()}
extra_frames["sales_daily"]["legacy_col"] = []
rep2 = pl.validate_raw_schemas(extra_frames)
check("3c. unexpected column REPORTED (never silently dropped)",
      "legacy_col" in rep2["sales_daily"]["unexpected_columns"])

dup_frames = {k: df.copy() for k, df in empty_frames.items()}
dup_frames["calendar"] = dup_frames["calendar"].rename(
    columns={"week": "date"})  # duplicate 'date' header
try:
    pl.validate_raw_schemas(dup_frames)
    check("3d. duplicate column headers raise SchemaValidationError", False)
except SchemaValidationError:
    check("3d. duplicate column headers raise SchemaValidationError", True)

# 1) current synthetic inputs are accepted without changing the contract ------
with tempfile.TemporaryDirectory() as td:
    synthetic_processed = Path(td) / "processed"
    synthetic_result = pl.run_pipeline(DATA_RAW, synthetic_processed)
    check("1. synthetic development inputs pass D1 in isolation",
          synthetic_result["status"] == "success"
          and synthetic_result["processed_dir"] == str(synthetic_processed))
    check("1b. synthetic D1 outputs use the official filenames",
          sorted(Path(path).name for path in synthetic_result["outputs_written"])
          == sorted(expected for expected in [
              "sales_daily_clean.csv", "sku_master_clean.csv",
              "calendar_clean.csv", "inventory_snapshots_clean.csv",
              "sales_analysis_ready.csv", "d1_data_quality_report.json",
          ]))

# 2) missing official inputs handled safely ----------------------------------
with tempfile.TemporaryDirectory() as td:
    empty = Path(td) / "empty"
    empty_processed = Path(td) / "empty_processed"
    empty.mkdir()
    try:
        pl.run_pipeline(empty, empty_processed)
        check("2a. isolated empty dir -> MissingOfficialInputsError", False)
    except MissingOfficialInputsError as e:
        names_ok = all(n in str(e) for n in
                       ("sales_daily", "sku_master", "calendar", "inventory_snapshots"))
        check("2a. isolated empty dir -> MissingOfficialInputsError", names_ok, str(e)[:80])

    # 2) no legacy fallback: legacy-named files alone must NOT satisfy the gate
    leg = Path(td) / "legacy_only"
    legacy_processed = Path(td) / "legacy_processed"
    leg.mkdir()
    for n in ("sales_daily.csv", "sku_master.csv", "inventory_snapshots.csv"):
        (leg / n).write_bytes(b"")           # placeholder, never read
    try:
        pl.run_pipeline(leg, legacy_processed)
        check("3. legacy filenames alone do NOT satisfy gate", False)
    except MissingOfficialInputsError:
        check("3. legacy filenames alone do NOT satisfy gate", True)

    # Only inspect output destinations; the CSV placeholders above are inputs
    # created by this verifier and must not count as pipeline writes.
    leftovers = [
        path for output_dir in (empty_processed, legacy_processed)
        for path in output_dir.rglob("*") if path.is_file()
    ]
    check("2b. failed runs write NOTHING", len(leftovers) == 0, str(leftovers)[:60])

# 4) deterministic output filenames ------------------------------------------
expected = [
    "sales_daily_clean.csv", "sku_master_clean.csv", "calendar_clean.csv",
    "inventory_snapshots_clean.csv", "sales_analysis_ready.csv",
]
check("4. output filenames deterministic",
      [OUTPUT_FILES[k] for k in OUTPUT_FILES] == expected
      and QUALITY_REPORT_FILE == "d1_data_quality_report.json")

# 7) no D2/D3/D4 execution inside D1 ----------------------------------------
src_txt = (REPO / "src" / "pipeline.py").read_text(encoding="utf-8")
bad_imports = [m for m in ("src.eda", "src.forecast", "src.risk", "import app") 
               if m in src_txt]
check("7. pipeline has no D2+ imports", not bad_imports, str(bad_imports))

# 6) CLI entry point ---------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    cli_raw = Path(td) / "raw"
    cli_processed = Path(td) / "processed"
    cli_raw.mkdir()
    r = subprocess.run([
        sys.executable, "-m", "src.pipeline",
        "--raw-dir", str(cli_raw),
        "--processed-dir", str(cli_processed),
    ], cwd=str(REPO), capture_output=True, text=True)
check("6a. CLI isolated empty dir exit code == 2",
      r.returncode == 2 and "Official FORESIGHT inputs are missing" in r.stderr,
      f"rc={r.returncode}")

r = subprocess.run([sys.executable, "-m", "src.pipeline", "--help"],
                   cwd=str(REPO), capture_output=True, text=True)
check("6b. CLI --help works", r.returncode == 0 and "--raw-dir" in r.stdout)

print()
failed = [n for n, ok, _ in results if not ok]
print(f"D1 VERIFICATION: {len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
print("ALL CHECKS PASSED")