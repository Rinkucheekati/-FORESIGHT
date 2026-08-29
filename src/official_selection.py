"""Create the reproducible official FORESIGHT transaction selection layer."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.paths import REPO_ROOT

SOURCE_DIR = REPO_ROOT / "data" / "raw" / "retail_contaminated_dataset"
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "official_selected"
SOURCE_FILE = SOURCE_DIR / "sales_transactions.csv"
OUTPUT_FILE = OUTPUT_DIR / "sales_transactions_25000.csv"
MANIFEST_FILE = OUTPUT_DIR / "selection_manifest.json"
TARGET_ROWS = 25_000
RANDOM_SEED = 42

SUPPORTING_FILES = (
    "sku_master.csv",
    "customer_master.csv",
    "store_master.csv",
    "promotions.csv",
    "inventory_snapshot.csv",
    "sku_inventory_flags.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_selection() -> dict:
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(f"Official source is missing: {SOURCE_FILE}")
    missing = [name for name in SUPPORTING_FILES if not (SOURCE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Official supporting files are missing: {missing}")

    source = pd.read_csv(SOURCE_FILE)
    if len(source) < TARGET_ROWS:
        raise ValueError("Official source contains fewer rows than the selection target")

    selected = source.sample(n=TARGET_ROWS, replace=False, random_state=RANDOM_SEED)
    selected = selected.reset_index(drop=True)
    if list(selected.columns) != list(source.columns):
        raise ValueError("Selected columns differ from the official source columns")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_FILE, index=False)

    manifest = {
        "project_name": "Project FORESIGHT - Demand & Inventory Intelligence",
        "source_file": str(SOURCE_FILE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_row_count": int(len(source)),
        "selected_row_count": int(len(selected)),
        "target_row_count": TARGET_ROWS,
        "random_seed": RANDOM_SEED,
        "selection_method": "pandas DataFrame.sample(n=25000, replace=False, random_state=42)",
        "source_sha256": sha256_file(SOURCE_FILE),
        "selected_file_sha256": sha256_file(OUTPUT_FILE),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_columns": list(source.columns),
        "supporting_files": list(SUPPORTING_FILES),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(create_selection(), indent=2, sort_keys=True))
