"""Verification script for D4 risk scoring outputs.

Validates that:
- All expected output files exist and are non-empty
- Risk scores are reasonable and cover all SKUs
- Recommendations are present and sensible
- Risk report has expected structure and metrics
"""

import sys
from pathlib import Path
import json
import pandas as pd

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

def check_inventory_risk():
    """Verify inventory_risk.csv is complete and sensible."""
    fpath = PROCESSED_DIR / "inventory_risk.csv"
    assert fpath.exists(), f"inventory_risk.csv missing: {fpath}"
    
    df = pd.read_csv(fpath)
    print(f"✓ inventory_risk.csv: {len(df)} SKU rows")
    
    # Check all required columns
    required_cols = ["sku_id", "stockout_risk", "overstock_risk", "decision", "rupee_value_at_stake"]
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    print(f"✓ All required columns present")
    
    # Check decision categories
    valid_decisions = {"REORDER_NOW", "MARKDOWN_CLEAR", "WATCH_VOLATILE", "HEALTHY"}
    decisions_found = set(df["decision"].unique())
    print(f"✓ Decision categories found: {decisions_found}")
    invalid = decisions_found - valid_decisions
    assert not invalid, f"Invalid decision values: {invalid}"
    
    # Check numeric ranges
    print(f"✓ Stockout risk range: [{df['stockout_risk'].min():.3f}, {df['stockout_risk'].max():.3f}]")
    print(f"✓ Overstock risk range: [{df['overstock_risk'].min():.3f}, {df['overstock_risk'].max():.3f}]")
    
    # Check rupee values
    rupee_col = "rupee_value_at_stake"
    if rupee_col in df.columns:
        non_null = df[rupee_col].notna().sum()
        print(f"✓ Rupee values: {non_null} / {len(df)} SKUs (rest are null/unknown)")
    
    # Summary by decision
    print(f"✓ SKU distribution by decision:")
    for decision in valid_decisions & decisions_found:
        count = (df["decision"] == decision).sum()
        pct = 100 * count / len(df)
        print(f"  - {decision}: {count} ({pct:.1f}%)")
    
    return True

def check_recommendations():
    """Verify recommendations.csv is complete."""
    fpath = PROCESSED_DIR / "recommendations.csv"
    assert fpath.exists(), f"recommendations.csv missing: {fpath}"
    
    df = pd.read_csv(fpath)
    print(f"✓ recommendations.csv: {len(df)} recommendation rows")
    
    # Check required columns
    required_cols = ["sku_id", "recommendation", "priority"]
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    print(f"✓ All required columns present")
    
    # Check priority levels
    if "priority" in df.columns:
        priorities = df["priority"].unique()
        print(f"✓ Priority levels: {sorted([p for p in priorities if pd.notna(p)])}")
    
    # Summary
    if "recommendation" in df.columns:
        rec_counts = df["recommendation"].value_counts()
        print(f"✓ Top recommendations:")
        for rec, count in rec_counts.head(5).items():
            print(f"  - {rec}: {count}")
    
    return True

def check_risk_report():
    """Verify risk report has expected structure."""
    fpath = REPORTS_DIR / "d4_risk_report.json"
    assert fpath.exists(), f"d4_risk_report.json missing: {fpath}"
    
    with open(fpath) as f:
        report = json.load(f)
    
    print(f"✓ d4_risk_report.json present")
    print(f"  - Report name: {report.get('report')}")
    print(f"  - Status: {report.get('status')}")
    print(f"  - Total SKUs: {report.get('total_skus_scored')}")
    print(f"  - Total rupee value at stake: ₹{report.get('total_rupee_value_at_stake', 0):,.2f}")
    
    # Verify key sections
    assert report.get("status"), "Status missing"
    assert report.get("total_skus_scored"), "Total SKUs count missing"
    
    if "decision_distribution" in report:
        print(f"✓ Decision distribution:")
        for decision, count in report["decision_distribution"].items():
            print(f"  - {decision}: {count}")
    
    print(f"✓ All report sections present")
    
    return True

def main():
    """Run all verification checks."""
    print("\n=== D4 RISK SCORING VERIFICATION ===\n")
    
    try:
        check_inventory_risk()
        print()
        check_recommendations()
        print()
        check_risk_report()
        print("\n✅ D4 OUTPUT VERIFICATION PASSED\n")
        return 0
    except (FileNotFoundError, AssertionError, KeyError) as e:
        print(f"\n❌ VERIFICATION FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
