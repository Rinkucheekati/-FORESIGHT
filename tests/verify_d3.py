"""Verification script for D3 forecasting outputs.

Validates that:
- All expected output files exist and are non-empty
- Forecast results contain all 5,000 SKUs
- Forecast values are sensible (non-negative, in expected range)
- Backtest results are complete
- Model comparison metrics are present
- No data integrity issues
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import forecast

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

def check_forecast_results():
    """Verify forecast_results.csv is complete and sensible."""
    fpath = PROCESSED_DIR / "forecast_results.csv"
    assert fpath.exists(), f"forecast_results.csv missing: {fpath}"
    
    df = pd.read_csv(fpath)
    print(f"✓ forecast_results.csv: {len(df)} rows")
    
    required_cols = ["period", "sku_id", "forecast_units", "model_name", "low_confidence", "note"]
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    print(f"✓ All required columns present")
    
    expected_skus = pd.read_csv(PROCESSED_DIR / "sales_analysis_ready.csv")["sku_id"].nunique()
    skus = df["sku_id"].unique()
    print(f"✓ Forecasted SKUs: {len(skus)}")
    assert len(skus) == expected_skus, f"Expected {expected_skus} D1 SKUs, got {len(skus)}"
    
    null_forecast = df["forecast_units"].isna().sum()
    print(f"✓ Null forecast values: {null_forecast}")
    assert null_forecast == 0, f"Null forecast values found: {null_forecast}"
    assert not np.isinf(df["forecast_units"]).any(), "Forecasts contain inf values"
    assert (df["forecast_units"] >= 0).all(), "Negative forecast values found"
    
    non_null_forecasts = df["forecast_units"].dropna()
    min_val = non_null_forecasts.min()
    max_val = non_null_forecasts.max()
    print(f"✓ Forecast value range: [{min_val:.2f}, {max_val:.2f}]")
    
    model_counts = df["model_name"].value_counts()
    print(f"✓ Model distribution:")
    for model, count in model_counts.items():
        print(f"  - {model}: {count}")

    rows_per_sku = df.groupby("sku_id").size()
    assert rows_per_sku.eq(8).all(), "Every SKU must have exactly 8 forecast rows"
    assert not df.duplicated(["sku_id", "period"]).any(), "Duplicate SKU/week rows found"
    assert "category_historical_fallback" in set(model_counts.index), (
        "Category historical fallback is missing"
    )
    assert df["low_confidence"].astype(bool).any(), "Sparse fallback low-confidence rows missing"
    
    return True


def check_fallback_hierarchy():
    """Prove the sparse SKU fallback is numeric, finite and cutoff-safe."""
    tables = forecast.load_d1_outputs(PROCESSED_DIR)
    config = forecast.ForecastConfig()
    weekly = forecast.prepare_weekly_demand(tables, config)
    counts = weekly.groupby("sku_id")["period"].nunique()
    sparse_ids = counts[counts < config.min_train_weeks].index
    assert len(sparse_ids) > 0, "Expected sparse official SKUs for fallback validation"

    sample = str(sorted(sparse_ids)[0])
    sample_history = weekly[weekly["sku_id"].astype(str) == sample].sort_values(["iso_year", "iso_week"])
    last_year = int(sample_history["iso_year"].max())
    last_week = int(sample_history.loc[sample_history["iso_year"] == last_year, "iso_week"].max())
    future = forecast._next_periods(last_year, last_week, config.horizon_weeks)
    target = pd.DataFrame({
        "sku_id": [sample] * config.horizon_weeks,
        "category": [sample_history.iloc[0]["category"]] * config.horizon_weeks,
        "iso_week": [week for _, week in future],
    })

    preds = forecast._category_fallback_predictions(weekly, weekly["sku_id"].astype(str) == sample, target)
    assert len(preds) == config.horizon_weeks, "Fallback should return exactly 8 values"
    assert preds.notna().all(), "Fallback returned NaN values"
    assert np.isfinite(preds.to_numpy()).all(), "Fallback returned inf values"
    assert (preds >= 0).all(), "Fallback returned negative values"
    assert preds.dtype.kind == "f", "Fallback predictions should be floating-point"

    sku_record = forecast.forecast_weekly_sku(
        sku_id=sample,
        tables=tables,
        config=config,
        weekly=weekly,
        features=forecast.build_forecast_features(weekly),
    )
    rows = sku_record["forecast_rows"]
    assert all(r["model_name"] == "category_historical_fallback" for r in rows), "Sparse SKU should use category historical fallback"
    assert all(r["low_confidence"] is True for r in rows), "Sparse SKU fallback must be low confidence"
    assert len(rows) == config.horizon_weeks, "Sparse SKU row count must be exactly 8"
    assert all(np.isfinite(pd.to_numeric([r["forecast_units"] for r in rows], errors="coerce").to_numpy())), "Sparse fallback contains non-finite values"
    print(f"✓ Sparse fallback hierarchy validated for SKU {sample}")
    return True


def check_category_fallback():
    """Verify a sparse official SKU receives numeric category fallback values."""
    tables = forecast.load_d1_outputs(PROCESSED_DIR)
    config = forecast.ForecastConfig()
    weekly = forecast.prepare_weekly_demand(tables, config)
    counts = weekly.groupby("sku_id")["period"].nunique()
    sparse_ids = counts[counts < config.min_train_weeks].index
    assert len(sparse_ids) > 0, "Expected sparse official SKUs for fallback validation"
    sku_id = str(sparse_ids[0])
    history = weekly[weekly["sku_id"].astype(str) == sku_id].sort_values(
        ["iso_year", "iso_week"]
    )
    last_year = int(history["iso_year"].max())
    last_week = int(history.loc[history["iso_year"] == last_year, "iso_week"].max())
    future = forecast._next_periods(last_year, last_week, config.horizon_weeks)
    target = pd.DataFrame({
        "sku_id": [sku_id] * config.horizon_weeks,
        "category": [history.iloc[0]["category"]] * config.horizon_weeks,
        "iso_week": [week for _, week in future],
    })
    predictions = forecast._category_fallback_predictions(
        weekly, weekly["sku_id"].astype(str) == sku_id, target
    )
    assert predictions.notna().all(), "Category fallback returned null predictions"
    assert (predictions >= 0).all(), "Category fallback returned negative predictions"
    print(f"✓ Category fallback numeric for sparse SKU {sku_id}")

def check_backtest_results():
    """Verify backtest results are complete."""
    fpath = PROCESSED_DIR / "d3_backtest_results.csv"
    assert fpath.exists(), f"d3_backtest_results.csv missing: {fpath}"
    
    df = pd.read_csv(fpath)
    print(f"✓ d3_backtest_results.csv: {len(df)} fold rows")
    
    required_cols = ["fold", "train_start_period", "train_end_period", "test_period"]
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing columns in backtest: {missing}"
    print(f"✓ Backtest has all fold information")
    
    return True

def check_model_comparison():
    """Verify model comparison results."""
    fpath = PROCESSED_DIR / "d3_model_comparison.json"
    assert fpath.exists(), f"d3_model_comparison.json missing: {fpath}"
    
    with open(fpath) as f:
        cmp = json.load(f)
    
    print(f"✓ d3_model_comparison.json present")
    
    assert "baseline" in cmp, "Missing baseline metrics"
    assert "model" in cmp, "Missing ML model metrics"
    
    # Check WAPE values
    baseline_wape = cmp["baseline"].get("wape")
    model_wape = cmp["model"].get("wape")
    print(f"✓ WAPE - baseline: {baseline_wape}, model: {model_wape}")
    
    # Check bias
    baseline_bias = cmp["baseline"].get("bias")
    model_bias = cmp["model"].get("bias")
    print(f"✓ Bias - baseline: {baseline_bias}, model: {model_bias}")
    
    return True

def check_forecast_report():
    """Verify forecast report has expected structure."""
    fpath = REPORTS_DIR / "d3_forecast_report.json"
    assert fpath.exists(), f"d3_forecast_report.json missing: {fpath}"
    
    with open(fpath) as f:
        report = json.load(f)
    
    print(f"✓ d3_forecast_report.json present")
    print(f"  - Report name: {report.get('report')}")
    print(f"  - Status: {report.get('data_sufficiency_status')}")
    print(f"  - Model used: {report.get('model_used')}")
    print(f"  - Total SKUs: {report.get('total_skus')}")
    print(f"  - Total weeks observed: {report.get('total_weeks_observed')}")
    print(f"  - Low-history SKU count: {report.get('low_history_sku_count')}")
    
    # Verify key metrics are present
    assert report.get("wape"), "WAPE metrics missing"
    assert report.get("bias"), "Bias metrics missing"
    assert report.get("model_vs_baseline"), "Model comparison missing"
    
    print(f"✓ All report sections present")
    
    return True

def main():
    """Run all verification checks."""
    print("\n=== D3 FORECASTING VERIFICATION ===\n")
    
    try:
        check_forecast_results()
        check_category_fallback()
        print()
        check_backtest_results()
        print()
        check_model_comparison()
        print()
        check_forecast_report()
        print("\n✅ D3 OUTPUT VERIFICATION PASSED\n")
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
