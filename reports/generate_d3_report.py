"""Run the D3 weekly forecasting workflow on the official D1 outputs.

Forecasts and metrics are computed by ``src.forecast`` from the official D1
analysis-ready outputs. This runner does not fabricate or manually alter data.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import forecast

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FORECAST_PATH = PROCESSED_DIR / "forecast_results.csv"
BACKTEST_PATH = PROCESSED_DIR / "d3_backtest_results.csv"
COMPARISON_PATH = PROCESSED_DIR / "d3_model_comparison.json"
REPORT_PATH = REPO_ROOT / "reports" / "d3_forecast_report.json"


def _jsonable(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    tables = forecast.load_d1_outputs(PROCESSED_DIR)
    config = forecast.ForecastConfig(
        model_params={"max_iter": 60, "max_leaf_nodes": 15}
    )
    config.validate()
    weekly = forecast.prepare_weekly_demand(tables, config)
    features = forecast.build_forecast_features(weekly)
    readiness = forecast.check_forecast_data_readiness(tables, config, PROCESSED_DIR)
    if readiness["status"] != "ready":
        raise forecast.InsufficientHistoryError(
            "; ".join(readiness.get("reasons", []))
        )

    backtest = forecast.rolling_origin_backtest(features, config)
    comparison = forecast.compare_model_to_baseline(backtest)
    selection = forecast.select_best_forecaster(comparison)
    report = forecast.create_forecast_report(
        readiness, backtest, comparison, selection, config,
        low_history_sku_count=readiness["details"]["skus_low_history"],
    )

    start_time = time.time()
    print("[D3] Starting shared-model forecast for official SKUs...", file=sys.stderr)
    sku_results = forecast.forecast_all_skus(
        weekly=weekly,
        features=features,
        config=config,
        forecast_run_id="fc_official_d3_20260827",
    )
    forecast_rows = [
        row for result in sku_results for row in result["forecast_rows"]
    ]
    total_skus = len(sku_results)
    forecast_frame = pd.DataFrame(forecast_rows)
    forecast_frame.to_csv(FORECAST_PATH, index=False)
    pd.DataFrame(backtest["folds"]).to_csv(BACKTEST_PATH, index=False)
    COMPARISON_PATH.write_text(
        json.dumps(_jsonable(comparison), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report["data_source_statement"] = (
        "Data source: the official retail dataset provided for Project "
        "FORESIGHT, reduced deterministically to 25,000 transactions (seed 42) "
        "and processed through the official D1 pipeline. D3 used the complete "
        "official D1 cleaned sales output without an additional row or SKU "
        "reduction. No transaction values were fabricated or manually altered."
    )
    report["output_files"] = {
        "forecast_results": str(FORECAST_PATH.relative_to(REPO_ROOT)),
        "backtest_results": str(BACKTEST_PATH.relative_to(REPO_ROOT)),
        "model_comparison": str(COMPARISON_PATH.relative_to(REPO_ROOT)),
    }
    REPORT_PATH.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    
    total_elapsed = time.time() - start_time
    print(
        f"[D3] ✅ Forecast complete! {total_skus} SKUs in {total_elapsed:.0f}s "
        f"({total_elapsed/total_skus:.2f}s per SKU)",
        file=sys.stderr
    )
    print(
        f"D3_OK skus={forecast_frame['sku_id'].nunique()} "
        f"horizon={config.horizon_weeks} folds={len(backtest['folds'])} "
        f"selected={selection['selected_model']} "
        f"baseline_wape={comparison['baseline']['wape']:.6f} "
        f"model_wape={comparison['model']['wape']:.6f}"
    )


if __name__ == "__main__":
    main()
