"""Run D4 inventory risk scoring on the official FORESIGHT data.

All calculations are delegated to ``src.risk``. D4 reads the official D1
analysis-ready outputs and the official D3 weekly forecasts (upstream: the
official retail dataset provided for Project FORESIGHT, reduced
deterministically to 25,000 transactions, seed 42). Risk scores, decisions,
recommendations, and rupee values are computed from that official data; they
are model-derived planning recommendations, not official Zidio or NorthBay
Living business results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import risk

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RISK_PATH = PROCESSED_DIR / "inventory_risk.csv"
RECOMMENDATIONS_PATH = PROCESSED_DIR / "recommendations.csv"
REPORT_PATH = REPO_ROOT / "reports" / "d4_risk_report.json"


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
    config = risk.RiskConfig()
    config.validate()
    scored = risk.score_all_skus(None, config, PROCESSED_DIR)
    report = risk.create_risk_report(scored, config)
    if scored.get("status") != risk.STATUS_READY:
        report["data_source_statement"] = (
            "Data source: the official retail dataset provided for Project "
            "FORESIGHT (deterministic 25,000-transaction selection, seed 42), "
            "processed through the official D1 pipeline. D4 did not reach "
            "READY status on this data, so no risk score, decision, or rupee "
            "value here is a complete or official result."
        )
        REPORT_PATH.write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise risk.InsufficientDataError("D4 did not reach READY status.")

    rows = pd.DataFrame(scored["scored_skus"])
    rows = rows.sort_values(
        ["rupee_value_at_stake", "stockout_risk", "overstock_risk", "sku_id"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    rows.to_csv(RISK_PATH, index=False)

    rows = rows.copy()
    rows["recommendation"] = rows["decision"].map({
        "REORDER_NOW": "Reorder now",
        "MARKDOWN_CLEAR": "Markdown clear",
        "WATCH_VOLATILE": "Watch volatile",
        "HEALTHY": "Healthy",
    }).fillna(rows["decision"])

    priority_order = {"REORDER_NOW": "P1", "MARKDOWN_CLEAR": "P2", "WATCH_VOLATILE": "P3", "HEALTHY": "P4"}
    rows["priority"] = rows["decision"].map(priority_order).fillna("P4")

    recommendation_columns = [
        "sku_id", "category", "subcategory", "decision", "recommendation",
        "priority", "decision_reason", "rupee_value_at_stake",
        "stockout_risk", "overstock_risk", "coverage_weeks",
        "demand_rate_weekly", "forecast_rate_weekly",
        "preferred_demand_source", "data_quality_flag",
    ]
    rows[recommendation_columns].to_csv(RECOMMENDATIONS_PATH, index=False)

    report["data_source_statement"] = (
        "Data source: the official retail dataset provided for Project "
        "FORESIGHT (deterministic 25,000-transaction selection, seed 42), "
        "processed through the official D1 pipeline with the official D3 "
        "weekly forecasts. Risk scores, decisions, recommendations, and "
        "rupee values are computed from this official data; they are "
        "model-derived planning recommendations, not official Zidio or "
        "NorthBay Living business results."
    )
    report["output_files"] = {
        "inventory_risk": str(RISK_PATH.relative_to(REPO_ROOT)),
        "recommendations": str(RECOMMENDATIONS_PATH.relative_to(REPO_ROOT)),
    }
    report["prioritised_sku_count"] = len(rows)
    report["recommendation_counts"] = {
        decision: int((rows["decision"] == decision).sum())
        for decision in report["decision_summary"]["grid"]
    }
    report["decision_summary"]["counts"] = report["recommendation_counts"]
    REPORT_PATH.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"D4_OK skus={len(rows)} decisions="
        f"{report['decision_summary']['counts']} "
        f"rupee_at_stake={report['rupee_value_summary']['total_rupee_value_at_stake']:.2f}"
    )


if __name__ == "__main__":
    main()
