"""Run D2 EDA on the current synthetic development inputs.

This creates report artifacts only from D1 outputs. It does not create or
claim official Zidio or NorthBay Living business results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import eda

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORT_DIR = REPO_ROOT / "reports"
CHART_DIR = REPORT_DIR / "d2_charts"


def _jsonable(value):
    if isinstance(value, (pd.Timestamp,)):
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


def _save_chart(name: str, title: str, x, y, kind: str = "line") -> str:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / f"{name}.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    if kind == "bar":
        ax.bar([str(item) for item in x], y, color="#176b87")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.plot(x, y, color="#176b87", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Units")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(REPO_ROOT))


def main() -> None:
    tables = eda.load_analysis_ready_data(PROCESSED_DIR)
    quality = eda.summarize_data_quality(tables)
    demand = eda.analyze_demand_patterns(tables)
    seasonality = eda.analyze_seasonality(tables)
    trend = eda.analyze_trend(tables)
    movers = eda.top_movers(tables)
    dead_stock = eda.dead_stock(tables)
    drivers = eda.analyze_drivers(tables)
    insights = eda.generate_business_insights(
        tables, demand, seasonality, trend, movers, dead_stock, drivers
    )
    report = eda.create_d2_report(
        tables, quality, demand, seasonality, trend, movers,
        dead_stock, drivers, insights,
    )

    sales = tables["sales_daily"].copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales_daily = sales.groupby("date", as_index=False)["units_sold"].sum()
    sales_daily["week_start"] = sales_daily["date"].dt.to_period("W").dt.start_time
    weekly = sales_daily.groupby("week_start", as_index=False)["units_sold"].sum()
    monthly = sales_daily.assign(month=sales_daily["date"].dt.to_period("M").astype(str))
    monthly = monthly.groupby("month", as_index=False)["units_sold"].sum()
    report["weekly_trend"] = weekly.to_dict(orient="records")
    report["monthly_trend"] = monthly.to_dict(orient="records")
    report["synthetic_data_disclaimer"] = (
        "Synthetic development data only. No finding, metric, or recommendation "
        "in this report represents official Zidio or NorthBay Living results."
    )
    report["charts"] = [
        _save_chart("daily_demand", "Daily demand", sales_daily["date"], sales_daily["units_sold"]),
        _save_chart("weekly_demand", "Weekly demand", weekly["week_start"].dt.strftime("%Y-%m-%d"), weekly["units_sold"]),
        _save_chart("monthly_demand", "Monthly demand", monthly["month"], monthly["units_sold"], "bar"),
        _save_chart("top_movers", "Top movers", [row["sku_id"] for row in movers["movers"]], [row["total_units"] for row in movers["movers"]], "bar"),
    ]

    (REPORT_DIR / "d2_eda_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    memo = eda.create_d2_memo(report)
    memo += "\n\n> **Synthetic development data only:** these computed findings are temporary development outputs and must not be reported as official Zidio or NorthBay Living results.\n"
    memo += "\n## Computed Insights\n\n"
    for index, insight in enumerate(insights, start=1):
        memo += f"{index}. **{insight['observation']}** {insight['evidence']} {insight['business_implication']} Recommended action: {insight['recommended_action']}\n"
    memo += "\n## Artifacts\n\n- `d2_eda_report.json` contains the computed report and trend tables.\n- `d2_charts/` contains the generated PNG charts.\n"
    (REPORT_DIR / "d2_eda_memo.md").write_text(memo, encoding="utf-8")
    print(f"D2_OK insights={len(insights)} charts={len(report['charts'])}")


if __name__ == "__main__":
    main()
