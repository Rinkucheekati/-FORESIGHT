# FORESIGHT D2 — EDA & Data-Quality Memo

## Executive summary
See the D2 report structure for computed summaries.

## Data quality
See `src.eda.summarize_data_quality` outputs.

## Demand patterns
See `src.eda.analyze_demand_patterns` outputs.

## Seasonality / trend / movers / dead stock / drivers
See `src.eda` analysis outputs; no conclusions are hard-coded.

## Business insights
Insights are generated only from actual computed D2 results.

## Limitations
Results reflect only the official dataset provided.

> **Synthetic development data only:** these computed findings are temporary development outputs and must not be reported as official Zidio or NorthBay Living results.

## Computed Insights

1. **SKU-001 is the highest-volume SKU.** It represents 6.5% of 325,812 total units sold. A concentrated demand contribution makes this SKU important to service-level planning. Recommended action: Prioritize SKU-001 in forecast review and inventory monitoring.
2. **Spring has the highest observed seasonal demand.** The period total is 119,803 units in the computed season aggregation. Demand planning should account for calendar-period variation rather than use one flat average. Recommended action: Review replenishment coverage before the Spring demand period.
3. **5 SKU(s) meet the computed dead/slow-stock rule.** The rule flags zero sales or bottom-quartile movement with no observed stock depletion. Capital may be tied up in inventory with limited observed movement. Recommended action: Review these SKUs for markdown, assortment, or replenishment-policy changes.
4. **Aggregate daily demand is increasing over the observed window.** The fitted daily slope is 0.013 units per day. The time trend should be monitored when setting future inventory targets. Recommended action: Compare future forecasts with this trend signal during review.

## Artifacts

- `d2_eda_report.json` contains the computed report and trend tables.
- `d2_charts/` contains the generated PNG charts.
