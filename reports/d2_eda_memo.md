# Project FORESIGHT D2: Data Quality and EDA Memo

## Scope
This memo is calculated only from verified D1 outputs. D2 does not read raw transactions or legacy data.

## Data Quality
Compact summary of the official D1 outputs. The complete per-column
detail (columns, per-column missing values, per-SKU row counts,
duplicate rows, date coverage) is preserved unchanged in
`reports/d2_eda_report.json` under `executive_summary.data_quality`.

| Table | Rows | Columns | Duplicate rows | Missing values | Date coverage | Unique SKUs |
|---|---:|---:|---:|---:|---|---:|
| analysis_ready | 24,109 | 16 | 0 | 49,291 | 2022-01-01 → 2025-12-31 (1,461 valid days) | 4,616 |
| calendar | 1,461 | 6 | 0 | 1,824 | 2022-01-01 → 2025-12-31 (1,461 valid days) | n/a |
| inventory_snapshots | 4,495 | 6 | 0 | 0 | 2025-12-31 → 2025-12-31 (1 valid day) | 4,495 |
| sales_daily | 24,109 | 6 | 0 | 19,085 | 2022-01-01 → 2025-12-31 (1,461 valid days) | 4,616 |
| sku_master | 5,000 | 6 | 0 | 384 | n/a | 5,000 |

Missing values by column (non-zero only):
- `analysis_ready.promo_flag`: 19085 missing
- `analysis_ready.is_holiday`: 24109 missing
- `analysis_ready.promo_event`: 6097 missing
- `calendar.is_holiday`: 1461 missing
- `calendar.promo_event`: 363 missing
- `sales_daily.promo_flag`: 19085 missing
- `sku_master.launch_date`: 384 missing

## Computed Findings
1. **SKU04321 is the highest-volume SKU.** It represents 6.5% of 46,940 total units sold. A concentrated demand contribution makes this SKU important to service-level planning. Recommended action: Prioritize SKU04321 in forecast review and inventory monitoring.
2. **Season Autumn has the highest observed demand.** The period total is 12,760 units in the computed season aggregation. Demand planning should account for calendar-period variation rather than use one flat average. Recommended action: Review replenishment coverage before the Autumn demand period.
3. **1240 SKU(s) meet the computed dead/slow-stock rule.** The rule flags zero sales or bottom-quartile movement with no observed stock depletion. Capital may be tied up in inventory with limited observed movement. Recommended action: Review these SKUs for markdown, assortment, or replenishment-policy changes.
4. **Aggregate daily demand is increasing over the observed window.** The fitted daily slope is 0.007 units per day. The time trend should be monitored when setting future inventory targets. Recommended action: Compare future forecasts with this trend signal during review.

## Downstream Implications
- D3 should account for weekly demand behaviour, concentration, sparse observations, promotion coverage, and unavailable holiday labels.
- D4 should use inventory coverage and variability as decision context and preserve the distinction between observed demand and inventory position.

## Charts
- `reports\d2_charts\daily_demand.png`
- `reports\d2_charts\weekly_demand.png`
- `reports\d2_charts\weekly_revenue.png`
- `reports\d2_charts\category_contribution.png`
- `reports\d2_charts\top_sku_concentration.png`
- `reports\d2_charts\promotion_impact.png`
- `reports\d2_charts\weekly_seasonality.png`
- `reports\d2_charts\demand_variability.png`
