"""Generate Project FORESIGHT D2 from verified D1 outputs only."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import eda

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CHARTS = REPORTS / "d2_charts"


def _jsonable(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _save_line(name, title, frame, x, y, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame[x], frame[y], color="#176b87", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def _save_bar(name, title, frame, x, y, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [str(value) for value in frame[x].tolist()]
    values = pd.to_numeric(frame[y], errors="coerce").fillna(0.0).to_numpy()
    ax.bar(labels, values, color="#176b87")
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def _fmt_int(value) -> str:
    """Format an integer with thousands separators; 'n/a' when absent."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _quality_summary_lines(quality: dict) -> list:
    """Compact, readable data-quality summary table for the memo.

    Replaces the former raw ``json.dumps`` of the full quality structure,
    whose per-SKU ``rows_per_sku`` maps made the memo hundreds of
    kilobytes. Every number below is computed from the same ``quality``
    object that is stored, in full and unchanged, in
    ``reports/d2_eda_report.json`` under ``executive_summary.data_quality``
    — this table summarizes it; nothing is invented or altered.
    """
    tables = quality.get("tables", {})
    lines = [
        "Compact summary of the official D1 outputs. The complete per-column",
        "detail (columns, per-column missing values, per-SKU row counts,",
        "duplicate rows, date coverage) is preserved unchanged in",
        "`reports/d2_eda_report.json` under `executive_summary.data_quality`.",
        "",
        "| Table | Rows | Columns | Duplicate rows | Missing values | Date coverage | Unique SKUs |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    missing_detail = []
    invalid_detail = []
    for name in sorted(tables):
        info = tables.get(name) or {}
        missing = info.get("missing_values") or {}
        missing_total = sum(int(v) for v in missing.values())
        missing_cols = [str(c) for c, v in missing.items() if int(v) > 0]
        date_cov = info.get("date_coverage") or {}
        if date_cov.get("present"):
            if date_cov.get("first") and date_cov.get("last"):
                valid_days = date_cov.get("valid_dates")
                day_word = "day" if valid_days == 1 else "days"
                date_txt = (
                    f"{date_cov['first']} \u2192 {date_cov['last']} "
                    f"({_fmt_int(valid_days)} valid {day_word})"
                )
            else:
                date_txt = "no valid dates"
        else:
            date_txt = "n/a"
        sku_cov = info.get("sku_coverage") or {}
        sku_txt = (
            _fmt_int(sku_cov.get("unique_skus")) if sku_cov.get("present") else "n/a"
        )
        lines.append(
            f"| {name} | {_fmt_int(info.get('row_count'))} | "
            f"{_fmt_int(info.get('column_count'))} | "
            f"{_fmt_int(info.get('duplicate_rows'))} | "
            f"{_fmt_int(missing_total)} | {date_txt} | {sku_txt} |"
        )
        for col in missing_cols:
            missing_detail.append(f"- `{name}.{col}`: {missing[col]} missing")
        for col, cnt in sorted((info.get("invalid_value_counts") or {}).items()):
            if int(cnt) > 0:
                invalid_detail.append(f"- `{name}.{col}`: {cnt} invalid")

    if missing_detail:
        lines += ["", "Missing values by column (non-zero only):"] + missing_detail
    else:
        lines += ["", "Missing values by column: none in any official D1 output."]
    if invalid_detail:
        lines += ["", "Invalid values flagged by D1:"] + invalid_detail
    return lines


def main() -> None:
    tables = eda.load_analysis_ready_data(PROCESSED)
def main() -> None:
    tables = eda.load_analysis_ready_data(PROCESSED)
    quality = eda.summarize_data_quality(tables)
    demand = eda.analyze_demand_patterns(tables)
    seasonality = eda.analyze_seasonality(tables)
    trend = eda.analyze_trend(tables)
    movers = eda.top_movers(tables)
    dead = eda.dead_stock(tables)
    drivers = eda.analyze_drivers(tables)
    insights = eda.generate_business_insights(
        tables, demand, seasonality, trend, movers, dead, drivers
    )

    sales = tables["analysis_ready"].copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales["revenue"] = pd.to_numeric(sales["revenue"], errors="coerce")
    sales["units_sold"] = pd.to_numeric(sales["units_sold"], errors="coerce")
    sales["promo_flag"] = pd.to_numeric(sales["promo_flag"], errors="coerce")

    daily = sales.groupby("date", as_index=False).agg(
        units_sold=("units_sold", "sum"), revenue=("revenue", "sum")
    )
    daily["week"] = daily["date"].dt.to_period("W").astype(str)
    weekly = daily.groupby("week", as_index=False).agg(
        units_sold=("units_sold", "sum"), revenue=("revenue", "sum")
    )
    category = sales.groupby("category", dropna=False, as_index=False).agg(
        units_sold=("units_sold", "sum"), revenue=("revenue", "sum")
    ).sort_values("units_sold", ascending=False)
    sku = sales.groupby("sku_id", as_index=False).agg(
        units_sold=("units_sold", "sum"), observed_days=("date", "nunique")
    ).sort_values("units_sold", ascending=False)
    total_units = float(sku["units_sold"].sum())
    top_10_share = float(sku.head(10)["units_sold"].sum() / total_units) if total_units else 0.0
    promo = sales.groupby("promo_flag", dropna=False, as_index=False).agg(
        rows=("sku_id", "size"), units_sold=("units_sold", "sum"), revenue=("revenue", "sum")
    )
    variability = sales.groupby("sku_id", as_index=False)["units_sold"].agg(
        mean_daily="mean", std_daily="std", observed_days="count"
    )
    variability["cv"] = variability["std_daily"] / variability["mean_daily"].replace(0, pd.NA)
    variability = variability.sort_values("cv", ascending=False, na_position="last")
    inventory = tables["inventory_snapshots"].copy()
    inventory_skus = set(inventory["sku_id"].astype(str))
    sales_skus = set(sales["sku_id"].astype(str))
    inventory_summary = {
        "inventory_rows": int(len(inventory)),
        "inventory_unique_skus": int(inventory["sku_id"].nunique()),
        "sales_skus_with_inventory": int(len(sales_skus & inventory_skus)),
        "sales_sku_inventory_coverage": float(len(sales_skus & inventory_skus) / len(sales_skus)) if sales_skus else 0.0,
        "total_on_hand_units": float(pd.to_numeric(inventory["on_hand_units"], errors="coerce").sum()),
    }

    if CHARTS.exists():
        shutil.rmtree(CHARTS)
    CHARTS.mkdir(parents=True, exist_ok=True)
    charts = [
        _save_line("daily_demand", "Daily demand trend", daily, "date", "units_sold", "Units sold"),
        _save_line("weekly_demand", "Weekly demand behaviour", weekly, "week", "units_sold", "Units sold"),
        _save_line("weekly_revenue", "Weekly revenue trend", weekly, "week", "revenue", "Revenue"),
        _save_bar("category_contribution", "Category demand contribution", category, "category", "units_sold", "Units sold"),
        _save_bar("top_sku_concentration", "Top SKU demand concentration", sku.head(20), "sku_id", "units_sold", "Units sold"),
        _save_bar("promotion_impact", "Demand and revenue by promotion flag", promo, "promo_flag", "units_sold", "Units sold"),
        _save_bar("weekly_seasonality", "Weekly seasonal pattern", sales.assign(week_num=sales["date"].dt.isocalendar().week).groupby("week_num", as_index=False)["units_sold"].sum(), "week_num", "units_sold", "Units sold"),
        _save_bar("demand_variability", "Highest demand variability", variability.head(20), "sku_id", "cv", "Coefficient of variation"),
    ]

    report = eda.create_d2_report(
        tables, quality, demand, seasonality, trend, movers, dead, drivers, insights
    )
    report["data_source_statement"] = "D2 uses only verified D1 outputs under data/processed/. No raw transaction file, legacy data, notebook, or fabricated data was read."
    report["rows_analyzed"] = {name: int(len(frame)) for name, frame in tables.items()}
    report["metrics"] = {
        "date_range": {"first": str(sales["date"].min().date()), "last": str(sales["date"].max().date())},
        "category_count": int(sales["category"].nunique(dropna=True)),
        "subcategory_count": int(sales["subcategory"].nunique(dropna=True)),
        "top_10_sku_unit_share": top_10_share,
        "promotion": promo.to_dict(orient="records"),
        "highest_variability_skus": variability.head(20),
        "intermittent_sku_count": int((sku["observed_days"] < sales["date"].nunique()).sum()),
        "holiday_status": "unavailable in D1 source; is_holiday remains missing",
        "inventory": inventory_summary,
    }
    report["charts"] = charts
    report["observations_for_downstream"] = [
        "Use observed weekly and category patterns as D3 forecast signals.",
        "Review concentrated top-SKU demand and high-variability SKUs in D3 and D4.",
        "Treat promotion and inventory coverage as context; correlation is not causation.",
    ]
    report_path = REPORTS / "d2_eda_report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")

    memo = [
        "# Project FORESIGHT D2: Data Quality and EDA Memo",
        "",
        "## Scope",
        "This memo is calculated only from verified D1 outputs. D2 does not read raw transactions or legacy data.",
        "",
        "## Data Quality",
        *_quality_summary_lines(quality),
        "",
        "## Computed Findings",
    ]
    memo.extend(f"{i}. **{item['observation']}** {item['evidence']} {item['business_implication']} Recommended action: {item['recommended_action']}" for i, item in enumerate(insights, 1))
    memo.extend([
        "",
        "## Downstream Implications",
        "- D3 should account for weekly demand behaviour, concentration, sparse observations, promotion coverage, and unavailable holiday labels.",
        "- D4 should use inventory coverage and variability as decision context and preserve the distinction between observed demand and inventory position.",
        "",
        "## Charts",
        *[f"- `{path}`" for path in charts],
    ])
    (REPORTS / "d2_eda_memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"D2_OK rows={len(sales)} charts={len(charts)} insights={len(insights)}")


if __name__ == "__main__":
    main()
