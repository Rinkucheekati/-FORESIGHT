"""Project FORESIGHT — official Streamlit planning dashboard (D5).

This is the OFFICIAL Zidio FORESIGHT dashboard. It replaces the legacy
Mini-FORESIGHT demo dashboard entirely: no legacy data, captions, or metrics
are used anywhere in this file.

Architecture
------------
``app/app.py`` is presentation-only. All business logic lives in the official
source modules and is REUSED, never duplicated here:

    src.eda       D2 — analysis-ready loading + demand/seasonality/trend/
                  movers/dead-stock/drivers analyses
    src.forecast  D3 — weekly SKU forecasts, rolling-origin backtest,
                  WAPE/bias, model-vs-baseline selection
    src.risk      D4 — inventory position, stockout/overstock risk,
                  rupee value at stake, four-cell decision grid

Data gating
-----------
The dashboard renders a friendly ``DATA NOT AVAILABLE`` state whenever the
official D1 outputs are absent (``data/processed/*_clean.csv``). It never
falls back to ``data/raw/``, never shows placeholder charts or KPIs, and every
unavailable metric is displayed as ``Not available``.

Run from the repository root::

    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# --- Repository-relative import bootstrap (no hard-coded absolute paths) ----
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import DATA_PROCESSED  # noqa: E402
import src.eda as eda  # noqa: E402
import src.risk as risk  # noqa: E402
import src.forecast as fc  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants / page text
# --------------------------------------------------------------------------- #

PROCESSED_DIR_STR = str(DATA_PROCESSED)

DATA_NOT_AVAILABLE_TITLE = "DATA NOT AVAILABLE"
DATA_NOT_AVAILABLE_MESSAGE = (
    "Official FORESIGHT row-level data is not currently available. Dashboard "
    "metrics and model results will appear automatically when the official "
    "extracts are supplied and processed through D1."
)
SYNTHETIC_DATA_MESSAGE = (
    "SYNTHETIC DEVELOPMENT DATA — temporary development results only. "
    "These forecasts, risks, metrics, and recommendations are not official "
    "Zidio or NorthBay Living results."
)

DECISION_ACTION_TEMPLATES = {
    "REORDER_NOW": (
        "Inventory exposure indicates elevated stockout risk; review "
        "replenishment."
    ),
    "MARKDOWN_CLEAR": (
        "Inventory exposure indicates elevated overstock risk; review "
        "markdown/clearance action."
    ),
    "WATCH_VOLATILE": (
        "Demand/inventory conditions require monitoring due to elevated "
        "volatility or uncertainty."
    ),
    "HEALTHY": (
        "No immediate inventory action indicated by the current risk framework."
    ),
}

PRIMARY_METRIC_LABEL = "PRIMARY METRIC = WAPE"
SECONDARY_METRIC_LABEL = "SECONDARY METRIC = Bias"

RISK_CFG_FIELDS = (
    "coverage_target_weeks", "overstock_coverage_weeks",
    "stockout_ratio_threshold", "volatility_cv_high",
    "min_history_weeks", "low_history_weeks",
    "reorder_now_stockout_min", "markdown_clear_overstock_min",
)
FCST_CFG_FIELDS = (
    "horizon_weeks", "seasonal_period", "cv_folds",
    "min_train_weeks", "min_obs_per_sku", "random_seed",
)




# --------------------------------------------------------------------------- #
# Default engine configurations (single source for this dashboard session)
# --------------------------------------------------------------------------- #

DEFAULT_RISK_CONFIG = risk.RiskConfig()
DEFAULT_FCST_CONFIG = fc.ForecastConfig()


def official_signature(processed_dir_str: str = PROCESSED_DIR_STR) -> Optional[str]:
    """Stable signature of the official input files (name/mtime/size).

    Cache keys embed this so every cached artifact refreshes automatically
    once D1 regenerates its outputs.
    """
    pdir = Path(processed_dir_str)
    names = [
        "sales_daily_clean.csv", "sku_master_clean.csv",
        "calendar_clean.csv", "inventory_snapshots_clean.csv",
        "forecast_results.csv",
    ]
    parts: List[str] = []
    for n in names:
        f = pdir / n
        if f.is_file():
            st_ = f.stat()
            parts.append(f"{n}:{int(st_.st_mtime)}:{st_.st_size}")
        else:
            parts.append(f"{n}:missing")
    return None if all(p.endswith(":missing") for p in parts[:4]) else ";".join(parts)


@st.cache_data(show_spinner="Loading official FORESIGHT data…")
def load_tables_cached(sig: str, processed_dir_str: str) -> Dict[str, pd.DataFrame]:
    """Load ONLY the official D1 analysis-ready outputs via the D2 loader."""
    return eda.load_analysis_ready_data(Path(processed_dir_str))


@st.cache_data(show_spinner="Preparing D2 demand analyses…")
def eda_analyses_cached(sig: str, processed_dir_str: str) -> Dict[str, Any]:
    """Official D2 analyses computed on the analysis-ready tables."""
    tables = load_tables_cached(sig, processed_dir_str)
    demand = eda.analyze_demand_patterns(tables)
    seasonality = eda.analyze_seasonality(tables)
    trend = eda.analyze_trend(tables)
    movers = eda.top_movers(tables)
    dead_stock_res = eda.dead_stock(tables)
    drivers = eda.analyze_drivers(tables)
    insights = eda.generate_business_insights(
        tables, demand, seasonality, trend, movers, dead_stock_res, drivers,
    )
    return {
        "quality": eda.summarize_data_quality(tables),
        "demand": demand,
        "seasonality": seasonality,
        "trend": trend,
        "movers": movers,
        "dead_stock": dead_stock_res,
        "drivers": drivers,
        "insights": insights,
    }


def fmt_int(v: Any) -> str:
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return "Not available"


def fmt_num(v: Any, nd: int = 2) -> str:
    try:
        f = float(v)
        return f"{f:,.{nd}f}" if f == f else "Not available"
    except (TypeError, ValueError):
        return "Not available"


def fmt_pct(v: Any, nd: int = 1) -> str:
    try:
        f = float(v)
        return f"{f * 100:.{nd}f}%" if f == f else "Not available"
    except (TypeError, ValueError):
        return "Not available"


def fmt_pct_raw(v: Any, nd: int = 1) -> str:
    """Format a value already expressed in percent units."""
    try:
        f = float(v)
        return f"{f:.{nd}f}%" if f == f else "Not available"
    except (TypeError, ValueError):
        return "Not available"


def fmt_rupee(v: Any) -> str:
    try:
        f = float(v)
        return ("₹" + f"{f:,.0f}") if f == f else "Not available"
    except (TypeError, ValueError):
        return "Not available"


def kpi(label: str, value: Any, formatter=fmt_int, help_text: str = "") -> None:
    """Render one honest KPI card; unavailable values never fake numbers."""
    st.metric(
        label=label,
        value=value if isinstance(value, str) else formatter(value),
        help=help_text or None,
    )


def info_state(container_title: str = DATA_NOT_AVAILABLE_TITLE,
               message: str = DATA_NOT_AVAILABLE_MESSAGE,
               extra: Optional[List[str]] = None) -> None:
    st.info(f"**{container_title}**\n\n{message}")
    for line in extra or []:
        st.caption(line)


def na_cell() -> str:
    return "Not available"



def risk_cfg_tuple(cfg: risk.RiskConfig) -> Tuple[Any, ...]:
    return tuple(getattr(cfg, f) for f in RISK_CFG_FIELDS)


def fcst_cfg_tuple(cfg: fc.ForecastConfig) -> Tuple[Any, ...]:
    return tuple(getattr(cfg, f) for f in FCST_CFG_FIELDS)


@st.cache_data(show_spinner="Scoring inventory risk with D4…")
def d4_scored_cached(sig: str, cfg_t: Tuple[Any, ...],
                     processed_dir_str: str) -> Dict[str, Any]:
    """Official D4 scoring (cached; recomputed only when inputs/config change)."""
    cfg = risk.RiskConfig(**dict(zip(RISK_CFG_FIELDS, cfg_t)))
    return risk.score_all_skus(None, cfg, Path(processed_dir_str))


@st.cache_data(show_spinner="Running D3 weekly backtest (first run trains models)…")
def d3_evaluation_cached(sig: str, cfg_t: Tuple[Any, ...],
                         processed_dir_str: str) -> Dict[str, Any]:
    """Official D3 pipeline: readiness → weekly features → rolling-origin
    backtest → baseline comparison → honest model selection → report."""
    cfg = fc.ForecastConfig(**dict(zip(FCST_CFG_FIELDS, cfg_t)))
    readiness = fc.check_forecast_data_readiness(None, cfg, Path(processed_dir_str))
    if readiness["status"] != "ready":
        report = fc.create_forecast_report(readiness=readiness, config=cfg)
        return {"readiness": readiness, "report": report,
                "backtest": None, "comparison": None, "selection": None}
    tables = fc.load_d1_outputs(Path(processed_dir_str))
    weekly = fc.prepare_weekly_demand(tables, cfg)
    features = fc.build_forecast_features(weekly)
    backtest = fc.rolling_origin_backtest(features, cfg)
    comparison = fc.compare_model_to_baseline(backtest)
    selection = fc.select_best_forecaster(comparison)
    low_hist = int(readiness.get("details", {}).get("skus_low_history", 0))
    report = fc.create_forecast_report(
        readiness=readiness, backtest=backtest, comparison=comparison,
        selection=selection, config=cfg, low_history_sku_count=low_hist,
    )
    return {"readiness": readiness, "report": report, "backtest": backtest,
            "comparison": comparison, "selection": selection}


@st.cache_data(show_spinner="Forecasting selected SKU via D3…")
def sku_forecast_cached(sig: str, sku_id: str, cfg_t: Tuple[Any, ...],
                        processed_dir_str: str) -> Dict[str, Any]:
    """Official per-SKU forecast through the D3 interface (cached per SKU)."""
    cfg = fc.ForecastConfig(**dict(zip(FCST_CFG_FIELDS, cfg_t)))
    return fc.forecast_weekly_sku(sku_id, None, cfg, Path(processed_dir_str))


def get_official_state() -> Dict[str, Any]:
    """Single entry point: readiness + cached official artifacts for this run."""
    sig = official_signature()
    state: Dict[str, Any] = {
        "sig": sig, "available": sig is not None, "tables": None,
        "risk_readiness": None, "d4": None, "d3": None, "eda": None,
        "load_error": None, "d4_error": None, "d3_error": None,
    }
    try:
        state["risk_readiness"] = risk.check_risk_data_readiness(
            None, DEFAULT_RISK_CONFIG
        )
    except Exception as exc:  # noqa: BLE001
        state["risk_readiness"] = {"status": "ERROR", "reasons": [str(exc)]}
    if not state["available"]:
        return state
    try:
        state["tables"] = load_tables_cached(sig, PROCESSED_DIR_STR)
        state["eda"] = eda_analyses_cached(sig, PROCESSED_DIR_STR)
    except Exception as exc:  # noqa: BLE001 - keep dashboard usable
        state["load_error"] = str(exc)
    try:
        state["d4"] = d4_scored_cached(sig, risk_cfg_tuple(DEFAULT_RISK_CONFIG),
                                       PROCESSED_DIR_STR)
    except Exception as exc:  # noqa: BLE001
        state["d4_error"] = str(exc)
    try:
        state["d3"] = d3_evaluation_cached(sig, fcst_cfg_tuple(DEFAULT_FCST_CONFIG),
                                           PROCESSED_DIR_STR)
    except Exception as exc:  # noqa: BLE001
        state["d3_error"] = str(exc)
    return state

# --------------------------------------------------------------------------- #
# Page: EXECUTIVE OVERVIEW
# --------------------------------------------------------------------------- #


def page_executive(state: Dict[str, Any]) -> None:
    st.header("📋 Executive Overview")
    st.caption(
        "Official FORESIGHT KPIs — every value is computed from official D1–D4 "
        "results; unavailable metrics display 'Not available'."
    )

    d4 = state.get("d4") or {}
    scored = d4.get("scored_skus", []) if d4.get("status") == "READY" else []
    d3rep = (state.get("d3") or {}).get("report") or {}
    master = (state.get("tables") or {}).get("sku_master")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Total SKUs",
            int(master["sku_id"].nunique()) if master is not None else na_cell())
        kpi("Forecast horizon (weeks)",
            d3rep.get("forecast_horizon_weeks", "Not available"), fmt_int)
    with c2:
        model = d3rep.get("selected_model")
        kpi("Forecast model selected",
            model.replace("_", " ").title() if isinstance(model, str)
            else "Not available")
        wape_m = (d3rep.get("wape") or {}).get("model")
        wape_b = (d3rep.get("wape") or {}).get("baseline")
        kpi("WAPE (model vs baseline)",
            f"{fmt_pct_raw(wape_m)} vs {fmt_pct_raw(wape_b)}"
            if wape_m is not None and wape_b is not None else "Not available")
    with c3:
        bias_m = (d3rep.get("bias") or {}).get("model")
        kpi("Bias (secondary)",
            fmt_num(bias_m) if bias_m is not None else "Not available")
        total_rupee = (
            sum((r.get("rupee_value_at_stake") or 0.0) for r in scored)
            if scored else None
        )
        kpi("Rupee value at stake", total_rupee, fmt_rupee)

    st.subheader("Decision mix (D4 four-cell grid)")
    if scored and any(r.get("decision") for r in scored):
        decisions = [r["decision"] for r in scored]
        order = ["REORDER_NOW", "MARKDOWN_CLEAR", "WATCH_VOLATILE", "HEALTHY"]
        cols = st.columns(4)
        for col, dec in zip(cols, order):
            with col:
                kpi(dec.replace("_", " ").title(), int(decisions.count(dec)))
        st.bar_chart(pd.DataFrame({"SKUs": pd.Series(decisions).value_counts()}))
    else:
        info_state(extra=[
            "Inventory-risk decision mix appears here once official data is "
            "processed through D1."
        ])

    st.divider()
    st.caption(f"{PRIMARY_METRIC_LABEL} · {SECONDARY_METRIC_LABEL}")

# --------------------------------------------------------------------------- #
# Page: DATA QUALITY / READINESS
# --------------------------------------------------------------------------- #


def _availability_table() -> pd.DataFrame:
    pdir = Path(PROCESSED_DIR_STR)
    rows = []
    for label, fname in [
        ("sales_daily (D1)", "sales_daily_clean.csv"),
        ("sku_master (D1)", "sku_master_clean.csv"),
        ("calendar (D1)", "calendar_clean.csv"),
        ("inventory_snapshots (D1)", "inventory_snapshots_clean.csv"),
    ]:
        f = pdir / fname
        rows.append({"official input": label, "file": fname,
                     "present": "yes" if f.is_file() else "missing"})
    rows.append({
        "official input": "D3 weekly forecast output",
        "file": "forecast_results.csv (official ISO-week schema)",
        "present": "validated by schema when present",
    })
    return pd.DataFrame(rows)


def page_data_quality(state: Dict[str, Any]) -> None:
    st.header("🧪 Data Quality & Readiness")
    rr = state.get("risk_readiness") or {}
    status = rr.get("status", "UNKNOWN")

    st.subheader("Official input availability")
    st.dataframe(_availability_table(), use_container_width=True, hide_index=True)

    st.subheader("Readiness status")
    badge = {"READY": "✅ READY",
             "INSUFFICIENT_DATA": "⚠️ INSUFFICIENT DATA",
             "DATA_NOT_AVAILABLE": "⛔ DATA NOT AVAILABLE"}.get(status, status)
    st.markdown(f"**{badge}**")
    for r in rr.get("reasons", [])[:8]:
        st.caption(f"• {r}")

    details = rr.get("details") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs in master", details.get("total_skus_in_master", na_cell()))
    c2.metric("SKUs with sufficient history",
              details.get("skus_with_sufficient_history", na_cell()))
    c3.metric("Low-history SKUs", details.get("skus_low_history", na_cell()))
    st.caption("D3 forecast integration: "
               + str(details.get("forecast_status", "not requested")))

    quality = (state.get("eda") or {}).get("quality")
    if quality:
        st.subheader("Per-table quality summary (computed by D2)")
        rows = []
        for tname, info in quality.get("tables", {}).items():
            miss_total = int(sum(info.get("missing_values", {}).values()))
            dcov = info.get("date_coverage") or {}
            scov = info.get("sku_coverage") or {}
            rows.append({
                "table": tname,
                "rows": info.get("row_count"),
                "cols": info.get("column_count"),
                "missing cells": miss_total,
                "duplicate rows": info.get("duplicate_rows"),
                "date coverage": (
                    f"{dcov.get('first')} → {dcov.get('last')}"
                    if dcov.get("present") else "n/a"
                ),
                "unique SKUs": scov.get("unique_skus", "n/a"),
            })
        df_q = pd.DataFrame(rows)
        st.dataframe(df_q, use_container_width=True, hide_index=True)
        st.download_button(
            "Download data-quality summary (CSV)",
            data=df_q.to_csv(index=False).encode("utf-8"),
            file_name="foresight_data_quality_summary.csv",
            mime="text/csv",
        )
    else:
        info_state(extra=[
            "Quality summaries populate automatically after D1 processing."
        ])

    d4_issues = ((state.get("d4") or {}).get("issues")) or []
    if d4_issues:
        st.subheader("Inventory input anomalies flagged by D4")
        st.dataframe(pd.DataFrame(d4_issues), use_container_width=True,
                     hide_index=True)

# --------------------------------------------------------------------------- #
# Page: DEMAND ANALYSIS (official observations only; insights from D2 engine)
# --------------------------------------------------------------------------- #


def page_demand_analysis(state: Dict[str, Any], filters: Dict[str, Any]) -> None:
    st.header("📈 Demand Analysis")
    if not state.get("eda"):
        info_state(extra=["Demand analytics populate automatically after D1."])
        return
    ed = state["eda"]
    tables = state.get("tables") or {}
    sales = tables.get("sales_daily")

    demand = ed["demand"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total units sold", fmt_int(demand.get("total_units_sold")))
    dated = demand.get("dated") or {}
    c2.metric("Days observed", dated.get("unique_days", na_cell()))
    c3.metric("SKUs selling",
              len((demand.get("sku_level") or pd.DataFrame())) or na_cell())

    st.subheader("Total demand over time")
    daily = demand.get("daily_demand")
    if daily is not None and len(daily):
        st.line_chart(daily)
        st.caption("Daily total units sold across all SKUs (official sales).")
    else:
        info_state()

    st.subheader("Weekly demand trend")
    weekly = fc.prepare_weekly_demand(tables) if tables else None
    if weekly is not None and len(weekly):
        if filters.get("season") and "season" in weekly.columns:
            weekly = weekly[weekly["season"].astype(str) == str(filters["season"])]
        if filters.get("promo") is not None and "promo" in weekly.columns:
            weekly = weekly[pd.to_numeric(weekly["promo"], errors="coerce")
                            == float(filters["promo"])]
        wk = (weekly.groupby("period")["units_sold"].sum())
        st.bar_chart(wk)
        st.caption("Official weekly grain (ISO year-week), identical to D3. "
                   "Season/promotion filters apply to this chart.")
    else:
        info_state()

    sku_level = demand.get("sku_level")
    if isinstance(sku_level, pd.DataFrame) and len(sku_level):
        st.subheader("SKU demand comparison")
        st.bar_chart(sku_level.set_index("sku_id")["total_units"])
        show = sku_level.copy()
        for col in ("avg_daily", "std_daily"):
            if col in show.columns:
                show[col] = show[col].round(3)
        st.dataframe(show, use_container_width=True, hide_index=True)

    cat = demand.get("category_subcategory")
    if isinstance(cat, pd.DataFrame) and len(cat):
        st.subheader("Category / subcategory demand")
        label_col = cat.columns[0]
        st.bar_chart(cat.set_index(label_col)["total_units"])

    st.subheader("Seasonality & promotion patterns (official calendar fields)")
    seas = ed["seasonality"] or {}
    found_any = False
    for field in ("week", "month", "season", "is_holiday", "promo_event"):
        block = seas.get(field)
        if block and block.get("sales_by_period"):
            found_any = True
            with st.expander(f"Demand by {field}", expanded=(field == "week")):
                ser = pd.Series(block["sales_by_period"], dtype=float)
                st.bar_chart(ser)
    if not found_any:
        info_state(extra=["Calendar-based seasonality appears once official "
                          "calendar data is processed."])

    # Promotion-related pattern from official promo_flag on sales
    if sales is not None and {"promo_flag", "units_sold"} <= set(sales.columns):
        with st.expander("Promotion vs non-promotion demand (official promo_flag)"):
            grp = sales.groupby(sales["promo_flag"].fillna(0))["units_sold"].mean()
            st.bar_chart(grp)
            st.caption("Average units per record by promotion flag — descriptive "
                       "only; no causal claim.")

    st.subheader("Top movers (D2 methodology)")
    movers = ed["movers"] or {}
    mv = pd.DataFrame(movers.get("movers", []))
    if len(mv):
        st.dataframe(mv, use_container_width=True, hide_index=True)
        st.caption(f"Methodology: {movers.get('methodology', 'documented in D2')}")
    else:
        info_state(extra=["Movers appear after official D1 outputs exist."])

    st.subheader("Dead stock (D2 definition)")
    dead = ed["dead_stock"] or {}
    dead_rows = dead.get("flagged_skus", [])
    if dead_rows:
        st.dataframe(pd.DataFrame(dead_rows), use_container_width=True,
                     hide_index=True)
        st.caption(f"Definitions: {dead.get('definitions')}")
    else:
        st.info("No SKUs meet the documented dead-stock definitions in the "
                "current official data.")

    st.subheader("Drivers / correlations (descriptive)")
    drivers = ed["drivers"] or {}
    corr = drivers.get("correlation_matrix")
    if corr:
        st.dataframe(pd.DataFrame(corr), use_container_width=True)
        if drivers.get("sales_vs_inventory_corr") is not None:
            st.caption("Sales vs on-hand inventory correlation: "
                       + fmt_num(drivers["sales_vs_inventory_corr"], 3))
        st.caption("Correlations are descriptive; they do not imply causation.")

    st.subheader("Business insights (generated by the D2 engine)")
    insights = ed.get("insights") or []
    if insights:
        for ins in insights:
            with st.container(border=True):
                st.markdown(f"**Observation:** {ins.get('observation','')}")
                st.markdown(f"- **Evidence:** {ins.get('evidence','')}")
                st.markdown(f"- **Implication:** {ins.get('business_implication','')}")
                st.markdown(f"- **Recommended action:** {ins.get('recommended_action','')}")
    else:
        st.info("The D2 insight engine produced no insights for the current "
                "data. Insights appear only when supported by computed results.")

# --------------------------------------------------------------------------- #
# Page: FORECAST (official D3 results only)
# --------------------------------------------------------------------------- #


def page_forecast(state: Dict[str, Any], filters: Dict[str, Any]) -> None:
    st.header("🔮 Demand Forecast")
    d3 = state.get("d3") or {}
    rep = d3.get("report") or {}

    if d3.get("d3_error"):
        st.error("D3 evaluation could not run with the current official data.")
        st.caption(d3["d3_error"])
        return

    c1, c2 = st.columns(2)
    c1.metric("Forecast horizon",
              f"{rep.get('forecast_horizon_weeks','Not available')} weeks"
              if rep.get("forecast_horizon_weeks") else "Not available")
    model = rep.get("selected_model")
    c2.metric("Selected model",
              model.replace("_", " ").title() if isinstance(model, str)
              else "Not available")

    if rep.get("data_sufficiency_status") != "ready":
        info_state(
            message=rep.get("unavailability_statement", DATA_NOT_AVAILABLE_MESSAGE),
            extra=(rep.get("readiness_reasons") or [])[:6]
            or (state.get("d3") or {}).get("readiness", {}).get("reasons", []),
        )
        return

    st.subheader("Model vs seasonal-naive baseline")
    wape_block = rep.get("wape") or {}
    bias_block = rep.get("bias") or {}
    cmp_block = rep.get("model_vs_baseline") or {}
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("WAPE — model", fmt_pct_raw(wape_block.get("model")))
    k2.metric("WAPE — baseline", fmt_pct_raw(wape_block.get("baseline")))
    imp = cmp_block.get("improvement_vs_baseline_wape")
    k3.metric("WAPE improvement", fmt_pct(imp) if imp is not None else "Not available")
    k4.metric("Bias — model", fmt_num(bias_block.get("model")))

    if wape_block.get("model") is not None and wape_block.get("baseline") is not None:
        st.bar_chart(pd.DataFrame({
            "WAPE (%)": {"ML model": wape_block["model"],
                         "Seasonal-naive": wape_block["baseline"]}
        }))
    st.caption(f"{PRIMARY_METRIC_LABEL} · {SECONDARY_METRIC_LABEL}. "
               "Bias sign convention: positive = over-forecast.")

    bt = d3.get("backtest") or {}
    folds = bt.get("folds") or []
    if folds:
        st.subheader("Rolling-origin backtest folds")
        fold_df = pd.DataFrame(folds)
        show_cols = [c for c in ["fold", "train_end_period", "test_start_period",
                                 "test_end_period", "n_train_rows", "n_test_rows",
                                 "baseline_evaluable_rows", "model_evaluable"]
                     if c in fold_df.columns]
        st.dataframe(fold_df[show_cols], use_container_width=True, hide_index=True)
        meth = bt.get("methodology") or {}
        st.caption("Methodology: {type} · horizon {h}w · seasonal period {p}w · "
                   "{f} folds · seed {s}".format(
                       type=meth.get("type"), h=meth.get("horizon_weeks"),
                       p=meth.get("seasonal_period_weeks"),
                       f=meth.get("cv_folds"), s=meth.get("random_seed")))

    _sku_forecast_selector(state, filters)

def _sku_forecast_selector(state: Dict[str, Any], filters: Dict[str, Any]) -> None:
    """Per-SKU weekly forecast via the official D3 interface (cached per SKU)."""
    st.subheader("SKU-level weekly forecast")
    master = (state.get("tables") or {}).get("sku_master")
    if master is None or not len(master):
        info_state(extra=["SKU selection appears once official D1 outputs exist."])
        return
    skus = sorted(master["sku_id"].astype(str).unique())
    default = filters.get("sku") if filters.get("sku") in skus else skus[0]
    sku_id = st.selectbox("Select SKU", skus, index=skus.index(default),
                          key="fc_sku_select")
    try:
        res = sku_forecast_cached(state["sig"], str(sku_id),
                                  fcst_cfg_tuple(DEFAULT_FCST_CONFIG),
                                  PROCESSED_DIR_STR)
    except Exception as exc:  # noqa: BLE001 - explain, never crash
        st.error("The official D3 engine could not produce this forecast.")
        st.caption(str(exc))
        return

    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("Model used",
                 res.get("model_name", "n/a").replace("_", " ").title())
    meta2.metric("Horizon (weeks)", res.get("horizon_weeks", "n/a"))
    meta3.metric("Confidence",
                 "Low (fallback/low history)"
                 if res.get("low_history_flagged") or res.get("fallback_used")
                 else "Standard")

    rows = res.get("forecast_rows") or []
    fdf = pd.DataFrame(rows)
    if fdf.empty:
        st.info("D3 returned no forecast rows for this SKU (insufficient "
                "official history and fallback disabled).")
        return
    if res.get("note"):
        st.caption(res["note"])

    st.dataframe(fdf[["period", "forecast_units", "model_name"]],
                 use_container_width=True, hide_index=True)

    # History vs forecast chart (weekly actuals + forecast overlay).
    tables = state.get("tables") or {}
    sales = tables.get("sales_daily")
    if sales is not None:
        weekly = fc.prepare_weekly_demand(tables)
        hist = (weekly[weekly["sku_id"].astype(str) == str(sku_id)]
                .groupby("period")["units_sold"].sum())
        fcst = fdf.set_index("period")["forecast_units"]
        combined = pd.DataFrame({"actual": hist}).join(
            pd.DataFrame({"forecast": fcst}), how="outer"
        )
        st.line_chart(combined)
        st.caption("Official weekly actuals followed by the D3 forecast for "
                   "the configured horizon.")

    st.download_button(
        "Download this SKU forecast (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name=f"foresight_forecast_{sku_id}.csv", mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Page: MODEL PERFORMANCE
# --------------------------------------------------------------------------- #


def page_model_performance(state: Dict[str, Any]) -> None:
    st.header("📏 Model Performance")
    d3 = state.get("d3") or {}
    rep = d3.get("report") or {}

    if rep.get("data_sufficiency_status") != "ready":
        info_state(message=rep.get("unavailability_statement",
                                   DATA_NOT_AVAILABLE_MESSAGE))
        return

    cmpb = rep.get("model_vs_baseline") or {}
    sel = rep.get("selection_reason") or ""
    k1, k2, k3 = st.columns(3)
    k1.metric("Selected model",
              (rep.get("selected_model") or "n/a").replace("_", " ").title())
    imp = cmpb.get("improvement_vs_baseline_wape")
    k2.metric("WAPE improvement vs baseline",
              fmt_pct(imp) if imp is not None else "Not available")
    k3.metric("Same evaluation windows",
              "Yes" if cmpb.get("same_evaluation_windows") else "Unknown")

    m1, m2, m3, m4 = st.columns(4)
    wb = rep.get("wape") or {}
    bb = rep.get("bias") or {}
    mb = rep.get("mape") or {}
    m1.metric("ML WAPE", fmt_pct_raw(wb.get("model")))
    m2.metric("Baseline WAPE", fmt_pct_raw(wb.get("baseline")))
    m3.metric("ML bias", fmt_num(bb.get("model")))
    m4.metric("Baseline bias", fmt_num(bb.get("baseline")))

    st.caption(f"MAPE (secondary only) — model {fmt_pct(mb.get('model'))} · "
               f"baseline {fmt_pct(mb.get('baseline'))}")
    st.markdown(f"**{PRIMARY_METRIC_LABEL}** · **{SECONDARY_METRIC_LABEL}**")
    with st.expander("Why this model was selected (honest rule)"):
        st.write(sel or "Selection reason unavailable.")
        st.caption(rep.get("selection_rule",
                           "select ML only if model WAPE < baseline WAPE "
                           "on the same folds"))

    folds = ((d3.get("backtest") or {}).get("folds")) or []
    if folds:
        st.subheader("Backtest folds")
        st.dataframe(pd.DataFrame(folds), use_container_width=True,
                     hide_index=True)
    skipped = (d3.get("backtest") or {}).get("skipped_windows") or []
    if skipped:
        with st.expander("Windows not evaluable (reported honestly)"):
            st.dataframe(pd.DataFrame(skipped), use_container_width=True,
                         hide_index=True)

    lims = rep.get("limitations") or []
    if lims:
        with st.expander("Limitations"):
            for l in lims:
                st.caption(f"• {l}")

# --------------------------------------------------------------------------- #
# Page: INVENTORY RISK (official D4 output)
# --------------------------------------------------------------------------- #


def _scored_df(d4: Dict[str, Any]) -> pd.DataFrame:
    rows = d4.get("scored_skus", []) if d4.get("status") == "READY" else []
    return pd.DataFrame(rows)


def page_inventory_risk(state: Dict[str, Any], filters: Dict[str, Any]) -> None:
    st.header("⚠️ Inventory Risk")
    d4 = state.get("d4") or {}
    if state.get("d4_error"):
        st.error("D4 scoring could not run with the current official data.")
        st.caption(state["d4_error"])
        return
    if d4.get("status") != "READY":
        info_state(extra=(d4.get("reasons") or [])[:6])
        return

    df = _scored_df(d4)
    if df.empty:
        info_state()
        return

    # Apply shared filters.
    mask = pd.Series(True, index=df.index)
    if filters.get("category"):
        mask &= df["category"] == filters["category"]
    if filters.get("subcategory") and "subcategory" in df.columns:
        mask &= df["subcategory"] == filters["subcategory"]
    if filters.get("decision"):
        mask &= df["decision"] == filters["decision"]
    if filters.get("sku"):
        mask &= df["sku_id"] == filters["sku"]
    view = df[mask].copy()
    if view.empty:
        st.warning("No SKUs match the current filters.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SKUs scored", len(view))
    k2.metric("Mean stockout risk", fmt_num(view["stockout_risk"].mean(), 3))
    k3.metric("Mean overstock risk", fmt_num(view["overstock_risk"].mean(), 3))
    total_rupee = pd.to_numeric(view["rupee_value_at_stake"],
                                errors="coerce").sum()
    k4.metric("Rupee value at stake", fmt_rupee(total_rupee))

    display_cols = [
        "sku_id", "category", "inventory_position", "on_hand_units",
        "on_order_units", "lead_time_days", "calculated_risk_threshold",
        "coverage_weeks", "stockout_risk", "overstock_risk", "volatility_cv",
        "unit_cost", "rupee_value_at_stake", "data_quality_flag",
    ]
    show = view[[c for c in display_cols if c in view.columns]].copy()
    for col in ("coverage_weeks", "volatility_cv"):
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
    for col in ("stockout_risk", "overstock_risk"):
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
    st.dataframe(show, use_container_width=True, hide_index=True)

    if {"sku_id", "stockout_risk", "overstock_risk"}.issubset(view.columns):
        chart_df = view.set_index("sku_id")[["stockout_risk", "overstock_risk"]]
        st.bar_chart(chart_df)

    rep = risk.create_risk_report(d4, DEFAULT_RISK_CONFIG)
    with st.expander("D4 report summary (incl. configuration assumptions)"):
        st.json({k: v for k, v in rep.items()
                 if k in ("data_status", "inventory_summary",
                          "stockout_risk_summary", "overstock_risk_summary",
                          "rupee_value_summary", "decision_summary",
                          "low_history_summary", "configuration_assumptions",
                          "limitations")},
                 expanded=False)

    st.download_button(
        "Download inventory risk table (CSV)",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="foresight_risk_scores.csv", mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# Page: RECOMMENDATIONS / FOUR-CELL DECISIONS
# --------------------------------------------------------------------------- #


def page_decisions(state: Dict[str, Any], filters: Dict[str, Any]) -> None:
    st.header("🧭 Recommendations — Four-Cell Decisions")
    d4 = state.get("d4") or {}
    if d4.get("status") != "READY":
        info_state(extra=["Official four-cell decisions appear here after D1/D4."])
        return
    df = _scored_df(d4)
    if df.empty:
        info_state()
        return

    mask = pd.Series(True, index=df.index)
    if filters.get("decision"):
        mask &= df["decision"] == filters["decision"]
    if filters.get("category"):
        mask &= df["category"] == filters["category"]
    if filters.get("sku"):
        mask &= df["sku_id"] == filters["sku"]
    view = df[mask].copy()
    if view.empty:
        st.warning("No SKUs match the current filters.")
        return

    order = {"REORDER_NOW": 0, "MARKDOWN_CLEAR": 1,
             "WATCH_VOLATILE": 2, "HEALTHY": 3}
    view["_o"] = view["decision"].map(order).fillna(9)
    view = view.sort_values(["_o", "rupee_value_at_stake"],
                            ascending=[True, False]).drop(columns="_o")

    st.caption("Decisions come exclusively from the D4 engine "
               "(REORDER_NOW · MARKDOWN_CLEAR · WATCH_VOLATILE · HEALTHY).")
    cols = ["sku_id", "category", "inventory_position", "coverage_weeks",
            "stockout_risk", "overstock_risk", "volatility_cv",
            "rupee_value_at_stake", "decision"]
    show = view[[c for c in cols if c in view.columns]].copy()
    for c in ("coverage_weeks", "stockout_risk", "overstock_risk",
              "volatility_cv"):
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(3)
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Recommended actions")
    for _, row in view.iterrows():
        dec = str(row.get("decision"))
        emoji = {"REORDER_NOW": "🔴", "MARKDOWN_CLEAR": "🟠",
                 "WATCH_VOLATILE": "🟡", "HEALTHY": "🟢"}.get(dec, "•")
        with st.container(border=True):
            st.markdown(f"{emoji} **{row.get('sku_id')}** — "
                        f"{str(dec).replace('_',' ').title()}")
            st.write(DECISION_ACTION_TEMPLATES.get(
                dec, "Review this SKU with the planning team."))
            rv = row.get("rupee_value_at_stake")
            bits = [f"inventory position {fmt_int(row.get('inventory_position'))}",
                    f"stockout risk {fmt_num(row.get('stockout_risk'))}",
                    f"overstock risk {fmt_num(row.get('overstock_risk'))}"]
            if rv is not None:
                bits.append(f"value at stake {fmt_rupee(rv)}")
            st.caption(" · ".join(bits))

    st.download_button(
        "Download decision recommendations (CSV)",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="foresight_decisions.csv", mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# SKU DRILL-DOWN
# --------------------------------------------------------------------------- #


def page_drilldown(state: Dict[str, Any], filters: Dict[str, Any]) -> None:
    st.header("🔎 SKU Drill-down")
    tables = state.get("tables") or {}
    master = tables.get("sku_master")
    if master is None or not len(master):
        info_state(extra=["Drill-down becomes available after D1 outputs exist."])
        return

    skus = sorted(master["sku_id"].astype(str).unique())
    default = filters.get("sku") if filters.get("sku") in skus else skus[0]
    sku_id = st.selectbox("Select SKU", skus, index=skus.index(default),
                          key="dd_sku_select")

    mrow = master[master["sku_id"].astype(str) == str(sku_id)]
    if not mrow.empty:
        m = mrow.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", m.get("category", na_cell()))
        c2.metric("Subcategory", m.get("subcategory", na_cell()))
        lp = m.get("list_price")
        c3.metric("List price",
                  fmt_num(lp) if pd.notna(lp) else na_cell())

    sales = tables.get("sales_daily")
    if sales is not None and state.get("sig"):
        weekly = fc.prepare_weekly_demand(tables)
        hist = (weekly[weekly["sku_id"].astype(str) == str(sku_id)]
                .groupby("period")["units_sold"].sum())
        if len(hist):
            st.subheader("Historical weekly demand")
            st.line_chart(hist)

    d4 = state.get("d4") or {}
    recs = [r for r in (d4.get("scored_skus", [])
                        if d4.get("status") == "READY" else [])
            if str(r.get("sku_id")) == str(sku_id)]
    if recs:
        r = recs[0]
        st.subheader("Inventory position & risk (D4)")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("On-hand units", fmt_int(r.get("on_hand_units")))
        g2.metric("On-order units", fmt_int(r.get("on_order_units")))
        g3.metric("Inventory position", fmt_int(r.get("inventory_position")))
        g4.metric("Lead time (days)", fmt_int(r.get("lead_time_days")))
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Lead-time demand", fmt_num(r.get("calculated_risk_threshold")))
        h2.metric("Coverage (weeks)", fmt_num(r.get("coverage_weeks")))
        h3.metric("Stockout risk", fmt_num(r.get("stockout_risk")))
        h4.metric("Overstock risk", fmt_num(r.get("overstock_risk")))
        v1, v2, v3 = st.columns(3)
        v1.metric("Volatility (CV)", fmt_num(r.get("volatility_cv")))
        v2.metric("Rupee value at stake", fmt_rupee(r.get("rupee_value_at_stake")))
        flag = r.get("data_quality_flag")
        v3.metric("Confidence", "Standard" if flag == "ok" else str(flag))
        st.success(f"Decision: **{r.get('decision')}** — "
                   f"{r.get('decision_reason','')}")
    else:
        st.info("No official D4 record for this SKU yet (data gated).")

    if state.get("sig") and (state.get("risk_readiness") or {}).get("status") == "READY":
        try:
            res = sku_forecast_cached(state["sig"], str(sku_id),
                                      fcst_cfg_tuple(DEFAULT_FCST_CONFIG),
                                      PROCESSED_DIR_STR)
            rows = res.get("forecast_rows") or []
            if rows:
                fdf = pd.DataFrame(rows)
                st.subheader("D3 forecast")
                st.dataframe(fdf[["period", "forecast_units", "model_name"]],
                             use_container_width=True, hide_index=True)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"D3 forecast unavailable for this SKU: {exc}")

# --------------------------------------------------------------------------- #
# Sidebar filters (operate on official data only)
# --------------------------------------------------------------------------- #


def _build_filters(tables: Dict[str, Any]) -> Dict[str, Any]:
    master = (tables or {}).get("sku_master")
    calendar = (tables or {}).get("calendar")

    skus = sorted(master["sku_id"].astype(str).unique()) if master is not None else []
    cats = sorted(master["category"].dropna().astype(str).unique()) if (
        master is not None and "category" in master.columns) else []
    subcats_all = (master["subcategory"].dropna().astype(str).unique().tolist()
                   if master is not None and "subcategory" in master.columns else [])

    with st.sidebar.expander("🔎 Filters", expanded=True):
        sku = st.selectbox("SKU", ["All"] + skus, key="flt_sku")
        category = st.selectbox("Category", ["All"] + cats, key="flt_cat")
        cat_mask = category if category != "All" else None
        sub_opts = sorted(subcats_all)
        subcategory = st.selectbox("Subcategory", ["All"] + sub_opts,
                                   key="flt_sub")
        decision = st.selectbox(
            "Decision", ["All", "REORDER_NOW", "MARKDOWN_CLEAR",
                         "WATCH_VOLATILE", "HEALTHY"], key="flt_dec")
        seasons = (sorted(calendar["season"].dropna().astype(str).unique())
                   if calendar is not None and "season" in calendar.columns else [])
        season = st.selectbox("Season", ["All"] + seasons, key="flt_season")
        promo = st.selectbox("Promotion state", ["All", "Promo", "No promo"],
                             key="flt_promo")

    return {
        "sku": None if sku == "All" else sku,
        "category": cat_mask,
        "subcategory": None if subcategory == "All" else subcategory,
        "decision": None if decision == "All" else decision,
        "season": None if season == "All" else season,
        "promo": None if promo == "All" else (1 if promo == "Promo" else 0),
    }


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(
        page_title="Project FORESIGHT — Demand & Inventory Intelligence",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📦 Project FORESIGHT")
    st.caption("Demand & Inventory Intelligence — data-gated on D1–D4 outputs.")
    st.warning(SYNTHETIC_DATA_MESSAGE)

    state = get_official_state()
    if state.get("load_error"):
        st.error("Official data could not be loaded.")
        st.caption(state["load_error"])
    if not state["available"]:
        info_state(extra=[
            "Expected official inputs under data/processed/: "
            "sales_daily_clean.csv · sku_master_clean.csv · "
            "calendar_clean.csv · inventory_snapshots_clean.csv",
            "This dashboard never reads legacy demo data or data/raw/.",
        ])

    filters = _build_filters(state.get("tables"))

    pages = {
        "Executive Overview": lambda: page_executive(state),
        "Data Quality": lambda: page_data_quality(state),
        "Demand Analysis": lambda: page_demand_analysis(state, filters),
        "Forecast": lambda: page_forecast(state, filters),
        "Model Performance": lambda: page_model_performance(state),
        "Inventory Risk": lambda: page_inventory_risk(state, filters),
        "Decisions": lambda: page_decisions(state, filters),
        "SKU Drill-down": lambda: page_drilldown(state, filters),
    }
    choice = st.sidebar.radio("Navigation", list(pages.keys()),
                              key="nav_pages")
    st.sidebar.divider()
    st.sidebar.caption(f"{PRIMARY_METRIC_LABEL}\n\n{SECONDARY_METRIC_LABEL}")

    try:
        pages[choice]()
    except Exception as exc:  # noqa: BLE001 - never crash the dashboard
        st.error("This section could not be rendered with the current data.")
        st.caption(f"Details: {exc}")

    st.divider()
    st.caption("Project FORESIGHT · D5 planning dashboard · all figures derive "
               "from official D1–D4 modules; unavailable results display "
               "'Not available'.")


if __name__ == "__main__":
    main()
