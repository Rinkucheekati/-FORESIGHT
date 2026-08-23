"""
Mini-FORESIGHT — Streamlit Dashboard
====================================
Interactive dashboard that brings together all the outputs from the
Mini-FORESIGHT pipeline:

1. Executive Overview
2. Sales Analysis
3. Demand Forecast
4. Inventory Risk
5. Recommendations
6. Model Performance

Run with:
    streamlit run app/app.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mini-FORESIGHT Dashboard",
    page_icon="📦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@st.cache_data
def load_data():
    """Load all processed datasets used by the dashboard."""
    sales = pd.read_csv(PROCESSED_DIR / "sales_daily_clean.csv")
    sku_master = pd.read_csv(PROCESSED_DIR / "sku_master_clean.csv")
    inventory = pd.read_csv(PROCESSED_DIR / "inventory_snapshots_clean.csv")
    forecast = pd.read_csv(PROCESSED_DIR / "forecast_results.csv")
    risk = pd.read_csv(PROCESSED_DIR / "inventory_risk.csv")
    wape = pd.read_csv(PROCESSED_DIR / "wape_results.csv")
    recommendations = pd.read_csv(PROCESSED_DIR / "recommendations.csv")
    features = pd.read_csv(PROCESSED_DIR / "features.csv")

    # Convert dates to datetime
    sales["date"] = pd.to_datetime(sales["date"])
    inventory["date"] = pd.to_datetime(inventory["date"])
    forecast["date"] = pd.to_datetime(forecast["date"])
    features["date"] = pd.to_datetime(features["date"])

    return {
        "sales": sales,
        "sku_master": sku_master,
        "inventory": inventory,
        "forecast": forecast,
        "risk": risk,
        "wape": wape,
        "recommendations": recommendations,
        "features": features,
    }


data = load_data()

sales = data["sales"]
sku_master = data["sku_master"]
inventory = data["inventory"]
forecast = data["forecast"]
risk = data["risk"]
wape = data["wape"]
recommendations = data["recommendations"]
features = data["features"]


def compute_overall_wape():
    """Recompute the overall WAPE exactly as in notebook 06.

    Uses the same chronological 70/30 split and Random Forest settings,
    then computes WAPE across all test rows combined (not the mean of
    per-SKU WAPEs).
    """
    feature_columns = [
        "lag_1", "lag_2", "lag_3",
        "rolling_mean_3", "rolling_mean_7",
        "day_of_week", "is_weekend",
    ]
    target = "units_sold"

    feats = features.sort_values(["sku_id", "date"]).reset_index(drop=True)

    all_actual = []
    all_ml = []
    all_naive = []

    for sku in sorted(feats["sku_id"].unique()):
        sku_df = feats[feats["sku_id"] == sku].sort_values("date").reset_index(drop=True)
        split_idx = int(len(sku_df) * 0.7)
        train_sku = sku_df.iloc[:split_idx]
        test_sku = sku_df.iloc[split_idx:]

        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        model.fit(train_sku[feature_columns], train_sku[target])

        actual = test_sku[target].values
        ml_pred = model.predict(test_sku[feature_columns])

        # Naive baseline: tomorrow = today. Compute on the full SKU series
        # BEFORE splitting so the first test row uses the last training value.
        sku_df["naive"] = sku_df[target].shift(1)
        naive_pred = sku_df.iloc[split_idx:]["naive"].values

        all_actual.extend(actual)
        all_ml.extend(ml_pred)
        all_naive.extend(naive_pred)

    all_actual = np.array(all_actual)
    all_ml = np.array(all_ml)
    all_naive = np.array(all_naive)

    ml_wape = (np.sum(np.abs(all_actual - all_ml)) / np.sum(all_actual)) * 100
    naive_wape = (np.sum(np.abs(all_actual - all_naive)) / np.sum(all_actual)) * 100

    return ml_wape, naive_wape


overall_ml_wape, overall_naive_wape = compute_overall_wape()

# ---------------------------------------------------------------------------
# Derived values for KPI cards
# ---------------------------------------------------------------------------
total_skus = sku_master["sku_id"].nunique()
high_risk_count = int((risk["risk_level"] == "High").sum())
total_reorder_qty = recommendations["recommended_reorder_qty"].sum()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("📦 Mini-FORESIGHT Dashboard")
st.caption("A beginner-friendly retail demand forecasting and inventory project")

# ===========================================================================
# 1. Executive Overview
# ===========================================================================
st.header("📊 Executive Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total SKUs", total_skus)
col2.metric("ML WAPE", f"{overall_ml_wape:.1f}%")
col3.metric("High Risk SKUs", high_risk_count)
col4.metric("Total Reorder Qty", f"{total_reorder_qty:.0f} units")

st.divider()

# ===========================================================================
# 2. Sales Analysis
# ===========================================================================
st.header("📈 Sales Analysis")

# Daily sales trend
daily_total = sales.groupby("date")["units_sold"].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily_total["date"], daily_total["units_sold"], marker="o")
ax.set_title("Daily Sales Trend (All SKUs)")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
plt.xticks(rotation=45)
st.pyplot(fig)

# Sales by SKU
sales_by_sku = (
    sales.groupby("sku_id")["units_sold"]
    .sum()
    .reset_index()
    .merge(sku_master[["sku_id", "product_name"]], on="sku_id")
)
sales_by_sku["label"] = sales_by_sku["sku_id"] + " - " + sales_by_sku["product_name"]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(sales_by_sku["label"], sales_by_sku["units_sold"])
ax.set_title("Total Sales by SKU")
ax.set_xlabel("SKU")
ax.set_ylabel("Total Units Sold")
plt.xticks(rotation=15)
st.pyplot(fig)

# Sales by weekday
sales_weekday = sales.copy()
sales_weekday["weekday"] = sales_weekday["date"].dt.day_name()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_sales = (
    sales_weekday.groupby("weekday")["units_sold"]
    .sum()
    .reindex(weekday_order)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(weekday_sales["weekday"], weekday_sales["units_sold"])
ax.set_title("Sales by Weekday")
ax.set_xlabel("Weekday")
ax.set_ylabel("Total Units Sold")
plt.xticks(rotation=30)
st.pyplot(fig)

st.divider()

# ===========================================================================
# 3. Demand Forecast
# ===========================================================================
st.header("🔮 Demand Forecast")
st.caption("3-day machine-learning forecast (2025-01-15 to 2025-01-17)")

forecast_pivot = forecast.pivot(index="date", columns="sku_id", values="forecast_units")
forecast_pivot = forecast_pivot.round(2)
forecast_pivot.index = forecast_pivot.index.strftime("%b %d")

st.dataframe(forecast_pivot, use_container_width=True)

st.divider()

# ===========================================================================
# 4. Inventory Risk
# ===========================================================================
st.header("⚠️ Inventory Risk")

risk_display = risk[
    ["sku_id", "product_name", "current_stock", "days_of_stock", "risk_level"]
].copy()
risk_display["days_of_stock"] = risk_display["days_of_stock"].round(2)
risk_display.columns = ["SKU", "Product", "Stock", "Coverage (days)", "Risk"]

st.dataframe(risk_display, use_container_width=True)

st.divider()

# ===========================================================================
# 5. Recommendations
# ===========================================================================
st.header("🛒 Recommendations")
st.caption("Reorder recommendations based on forecast, lead time, and safety stock")

for _, row in recommendations.iterrows():
    urgency = row["urgency"]
    if urgency == "High":
        emoji = "🔴"
    elif urgency == "Medium":
        emoji = "🟠"
    else:
        emoji = "🟢"

    with st.container(border=True):
        st.subheader(f"{emoji} {row['sku_id']} | {row['product_name']} — {urgency.upper()} URGENCY")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Stock", f"{row['current_stock']:.0f}")
        c2.metric("Reorder Quantity", f"{row['recommended_reorder_qty']:.2f}")
        c3.metric("Action", row["recommendation"])

st.divider()

# ===========================================================================
# 6. Model Performance
# ===========================================================================
st.header("📏 Model Performance")
st.caption("Overall WAPE comparison: ML model vs naive baseline")

fig, ax = plt.subplots(figsize=(8, 4))
methods = ["ML Model", "Naive Baseline"]
values = [overall_ml_wape, overall_naive_wape]
bars = ax.bar(methods, values, color=["#2e86de", "#e67e22"])
ax.set_title("Overall WAPE: ML Model vs Naive Baseline")
ax.set_ylabel("WAPE (%)")
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
    )
st.pyplot(fig)

st.caption(
    "The ML model reduces overall WAPE from "
    f"{overall_naive_wape:.1f}% to {overall_ml_wape:.1f}%, "
    "confirming the engineered features add value over a simple naive baseline."
)

st.divider()

st.caption("Mini-FORESIGHT — end-to-end demand forecasting & inventory project")