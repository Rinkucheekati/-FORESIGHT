# 📦 Mini-FORESIGHT

### Retail Demand Forecasting & Inventory Optimization

Mini-FORESIGHT is an end-to-end machine learning project that forecasts short-term product demand and converts those forecasts into actionable inventory reorder recommendations.

The project demonstrates a complete data science workflow:

Raw Data → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Demand Forecasting → Model Evaluation → Inventory Risk Analysis → Reorder Recommendations → Streamlit Dashboard

---

## 🎯 Project Overview

Retail businesses need to maintain enough inventory to satisfy customer demand without holding excessive stock.

If inventory is too low:

- Products may go out of stock
- Sales opportunities may be lost
- Customer satisfaction can decrease

If inventory is too high:

- Capital is tied up in inventory
- Storage costs increase
- Products may remain unsold

Mini-FORESIGHT addresses this problem by combining historical sales data, machine learning forecasting, inventory information, and reorder calculations.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze historical daily sales.
2. Clean and prepare the datasets.
3. Understand sales patterns using exploratory data analysis.
4. Create time-series features for forecasting.
5. Build a Random Forest demand forecasting model.
6. Generate a 3-day demand forecast for each SKU.
7. Compare the ML model with a naive forecasting baseline.
8. Evaluate forecasting performance using WAPE.
9. Identify inventory risk based on stock coverage and lead time.
10. Calculate recommended reorder quantities.
11. Present the results through an interactive Streamlit dashboard.

---

## 📊 Dataset

The project contains three main raw datasets.

### 1. Sales Data

`data/raw/sales_daily.csv`

Contains daily sales for each product.

| Column | Description |
|---|---|
| date | Sales date |
| sku_id | Product identifier |
| units_sold | Number of units sold |

The dataset contains:

- 14 days
- 3 SKUs
- 42 records

---

### 2. SKU Master

`data/raw/sku_master.csv`

Contains product information.

| Column | Description |
|---|---|
| sku_id | Product identifier |
| product_name | Product name |
| category | Product category |
| unit_price | Product price |
| lead_time_days | Supplier replenishment time |

Products:

| SKU | Product |
|---|---|
| SKU001 | Wooden Chair |
| SKU002 | Table Lamp |
| SKU003 | Cushion Set |

---

### 3. Inventory Snapshots

`data/raw/inventory_snapshots.csv`

Contains daily inventory information.

| Column | Description |
|---|---|
| date | Inventory date |
| sku_id | Product identifier |
| opening_stock | Stock at beginning of day |
| units_received | Units received |
| units_sold | Units sold |
| closing_stock | Stock at end of day |

Inventory consistency was verified using:

`closing_stock = opening_stock + units_received - units_sold`

---

# 🔄 Project Workflow

## Step 1 — Data Understanding

The raw datasets were inspected for:

- Dataset dimensions
- Data types
- Missing values
- Duplicate records
- SKU consistency
- Date ranges
- Sales distributions

---

## Step 2 — Data Cleaning

The datasets were cleaned and stored in:

`data/processed/`

Cleaning included:

- Date conversion
- Sorting
- Duplicate checks
- Missing-value checks
- Data consistency validation
- Cross-dataset validation

---

## Step 3 — Exploratory Data Analysis

The project analyzes:

- Daily sales trends
- Sales by SKU
- Sales by weekday
- SKU-wise weekday demand
- Inventory levels
- Relationship between opening stock and sales

The best sales day was:

**January 12, 2025 — 13 units**

The worst sales day was:

**January 2, 2025 — 8 units**

Total historical sales by SKU:

| SKU | Product | Units Sold |
|---|---|---:|
| SKU001 | Wooden Chair | 29 |
| SKU002 | Table Lamp | 50 |
| SKU003 | Cushion Set | 59 |

---

# ⚙️ Feature Engineering

The forecasting model uses the following features:

- `lag_1`
- `lag_2`
- `lag_3`
- `rolling_mean_3`
- `rolling_mean_7`
- `day_of_week`
- `is_weekend`

### Lag Features

Lag features represent previous demand.

For example:

`lag_1` = yesterday's sales

`lag_2` = sales two days ago

`lag_3` = sales three days ago

### Rolling Features

`rolling_mean_3` represents recent average demand over the previous 3 days.

`rolling_mean_7` represents recent average demand over the previous 7 days.

These features help the model understand recent demand behavior.

---

# 🤖 Demand Forecasting

A **Random Forest Regressor** was used for demand forecasting.

The dataset was split chronologically rather than randomly to preserve the time-series nature of the problem.

The model generated a 3-day forecast for:

**January 15 → January 17, 2025**

### Forecast

| SKU | Jan 15 | Jan 16 | Jan 17 |
|---|---:|---:|---:|
| SKU001 | 2.16 | 1.92 | 2.27 |
| SKU002 | 3.89 | 3.50 | 3.96 |
| SKU003 | 4.44 | 3.99 | 4.29 |

---

# 📏 Model Evaluation

The ML model was compared against a naive baseline.

The evaluation metric used was:

**WAPE — Weighted Absolute Percentage Error**

### Results

| Model | Overall WAPE |
|---|---:|
| Random Forest | **24.3%** |
| Naive Baseline | **40.0%** |

The Random Forest model reduced WAPE from **40.0% to 24.3%**.

This indicates that the engineered features and ML model performed better than the simple naive baseline on the available test data.

> Note: The dataset is intentionally small and contains only 14 days of history. Therefore, the results are primarily educational and should not be interpreted as production-level model performance.

---

# ⚠️ Inventory Risk Analysis

Inventory risk was calculated using:

- Current stock
- Historical average demand
- Lead time
- Lead-time demand
- Days of stock coverage
- Forecast demand

### Current Inventory

| SKU | Product | Current Stock |
|---|---|---:|
| SKU001 | Wooden Chair | 11 |
| SKU002 | Table Lamp | 5 |
| SKU003 | Cushion Set | 6 |

All three SKUs were classified as **High Risk**.

SKU002 has the lowest stock coverage:

**1.40 days**

and the highest lead-time demand:

**17.86 units**

---

# 🛒 Reorder Recommendations

A simple safety-stock strategy was used.

### Safety Stock

Safety stock was calculated as:

`Safety Stock = Lead-Time Demand × 20%`

### Reorder Point

`Reorder Point = Lead-Time Demand + Safety Stock`

### Recommended Reorder Quantity

`Reorder Quantity = Reorder Point - Current Stock`

The calculated recommendations are:

| SKU | Product | Current Stock | Reorder Qty | Urgency |
|---|---|---:|---:|---|
| SKU002 | Table Lamp | 5 | **16.43** | 🔴 High |
| SKU003 | Cushion Set | 6 | **9.17** | 🔴 High |
| SKU001 | Wooden Chair | 11 | **6.40** | 🔴 High |

Total recommended reorder quantity:

**32 units**

---

# 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard contains six sections:

### 1. Executive Overview

Displays:

- Total SKUs
- ML WAPE
- High-risk SKUs
- Total reorder quantity

### 2. Sales Analysis

Displays:

- Daily sales trend
- Sales by SKU
- Sales by weekday

### 3. Demand Forecast

Displays the 3-day machine-learning forecast.

### 4. Inventory Risk

Displays:

- Current stock
- Stock coverage
- Risk level

### 5. Recommendations

Displays SKU-level reorder recommendations and urgency.

### 6. Model Performance

Compares:

- ML model WAPE
- Naive baseline WAPE

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Git
- GitHub

---

# 📁 Project Structure

```text
mini-foresight/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_forecasting.ipynb
│   ├── 06_baseline_wape.ipynb
│   ├── 07_inventory_risk.ipynb
│   └── 08_recommendations.ipynb
│
├── requirements.txt
├── README.md
└── .gitignoreect
