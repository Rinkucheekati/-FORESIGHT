# Synthetic Development Data

> **Development/Dummy Data - Synthetic data used temporarily until official Zidio extracts are supplied by the mentor. No business conclusions represent NorthBay Living.**

This directory contains a reproducible generator for local development and testing of the FORESIGHT D1-D7 workflow. It does not use the Kaggle dataset, download external data, or represent internship results.

## Generate the inputs

From the repository root:

```text
.venv\Scripts\python.exe dev_data\generate_dummy_data.py
```

The generator writes the four official-shaped input files to `data/raw/`:

- `sales_daily.csv`
- `sku_master.csv`
- `calendar.csv`
- `inventory_snapshots.csv`

The data contains 80 weeks of daily history and 30 SKUs across multiple categories and subcategories. Profiles include high demand, seasonal, volatile, slow/dead stock, overstock, and healthy inventory situations. A fixed seed makes regeneration deterministic.

## Important boundary

These files are synthetic development inputs only. Do not report their forecasts, metrics, risk scores, monetary values, or EDA findings as official Zidio or NorthBay Living results. When the mentor supplies the real extracts, replace these four files in `data/raw/` and rerun the unchanged official pipeline.
