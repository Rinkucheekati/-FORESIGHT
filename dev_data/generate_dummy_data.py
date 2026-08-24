"""Generate the synthetic FORESIGHT development dataset.

This module is intentionally self-contained: it uses no external files,
network calls, or internship data. It writes the four official CSV schemas to
``data/raw`` by default so the files are drop-in development inputs only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Support both ``python -m dev_data.generate_dummy_data`` and the documented
# direct invocation from the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import DATA_RAW, OFFICIAL_SCHEMAS

SEED = 20260824
WEEKS = 80
N_SKUS = 30


def _sku_master() -> pd.DataFrame:
    categories = {
        "Home": ["Kitchen", "Bedding", "Decor"],
        "Apparel": ["Basics", "Outerwear", "Accessories"],
        "Personal Care": ["Bath", "Hair", "Wellness"],
        "Electronics": ["Audio", "Mobile", "Accessories"],
    }
    rows = []
    category_names = list(categories)
    for index in range(N_SKUS):
        category = category_names[index % len(category_names)]
        subcategory = categories[category][(index // len(category_names)) % 3]
        list_price = round(12 + (index % 10) * 8.5 + (index // 10) * 3.25, 2)
        rows.append(
            {
                "sku_id": f"SKU-{index + 1:03d}",
                "category": category,
                "subcategory": subcategory,
                "launch_date": (pd.Timestamp("2023-10-01") + pd.Timedelta(days=index * 2)).date(),
                "unit_cost": round(list_price * (0.42 + (index % 5) * 0.035), 2),
                "list_price": list_price,
            }
        )
    return pd.DataFrame(rows)


def _calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    holidays = {pd.Timestamp("2024-02-14"), pd.Timestamp("2024-05-27"),
                pd.Timestamp("2024-07-04"), pd.Timestamp("2024-11-29"),
                pd.Timestamp("2024-12-25"), pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-05-26"), pd.Timestamp("2025-07-04"),
                pd.Timestamp("2025-11-28"), pd.Timestamp("2025-12-25")}
    promo_dates = set()
    for start in dates[::56]:
        promo_dates.update(pd.date_range(start, periods=5, freq="D"))
    rows = []
    for date in dates:
        iso = date.isocalendar()
        month = date.month
        season = "Winter" if month in (12, 1, 2) else "Spring" if month in (3, 4, 5) else "Summer" if month in (6, 7, 8) else "Autumn"
        rows.append({
            "date": date.date(),
            "week": int(iso.week),
            "month": month,
            "season": season,
            "is_holiday": int(date in holidays),
            "promo_event": "Seasonal promotion" if date in promo_dates else "",
        })
    return pd.DataFrame(rows)


def generate_dataset(output_dir: Path = DATA_RAW) -> Dict[str, Path]:
    """Generate and write all four official-shaped development extracts."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=WEEKS * 7, freq="D")
    master = _sku_master()
    calendar = _calendar(dates)
    promo_by_date = calendar.set_index("date")["promo_event"].ne("")
    holiday_by_date = calendar.set_index("date")["is_holiday"].astype(bool)

    sales_rows = []
    inventory_rows = []
    for index, sku in master.iterrows():
        profile = (
            "high_demand" if index < 6 else
            "seasonal" if index < 12 else
            "volatile" if index < 18 else
            "slow_stock" if index < 24 else
            "overstock" if index < 27 else "healthy"
        )
        base = {"high_demand": 34, "seasonal": 18, "volatile": 22,
            "slow_stock": 0.5, "overstock": 9, "healthy": 13}[profile]
        lead_time = int(3 + index % 10)
        reorder_point = int(np.ceil(base * lead_time * 1.25))
        inventory_level = int(base * ({"high_demand": 0.35, "seasonal": 1.6,
                                       "volatile": 0.8, "slow_stock": 100,
                                       "overstock": 70, "healthy": 5}[profile]))

        for day_number, date in enumerate(dates):
            weekday_factor = [0.82, 0.9, 0.96, 1.0, 1.12, 1.28, 1.18][date.weekday()]
            annual_factor = 1 + 0.18 * np.sin(2 * np.pi * day_number / 365.25)
            seasonal_factor = (1.45 if profile == "seasonal" and date.month in (11, 12)
                               else 0.72 if profile == "seasonal" and date.month in (6, 7)
                               else 1.0)
            promo = int(bool(promo_by_date.get(date.date(), False)))
            holiday = int(bool(holiday_by_date.get(date.date(), False)))
            event_factor = 1.25 if promo else 1.0
            event_factor *= 1.3 if holiday else 1.0
            volatility = 1.0 if profile == "volatile" else 0.12
            expected = max(0.05, base * weekday_factor * annual_factor * seasonal_factor * event_factor)
            if profile == "volatile" and rng.random() < 0.15:
                expected *= float(rng.uniform(0.05, 3.5))
            units = int(rng.poisson(expected * float(rng.lognormal(-volatility**2 / 2, volatility))))
            price = round(float(sku["list_price"]) * (0.85 if promo else 1.0), 2)
            sales_rows.append({
                "date": date.date(), "sku_id": sku["sku_id"],
                "units_sold": units, "revenue": round(units * price, 2),
                "unit_price": price, "promo_flag": promo,
            })

            if profile == "high_demand":
                on_hand = 0 if (day_number + index) % 17 in (0, 1) else int(rng.integers(1, 12))
                on_order = int(rng.integers(0, 35))
            elif profile == "overstock":
                on_hand = inventory_level + int(rng.integers(-8, 18))
                on_order = int(rng.integers(20, 80))
            elif profile == "slow_stock":
                on_hand = inventory_level + int(rng.integers(-3, 5))
                on_order = int(rng.integers(0, 8))
            elif profile == "volatile":
                on_hand = max(0, int(base * rng.uniform(0.2, 2.0)))
                on_order = int(rng.integers(0, 45))
            else:
                on_hand = max(0, int(inventory_level + rng.normal(0, max(2, base * 0.25))))
                on_order = int(rng.integers(0, max(5, int(base * 2))))
            inventory_rows.append({
                "date": date.date(), "sku_id": sku["sku_id"],
                "on_hand_units": on_hand, "on_order_units": on_order,
                "lead_time_days": lead_time, "reorder_point": reorder_point,
            })

    tables = {
        "sales_daily": pd.DataFrame(sales_rows),
        "sku_master": master,
        "calendar": calendar,
        "inventory_snapshots": pd.DataFrame(inventory_rows),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        table = table.loc[:, OFFICIAL_SCHEMAS[name]]
        destination = output_dir / f"{name}.csv"
        table.to_csv(destination, index=False)
        paths[name] = destination
    return paths


if __name__ == "__main__":
    for name, path in generate_dataset().items():
        print(f"{name}: {path}")
